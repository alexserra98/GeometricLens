from dadapy.data import Data
from dadapy.plot import plot_inf_imb_plane
from dadapy.metric_comparisons import MetricComparisons
import numpy as np
import torch
from helm.benchmark.adaptation.scenario_state import ScenarioState
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable, Set, Type
from einops import reduce
import einsum
import torch.nn.functional as F
import functools
from sklearn.neighbors import NearestNeighbors
from .utils import compute_id, hidden_states_collapse, RunMeta, InstanceHiddenStates, Match, Layer, neig_overlap
from utils_nobatch import exact_match, quasi_exact_match
import tqdm
from generation import RequestResult
import pandas as pd
import warnings

#TODO use a factory to create RunMetrics objects

# Define a type hint 
Array = Type[np.ndarray]
Tensor = Type[torch.Tensor]

@dataclass
class InstanceResult():
  dataset: str
  train_instances: int
  intrinsic_dim: Dict[str, Dict[str, np.ndarray]]
  metrics: Dict[str, Dict[str, float]]
  letter_nn: Optional[Dict[str, Dict[str, np.ndarray]]] = None
  
  
class ShotMetrics():
  def __init__(self, requests_results: List[RequestResult], dataset:str, train_instances: int, model:str):
    self.requests_results = requests_results
    self.train_instances = train_instances
    self.dataset = dataset
    self.model = model
    self.basic_metric, self.hidden_states = self.set_dataframes()

  def set_dataframes(self):
    """
    Compute the per instance metric
    Output
    ----------
    List[Dict[metric, value]]
    """
    basic_metrics_dict = {"loss":[], "perplexity":[], "std_exact_match":[],"std_quasi_exact_match":[], "ref_exact_match":[]}
    hidden_states_dict = {"hidden_states": [],"layer": [], "match": [], "answered_letter": [], "gold_letter": []}
    for request_result in self.requests_results:
      basic_metrics_dict["loss"].append(request_result.loss)
      basic_metrics_dict["perplexity"].append(np.exp(request_result.loss))
      basic_metrics_dict["std_exact_match"].append(exact_match(request_result.preds["std_pred"]["letter"], request_result.gold["letter"]))
      basic_metrics_dict["ref_exact_match"].append(exact_match(request_result.preds["only_ref_pred"]["letter"],request_result.gold["letter"]))
      basic_metrics_dict["std_quasi_exact_match"].append(quasi_exact_match(request_result.preds["std_pred"]["letter"], request_result.gold["letter"]))
      for layer in ["last","sum"]:
        hidden_states_dict["hidden_states"].append(request_result.hidden_states[layer])
        hidden_states_dict["layer"].append(layer)
        match = "correct" if basic_metrics_dict["std_exact_match"][-1] else "wrong"
        hidden_states_dict["match"].append(match)
        hidden_states_dict["answered_letter"].append(request_result.preds["only_ref_pred"]["letter"])
        hidden_states_dict["gold_letter"].append(request_result.gold["letter"])
    basic_metrics = pd.DataFrame(basic_metrics_dict)
    hidden_states = pd.DataFrame(hidden_states_dict)
    return basic_metrics, hidden_states
  def evaluate(self): 
    basic_metric = self.basic_metric_mean()
    hidden_states = self.construct_hidden_states()
    intrinsic_dim = self.intrinsic_dim(hidden_states)
    if self.dataset == "commonsenseqa" and "llama" in self.model:
      letter_overlap = self.get_all_letter_overlaps(hidden_states) 
      return InstanceResult(self.dataset, self.train_instances, intrinsic_dim, basic_metric, letter_overlap)
    return InstanceResult(self.dataset, self.train_instances, intrinsic_dim, basic_metric)
    
  #TODO use property decorator
  def basic_metric_mean(self):
    output_dict = {column_name: self.basic_metric[column_name].mean() for column_name in self.basic_metric.columns}
    return output_dict  
  def construct_hidden_states(self):
    hidden_states = HiddenStates(self.hidden_states)
    return hidden_states
  def intrinsic_dim(self,hidden_states):
    intrinsic_dim = hidden_states.get_instances_id()
    return intrinsic_dim 
  def get_all_letter_overlaps(self,hidden_states):
    return hidden_states.layer_overlap_letter()

  def hidden_states_metrics(self):
    hidden_states = self.construct_hidden_states()
    intrinsic_dim = self.intrinsic_dim(hidden_states)
    letter_overlap = self.get_all_letter_overlaps(hidden_states)
    return intrinsic_dim, letter_overlap 

  def instance_result(self):
    intrinsic_dim = self.construct_hidden_states().get_instances_id()
    metrics = self.basic_metric_mean()
    return InstanceResult(intrinsic_dim, metrics)

class HiddenStates():
  def __init__(self,hidden_states):
    self.hidden_states = hidden_states
  
  #TODO use a decorator
  def preprocess_hiddenstates(self,func, query, *args, **kwargs):
    hidden_states_collapsed = hidden_states_collapse(self.hidden_states, query)

    return func(hidden_states_collapsed, *args, **kwargs)

  def set_hidden_states(self, exact_matches) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Output
    ----------
    {method: method(num_instances, num_layers, model_dim)}
    """ 
    instances_hiddenstates = self._get_instances_hidden_states(exact_matches) 
    hidden_states = {match.value: {layer.value: hidden_states_collapse(instances_hiddenstates, {"layer":layer.value,"match": match.value}) 
                            for layer in [Layer.LAST,Layer.SUM] } 
                            for match in [Match.CORRECT, Match.WRONG, Match.ALL]}
    return hidden_states
  
  def get_instances_id(self) -> np.ndarray:
    """
    Compute the ID of all instances using gride algorithm
    Output
    ----------
    Dict[str, np.array(num_layers)]
    """
    id = {match.value: 
            {layer.value: compute_id(self.hidden_states,{"match":match.value,"layer":layer.value}, "gride") 
            for layer in [Layer.LAST, Layer.SUM]} 
            for match in [Match.CORRECT, Match.WRONG, Match.ALL]}
    return id
  
  def nearest_neighbour(self, hidden_states, k: int ) -> np.ndarray:
    """
    Compute the nearest neighbours of each instance in the run per layer
    using the provided methodv
    Output
    ----------
    Array(num_layers, num_instances, k_neighbours)
    """

    assert k <= hidden_states.shape[0], "K must be smaller than the number of instances"
    layers = hidden_states.shape[1]
    neigh_matrix_list = []
    for i in range(layers):
      neigh = NearestNeighbors(n_neighbors=k)
      neigh.fit(hidden_states[:,i,:])
      dist, indices = neigh.kneighbors(hidden_states[:,i,:])
      indices = np.delete(indices, 0, 1) # removing the first column which is the instance itself
      neigh_matrix_list.append(indices)
    
    return np.stack(neigh_matrix_list)
  
  def layer_overlap_letter(self):
    nn = []
    nn_half1 = []
    nn_half2 = []
    
    half = True
    for letter in ["A","B","C","D"]:
      k = max(int(half*0.8),2)
      hidden_states = hidden_states_collapse(self.hidden_states,
                                             {"match":Match.ALL.value, 
                                              "layer":Layer.LAST.value, 
                                              "answered_letter":letter})
      nn.append(self.nearest_neighbour(hidden_states, k=k))
      half = int(hidden_states.shape[0]/2)
      nn_half1.append(self.nearest_neighbour(hidden_states[:half], k=k))
      nn_half2.append(self.nearest_neighbour(hidden_states[half:], k=k))
    overlap = np.empty([4,4])
    warnings.warn("Computing overlap for -5th layer and using 80 %% of the instances")
    for i in tqdm.tqdm(range(4), desc = "Computing overlap"):
      for j in range (4):
        if i==j:
          overlap[i,j] = neig_overlap(nn_half1[i][-5,], nn_half2[j][-5])
        else:
          overlap[i,j] = neig_overlap(nn[i][-5,], nn[j][-5])
    
    return overlap

  
      
class DatasetMetrics():
  """
  Geometry stores all the runs in a version and collect methods to compute geometric information of the representations
  across runs
  Methods
  ----------
  _instances_overlap: compute the overlap between two representations
  neig_overlap: compute the overlap between two representations
  """

  def __init__(self, instances_result: List[InstanceResult] ) -> None:
    """
    Parameters
    ----------
    nearest_neig : List[Dict]
        List of dictionaries containing the nearest neighbours of each layer of the two runs
        Dict[k-method, Array(num_layers, num_instances, k_neighbours)]
    """
    self.instances_results = instances_result 

  def get_last_layer_id_diff(self):
    """
    Compute difference in last layer ID among 0-shot and few-shot
    Output
    ----------
    Dict[k-method, Dict[accuracy, Array(neigh_order)]]
    """
    num_runs = len(self.instances_results)
    metric_id = {}
    zero_shot = next((x for x in self.instances_results if x.train_instances==0), None)
    
    for method in [Layer.LAST.value, Layer.SUM.value]:
        metric_id[method] = {"id_diff": [], "metric_diff": {key: [] for key in self.instances_results[0].metrics.keys()}} 
        for i_result in self.instances_results:
          if i_result.train_instances == 0:
            continue
          zero_shot_id = zero_shot.intrinsic_dim[Match.ALL.value][method]
          i_result_id = i_result.intrinsic_dim[Match.ALL.value][method]
          nn_order = min(zero_shot.shape[1], i_result_id.shape[1])
          id_diff = np.abs(zero_shot_id[-5][:nn_order] - i_result_id[-5][:nn_order])
          metric_id[method]["id_diff"].append(id_diff)
          for metric in i_result.metrics.keys():
            metric_id[method]["metric_diff"][metric].append(np.abs(zero_shot.metrics[metric] - i_result.metrics[metric]))
    return metric_id

  def get_all_overlaps(self) -> Dict:
    """
    Compute the overlap between all the runs
    Output
    ----------
    Dict[k-method, np.ndarray[i,j, Array(num_layers, num_layers)]]
    """
    overlaps = {}
    num_runs = len(self.nearest_neig)
    num_layers = self.nearest_neig[0]["last"].shape[0]
    for method in ["last", "sum"]:
      overlaps[method] = {}
      desc = "Computing overlap for method " + method
      for i in tqdm.tqdm(range(num_runs-1), desc = desc):
        for j in range(i+1,num_runs):
          # indexes are not 0-based because it's easier to mange with plolty sub_trace
          overlaps[method][(i+1,j+1)] = self._instances_overlap(i,j,method)
    return overlaps
    
  def _instances_overlap(self, i,j,method) -> np.ndarray:
    nn1 = self.nearest_neig[i][method]
    nn2 = self.nearest_neig[j][method]
    assert nn1.shape == nn2.shape, "The two nearest neighbour matrix must have the same shape" 
    layers_len = nn1.shape[0]
    overlaps = np.empty([layers_len, layers_len])
    for i in range(layers_len):
      for j in range(layers_len):
        # WARNING : the overlap is computed with K=K THE OTHER OCC IS IN RUNGEOMETRY
        overlaps[i][j] = neig_overlap(nn1[i], nn2[j])
    return overlaps
  
  

        
#class InstanceMetrics():
#  """
#  RunGeometry stores the hidden states of all instances with some metadata
#  and collect methods to compute geometric information of the representations
#  
#  Methods
#  ----------
#  _get_instances_hidden_states: collect hidden states of all instances
#  _get_instances_id: compute the ID of all instances
#  hidden_states_collapse: collect hidden states of all instances and collapse them in one tensor
#  nearest_neighbour: compute the nearest neighbours of each instance in the run per layer
#  """
#  def __init__(self, per_instance_data: List[Dict]):
#    self._per_instance_data = per_instance_data 
#    #self._instances_hiddenstates = self._get_instances_hidden_states(raw_hidden_states,raw_metrics)
#    #self.hidden_states = self.set_hidden_states(self._instances_hiddenstates)
#    #self.run_meta = self._set_run_meta()
#    #self._metrics = self.set_metrics(raw_metrics)
#    #self._instances_id = self.instances_id_set()
#    #self._dict_nn = self.set_dict_nn(k=int(self.run_meta.num_instances*0.8))
#    
#  def compute_metrics(self):
#    metrics = ["loss", "perplexity", "std_exact_match", "std_quasi_exact_match", "ref_exact_match"]
#    averages = {metric: sum(instance[metric] for instance in self._per_instance_data) / len(self._per_instance_data) for metric in metrics}
#    return averages 
#  
#  def _set_run_meta(self):
#    num_layers = self.hidden_states["all"]["last"].shape[1]
#    num_instances = self.hidden_states["all"]["last"].shape[0]
#    model_dim = self.hidden_states["all"]["last"].shape[2]
#    return RunMeta(num_layers, num_instances, model_dim)
#  
#  def nearest_neighbour(self, method: str, k: int) -> np.ndarray:
#    """
#    Compute the nearest neighbours of each instance in the run per layer
#    using the provided methodv
#    Output
#    ----------
#    Array(num_layers, num_instances, k_neighbours)
#    """
#    hidden_states = self.set_hidden_states()
#    hidden_states = hidden_states["all"][method]
#    assert k <= hidden_states.shape[0], "K must be smaller than the number of instances"
#    layers = self.run_meta.num_layers
#    neigh_matrix_list = []
#    for i in range(layers):
#      neigh = NearestNeighbors(n_neighbors=k)
#      neigh.fit(hidden_states[:,i,:])
#      dist, indices = neigh.kneighbors(hidden_states[:,i,:])
#      indices = np.delete(indices, 0, 1) # removing the first column which is the instance itself
#      neigh_matrix_list.append(indices)
#    
#    return np.stack(neigh_matrix_list)



