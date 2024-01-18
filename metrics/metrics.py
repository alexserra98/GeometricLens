import numpy as np
from pandas.core.api import DataFrame as DataFrame
import torch

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional,  Type
from .utils import  Match, Layer, neig_overlap, exact_match, quasi_exact_match, layer_overlap

import tqdm
from inference_id.generation.generation import  ScenarioResult
import pandas as pd

from .hidden_states import HiddenStates
from enum import Enum
from abc import ABC, abstractmethod

from warnings import warn

#Define a type hint 
Array = Type[np.ndarray]
Tensor = Type[torch.Tensor]

class Match(Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    ALL = "all"
class Layer(Enum):
    LAST = "last"
    SUM = "sum"
    
@dataclass
class InstanceResult():
  dataset: str
  train_instances: int
  intrinsic_dim: Dict[str, Dict[str, np.ndarray]]
  metrics: Dict[str, Dict[str, float]]
  label_nn: Optional[Dict[str, Dict[str, np.ndarray]]] = None
  
  
class ShotMetrics():
  """
  Class to compute the metrics of a run. It takes a list of request results computed by generate.py
  and compute the per instance metrics 
  """
  def __init__(self, scenario_result: ScenarioResult):
    self.requests_results = scenario_result.requests_results
    self.train_instances = scenario_result.train_instances
    self.dataset = scenario_result.dataset
    self.model = scenario_result.model_name
    self.basic_metric, self.hidden_states = self.set_dataframes()

  def set_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate in two different dataframes the data for basic metrics and the hidden states of all instances
    Output
    ----------
    basic_metrics: pd.DataFrame(num_instances, num_metrics)
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
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
  
  def evaluate(self) -> InstanceResult: 
    """
    Compute all the implemented metrics
    Output
    ----------
    InstanceResult object
    """
    basic_metric = self.basic_metric_mean()
    hidden_states = HiddenStates(self.hidden_states)
    intrinsic_dim = hidden_states.get_instances_id()
    return InstanceResult(self.dataset, self.train_instances, intrinsic_dim, basic_metric)
  def compute_nn(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute the nearest neighbours of each instance in the run per layer
    using the provided methodv
    Output
    ----------
    Dict[k-method, Array(num_layers, num_instances, k_neighbours)]
    """
    hidden_states = HiddenStates(self.hidden_states)
    k = 20
    warn("Computing nearest neighbours with k=20")
    return hidden_states.get_nearest_neighbour(k) 
  #TODO use property decorator
  def basic_metric_mean(self) -> Dict[str, float]:
    output_dict = {column_name: self.basic_metric[column_name].mean() for column_name in self.basic_metric.columns}
    return output_dict  


class LabelOverlap(ABC):
  def __init__(self, scenario_results: List[ScenarioResult], label: str) -> None:
    """
    Compute overlap between MMLU subjects
    """
    self.scenario_results = scenario_results
    self.hidden_states = self.set_dataframes()
    self.label = label
  
  @abstractmethod
  def set_dataframes(self) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    pass
  
  def compute_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
    hidden_states = HiddenStates(self.hidden_states)
    return hidden_states.layer_overlap_label(self.label)

class SubjectOverlap(LabelOverlap):
  
  def set_dataframes(self) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    hidden_states_dict = {"hidden_states": [],"layer": [], "subject":[]}
    for scenario_result in self.scenario_results:
      for request_result in scenario_result.requests_results:
        for layer in ["last","sum"]:
          hidden_states_dict["hidden_states"].append(request_result.hidden_states[layer])
          hidden_states_dict["layer"].append(layer)
          hidden_states_dict["subject"].append(scenario_result.dataset)
    hidden_states = pd.DataFrame(hidden_states_dict)
    return  hidden_states 
  
class LetterOverlap(LabelOverlap):
  def set_dataframes(self) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    hidden_states_dict = {"hidden_states": [],"layer": [], "letter":[]}
    for scenario_result in self.scenario_results:
      for request_result in scenario_result.requests_results:
        for layer in ["last","sum"]:
          hidden_states_dict["hidden_states"].append(request_result.hidden_states[layer])
          hidden_states_dict["layer"].append(layer)
          hidden_states_dict["letter"].append(request_result.preds['only_ref_pred']['letter'])
    hidden_states = pd.DataFrame(hidden_states_dict)
    return  hidden_states 
  

class BaseFinetuneOverlap():
  def __init__(self, data: Dict) -> None:
    """
    Compute overlap between base and finetuned model
    """
    self.data = data
#  def get_couples(self, list_of_models: List) -> List[Tuple[str, str]]:
#    """
#    Get all the couples of base and finetuned model
#    """
#    base = list(filter(lambda x: "chat" not in x, list_of_models))
#    finetuned = list(filter(lambda x: "chat" in x, list_of_models))
#    couples = [(b,f) for f in finetuned for b in base if "chat" in list(set(b.split("-"))-set(f.split("-")))]
#    import pdb; pdb.set_trace()
#    return couples
#  
  def compute_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
    overlaps = {}
    for couples in tqdm.tqdm([("Llama-2-7b-hf","Llama-2-7b-chat-hf"),("Llama-2-13b-hf","Llama-2-13b-chat-hf")]):
      overlaps[couples] = {}
      for dataset in self.data[couples[0]].keys():
        overlaps[couples][dataset] = {}
        for method in ["last", "sum"]:
          overlaps[couples][dataset][method] = np.empty([6,6,self.data[couples[0]][dataset]['0']["all"][method].shape[0]])
          for train_instances_i in self.data[couples[0]][dataset].keys():
            for train_instances_j in self.data[couples[0]][dataset].keys():
                overlaps[couples][dataset][method][int(train_instances_i),int(train_instances_j)] = layer_overlap(self.data[couples[0]][dataset][train_instances_i]["all"][method],
                                                                              self.data[couples[1]][dataset][train_instances_j]["all"][method])
    return overlaps
    
  
class DatasetMetrics():
  """
  Class to compute metrics across all the runs of a dataset. Generally used to compare between different n-shots
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

  def get_last_layer_id_diff(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute difference in last layer ID among 0-shot and few-shot
    Output
    ----------
    Dict[layer, Dict[metric, Array(neigh_order)]]
    """
    metric_id = {}
    zero_shot = next((x for x in self.instances_results if x.train_instances=='0'), None)
    
    for method in [Layer.LAST.value, Layer.SUM.value]:
        metric_id[method] = {"id_diff": [], "metric_diff": {key: [] for key in self.instances_results[0].metrics.keys()}} 
        for i_result in self.instances_results:
          if i_result.train_instances == 0:
            continue
          zero_shot_id = zero_shot.intrinsic_dim[Match.ALL.value][method]
          i_result_id = i_result.intrinsic_dim[Match.ALL.value][method]
          # the intrinisc dimension is calculated up to the nearest neighbour order that change between runs
          nn_order = min(zero_shot_id.shape[1], i_result_id.shape[1])
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

# class HiddenStates():
#   def __init__(self,hidden_states):
#     self.hidden_states = hidden_states
  
#   #TODO use a decorator
#   def preprocess_hiddenstates(self,func, query, *args, **kwargs):
#     hidden_states_collapsed = hidden_states_collapse(self.hidden_states, query)

#     return func(hidden_states_collapsed, *args, **kwargs)

#   def set_hidden_states(self, exact_matches) -> Dict[str, Dict[str, np.ndarray]]:
#     """
#     Output
#     ----------
#     {method: method(num_instances, num_layers, model_dim)}
#     """ 
#     instances_hiddenstates = self._get_instances_hidden_states(exact_matches) 
#     hidden_states = {match.value: {layer.value: hidden_states_collapse(instances_hiddenstates, {"layer":layer.value,"match": match.value}) 
#                             for layer in [Layer.LAST,Layer.SUM] } 
#                             for match in [Match.CORRECT, Match.WRONG, Match.ALL]}
#     return hidden_states
  
#   def get_instances_id(self) -> np.ndarray:
#     """
#     Compute the ID of all instances using gride algorithm
#     Output
#     ----------
#     Dict[str, np.array(num_layers)]
#     """
#     id = {match.value: 
#             {layer.value: compute_id(self.hidden_states,{"match":match.value,"layer":layer.value}, "gride") 
#             for layer in [Layer.LAST, Layer.SUM]} 
#             for match in [Match.CORRECT, Match.WRONG, Match.ALL]}
#     return id
  
#   def nearest_neighbour(self, hidden_states, k: int ) -> np.ndarray:
#     """
#     Compute the nearest neighbours of each instance in the run per layer
#     using the provided methodv
#     Output
#     ----------
#     Array(num_layers, num_instances, k_neighbours)
#     """

#     assert k <= hidden_states.shape[0], "K must be smaller than the number of instances"
#     layers = hidden_states.shape[1]
#     neigh_matrix_list = []
#     for i in range(layers):
#       neigh = NearestNeighbors(n_neighbors=k)
#       neigh.fit(hidden_states[:,i,:])
#       dist, indices = neigh.kneighbors(hidden_states[:,i,:])
#       indices = np.delete(indices, 0, 1) # removing the first column which is the instance itself
#       neigh_matrix_list.append(indices)
    
#     return np.stack(neigh_matrix_list)
  
#   def layer_overlap_letter(self):
#     nn = []
#     nn_half1 = []
#     nn_half2 = []
    
#     half = True
#     for letter in ["A","B","C","D"]:
#       k = max(int(half*0.8),2)
#       hidden_states = hidden_states_collapse(self.hidden_states,
#                                              {"match":Match.ALL.value, 
#                                               "layer":Layer.LAST.value, 
#                                               "answered_letter":letter})
#       nn.append(self.nearest_neighbour(hidden_states, k=k))
#       half = int(hidden_states.shape[0]/2)
#       nn_half1.append(self.nearest_neighbour(hidden_states[:half], k=k))
#       nn_half2.append(self.nearest_neighbour(hidden_states[half:], k=k))
#     overlap = np.empty([4,4])
#     warnings.warn("Computing overlap for -5th layer and using 80 %% of the instances")
#     for i in tqdm.tqdm(range(4), desc = "Computing overlap"):
#       for j in range (4):
#         if i==j:
#           overlap[i,j] = neig_overlap(nn_half1[i][-5,], nn_half2[j][-5])
#         else:
#           overlap[i,j] = neig_overlap(nn[i][-5,], nn[j][-5])
    
#     return overlap

