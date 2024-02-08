from metrics.hidden_states_metrics import HiddenStatesMetrics
from .utils import  hidden_states_collapse
from .utils import hidden_states_collapse, HiddenPrints
from metrics.query import DataFrameQuery
from common.globals_vars import _NUM_PROC

from dadapy.data import Data
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from functools import partial




_COMPARISON_METRICS = {"adjusted_rand_score":adjusted_rand_score, 
                      "adjusted_mutual_info_score":adjusted_mutual_info_score, 
                      "mutual_info_score":mutual_info_score}


class LabelClustering(HiddenStatesMetrics):
    def __init__(self, df: pd.DataFrame, tensor_storage, label: str):
      super().__init__(df, tensor_storage)
      self.label = label
    
    def main(self) -> pd.DataFrame:
      """
      Compute the overlap between the layers of instances in which the model answered with the same letter
      Output
      ----------
      Dict[layer: List[Array(num_layers, num_layers)]]
      """
      #The last token is always the same, thus its first layer activation (embedding) is always the same
      #iter_list=[0.05,0.10,0.20,0.50]
      iter_list=[0.9,1.68,2]
      rows = []
      
      for z in tqdm.tqdm(iter_list, desc = "Computing overlap"):
        for model in self.df["model_name"].unique().tolist():
          for method in ["last"]: #self.df["method"].unique().tolist():
            for train_instances in ["0","2","5"]:#self.df["train_instances"].unique().tolist():
              
              query = DataFrameQuery({"method":method,
                                      "model_name":model,
                                      "train_instances": train_instances}) 
                                    
              hidden_states, _, hidden_states_df= hidden_states_collapse(self.df,query, self.tensor_storage)
              
              row = [model, method, train_instances, z]
              
              label_per_row = self.constructing_labels(hidden_states_df, hidden_states)
              
              clustering_dict = self.parallel_compute(hidden_states, label_per_row, z)

              row.extend([clustering_dict["bincount"], 
                          clustering_dict["adjusted_rand_score"], 
                          clustering_dict["adjusted_mutual_info_score"], 
                          clustering_dict["mutual_info_score"]])
              rows.append(row)
                        
      df = pd.DataFrame(rows, columns = ["model",
                                        "method",
                                        "train_instances",
                                        "z",
                                        "clustering_bincount",
                                        "adjusted_rand_score",
                                        "adjusted_mutual_info_score",
                                        "mutual_info_score", ])
      return df
    
    def constructing_labels(self, hidden_states_df: pd.DataFrame, hidden_states: np.ndarray) -> np.ndarray:
      labels_literals = hidden_states_df[self.label].unique()
      labels_literals.sort()
    
      map_labels = {class_name: n for n,class_name in enumerate(labels_literals)}
      
      label_per_row = hidden_states_df[self.label].reset_index(drop=True)
      label_per_row = np.array([map_labels[class_name] for class_name in label_per_row])[:hidden_states.shape[0]]
      
      return label_per_row
    
    
    def parallel_compute(self, hidden_states: np.ndarray, label: np.array, z: float) -> dict:
      
      assert hidden_states.shape[0] == label.shape[0], "Label lenght don't mactch the number of instances"
      number_of_layers = hidden_states.shape[1]
      #k = 100 if not _DEBUG else 50
  
      process_layer = partial(self.process_layer, hidden_states=hidden_states, label=label, z=z)
  
      
      #Parallelize the computation of the metrics
      with Parallel(n_jobs=_NUM_PROC) as parallel:
        results = parallel(delayed(process_layer)(layer) for layer in range(1, number_of_layers))
      
      keys = list(_COMPARISON_METRICS.keys()); keys.append("bincount")
      output = {key: [] for key in keys}
      
      #Merge the results
      for layer_result in results:
          for key in output:
              output[key].append(layer_result[key])
      return output    
  
    
    def process_layer(self, layer, hidden_states, label, z) -> dict:
        #DADApy
        data = Data(hidden_states[:, layer, :])
        with HiddenPrints():
            data.remove_identical_points()
        data.compute_distances(maxk=100)
        clusters_assignement = data.compute_clustering_ADP(Z=z)
        
        layer_results = {}
        
        #Bincount
        unique_clusters, cluster_counts = np.unique(clusters_assignement, return_counts=True)
        bincount = np.zeros((len(unique_clusters), len(np.unique(label))))
        for unique_cluster in unique_clusters:
          bincount[unique_cluster] = np.bincount(label[clusters_assignement == unique_cluster], minlength=len(np.unique(label)))
        layer_results["bincount"] = bincount
        
        #Comparison metrics
        for key, func in _COMPARISON_METRICS.items():
            layer_results[key] = func(clusters_assignement, label)
        
        return layer_results
      
      

  
class PointClustering(HiddenStatesMetrics):

  
  def main(self) -> pd.DataFrame:
    """
    Compute the overlap between same dataset, same train instances, different models (pretrained and finetuned)
    
    Parameters
    ----------
    data: Dict[model, Dict[dataset, Dict[train_instances, Dict[method, Dict[match, Dict[layer, np.ndarray]]]]]]
    Output
    df: pd.DataFrame (k,dataset,method,train_instances_i,train_instances_j,overlap)
    """
    #warn("Computing overlap using k with  2 -25- 500")

    iter_list=[0.9,1.68,2]

    rows = []
    for z in tqdm.tqdm(iter_list, desc = "Computing overlaps k"):
      for couples in self.pair_names(self.df["model_name"].unique().tolist()):
        #import pdb; pdb.set_trace()
        for method in ["last"]:#self.df["method"].unique().tolist():
          if couples[0]==couples[1]:
            iterlist = [("0","5")]
          else:
            iterlist = [("0","0"),("0","5"),("5","5"),("5","0")]
          for shot in iterlist:
            train_instances_i, train_instances_j = shot
            query_i = DataFrameQuery({"method":method,
                    "model_name":couples[0], 
                    "train_instances": train_instances_i,})
            query_j = DataFrameQuery({"method":method,
                    "model_name":couples[1], 
                    "train_instances": train_instances_j,})
            hidden_states_i, _,_ = hidden_states_collapse(self.df,query_i, self.tensor_storage)
            hidden_states_j, _,_ = hidden_states_collapse(self.df,query_j, self.tensor_storage)
            clustering_out = self.parallel_compute(hidden_states_i, hidden_states_j, z)

            rows.append([z,
                         couples,
                         method,
                         train_instances_i,
                         train_instances_j, 
                         clustering_out["adjusted_rand_score"],
                         clustering_out["adjusted_mutual_info_score"],
                         clustering_out["mutual_info_score"]])
    df = pd.DataFrame(rows, columns = ["z",
                                       "couple",
                                       "method",
                                       "train_instances_i",
                                       "train_instances_j",
                                       "adjusted_rand_score",
                                       "adjusted_mutual_info_score",
                                       "mutual_info_score"])
    return df
  
  def pair_names(self,names_list):
    """
    Pairs base names with their corresponding 'chat' versions.

    Args:
    names_list (list): A list of strings containing names.

    Returns:
    list: A list of tuples, each containing a base name and its 'chat' version.
    """
    # Separating base names and 'chat' names
    difference = 'chat'
    base_names = [name for name in names_list if difference not in name]
    chat_names = [name for name in names_list if difference in name]
    base_names.sort()
    chat_names.sort()
    # Pairing base names with their corresponding 'chat' versions
    pairs = []
    for base_name, chat_name in zip(base_names, chat_names):
      pairs.append((base_name, base_name))
      pairs.append((chat_name, chat_name))
      pairs.append((base_name, chat_name))
    return pairs
  def parallel_compute(self, input_i: np.ndarray, input_j: np.ndarray, z: int) -> np.ndarray:
      assert input_i.shape[1] == input_j.shape[1], "The two runs must have the same number of layers"
      number_of_layers = input_i.shape[1]

      comparison_output = {key:[] for key in _COMPARISON_METRICS.keys()}
      
      process_layer = partial(self.process_layer, input_i=input_i, input_j=input_j, z=z, _COMPARISON_METRICS=_COMPARISON_METRICS)
      with Parallel(n_jobs=_NUM_PROC) as parallel:
        results = parallel(delayed(process_layer)(layer) for layer in range(1, number_of_layers))
      
      # Organize the results
      comparison_output = {key: [] for key in _COMPARISON_METRICS}
      for layer_result in results:
          for key in comparison_output:
              comparison_output[key].append(layer_result[key])
      return comparison_output
    
  def process_layer(self, layer, input_i, input_j, z):
    """
    Process a single layer.
    """
    data_i = Data(input_i[:, layer, :])
    data_j = Data(input_j[:, layer, :])

    with HiddenPrints():
        data_i.remove_identical_points()
        data_j.remove_identical_points()

    data_i.compute_distances(maxk=100)
    data_j.compute_distances(maxk=100)

    clusters_i = data_i.compute_clustering_ADP(Z=z)
    clusters_j = data_j.compute_clustering_ADP(Z=z)

    layer_results = {}
    for key, func in _COMPARISON_METRICS.items():
        layer_results[key] = func(clusters_i, clusters_j)

    return layer_results