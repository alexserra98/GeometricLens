import numpy as np
from .utils import  hidden_states_collapse
import tqdm
import pandas as pd
from .utils import hidden_states_collapse, HiddenPrints
from dadapy.data import Data
from metrics.query import DataFrameQuery
from common.globals_vars import _NUM_PROC
from joblib import Parallel, delayed
from metrics.hidden_states_metrics import HiddenStatesMetrics
import skdim
from functools import partial
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from dadapy.data import Data
from sklearn.metrics import mutual_info_score
import sys


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
        
      output = {key: [] for key in _COMPARISON_METRICS.keys().append("clustering_bincount")}
      
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
        layer_results["clustering_bincount"] = bincount
        
        #Comparison metrics
        for key, func in _COMPARISON_METRICS.items():
            layer_results[key] = func(clusters_assignement, label)
        
        return layer_results
      
      

  
