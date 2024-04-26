from .utils import  exact_match, quasi_exact_match, hidden_states_collapse
from metrics.query import DataFrameQuery
from metrics.intrinisic_dimension import IntrinsicDimension
from metrics.clustering import LabelClustering, PointClustering
from metrics.overlap import PointOverlap, LabelOverlap
from metrics.cka import CenteredKernelAlignement
from common.tensor_storage import TensorStorage
#from sklearn.feature_selection import mutual_info_regression MISSIN?
from dadapy.data import Data

import warnings
from typing import Dict, List
from pathlib  import Path
import tqdm

import numpy as np
import pandas as pd
  
class HiddenStates():
  def __init__(self,hidden_states: pd.DataFrame, 
               hidden_states_path: Path, 
               variations:dict = None):
    self.df = hidden_states
    self.tensor_storage = TensorStorage(hidden_states_path)
    self.variations = variations
 
  def intrinsic_dim(self) -> pd.DataFrame:
    intrinsic_dim = IntrinsicDimension(df = self.df, 
                                       tensor_storage = self.tensor_storage, 
                                       variations=self.variations)
    return intrinsic_dim.main()
      
  def shot_metrics(self,metrics_list=None) -> Dict[str, Dict[str, float]]:
    rows = []
    #import pdb; pdb.set_trace()
    for dataset in tqdm.tqdm(self.df["dataset"].unique().tolist()):
      for model in self.df["model_name"].unique().tolist():
        if "13" in model:
          continue
        for train_instances in self.df["train_instances"].unique().tolist():
          query = DataFrameQuery({"dataset":dataset, 
                                  "method":"last",
                                  "model_name":model,
                                  "train_instances": train_instances})
          _,_, hidden_states_df= hidden_states_collapse(self.df,query, self.tensor_storage)
          rows.append([dataset, 
                       model, 
                       train_instances, 
                       hidden_states_df["loss"].mean(), 
                       hidden_states_df.apply(lambda r: exact_match(r["std_pred"], r["letter_gold"]), axis=1).mean(), 
                       hidden_states_df.apply(lambda r: quasi_exact_match(r["std_pred"], r["letter_gold"]), axis=1).mean(), 
                       hidden_states_df.apply(lambda r: exact_match(r["only_ref_pred"], r["letter_gold"]), axis=1).mean()])
     
    df = pd.DataFrame(rows, columns = ["dataset",
                                        "model",
                                        "train_instances",
                                        "loss", 
                                        "exact_match", 
                                        "quasi_exact_match", 
                                        "only_ref_exact_match", 
                                        ])
    return df
  
  
  def label_clustering(self, label:str, **kwargs) -> pd.DataFrame:
    if "variation" in kwargs:
      variation = kwargs["variation"]
    if variation:
      label_clustering = LabelClustering(df = self.df, 
                                         tensor_storage = self.tensor_storage, 
                                         variation = variation)
    else:
      label_clustering = LabelClustering(df = self.df, 
                                         tensor_storage = self.tensor_storage)
    return label_clustering.main(label=label)
  
  def point_overlap(self) -> pd.DataFrame:
    point_overlap = PointOverlap(df = self.df, 
                                 tensor_storage = self.tensor_storage, 
                                 variations = self.variations)
    return point_overlap.main()
  
  def point_cluster(self) -> pd.DataFrame:
    point_cluster = PointClustering(df = self.df, 
                                    tensor_storage = self.tensor_storage, 
                                    variations = self.variations)
    return point_cluster.main()
  
  def label_overlap(self, label:str) -> pd.DataFrame:
    label_overlap = LabelOverlap(df = self.df, 
                                 tensor_storage = self.tensor_storage, 
                                 variations = self.variations, 
                                 storage_logic = "npy")
    return label_overlap.main(label=label)
 
  def cka(self) -> pd.DataFrame:
    ck_align = CenteredKernelAlignement(df = self.df, 
                                        tensor_storage = self.tensor_storage, 
                                        variations = self.variations)
    return ck_align.main()
  
  
 
    
 
 
 
 
 
 
 
  # def _clustering(hidden_states, k):
  #   cluster_assignemnts = []
  #   for num_layer in range(hidden_states.shape[1]):
  #     data = Data(hidden_states[:,num_layer,:])
  #     data.compute_distances(maxk=k)
  #     cluster_assignemnts.append(data.compute_clustering_ADP())
  #   return np.stack(cluster_assignemnts)  
  
  # def clustering(self,label):
  #   """
  #   Compute the clustering between the layers of instances in which the model answered with the same letter
  #   Output
  #   ----------
  #   Dict[layer: List[Array(num_layers, num_layers)]]
  #   """
  #   #The last token is always the same, thus its first layer activation (embedding) is always the same
  #   iter_list=[5,10,20,50]
  #   rows = []
  #   for k in tqdm.tqdm(iter_list, desc = "Computing overlap"):
  #     for model in self.df["model_name"].unique().tolist():
  #         for method in self.df["method"].unique().tolist():
  #           for train_instances in self.df["train_instances"].unique().tolist():
  #             query = DataFrameQuery({"match":Match.ALL.value, 
  #                                     "method":method,
  #                                     "model_name":model,
  #                                     "train_instances": train_instances})
  #             hidden_states,_, hidden_states_df= hidden_states_collapse(self.df,query, self.tensor_storage)
  #             label_per_row = hidden_states_df[label].reset_index(drop=True)
  #             clustering = self._clustering(hidden_states, label_per_row, k) 
  #             rows.append([k, model, method, train_instances,clustering, ])
                  
  #   df = pd.DataFrame(rows, columns = ["k","model","method","train_instances","clustering"])
  #   return df
  
  
