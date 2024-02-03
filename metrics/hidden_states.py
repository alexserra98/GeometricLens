from dadapy.data import Data
import numpy as np
from .utils import  hidden_states_collapse, Match
import tqdm
import pandas as pd
from typing import Dict, List
from .utils import  Match,  exact_match, quasi_exact_match, hidden_states_collapse
from dadapy.data import Data
from dadapy.data import Data
from metrics.query import DataFrameQuery
from pathlib  import Path
from common.tensor_storage import TensorStorage
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import mutual_info_score
import warnings

import os, sys

_DEBUG = False
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def hidden_states_collapse(df_hiddenstates: pd.DataFrame(), query: DataFrameQuery, tensor_storage: TensorStorage)-> np.ndarray:
    """
    Collect hidden states of all instances and collapse them in one tensor
    using the provided method

    Parameters
    ----------
    df_hiddenstates: pd.DataFrame(hiddens_states, match, layer, answered letter, gold letter)
                     dataframe containing the hidden states of all instances
    query: Dict[condition, value] --> what instances to select
    Output
    ----------
    (num_instances, num_layers, model_dim)
    """ 

    df_hiddenstates = query.apply_query(df_hiddenstates)
    hidden_states = []
    logits = []
    for model in df_hiddenstates["model_name"].unique(): 
      for dataset in df_hiddenstates["dataset"].unique():
        for train_instances in df_hiddenstates["train_instances"].unique():
          hidden_states.extend(tensor_storage.load_tensors(f'{model.replace("/","-")}/{dataset}/{train_instances}/hidden_states', df_hiddenstates["id_instance"].tolist()))
          #logits.extend(tensor_storage.load_tensors(f'{model.replace("/","-")}/{dataset}/{train_instances}/logits', df_hiddenstates["id_instance"].tolist()))
    return np.stack(hidden_states), df_hiddenstates



class HiddenStates():
  def __init__(self,hidden_states: pd.DataFrame(), hidden_states_path: Path):
    self.df = hidden_states
    self.tensor_storage = TensorStorage(hidden_states_path)
    
  def _compute_id(self, hidden_states: np.ndarray ,  algorithm = "gride") -> np.ndarray:
    """
    Collect hidden states of all instances and compute ID
    we employ two different approaches: the one of the last token, the sum of all tokens
    Parameters
    ----------
    hidden_states: np.array(num_instances, num_layers, model_dim)
    algorithm: 2nn or gride --> what algorithm to use to compute ID

    Output
    ---------- 
    Dict np.array(num_layers)
    """
    assert algorithm == "gride", "gride is the only algorithm supported"
    id_per_layer = []
    num_layers = hidden_states.shape[1]
    for i in range(1,num_layers):
        data = Data(hidden_states[:,i,:])
        with HiddenPrints():
          data.remove_identical_points()
        if algorithm == "2nn":
            #id_per_layer.append(layer.compute_id_2NN()[0])
            raise NotImplementedError
        elif algorithm == "gride":
            id_per_layer.append(data.return_id_scaling_gride(range_max = 500)[0])
    return  np.stack(id_per_layer[1:])
    
  def intrinsic_dim(self) -> pd.DataFrame:
    rows = []
    for model in tqdm.tqdm(self.df["model_name"].unique().tolist()):
      for method in self.df["method"].unique().tolist():
        for train_instances in self.df["train_instances"].unique().tolist():
          for match in ["correct", "incorrect", "all"]:
            query = DataFrameQuery({"method":method,
                                    "model_name":model,
                                    "train_instances": train_instances})
            if match != "correct":
              df = self.df[self.df.apply(lambda r: exact_match(r["std_pred"], r["letter_gold"]), axis=1)]
            elif match != "incorrect":
              df = self.df[self.df.apply(lambda r: not exact_match(r["std_pred"], r["letter_gold"]), axis=1)]
            else:
              df = self.df
            hidden_states, _= hidden_states_collapse(df,query, self.tensor_storage)
            id_per_layer = self._compute_id(hidden_states)
            rows.append([model, 
                        method,
                        match,
                        train_instances,
                        id_per_layer])
              
        
    return pd.DataFrame(rows, columns = ["model",
                                          "method",
                                          "match",  
                                          "train_instances",
                                          "id_per_layer"]) 
      

  
  #TODO use a decorator
  def preprocess_hiddenstates(self,func, query, *args, **kwargs):
    hidden_states_collapsed,_ = hidden_states_collapse(self.df, query)

    return func(hidden_states_collapsed, *args, **kwargs)
  
  
  def shot_metrics(self,metrics_list=None) -> Dict[str, Dict[str, float]]:
    rows = []
    for dataset in tqdm.tqdm(self.df["dataset"].unique().tolist()):
      for model in self.df["model_name"].unique().tolist():
        for train_instances in self.df["train_instances"].unique().tolist():
          query = DataFrameQuery({"match":Match.ALL.value,
                                  "dataset":dataset, 
                                  "method":"last",
                                  "model_name":model,
                                  "train_instances": train_instances})
          _, hidden_states_df= hidden_states_collapse(self.df,query, self.tensor_storage)
          rows.append([dataset, 
                       model, 
                       train_instances, 
                       hidden_states_df["loss"].mean(), 
                       hidden_states_df.apply(lambda r: exact_match(r["std_pred"], r["letter_gold"]), axis=1).mean(), 
                       hidden_states_df.apply(lambda r: quasi_exact_match(r["std_pred"], r["letter_gold"]), axis=1).mean(), 
                       hidden_states_df.apply(lambda r: exact_match(r["only_ref_pred"], r["letter_gold"]), axis=1).mean()])
            # ADD INTRINISC DIMENSION!      
    df = pd.DataFrame(rows, columns = ["dataset",
                                        "model",
                                        "train_instances",
                                        "loss", 
                                        "exact_match", 
                                        "quasi_exact_match", 
                                        "only_ref_exact_match", 
                                        ])
    return df

  
  def _label_overlap(self, hidden_states, labels, k=None, class_fraction = None) -> Dict[str, List[np.ndarray]]:
    overlaps = []
    if k is None and class_fraction is None:
      raise ValueError("You must provide either k or class_fraction")
    for num_layer in range(hidden_states.shape[1]):
      data = Data(hidden_states[:,num_layer,:])
      #data.compute_distances(maxk=k)
      #import pdb; pdb.set_trace()
      warnings.filterwarnings("ignore")
      overlap = data.return_label_overlap(labels,class_fraction=class_fraction,k=k)
      overlaps.append(overlap)
    return np.stack(overlaps)
      
  def label_overlap(self,label) -> Dict[str, List[np.ndarray]]:
    """
    Compute the overlap between the layers of instances in which the model answered with the same letter
    Output
    ----------
    Dict[layer: List[Array(num_layers, num_layers)]]
    """
    #The last token is always the same, thus its first layer activation (embedding) is always the same
    iter_list=[0.05,0.10,0.20,0.50]
    rows = []
    for class_fraction in tqdm.tqdm(iter_list, desc = "Computing overlap"):
      for model in self.df["model_name"].unique().tolist():
        for method in self.df["method"].unique().tolist():
          for train_instances in self.df["train_instances"].unique().tolist():
            query = DataFrameQuery({"match":Match.ALL.value, 
                                    "method":method,
                                    "model_name":model,
                                    "train_instances": train_instances}) 
                                    #{"balanced":label})
            hidden_states, hidden_states_df= hidden_states_collapse(self.df,query, self.tensor_storage)
            #assert hidden_states_df[label].value_counts().nunique() == 1, "There must be the same number of instances for each label - Class imbalance not supported"
            labels_literals = hidden_states_df[label].unique()
            labels_literals.sort()
            map_labels = {class_name: n for n,class_name in enumerate(labels_literals)}
            label_per_row = hidden_states_df[label].reset_index(drop=True)
            label_per_row = [map_labels[class_name] for class_name in label_per_row]
            label_per_row = np.array(label_per_row)
            #import pdb; pdb.set_trace() 
            label_per_row = label_per_row[:hidden_states.shape[0]]
            overlap = self._label_overlap(hidden_states, label_per_row, class_fraction=class_fraction) 
            rows.append([class_fraction, model, method, train_instances,overlap])
                    
    df = pd.DataFrame(rows, columns = ["k","model","method","train_instances","overlap"])
    return df
  
  
  def _point_overlap(self, data_i: np.ndarray, data_j: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the overlap between two runs
    
    Parameters
    ----------
    data_i: np.ndarray(num_instances, num_layers, model_dim)
    data_j: np.ndarray(num_instances, num_layers, model_dim)
    k: int
    
    Output
    ----------
    Array(num_layers)
    """
    assert data_i.shape[1] == data_j.shape[1], "The two runs must have the same number of layers"
    number_of_layers = data_i.shape[1]
    overlap_per_layer = []
    for layer in range(number_of_layers):
      data = Data(data_i[:,layer,:])
      warnings.filterwarnings("ignore")
      data.compute_distances(maxk=k)
      overlap_per_layer.append(data.return_data_overlap(data_j[:,layer,:]))
    
    return np.stack(overlap_per_layer)
  
  def pair_names(self,names_list):
    """
    Pairs base names with their corresponding 'chat' versions.

    Args:
    names_list (list): A list of strings containing names.

    Returns:
    list: A list of tuples, each containing a base name and its 'chat' version.
    """
    # Separating base names and 'chat' names
    difference = 'openai'
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
  
  def point_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute the overlap between same dataset, same train instances, different models (pretrained and finetuned)
    
    Parameters
    ----------
    data: Dict[model, Dict[dataset, Dict[train_instances, Dict[method, Dict[match, Dict[layer, np.ndarray]]]]]]
    Output
    df: pd.DataFrame (k,dataset,method,train_instances_i,train_instances_j,overlap)
    """
    #warn("Computing overlap using k with  2 -25- 500")
    comparison_metrics = {"adjusted_rand_score":adjusted_rand_score, 
                          "adjusted_mutual_info_score":adjusted_mutual_info_score, 
                          "mutual_info_score":mutual_info_score}
    iter_list = [5,10,30,100]
    if _DEBUG:
      iter_list = [2]
    rows = []
    for k in tqdm.tqdm(iter_list, desc = "Computing overlaps k"):
      for couples in self.pair_names(self.df["model_name"].unique().tolist()):
        #import pdb; pdb.set_trace()
        for method in self.df["method"].unique().tolist():
          for train_instances_i in ["0","5"]:#self.hidden_states["train_instances"].unique().tolist():
            for train_instances_j in ["0","5"]:#self.hidden_states["train_instances"].unique():
                query_i = DataFrameQuery({"method":method,
                        "model_name":couples[0], 
                        "train_instances": train_instances_i,})
                query_j = DataFrameQuery({"method":method,
                        "model_name":couples[1], 
                        "train_instances": train_instances_j,})
                hidden_states_i, _ = hidden_states_collapse(self.df,query_i, self.tensor_storage)
                hidden_states_j, _ = hidden_states_collapse(self.df,query_j, self.tensor_storage)
                clustering_out = self._clustering_overlap(hidden_states_i, hidden_states_j, k, comparison_metrics)

                rows.append([k,
                             couples,
                             method,
                             train_instances_i,
                             train_instances_j, 
                             clustering_out["adjusted_rand_score"],
                             clustering_out["adjusted_mutual_info_score"],
                             clustering_out["mutual_info_score"],
                             self._point_overlap(hidden_states_i, hidden_states_j, k)])
    df = pd.DataFrame(rows, columns = ["k",
                                       "couple",
                                       "method",
                                       "train_instances_i",
                                       "train_instances_j",
                                       "adjusted_rand_score",
                                       "adjusted_mutual_info_score",
                                       "mutual_info_score",
                                       "point_overlap"])
    return df
  
  def _clustering_overlap(self, input_i: np.ndarray, input_j: np.ndarray, k: int, comparison_metrics: dict) -> np.ndarray:
      assert input_i.shape[1] == input_j.shape[1], "The two runs must have the same number of layers"
      number_of_layers = input_i.shape[1]
      k = 100 if not _DEBUG else 50
      comparison_output = {key:[] for key in comparison_metrics.keys()}
      for layer in range(1,number_of_layers):
        data_i = Data(input_i[:,layer,:]); data_j = Data(input_j[:,layer,:])
        with HiddenPrints():
          data_i.remove_identical_points(); data_j.remove_identical_points()
        data_i.compute_distances(maxk=k); data_j.compute_distances(maxk=k)
        clusters_i = data_i.compute_clustering_ADP(Z=1.68)
        clusters_j = data_j.compute_clustering_ADP()
        for key, func in comparison_metrics.items():
          comparison_output[key].append(func(clusters_i, clusters_j))        
        
      for key in comparison_metrics.keys():
        comparison_output[key] = np.stack(comparison_output[key])
      return comparison_output
    
  def _clustering(hidden_states, k):
    cluster_assignemnts = []
    for num_layer in range(hidden_states.shape[1]):
      data = Data(hidden_states[:,num_layer,:])
      data.compute_distances(maxk=k)
      cluster_assignemnts.append(data.compute_clustering_ADP())
    return np.stack(cluster_assignemnts)  
  
  def clustering(self,label):
    """
    Compute the clustering between the layers of instances in which the model answered with the same letter
    Output
    ----------
    Dict[layer: List[Array(num_layers, num_layers)]]
    """
    #The last token is always the same, thus its first layer activation (embedding) is always the same
    iter_list=[5,10,20,50]
    rows = []
    for k in tqdm.tqdm(iter_list, desc = "Computing overlap"):
      for model in self.df["model_name"].unique().tolist():
          for method in self.df["method"].unique().tolist():
            for train_instances in self.df["train_instances"].unique().tolist():
              query = DataFrameQuery({"match":Match.ALL.value, 
                                      "method":method,
                                      "model_name":model,
                                      "train_instances": train_instances})
              hidden_states, hidden_states_df= hidden_states_collapse(self.df,query, self.tensor_storage)
              label_per_row = hidden_states_df[label].reset_index(drop=True)
              clustering = self._clustering(hidden_states, label_per_row, k) 
              rows.append([k, model, method, train_instances,clustering, ])
                  
    df = pd.DataFrame(rows, columns = ["k","model","method","train_instances","clustering"])
    return df
  
  
