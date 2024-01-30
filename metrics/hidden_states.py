from dadapy.data import Data
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable, Set, Type
from einops import reduce
from sklearn.neighbors import NearestNeighbors
from .utils import compute_id, hidden_states_collapse, Match, Layer, label_neig_overlap, class_imbalance
import tqdm
import pandas as pd
from collections import namedtuple

from dataclasses import asdict
from dadapy.data import Data
from dadapy.metric_comparisons import MetricComparisons
from warnings import warn
from metrics.query import DataFrameQuery
from pathlib  import Path
from common.tensor_storage import TensorStorage

def compute_id(hidden_states: np.ndarray ,query: Dict,  algorithm = "2nn") -> np.ndarray:
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
    hidden_states,_ = hidden_states_collapse(hidden_states,query)
    # Compute ID
    id_per_layer = []
    layers = hidden_states.shape[1]
    print(f"Computing ID for {layers} layers")
    for i in range(1,layers): #iterate over layers
        # (num_instances, model_dim)
        layer = Data(hidden_states[:,i,:])
        #print(f"layer {i} with {hidden_states.shape}")
        layer.remove_identical_points()
        if algorithm == "2nn":
            #id_per_layer.append(layer.compute_id_2NN()[0])
            raise NotImplementedError
        elif algorithm == "gride":
            id_per_layer.append(layer.return_id_scaling_gride(range_max = 1000)[0])
        #print(f"layer {i} with {id_per_layer[-1].shape}")
    return  np.stack(id_per_layer[1:])

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
    hidden_states = tensor_storage.load_tensors("hidden_states", df_hiddenstates["id_instance"].tolist())
    return np.stack(hidden_states), df_hiddenstates



class HiddenStates():
  def __init__(self,hidden_states: pd.DataFrame(), hidden_states_path: Path):
    self.df = hidden_states
    self.tesnsor_storage = TensorStorage(hidden_states_path) 
    
  
  #TODO use a decorator
  def preprocess_hiddenstates(self,func, query, *args, **kwargs):
    hidden_states_collapsed,_ = hidden_states_collapse(self.df, query)

    return func(hidden_states_collapsed, *args, **kwargs)

  
  def get_instances_id(self) -> np.ndarray:
    """
    Compute the ID of all instances using gride algorithm
    Output
    ----------
    Dict[str, np.array(num_layers)]
    """
    id = {match.value: 
            {layer.value: compute_id(self.df,{"match":match.value,"layer":layer.value}, "gride") 
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
  
  def get_nearest_neighbour(self, k: int) -> np.ndarray:
    """
    Compute the nearest neighbours of each instance in the run per layer
    using the provided methodv
    Output
    ----------
    Dict[str, np.array(num_layers, num_instances, k_neighbours)]
    """
    nn = {match.value: 
            {layer.value: self.nearest_neighbour(self.df,{"match":match.value,"layer":layer.value}, k) 
            for layer in [Layer.LAST, Layer.SUM]} 
            for match in [Match.CORRECT, Match.WRONG, Match.ALL]}
    return nn
  



  
  def label_overlap(self, hidden_states, labels, k) -> Dict[str, List[np.ndarray]]:
    overlaps = []
    for num_layer in range(hidden_states.shape[1]):
      mc = Data(hidden_states[:,num_layer,:])
      mc.compute_distances()
      overlap = mc.return_label_overlap(labels, k)
      overlaps.append(overlap)
    return np.stack(overlaps)
      
  def layer_overlap_label(self,label) -> Dict[str, List[np.ndarray]]:
    """
    Compute the overlap between the layers of instances in which the model answered with the same letter
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
                                      "train_instances": train_instances}, 
                                      {"balanced":label})
              hidden_states, hidden_states_df= hidden_states_collapse(self.df,query, self.tesnsor_storage)
              assert hidden_states_df[label].value_counts().nunique() == 1, "There must be the same number of instances for each label - Class imbalance not supported"
              label_per_row = hidden_states_df[label].reset_index(drop=True)
              overlap = self.label_overlap(hidden_states, label_per_row, k) 
              rows.append([k, model, method, train_instances,overlap])
                  
    df = pd.DataFrame(rows, columns = ["k","model","method","train_instances","overlap"])
    return df
  
  
  #HORRIBLE CODE REPETITION overlap will become a class itself
  def layer_overlap_subject(self) -> Dict[str, List[np.ndarray]]:
    """
    Compute the overlap between the layers of instances in which the model answered with the same letter
    Output
    ----------
    Dict[layer: List[Array(num_layers, num_layers)]]
    """
    label="subject"
    warn("The last token is always the same, thus its first layer activation (embedding) is always the same")
    warn("Computing overlap using k with  2 -25- 500")
    iter_list=[k for k in range(0,125,25)]
    iter_list[0]=2
    iter_list.extend([k for k in range(100,550,50)])
    iter_list=[k for k in range(0,125,25)]
    iter_list[0]=2
    rows = []
    for k in tqdm.tqdm(iter_list, desc = "Computing overlap"):
      for model in self.df["model"].unique().tolist():
        for method in self.df["layer"].unique().tolist():
          for train_instances in self.df["train_instances"].unique().tolist():
            
            hidden_states, hidden_states_df= hidden_states_collapse(self.df,{"match":Match.ALL.value, 
                                                                                        "layer":method,
                                                                                        "model":model,
                                                                                        "train_instances": train_instances, 
                                                                                        "balanced":label})
            assert hidden_states_df[label].value_counts().nunique() == 1, "There must be the same number of instances for each label - Class imbalance not supported"
            subject_per_row = hidden_states_df[label].reset_index(drop=True)
            overlap = self.label_overlap(hidden_states, subject_per_row, k) 
            rows.append([k, model,  method, train_instances,overlap])
                
    df = pd.DataFrame(rows, columns = ["k","model","method","train_instances","overlap"])
    return df

  
  
  def point_overlap(self, data_i: np.ndarray, data_j: np.ndarray, k: int) -> np.ndarray:
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
      mc = MetricComparisons(data_i[:,layer,:])
      overlap_per_layer.append(mc.return_data_overlap(data_j[:,layer,:], k = k))
    
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
    base_names = [name for name in names_list if 'chat' not in name]
    chat_names = [name for name in names_list if 'chat' in name]
    base_names.sort()
    chat_names.sort()
    # Pairing base names with their corresponding 'chat' versions
    pairs = []
    for base_name, chat_name in zip(base_names, chat_names):
      pairs.append((base_name, base_name))
      pairs.append((chat_name, chat_name))
      pairs.append((base_name, chat_name))

    return pairs
  
  def layer_point_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute the overlap between same dataset, same train instances, different models (pretrained and finetuned)
    
    Parameters
    ----------
    data: Dict[model, Dict[dataset, Dict[train_instances, Dict[method, Dict[match, Dict[layer, np.ndarray]]]]]]
    Output
    df: pd.DataFrame (k,dataset,method,train_instances_i,train_instances_j,overlap)
    """
    #warn("Computing overlap using k with  2 -25- 500")
    iter_list = [5,10,30,100]
    rows = []
    for k in tqdm.tqdm(iter_list, desc = "Computing overlaps k"):
      for couples in tqdm.tqdm(self.pair_names(self.df["model"].unique().tolist()), desc = "Computing overlap"):
        #import pdb; pdb.set_trace()
        for dataset in self.df["dataset"].unique().tolist():
          for method in self.df["layer"].unique().tolist():
            for train_instances_i in ["0","5"]:#self.hidden_states["train_instances"].unique().tolist():
              for train_instances_j in ["0","5"]:#self.hidden_states["train_instances"].unique():
                  hidden_states_i, _ = hidden_states_collapse(self.df,{"match":Match.ALL.value,
                                                                                              "layer":method, 
                                                                                              "model":couples[0], 
                                                                                              "dataset":dataset,
                                                                                              "train_instances": train_instances_i,})
                  hidden_states_j, _ = hidden_states_collapse(self.df,{"match":Match.ALL.value,
                                                                                              "layer":method, 
                                                                                              "model":couples[1], 
                                                                                              "dataset":dataset,
                                                                                              "train_instances": train_instances_j})
         
                  rows.append([k,couples,dataset,method,train_instances_i,train_instances_j, self.point_overlap(hidden_states_i, hidden_states_j, k)]) 
                  
    df = pd.DataFrame(rows, columns = ["k","couple","dataset","method","train_instances_i","train_instances_j","overlap"])
    return df

# def layer_overlap_label(self,label) -> Dict[str, List[np.ndarray]]:
#     """
#     Compute the overlap between the layers of instances in which the model answered with the same letter
#     Output
#     ----------
#     Dict[layer: List[Array(num_layers, num_layers)]]
#     """
#     overlaps = {}
#     Labels = namedtuple("Labels", "current_label, label_to_find")

#     for layer in [Layer.LAST, Layer.SUM]:
#       hidden_states, hidden_states_df= hidden_states_collapse(self.hidden_states,{"match":Match.ALL.value, "layer":layer.value, "balanced":label})
#       assert hidden_states_df[label].value_counts().nunique() == 1, "There must be the same number of instances for each label - Class imbalance not supported"
#       k = max(int(hidden_states.shape[0]*0.10),2) # number of nearest neighbours - Equal to the number of instances of a single label
#       nn = self.nearest_neighbour(hidden_states, k=k) # nearest neighbours matrix
#       # labelize the nearest neighbours matrix
#       # We substitute the indices with the labels of the associated instances, we keep a list_of_labels the track the label associated to each row
#       # Generally there are blocks of n consecutive rows with the same label
#       subject_per_row = hidden_states_df[label].reset_index(drop=True)
#       nn = self.labelize_nearest_neighbour(nn, subject_per_row)

#       overlaps[layer.value] = []
#       for num_layer in range(hidden_states.shape[1]):
#         overlap = np.empty([self.hidden_states[label].nunique(),self.hidden_states[label].nunique()])  
#         for n,i in enumerate(self.hidden_states[label].unique()):
#           for m,j in enumerate(self.hidden_states[label].unique()):
#             labels = Labels(i,j)
#             overlap[n,m] = label_neig_overlap(nn[num_layer],labels, subject_per_row)
#         overlap = (overlap - np.min(overlap)) / (np.max(overlap) - np.min(overlap))
#         overlap = overlap/overlap.shape[0]**2
#         overlaps[layer.value].append(overlap)
#     return overlaps
    
    
    # nn = []
    # nn_half1 = []
    # nn_half2 = []
    
    # half = True
    
    # for letter in ["A","B","C","D"]:
    #   k = max(int(half*0.8),2)
    #   hidden_states = hidden_states_collapse(self.hidden_states,
    #                                          {"match":Match.ALL.value, 
    #                                           "layer":Layer.LAST.value, 
    #                                           "answered_letter":letter})
    #   nn.append(self.nearest_neighbour(hidden_states, k=k))
    #   half = int(hidden_states.shape[0]/2)
    #   nn_half1.append(self.nearest_neighbour(hidden_states[:half], k=k))
    #   nn_half2.append(self.nearest_neighbour(hidden_states[half:], k=k))
    # overlap = np.empty([4,4])
    # warnings.warn("Computing overlap for -5th layer and using 80 %% of the instances")
    # for i in tqdm.tqdm(range(4), desc = "Computing overlap"):
    #   for j in range (4):
    #     if i==j:
    #       overlap[i,j] = neig_overlap(nn_half1[i][-5,], nn_half2[j][-5])
    #     else:
    #       overlap[i,j] = neig_overlap(nn[i][-5,], nn[j][-5])
    
    # return overlap
  
