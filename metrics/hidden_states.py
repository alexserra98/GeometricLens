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
from .utils import compute_id, hidden_states_collapse, Match, Layer, neig_overlap, label_neig_overlap

import tqdm
from inference_id.generation.generation import RequestResult
import pandas as pd
import warnings

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
    hidden_states = hidden_states_collapse(hidden_states,query)
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

def hidden_states_collapse(df_hiddenstates: pd.DataFrame(), query: Dict)-> np.ndarray:
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
    for condition in query.keys():
        if condition == "match" and query[condition] == Match.ALL.value:
            continue
        df_hiddenstates = df_hiddenstates[df_hiddenstates[condition] == query[condition]]
    hidden_states = df_hiddenstates["hidden_states"].tolist()
    return np.stack(hidden_states)



class HiddenStates():
  def __init__(self,hidden_states):
    self.hidden_states = hidden_states
  
  #TODO use a decorator
  def preprocess_hiddenstates(self,func, query, *args, **kwargs):
    hidden_states_collapsed = hidden_states_collapse(self.hidden_states, query)

    return func(hidden_states_collapsed, *args, **kwargs)

  
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
  
  def labelize_nearest_neighbour(self, neigh_matrix_list: np.ndarray, labels_dict: dict) -> np.ndarray:
    def substitute_with_dict(matrix, dictionary):
      vectorized_get = np.vectorize(dictionary.get)
      return vectorized_get(matrix)

    # Apply the function to each matrix
    substituted_matrices = substitute_with_dict(neigh_matrix_list, labels_dict)
    return np.stack(substituted_matrices)
  
  def layer_overlap_label(self,label) -> Dict[str, List[np.ndarray]]:
    """
    Compute the overlap between the layers of instances in which the model answered with the same letter
    Output
    ----------
    Dict[layer: List[Array(num_layers, num_layers)]]
    """
    overlaps = {}
    for layer in [Layer.LAST, Layer.SUM]:
      hidden_states = hidden_states_collapse(self.hidden_states,{"match":Match.ALL.value, "layer":layer.value})
      k = max(int(hidden_states.shape[0]*0.8),2)
      nn = self.nearest_neighbour(hidden_states, k=k)
      labels_dict = {i: self.hidden_states.iloc[i][label] for i in range(hidden_states.shape[0])}
      nn = self.labelize_nearest_neighbour(nn, labels_dict)
      overlaps[layer.value] = []
      for num_layer in tqdm.tqdm(range(hidden_states.shape[1]), desc = "Computing overlap"):
        overlap = np.empty([self.hidden_states[label].nunique(),self.hidden_states[label].nunique()])  
        for n,i in enumerate(self.hidden_states[label].unique()):
          for m,j in enumerate(self.hidden_states[label].unique()):
            overlap[n,m] = label_neig_overlap(nn[num_layer],[i,j], labels_dict)
        overlaps[layer.value].append(overlap)
    
    return overlaps
    
    
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
  
