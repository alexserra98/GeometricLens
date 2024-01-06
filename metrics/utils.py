from dadapy.data import Data
from dadapy.plot import plot_inf_imb_plane
from dadapy.metric_comparisons import MetricComparisons
import numpy as np
import torch
from dataclasses import dataclass
from einops import reduce
from collections import Counter
from enum import Enum
from typing import Dict, List
import pandas as pd
import functools

@dataclass
class RunMeta():
  num_layers: int
  num_instances: int
  model_dim: int

class Match(Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    ALL = "all"
class Layer(Enum):
    LAST = "last"
    SUM = "sum"

    
@dataclass
class InstanceHiddenStates():
  match: Match
  hidden_states: Dict[str, np.ndarray]


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
    return  np.stack(id_per_layer[1:]) #skip first layer - for some reason it outputs a wrong number of IDs



def hidden_states_collapse(df_hiddenstates: pd.DataFrame(), query: Dict)-> np.ndarray:
    """
    Collect hidden states of all instances and collapse them in one tensor
    using the provided method

    Parameters
    ----------
    instances_hiddenstates: List[InstanceHiddenSates]
    method: last or sum --> what method to use to collapse hidden states

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


def exact_match(answers, letters_gold):
    return answers.strip() == letters_gold

def quasi_exact_match(answers, letters_gold):
    is_in_string = answers.strip().lower() in letters_gold.lower()
    return is_in_string



def neig_overlap(X, Y):
    """
    Computes the neighborhood overlap between two representations.
    Parameters
    ----------
    X : 2D array of ints
        nearest neighbor index matrix of the first representation
    Y : 2D array of ints
        nearest neighbor index matrix of the second representation
    
    Returns
    -------
    overlap : float
        neighborhood overlap between the two representations
    """
    assert X.shape[0] == Y.shape[0]
    ndata = X.shape[0]
    # Is this correct?
    k = X.shape[1]
    iter = map(lambda x,y : np.intersect1d(x,y).shape[0]/k, X,Y)
    out = functools.reduce(lambda x,y: x+y, iter)
    return out/ndata

def label_neig_overlap(nn_matrix: np.ndarray, labels: List[str], labels_dict: Dict[str, int]) -> np.ndarray:
    """
    Computes the fraction of neighbours of label2 in the nearest neighbours of label1.
    Parameters
    ----------
    nn_matrix : 2D array of ints
        nearest neighbor labels matrix 
    labels : List[str]
        we want to compute the fraction of labels[1] in labels[0] neighbours
    """
    index_label =  [k for k, v in labels_dict.items() if v == labels[0]]
    nn_matrix = nn_matrix[index_label]==labels[1]
    import pdb; pdb.set_trace() 
    return nn_matrix.sum()/nn_matrix.shape[0]



