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

# def hidden_states_collapse(instances_hiddenstates,method)-> np.ndarray:
#     """
#     Collect hidden states of all instances and collapse them in one tensor
#     using the provided method

#     Parameters
#     ----------
#     instances_hiddenstates: List[HiddenGeometry]
#     method: last or sum --> what method to use to collapse hidden states

#     Output
#     ----------
#     (num_instances, num_layers, model_dim)
#     """ 
#     assert method in ["last", "sum"], "method must be last or sum"
#     hidden_states = []
#     for i in  instances_hiddenstates:
#         #collect only test question tokens
#         instance_hidden_states = i.hidden_states[:,-i.len_tokens_question:,:]
#         if method == "last":
#             hidden_states.append(instance_hidden_states[:,-1,:])
#         elif method == "sum":
#             hidden_states.append(reduce(instance_hidden_states, "l s d -> l d", "mean"))
            
#     # (num_instances, num_layers, model_dim)
#     hidden_states = torch.stack(hidden_states)
#     return hidden_states.detach().cpu().numpy()

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
    #import pdb; pdb.set_trace()
    for condition in query.keys():
        if condition == "match" and query[condition] == Match.ALL.value:
            continue
        df_hiddenstates = df_hiddenstates[df_hiddenstates[condition] == query[condition]]
    hidden_states = df_hiddenstates["hidden_states"].tolist()
    #import pdb; pdb.set_trace()
    return np.stack(hidden_states)

    # for some reason huggingface does not the full list of activations
    #counter = Counter([i.shape[0] for i in hidden_states])
    # Find the most common element
    #most_common_element = counter.most_common(1)[0][0]
    #hidden_states = list(filter(lambda x: x.shape[0] == most_common_element, hidden_states))
    # (num_instances, num_layers, model_dim)
    #hidden_states = torch.stack(hidden_states)
    #print(f'{len(hidden_states)} instances with {match} hidden states')
    #hidden_states = np.stack(hidden_states)
    #return hidden_states.detach().cpu().numpy()
def hidden_states_process(instance_hiddenstates: Dict)-> Dict:
    """
    Collect hidden states of all instances and collapse them in one tensor
    using the provided method

    Parameters
    ----------
    instances_hiddenstates: Dict.Keys: ["len_tokens_question", "hidden_states"] 
    Output
    ----------
    Dict.Keys: ["last", "sum"] --> (num_layers, model_dim)
    """ 
    len_tokens_question, hidden_states = instance_hiddenstates.values() 
    hidden_states = hidden_states.detach().cpu().numpy()
    out = {}
    out["last"] = hidden_states[:,-1,:]
    out["sum"] = reduce(hidden_states[:,-len_tokens_question:,:], "l s d -> l d", "mean")
    return out

def exact_match(instance, request_state):
    """
    Check if the generated answer is correct
    """
    gold = list(filter(lambda a: a.tags and a.tags[0] == "correct", instance.references))[0].output.text
    pred_index = request_state.result.completions[0].text.strip()
    pred = request_state.output_mapping.get(pred_index)
    if not pred:
        return "wrong"
    return "correct" if gold == pred else "wrong"

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
    lenght = min(X.shape[0],Y.shape[0])
    X = X[:lenght,:]
    Y = Y[:lenght,:]
    assert X.shape[0] == Y.shape[0]
    ndata = X.shape[0]
    # Is this correct?
    k = X.shape[1]
    iter = map(lambda x,y : np.intersect1d(x,y).shape[0]/k, X,Y)
    out = functools.reduce(lambda x,y: x+y, iter)
    return out/ndata

    # prompt = request_state.request.prompt
    # index_in_prompt = prompt.rfind("Question")
    # tokens_question = tokenizer(prompt[index_in_prompt:], return_tensors="pt", return_token_type_ids=False)
    # len_tokens_question = tokens_question["input_ids"].shape[1]
    # # generation
    # output = request_state.result.completions[0]
    # answer = tokenizer.decode(output.text).strip()
    # reference_index = request_state.output_mapping.get(answer, "incorrect")
    # if reference_index != "incorrect":
    #     result = list(filter(lambda a: a.output.text == reference_index, request_state.instance.references))
    #     if result and result[0].tags and result[0].tags[0] == "correct":
    #         return True
    # return False


