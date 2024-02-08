from dadapy.data import Data
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NamedTuple
import pandas as pd
import functools
from warnings import warn
from functools import partial
from metrics.query import DataFrameQuery
from common.tensor_storage import TensorStorage
import time
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
    
def softmax(logits):
    # Exponentiate each element
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    # Normalize each row to get probabilities
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
@dataclass
class InstanceHiddenStates():
  match: Match
  hidden_states: Dict[str, np.ndarray]


def hidden_states_collapse(df_hiddenstates: pd.DataFrame(), 
                           query: DataFrameQuery, 
                           tensor_storage: TensorStorage)-> np.ndarray:
    """
    Collect hidden states of all instances and collapse them in one tensor
    using the provided method

    Parameters
    ----------
    df_hiddenstates: pd.DataFrame(hiddens_states, match, layer, answered letter, gold letter)
                     dataframe containing the hidden states of all instances
    query: DataFrameQuery --> 
    Output
    ----------
    (num_instances, num_layers, model_dim)
    """ 

    df_hiddenstates = query.apply_query(df_hiddenstates)
    hidden_states = []
    start_time = time.time()
    rows = [row for _, row in df_hiddenstates.iterrows()]
    id_instances =  [row["id_instance"] for _, row in df_hiddenstates.iterrows()]
    hidden_state_path_rows =[[f'{row["model_name"].replace("/","-")}/{row["dataset"]}/{row["train_instances"]}/hidden_states', 
                              row["id_instance"]] 
                             for row in rows]
    hidden_state_path = pd.DataFrame(hidden_state_path_rows, columns = ["path", "id_instance"])
    id_instances_check = []
    hidden_states = []
    # for row in df_hiddenstates.iterrows():
    #     row = row[1]
    #     path = f'{row["model_name"].replace("/","-")}/{row["dataset"]}/{row["train_instances"]}/hidden_states'
    #     id_instances_check.append( row["id_instance"])
    #     hidden_states.append(tensor_storage.load_tensor(path, row["id_instance"] ))
    for path in hidden_state_path["path"].unique():
      id_instances_check.extend( hidden_state_path[hidden_state_path["path"] == path]["id_instance"])
      hidden_states.extend(tensor_storage.load_tensors(path, hidden_state_path[hidden_state_path["path"] == path]["id_instance"].tolist()))
    end_time = time.time()
    assert id_instances == id_instances_check, "The order of the instances is not the same"
    print(f" Tensor retrieval took: {end_time-start_time}\n")
    return np.stack(hidden_states),None, df_hiddenstates


def exact_match(answers, letters_gold):
    return answers.strip() == letters_gold

def quasi_exact_match(answers, letters_gold):
    is_in_string = answers.strip().lower() == letters_gold.strip().lower()
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

def label_neig_overlap(nn_matrix: np.ndarray, labels: NamedTuple, subject_per_row: pd.Series) -> np.ndarray:
    """
    Computes the fraction of neighbours of label2 in the nearest neighbours of label1.
    Parameters
    ----------
    nn_matrix : 2D array of ints
        nearest neighbor labels matrix 
    labels : List[str]
        we want to compute the fraction of labels[1] in labels[0] neighbours
    list_of_labels : pd.Series
        pandas Series of labels associated with each row in the nearest neighbour matrix
    """
    index_label =  subject_per_row.index[subject_per_row==labels.current_label].tolist()
    nn_matrix = nn_matrix[index_label]==labels.label_to_find
    out = nn_matrix.sum()/nn_matrix.shape[0]
    return out


    
def layer_overlap(nn1, nn2) -> np.ndarray:
    assert nn1.shape == nn2.shape, "The two nearest neighbour matrix must have the same shape" 
    layers_len = nn1.shape[0]
    overlaps = np.empty([layers_len])
    for i in range(layers_len):
        overlaps[i] = neig_overlap(nn1[i], nn2[i])
    return overlaps
  
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout