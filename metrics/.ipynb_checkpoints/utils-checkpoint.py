from metrics.query import DataFrameQuery
from common.tensor_storage import TensorStorage
from common.globals_vars import _DEBUG

from dadapy.data import Data

import numpy as np
import torch
import pandas as pd
from einops import rearrange

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NamedTuple
import functools
from warnings import warn
from functools import partial
import time
import sys
import os
from pathlib import Path
import os
import re
import pickle

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

class TensorStorageManager:
    def __init__(self, storage_config_h5: TensorStorage = None, storage_config_npy: Path = None):
        # Initialization with storage configurations or connections
        self.storage_h5 = storage_config_h5
        self.storage_npy = storage_config_npy


    def retrieve_from_storage_h5(self, df_hiddenstates: pd.DataFrame, query: DataFrameQuery = None) -> tuple:
        """
        Collect hidden states of all instances and collapse them in one tensor
        using the provided method.

        Parameters
        ----------
        df_hiddenstates : pd.DataFrame
            DataFrame containing the hidden states of all instances
            with columns: hiddens_states, match, layer, answered letter, gold letter.
        query : DataFrameQuery
            A query object to filter df_hiddenstates; not used if None.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, pd.DataFrame)
            Returns a tuple containing arrays of hidden states, logits, and the original DataFrame.
        """
        if query is not None:
            df_hiddenstates = query.apply_query(df_hiddenstates)

        # Generate paths using vectorized operations
        df_hiddenstates['path_hidden_states'] = df_hiddenstates.apply(
            lambda row: f"{row['model_name'].replace('/', '-')}/{row['dataset']}/{row['train_instances']}/hidden_states",
            axis=1)
        df_hiddenstates['path_logits'] = df_hiddenstates.apply(
            lambda row: f"{row['model_name'].replace('/', '-')}/{row['dataset']}/{row['train_instances']}/logits",
            axis=1)

        hidden_states, logits = [], []
        start_time = time.time()

        # Process each unique path only once
        for path in df_hiddenstates['path_hidden_states'].unique():
            ids = df_hiddenstates.loc[df_hiddenstates['path_hidden_states'] == path, 'id_instance']
            hidden_states.append(self.storage_h5.load_tensors(path, ids.tolist()))

        for path in df_hiddenstates['path_logits'].unique():
            ids = df_hiddenstates.loc[df_hiddenstates['path_logits'] == path, 'id_instance']
            logits.append(self.storage_h5.load_tensors(path, ids.tolist()))

        end_time = time.time()

        # Ensure order is preserved using try-except
        try:
            hidden_states = np.concatenate(hidden_states)
            logits = np.concatenate(logits)
            assert (df_hiddenstates['id_instance'].tolist() == ids.tolist()), "The order of the instances is not the same"
        except AssertionError as e:
            print(f"Error in tensor retrieval: {str(e)}")
            # Handle error or reprocess as needed
            # This could involve reordering the arrays based on the original DataFrame order
            # Or any other corrective action as deemed necessary

        # Debugging information
        if _DEBUG:
            print(f"Tensor retrieval took: {end_time - start_time:.2f} seconds")

        return hidden_states, logits, df_hiddenstates
        
    def retrieve_from_storage_npy(self, query):
        query_dict = query.query
        #TODO - Adapt path to query
        #import pdb; pdb.set_trace()
        adapted_name = query_dict["model_name"][11:].lower()
        adapted_name = adapted_name[:-3]
        storage_path = Path(f"/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens/repo/results/mmlu/{adapted_name}/{query_dict['train_instances']}shot")
        files = os.listdir(storage_path)

        # Filter files with the specific pattern and range check
        pattern = re.compile(r'l(\d+)_target\.pt')
        filtered_files = [file for file in files if pattern.match(file)]

        # Sort files based on the number in the filename
        filtered_files.sort(key=lambda x: int(pattern.match(x).group(1)))

        # Load tensors and add them to a list
        tensors = [torch.load(os.path.join(storage_path, file)) for file in filtered_files]

        # Stack all tensors along a new dimension
        stacked_tensor = torch.stack(tensors[:-1])
        stacked_tensor = rearrange(stacked_tensor, 'l n d -> n l d')
        logits = tensors[-1]

        #retrieve statistics
        with open(Path(storage_path,"statistics_target.pkl"), "rb") as f:
            stat_target = pickle.load(f)
            
        for key in stat_target.keys():
            if isinstance(stat_target[key],float):
                continue
            stat_target[key] = stat_target[key][:14039] 
        
        #import pdb; pdb.set_trace()    
        df = pd.DataFrame(stat_target)
        df = df.rename(columns={"subjects":"dataset", "predictions":"std_pred","answers":"letter_gold", "contrained_predictions":"only_ref_pred",})
        df["train_instances"] = query_dict["train_instances"]
        df["model_name"] = query_dict["model_name"] #"meta-llama-Llama-2-7b-hf"
        df["method"] = "last"
        
        #df = pd.read_pickle("/orfeo/scratch/dssc/zenocosini/mmlu_result/transposed_dataset/df.pkl")

        return stacked_tensor.float().numpy()[:14039], logits.float().numpy()[:14039], df
   
    def retrieve_tensor(self, query, criteria):
        # Decision logic to choose the storage
        if criteria == 'h5':
            return self.retrieve_from_storage_h5(query)
        elif criteria == 'npy':
            return self.retrieve_from_storage_npy(query)
        
        
def hidden_states_collapse( df_hiddenstates: pd.DataFrame(), 
                            tensor_storage: TensorStorage,
                            query: DataFrameQuery = None) -> np.ndarray:

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
    hidden_states: (num_instances, num_layers, model_dim),
    none - it is a placeholder for future use
    df_hiddenstates: pd.DataFrame(hiddens_states, match, layer, answered letter, gold letter)
                     dataframe containing the hidden states of all instances
    """ 
    if  query is not None:
        df_hiddenstates = query.apply_query(df_hiddenstates)
    
    hidden_states = []
    start_time = time.time()
    rows = [row for _, row in df_hiddenstates.iterrows()]
    id_instances =  [row["id_instance"] for _, row in df_hiddenstates.iterrows()]
    hidden_state_path_rows =[[f'{row["model_name"].replace("/","-")}/{row["dataset"]}/{row["train_instances"]}/hidden_states', 
                              row["id_instance"]] 
                             for row in rows]
    logits_path_rows =[[f'{row["model_name"].replace("/","-")}/{row["dataset"]}/{row["train_instances"]}/logits', 
                              row["id_instance"]] 
                             for row in rows]
    hidden_state_path = pd.DataFrame(hidden_state_path_rows, columns = ["path", "id_instance"])
    logits_path = pd.DataFrame(logits_path_rows, columns = ["path", "id_instance"])
    id_instances_check = []
    hidden_states = []
    logits = []
        #import pdb; pdb.set_trace()
    for path,path_logits in zip(hidden_state_path["path"].unique(),logits_path["path"].unique()):
      id_instances_check.extend( hidden_state_path[hidden_state_path["path"] == path]["id_instance"])
      hidden_states.extend(tensor_storage.load_tensors(path, hidden_state_path[hidden_state_path["path"] == path]["id_instance"].tolist()))

      logits.extend(tensor_storage.load_tensors(path_logits, logits_path[logits_path["path"] == path_logits]["id_instance"].tolist()))
    end_time = time.time()
    if id_instances != id_instances_check:
      indices = [id_instances_check.index(i) for i in id_instances]
      hidden_states = [hidden_states[i] for i in indices]
      logits = [logits[i] for i in indices]
      id_instances_check = [id_instances_check [i] for i in indices]

      #print("The order of the instances is not the same, switch to long method" )
      #hidden_states = []
      #id_instances_check = []
      #for row in df_hiddenstates.iterrows():
      #  row = row[1]
      #  path = f'{row["model_name"].replace("/","-")}/{row["dataset"]}/{row["train_instances"]}/hidden_states'
      #  id_instances_check.append( row["id_instance"])
      #  hidden_states.append(tensor_storage.load_tensor(path, row["id_instance"] ))


    assert id_instances == id_instances_check, "The order of the instances is not the same"
    print(f" Tensor retrieval took: {end_time-start_time}\n")
    return np.stack(hidden_states), np.stack(logits), df_hiddenstates


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

def angular_distance(mat):
    """
    Computes distance based on angles between vectors, over the rows of the matrix
    """
    dot_product = mat @ mat.T
    norm_vector = np.linalg.norm(mat, axis=1)
    stacked_vector = np.tile(norm_vector, (100, 1))
    norm_product = stacked_vector.T*stacked_vector

    cosine_similarity = dot_product / norm_product
    cosine_similarity = np.clip(cosine_similarity, -1, 1)
    distances = np.arccos(cosine_similarity) / np.pi
    distances.sort(axis=1)
    return mat

