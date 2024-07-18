import sys
sys.path.append("/u/dssc/zenocosini/helm_suite/MCQA_Benchmark")
from metrics.utils import *
from metrics.query import DataFrameQuery
from common.tensor_storage import TensorStorage
from common.metadata_db import MetadataDB
from common.utils import *

#from sklearn.feature_selection import mutual_info_regression MISSIN?
from dadapy.data import Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import KernelPCA
from sklearn.metrics import pairwise_distances
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from IPython.display import display
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from pathlib import Path
import pickle
from collections import Counter

def set_dataframes(db) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    df = pd.read_sql("SELECT * FROM metadata", db.conn)
    df["train_instances"] = df["train_instances"].astype(str)
    df.drop(columns=["id"],inplace = True)
    #import pdb; pdb.set_trace()
    df.drop_duplicates(subset = ["id_instance"],inplace = True, ignore_index = True) # why there are duplicates???
    return df

def find_identical_rows(matrix):
    # View the rows as a structured array to handle them as tuples
    dtype = [('row', matrix.dtype, matrix.shape[1])]
    structured_array = matrix.view(dtype)
    
    # Find unique rows and their indices
    _, inverse_indices = np.unique(structured_array, return_inverse=True)
    
    # Find where each row appears for the first time and their counts
    unique_rows, counts = np.unique(inverse_indices, return_counts=True)
    
    # Filter out unique rows, keeping only duplicates
    repeated_indices = [np.where(inverse_indices == i)[0] for i in unique_rows[counts > 1]]
    
    return repeated_indices

def filter_indentical_rows(hidden_states):
    repeated_indices = find_identical_rows(hidden_states)
    filtered_hidden_states = np.delete(hidden_states, np.concatenate(repeated_indices))
    return filtered_hidden_states, repeated_indices

def retrieve_labels(df: pd.DataFrame, query: dict) -> pd.DataFrame:
    """
    Retrieve labels from the metadata db
    """
    query = DataFrameQuery(query)
    db = MetadataDB()
    db.connect()
    labels = db.retrieve_labels(query)
    db.close()
    return labels