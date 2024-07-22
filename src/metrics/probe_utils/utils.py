from metrics.utils import hidden_states_collapse, TensorStorageManager
from metrics.query import DataFrameQuery
from common.tensor_storage import TensorStorage
from common.metadata_db import MetadataDB
from common.utils import *

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


from pathlib import Path


_PATH = Path("/orfeo/scratch/dssc/zenocosini/mmlu_result/")
#_PATH = Path("/home/alexserra98/helm-suite/mmlu_result/")
_TENSOR_STORAGE = TensorStorage(Path(_PATH, "tensor_files"))

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

def retrieve_df(label,model, method="last", train_instances=0):
    metadata_db = MetadataDB(_PATH / "metadata.db")
    metadata_df = set_dataframes(metadata_db)

    dict_query = { "method":method,
                   "model_name":model,
                   "train_instances": train_instances}
    query = DataFrameQuery(dict_query)
    tsm = TensorStorageManager()
    X,_, df = tsm.retrieve_tensor(query, "npy")
    if label == "subject":
        y = np.asarray(df['dataset'].tolist())
    elif label == "letter":
        y = np.asarray(df['only_ref_pred'].tolist())
    return X, y

def stratified_split(df_hiddenstates):
    # Splitting the dataset into train and test sets for each subject
    train_frames = []
    test_frames = []

    for subject in df_hiddenstates['dataset'].unique():
        subject_data = df_hiddenstates[df_hiddenstates['dataset'] == subject]
        train, test = train_test_split(subject_data, test_size=0.2, random_state=42)
        train_frames.append(train)
        test_frames.append(test)

    train_df = pd.concat(train_frames)
    test_df = pd.concat(test_frames)

    return train_df, test_df

 
