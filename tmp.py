from metrics.utils import hidden_states_collapse
from metrics.query import DataFrameQuery
from common.tensor_storage import TensorStorage

# from sklearn.feature_selection import mutual_info_regression MISSIN?
from dadapy.data import Data

from pathlib import Path

import numpy as np
import pandas as pd

from common.metadata_db import MetadataDB
from common.utils import *
from pathlib import Path


def set_dataframes(db) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    df = pd.read_sql("SELECT * FROM metadata", db.conn)
    df["train_instances"] = df["train_instances"].astype(str)
    df.drop(columns=["id"], inplace=True)
    # import pdb; pdb.set_trace()
    df.drop_duplicates(
        subset=["id_instance"], inplace=True, ignore_index=True
    )  # why there are duplicates???
    return df


_PATH = Path("/orfeo/scratch/dssc/zenocosini/mmlu_result")
metadata_db = MetadataDB(_PATH / "metadata.db")
metadata_df = set_dataframes(metadata_db)
tensor_storage = TensorStorage(Path(_PATH, "tensor_files"))
