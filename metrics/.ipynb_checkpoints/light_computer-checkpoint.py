from metrics.utils import hidden_states_collapse
from metrics.query import DataFrameQuery
from common.tensor_storage import TensorStorage
from common.metadata_db import MetadataDB
from metrics.intrinisic_dimension import *
from metrics.hidden_states_metrics import *

from pathlib  import Path

import numpy as np

from common.metadata_db import MetadataDB
from common.utils import *
from pathlib import Path
import pickle

with open("/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens/repo/results/mmlu/llama-2-7b/0shot/statistics_target.pkl","rb") as f:
    statistic_target = pickle.load(f)
statistic_target
df = pd.DataFrame(statistic_target)
df = df.rename(columns={"subjects":"dataset", "predictions":"std_pred","answers":"letter_gold", "contrained_predictions":"only_ref_pred",})
df["train_instances"] = 0
df["model_name"] = "meta-llama-Llama-2-7b-hf"
df["method"] = "last"
df.to_pickle("/orfeo/scratch/dssc/zenocosini/mmlu_result/transposed_dataset/df.pkl")

id = IntrinsicDimension(df = df, tensor_storage = None, variations=None, storage_logic="npy")
result = id.main()
result.to_pickle(Path("/orfeo/scratch/dssc/zenocosini/mmlu_result/transposed_dataset/result",id.pkl'))