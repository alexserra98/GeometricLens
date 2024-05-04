import sys

sys.path.append("/u/dssc/zenocosini/helm_suite/MCQA_Benchmark")
from metrics.utils import *
from metrics.query import DataFrameQuery
from common.tensor_storage import TensorStorage
from common.metadata_db import MetadataDB
from common.utils import *

tsm = TensorStorageManager()
dict_query = {
    "method": "last",
    "model_name": "meta-llama/Llama-3-8b-ft-hf",
    "train_instances": "0",
}
query = DataFrameQuery(dict_query)
hidden_states, logits, hidden_states_stat = tsm.retrieve_tensor(query, "npy")
print(hidden_states.shape)
