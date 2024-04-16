from metrics.utils import hidden_states_collapse
from metrics.query import DataFrameQuery
from common.tensor_storage import TensorStorage
from common.metadata_db import MetadataDB
from common.utils import *
from metrics.utils import  exact_match, angular_distance

#from sklearn.feature_selection import mutual_info_regression MISSIN?
from dadapy.data import Data

from pathlib  import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from pathlib import Path
import pickle
import joblib
import logging
import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')

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

def tensor_retrieve(dict_query):
    query = DataFrameQuery(dict_query)
    hidden_states,logits, hidden_states_df= hidden_states_collapse(df_hiddenstates=metadata_df,query=query,tensor_storage=tensor_storage)
    return hidden_states,logits,hidden_states_df

def constructing_labels(label: str, hidden_states_df: pd.DataFrame, hidden_states: np.ndarray) -> np.ndarray:
    labels_literals = hidden_states_df[label].unique()
    labels_literals.sort()
    
    map_labels = {class_name: n for n,class_name in enumerate(labels_literals)}
    
    label_per_row = hidden_states_df[label].reset_index(drop=True)
    label_per_row = np.array([map_labels[class_name] for class_name in label_per_row])[:hidden_states.shape[0]]
    
    return label_per_row, map_labels

_PATH = Path("/orfeo/scratch/dssc/zenocosini/mmlu_result/")
result_path = Path(_PATH,"log_reg")
result_path.mkdir(exist_ok=True,parents=True)
metadata_db = MetadataDB(_PATH / "metadata.db")
metadata_df = set_dataframes(metadata_db)
tensor_storage = TensorStorage(Path(_PATH, "tensor_files"))

dict_query = { "method":"last",
              "model_name":"meta-llama/Llama-2-7b-hf",
              "train_instances": 0}
query = DataFrameQuery(dict_query)
df_hiddenstates = query.apply_query(metadata_df)

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


# Train set
X_train, _, train_df = hidden_states_collapse(df_hiddenstates=train_df,tensor_storage=tensor_storage)
y_train = np.asarray(train_df['only_ref_pred'].tolist())
# Test set 
X_test, _, test_df = hidden_states_collapse(df_hiddenstates=test_df,tensor_storage=tensor_storage)
y_test = np.asarray(test_df['only_ref_pred'].tolist())

models = []
performance_per_layer = []

classes = train_df['only_ref_pred'].unique()
weights = compute_class_weight('balanced', classes=classes, y=train_df['only_ref_pred'])
class_weight = dict(zip(classes, weights))
accuracies = {}

for layer in tqdm.tqdm(range(X_train.shape[1]), desc = "Computing probe..."):
    # Preparing the logistic regression model
    model = LogisticRegression(max_iter=5000, class_weight=class_weight)  # Increased max_iter to ensure convergence
    
    # Training the model
    model.fit(X_train[:,layer,:], y_train)
    
    # Testing the model
    predictions = model.predict(X_test[:,layer,:])
    
    # Calculating the accuracy of the model
    accuracy = accuracy_score(y_test, predictions)
    accuracies[layer] = accuracy

    joblib.dump(model, Path(result_path, f"model_layer{layer}.joblib"))

with open(Path(result_path, "accuracies.pkl"), "wb") as f:
    pickle.dump(accuracies, f)
# Pretty print the accuracies
logging.info(f'The accuracy of the logistic regression model at each layers is')
accuracies_df = pd.DataFrame(list(accuracies.items()), columns=['Layer', 'Accuracy'])
print(accuracies_df.to_string(index=False))

