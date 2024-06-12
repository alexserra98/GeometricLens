from metrics.hidden_states_metrics import HiddenStatesMetrics
from ..utils import  hidden_states_collapse, exact_match, angular_distance
from metrics.query import DataFrameQuery
from common.globals_vars import _NUM_PROC
from .utils_cka import *

from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from dadapy import Data

import tqdm
import pandas as pd
import numpy as np
import warnings
import time
from functools import partial
from joblib import Parallel, delayed

class CenteredKernelAlignement(HiddenStatesMetrics):
       
    def main(self) -> pd.DataFrame:
        """
        Compute the overlap between same dataset, same train instances, different models (pretrained and finetuned)
        
        Parameters
        ----------
        data: Dict[model, Dict[dataset, Dict[train_instances, Dict[method, Dict[match, Dict[layer, np.ndarray]]]]]]
        Output
        df: pd.DataFrame (k,dataset,method,train_instances_i,train_instances_j,overlap)
        """

        rows = []
        if self.variations["cka"]=="rbf":
            iter_list = [0.1, 1, 10]
        else:
            iter_list = [1]
        for k in tqdm.tqdm(iter_list, desc = "Computing overlaps k"):
            for couples in self.pair_names(self.df["model_name"].unique().tolist()):
                if '13' in couples[0]:
                    continue
                for method in ["last"]:#self.df["method"].unique().tolist():
                    if couples[0]==couples[1]:
                        iterlist = [("0","5")]
                    else:
                        iterlist = [("5","0"),("0","5"),("5","5"),("5","0")]
                    for shot in iterlist:
                        train_instances_i, train_instances_j = shot
                        query_i = DataFrameQuery({"method":method,
                                "model_name":couples[0], 
                                "train_instances": train_instances_i,})
                                #"dataset": "mmlu:miscellaneous"})
                        query_j = DataFrameQuery({"method":method,
                                "model_name":couples[1], 
                                "train_instances": train_instances_j,})
                                #"dataset": "mmlu:miscellaneous"})
                        
                        hidden_states_i, _, df_i = hidden_states_collapse(self.df,self.tensor_storage, query_i)
                        hidden_states_j, _, df_j = hidden_states_collapse(self.df, self.tensor_storage,query_j)

                        df_i.reset_index(inplace=True)
                        df_j.reset_index(inplace=True)

                        for row_i, row_j in zip(df_i.iterrows(), df_j.iterrows()):
                            assert row_i[1]["id_instance"].replace("chat-","")[:-1] == row_j[1]["id_instance"].replace("chat-","")[:-1], "The two runs must have the same instances"

                        #import pdb; pdb.set_trace()
                        df_i["exact_match"] = df_i.apply(lambda r: exact_match(r["std_pred"], r["letter_gold"]), axis=1)
                        df_j["exact_match"] = df_j.apply(lambda r: exact_match(r["std_pred"], r["letter_gold"]), axis=1)
                        # find the index of rows that have "exact_match" True in both df_i and df_j
                        indices_i = df_i[df_i["exact_match"] == True].index
                        indices_j = df_j[df_j["exact_match"] == True].index
                        # find the intersection of the two sets of indices
                        indices = indices_i.intersection(indices_j)
                        hidden_states_i = hidden_states_i[indices]
                        hidden_states_j = hidden_states_j[indices]
                
                        id_instance_i = list(map(lambda k: k[:64],df_i["id_instance"].tolist()))
                        id_instance_j = list(map(lambda k: k[:64],df_j["id_instance"].tolist()))
                    
                        if id_instance_i!=id_instance_j:
                           indices = [id_instance_j.index(id_instance_i[k]) for k in range(len(id_instance_i))] 
                           hidden_states_i=hidden_states_i[np.array(indices)]
                    
                        rows.append([k,
                                    couples,
                                    method,
                                    train_instances_i,
                                    train_instances_j, 
                                    self.parallel_compute(hidden_states_i, hidden_states_j, k)])
        df = pd.DataFrame(rows, columns = ["k",
                                           "couple",
                                           "method",
                                           "train_instances_i",
                                           "train_instances_j",
                                           "point_overlap"])
        return df
    
    def pair_names(self,names_list):
        """
        Pairs base names with their corresponding 'chat' versions.

        Args:
        names_list (list): A list of strings containing names.

        Returns:
        list: A list of tuples, each containing a base name and its 'chat' version.
        """
        # Separating base names and 'chat' names
        difference = 'chat'
        base_names = [name for name in names_list if difference not in name]
        chat_names = [name for name in names_list if difference in name]
        base_names.sort()
        chat_names.sort()
        # Pairing base names with their corresponding 'chat' versions
        pairs = []
        for base_name, chat_name in zip(base_names, chat_names):
            pairs.append((base_name, base_name))
            pairs.append((chat_name, chat_name))
            pairs.append((base_name, chat_name))
        return pairs
        
    def parallel_compute(self, data_i: np.ndarray, data_j:np.ndarray, k: int) -> np.ndarray:
        """
        Compute the overlap between two runs
        
        Parameters
        ----------
        data_i: np.ndarray(num_instances, num_layers, model_dim)
        data_j: np.ndarray(num_instances, num_layers, model_dim)
        k: int
        
        Output
        ----------
        Array(num_layers)
        """
        assert data_i.shape[1] == data_j.shape[1], "The two runs must have the same number of layers"
        number_of_layers = data_i.shape[1]
        process_layer = partial(self.process_layer, data_i = data_i, data_j = data_j, k=k) 
        with Parallel(n_jobs=_NUM_PROC) as parallel:
            results = parallel(delayed(process_layer)(layer) for layer in range(1,number_of_layers))
        #results = []
        #for layer in range(4,number_of_layers):
        #    results.append(process_layer(layer))
        
        overlaps = list(results)
        
        return np.stack(overlaps)
    
    def process_layer(self, layer, data_i, data_j, k) -> np.ndarray:
        """
        Process a single layer.
        """
        data_i = data_i[:,layer,:]
        data_j = data_j[:,layer,:]
        data_i = data_i/np.linalg.norm(data_i,axis=1,keepdims=True)
        data_j = data_j/np.linalg.norm(data_j,axis=1,keepdims=True)
        
        if self.variations["cka"]=="rbf":
            data = Data(data_i)
            data.compute_distances()
            avg_dist = np.average(data.distances[:,1])
            sigma = avg_dist * k
            cka_out = cka(gram_rbf(data_i, sigma), gram_rbf(data_j, sigma))
        else:
            # CKA from examples
            # cka = cka(gram_linear(X), gram_linear(Y))
            # CKA from features
            cka_out = feature_space_linear_cka(data_i, data_j)
        
        return cka_out
    
    
