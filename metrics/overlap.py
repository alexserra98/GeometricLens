from metrics.hidden_states_metrics import HiddenStatesMetrics
from .utils import  hidden_states_collapse
from metrics.query import DataFrameQuery
from common.globals_vars import _NUM_PROC

from dadapy.data import Data
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

import tqdm
import pandas as pd
import numpy as np
import warnings
import time
from functools import partial
from joblib import Parallel, delayed

class PointOverlap(HiddenStatesMetrics):
       
    def main(self) -> pd.DataFrame:
        """
        Compute the overlap between same dataset, same train instances, different models (pretrained and finetuned)
        
        Parameters
        ----------
        data: Dict[model, Dict[dataset, Dict[train_instances, Dict[method, Dict[match, Dict[layer, np.ndarray]]]]]]
        Output
        df: pd.DataFrame (k,dataset,method,train_instances_i,train_instances_j,overlap)
        """
        #warn("Computing overlap using k with  2 -25- 500")
        comparison_metrics = {"adjusted_rand_score":adjusted_rand_score, 
                              "adjusted_mutual_info_score":adjusted_mutual_info_score, 
                              "mutual_info_score":mutual_info_score}
        
        iter_list = [5,10,30,100,500,1000]
        #iter_list = [30]
        #iter_list = [100,250,500,1000]
        rows = []
        for k in tqdm.tqdm(iter_list, desc = "Computing overlaps k"):
            for couples in self.pair_names(self.df["model_name"].unique().tolist()):
                if '13' in couples[0]:
                    continue
                for method in ["last"]:#self.df["method"].unique().tolist():
                    if couples[0]==couples[1] and "chat" in couples[0]:
                        
                        iterlist = [("4","5"),("0","5")]
                    else:
                        iterlist = [("5","0"),("0","5"),("5","5"),("5","0")]
                        continue
                    for shot in iterlist:
                        train_instances_i, train_instances_j = shot
                        query_i = DataFrameQuery({"method":method,
                                "model_name":couples[0], 
                                "train_instances": train_instances_i,
                                "dataset": "mmlu:miscellaneous"})
                        query_j = DataFrameQuery({"method":method,
                                "model_name":couples[1], 
                                "train_instances": train_instances_j,
                                "dataset": "mmlu:miscellaneous"})
                        
                        
                        hidden_states_i, _, df_i = hidden_states_collapse(self.df,query_i, self.tensor_storage)
                        hidden_states_j, _, df_j = hidden_states_collapse(self.df,query_j, self.tensor_storage)
                        #id_instance_i = list(map(lambda k: k[:64],df_i["id_instance"].tolist()))
                        #id_instance_j = list(map(lambda k: k[:64],df_j["id_instance"].tolist()))
                        #
                        #if id_instance_i != id_instance_j:
                        #    id_instance_i = sorted(id_instance_i)
                        #    id_instance_j = sorted(id_instance_j)    
                        #    print(f'{couples}--{shot}--{k}')
                        #    assert id_instance_i == id_instance_j, "The two runs must have the same instances"
                        
                        id_instance_i = list(map(lambda k: k[:64],df_i["id_instance"].tolist()))
                        id_instance_j = list(map(lambda k: k[:64],df_j["id_instance"].tolist()))
                        #import pdb; pdb.set_trace()
                        if id_instance_i!=id_instance_j:
                           indices = [id_instance_j.index(id_instance_i[k]) for k in range(len(id_instance_i))] 
                           hidden_states_i=hidden_states_i[np.array(indices)]
                        #df_i.reset_index(inplace=True)
                        #df_j.reset_indiex(inplace=True):q

                        #df_i = df_i.where(df_j.only_ref_pred == df_i.only_ref_pred)
                        #df_j = df_j.where(df_j.only_ref_pred == df_i.only_ref_pred) 
                        #df_i.dropna(inplace=True)
                        #df_j.dropna(inplace=True)
                        #hidden_states_i, _, df_i = hidden_states_collapse(df_i,query_i, self.tensor_storage)
                        #hidden_states_j, _, df_j = hidden_states_collapse(df_j,query_j, self.tensor_storage)

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

        import pdb;pdb.set_trace()
        with Parallel(n_jobs=_NUM_PROC) as parallel:
            results = parallel(delayed(process_layer)(layer) for layer in range(number_of_layers))
        #results = []
        #for layer in range(number_of_layers):
        #    results.append(process_layer(layer))
        overlaps = list(results)
        
        return np.stack(overlaps)
    
    def process_layer(self, layer, data_i, data_j, k) -> np.ndarray:
        """
        Process a single layer.
        """

        data = Data(data_i[:,layer,:])
        #warnings.filterwarnings("ignore")
        data.compute_distances(maxk=k)
        #print(f'{k} -- {data_j[:,layer,:]}')
        overlap = data.return_data_overlap(data_j[:,layer,:],k=k)
        return overlap
    
    
class LabelOverlap(HiddenStatesMetrics):
      
    def main(self,label) -> pd.DataFrame:
        """
        Compute the overlap between the layers of instances in which the model answered with the same letter
        Output
        ----------
        Dict[layer: List[Array(num_layers, num_layers)]]
        """
        self.label = label
        #The last token is always the same, thus its first layer activation (embedding) is always the same
        iter_list=[0.05,0.10,0.20,0.50]
        rows = []
        
        for class_fraction in tqdm.tqdm(iter_list, desc = "Computing overlap"):
            for model in self.df["model_name"].unique().tolist():
                if "13" in model:
                    continue 
                for method in ["last"]: #self.df["method"].unique().tolist():
                    for train_instances in ["0","2","5"]:#self.df["train_instances"].unique().tolist():
                        query = DataFrameQuery({"method":method,
                                                "model_name":model,
                                                "train_instances": train_instances}) 
                                    
                    hidden_states, logits, hidden_states_df= hidden_states_collapse(self.df,query, self.tensor_storage)
                    row = [model, method, train_instances, class_fraction]
                    label_per_row = self.constructing_labels(hidden_states_df, hidden_states)
                
                    overlap = self.parallel_compute(hidden_states, label_per_row, class_fraction=class_fraction) 
                    row.append(overlap)
                    rows.append(row)
                        
        df = pd.DataFrame(rows, columns = ["model",
                                        "method",
                                        "train_instances",
                                        "class_fraction",
                                        "overlap"]) 

        return df

    def constructing_labels(self, hidden_states_df: pd.DataFrame, hidden_states: np.ndarray) -> np.ndarray:
        labels_literals = hidden_states_df[self.label].unique()
        labels_literals.sort()
        
        map_labels = {class_name: n for n,class_name in enumerate(labels_literals)}
        
        label_per_row = hidden_states_df[self.label].reset_index(drop=True)
        label_per_row = np.array([map_labels[class_name] for class_name in label_per_row])[:hidden_states.shape[0]]
        
        return label_per_row

    def parallel_compute(self, hidden_states: np.ndarray, labels: np.array,  class_fraction: float ) -> np.ndarray:
        overlaps = []
        if class_fraction is None:
            raise ValueError("You must provide either k or class_fraction")
        start_time = time.time()
        process_layer = partial(self.process_layer, hidden_states = hidden_states,labels = labels, class_fraction = class_fraction)
        with Parallel(n_jobs=_NUM_PROC) as parallel:
            results = parallel(delayed(process_layer)(num_layer) for num_layer in range(hidden_states.shape[1]))
        end_time = time.time()
        print(f"Label overlap over batch of data took: {end_time-start_time}")
        overlaps = list(results)

        return np.stack(overlaps)

    def process_layer(self, num_layer, hidden_states, labels, class_fraction):
        """
        Process a single layer.
        """
        # Variation can be used to introduce temporary modifications in commonly used methods
        if not self.variations["overlap"]:
            data = Data(hidden_states[:, num_layer, :])
            #warnings.filterwarnings("ignore")
            overlap = data.return_label_overlap(labels, class_fraction=class_fraction)
        elif self.variations["overlap"] == "norm":
            hidden_states_norm = hidden_states[:, num_layer, :] / np.linalg.norm(hidden_states[:, num_layer, :], axis=1, keepdims=True)
            data = Data(hidden_states_norm)
            overlap = data.return_label_overlap(labels, class_fraction=class_fraction)
        else:
            raise ValueError("Unknown variation. It must be either None or 'norm'")
        
        return overlap
