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
from multiprocessing import Pool
import warnings
import time

class PointOverlap(HiddenStatesMetrics):
    def __init__(self, df: pd.DataFrame, tensor_storage, label: str):
      super().__init__(df, tensor_storage)
      self.label = label
    
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
        
        iter_list = [5,10,30,100]
        rows = []
        for k in tqdm.tqdm(iter_list, desc = "Computing overlaps k"):
            for couples in self.pair_names(self.df["model_name"].unique().tolist()):
                for method in ["last"]:#self.df["method"].unique().tolist():
                    if couples[0]==couples[1]:
                        iterlist = [("0","5")]
                    else:
                        iterlist = [("0","0"),("0","5"),("5","5"),("5","0")]
                    for shot in iterlist:
                        train_instances_i, train_instances_j = shot
                        query_i = DataFrameQuery({"method":method,
                                "model_name":couples[0], 
                                "train_instances": train_instances_i,})
                        query_j = DataFrameQuery({"method":method,
                                "model_name":couples[1], 
                                "train_instances": train_instances_j,})
                        hidden_states_i, _,_ = hidden_states_collapse(self.df,query_i, self.tensor_storage)
                        hidden_states_j, _,_ = hidden_states_collapse(self.df,query_j, self.tensor_storage)
                        
                        rows.append([k,
                                    couples,
                                    method,
                                    train_instances_i,
                                    train_instances_j, 
                                    self._point_overlap(hidden_states_i, hidden_states_j, k)])
        df = pd.DataFrame(rows, columns = ["k",
                                           "couple",
                                           "method",
                                           "train_instances_i",
                                           "train_instances_j",
                                           "point_overlap"])
        return df
    
    def parallel_compute(self, data_i: np.ndarray, data_j:np.ndarray) -> np.ndarray:
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
     
        with Pool(processes=_NUM_PROC) as pool:
            results = pool.starmap(self.process_layer, [(layer, data_i, data_j, k) for layer in range(number_of_layers)])

        overlaps = list(results)
        
        return np.stack(overlaps)
    
    def process_layer(self, layer, data_i, data_j, k) -> np.ndarray:
        """
        Process a single layer.
        """
        data = Data(data_i[:,layer,:])
        warnings.filterwarnings("ignore")
        data.compute_distances(maxk=k)
        overlap = data.return_data_overlap(data_j[:,layer,:])
        return overlap
    
    
class LabelOverlap():
  def __init__(self, df: pd.DataFrame, tensor_storage, label: str):
      super().__init__(df, tensor_storage)
      self.label = label
      
  def main(self) -> pd.DataFrame:
    """
    Compute the overlap between the layers of instances in which the model answered with the same letter
    Output
    ----------
    Dict[layer: List[Array(num_layers, num_layers)]]
    """
    #The last token is always the same, thus its first layer activation (embedding) is always the same
    iter_list=[0.05,0.10,0.20,0.50]
    rows = []
    
    for class_fraction in tqdm.tqdm(iter_list, desc = "Computing overlap"):
      for model in self.df["model_name"].unique().tolist():
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

def parallel_compute(self, hidden_states, labels,  class_fraction ) -> np.ndarray:
    overlaps = []
    if class_fraction is None:
      raise ValueError("You must provide either k or class_fraction")
    start_time = time.time()
    with Pool(processes=_NUM_PROC) as pool:
        results = pool.starmap(self.process_layer, [(num_layer, hidden_states, labels, class_fraction) for num_layer in range(hidden_states.shape[1])])
    end_time = time.time()
    print(f"Label overlap over batch of data took: {end_time-start_time}")
    overlaps = list(results)

    return np.stack(overlaps)

def process_layer(self, num_layer, hidden_states, labels, class_fraction):
    """
    Process a single layer.
    """
    data = Data(hidden_states[:, num_layer, :])

    warnings.filterwarnings("ignore")
    overlap = data.return_label_overlap(labels, class_fraction=class_fraction, k=k)
    return overlap