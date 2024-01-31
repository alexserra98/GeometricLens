import numpy as np
from pandas.core.api import DataFrame as DataFrame
import torch

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional,  Type
from .utils import  Match, Layer, neig_overlap, exact_match, quasi_exact_match, layer_overlap, hidden_states_collapse

import tqdm
from generation.generation import  ScenarioResult
import pandas as pd
from collections import namedtuple

from common.metadata_db import MetadataDB
from dataclasses import asdict
from dadapy.data import Data
from dadapy.metric_comparisons import MetricComparisons

from .hidden_states import HiddenStates
from enum import Enum
from abc import ABC, abstractmethod

from warnings import warn
from .query import DataFrameQuery
from pathlib import Path

#Define a type hint 
Array = Type[np.ndarray]
Tensor = Type[torch.Tensor]



class Match(Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    ALL = "all"
class Layer(Enum):
    LAST = "last"
    SUM = "sum"
    
@dataclass
class InstanceResult():
  dataset: str
  train_instances: int
  intrinsic_dim: Dict[str, Dict[str, np.ndarray]]
  metrics: Dict[str, Dict[str, float]]
  label_nn: Optional[Dict[str, Dict[str, np.ndarray]]] = None
  

class Metrics():
    def __init__(self, db: MetadataDB, metrics_list: List, path_result: Path) -> None:
      self.db = db
      self.metrics_list = metrics_list
      self.df = self.set_dataframes()
      self.path_result = path_result
    
    def set_dataframes(self) -> pd.DataFrame:
      """
      Aggregate in a dataframe the hidden states of all instances
      ----------
      hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
      """
      df = pd.read_sql("SELECT * FROM metadata", self.db.conn)
      df["train_instances"] = df["train_instances"].astype(str)
      return df
        
    def evaluate(self) -> InstanceResult:
      """
      Compute all the implemented metrics
      Output
      ----------
      pandas DataFrame
      """
      df_out = {}
      for metric in self.metrics_list:
        df_out[metric] = self._compute_metric(metric)
      return df_out
    
    def _compute_metric(self, metric) -> pd.DataFrame:
      if metric == "shot_metric":
        return self.shot_metrics()
      elif metric == "letter_overlap":
        return self._compute_letter_overlap()
      elif metric == "subject_overlap":
        return self._compute_subject_overlap()
      elif metric == "base_finetune_overlap":
        return self._compute_base_finetune_overlap()
      else:
        raise NotImplementedError
    
    def _compute_overlap(self, df,label) -> Dict[str, Dict[str, np.ndarray]]:
      hidden_states = HiddenStates(df, self.path_result)
      return hidden_states.layer_overlap_label(label)
    
    def _compute_letter_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
      return self._compute_overlap(self.df,"only_ref_pred")
    
    def _compute_subject_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
      return self._compute_overlap(self.df,"dataset")

    def shot_metrics(self):
      """
      Compute all the implemented metrics
      Output
      ----------
      InstanceResult object
      """
      hidden_states = HiddenStates(self.df, self.path_result)
      return hidden_states.shot_metrics()
    
    def _compute_base_finetune_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
      hidden_states = HiddenStates(self.df, self.path_result)
      return hidden_states.layer_point_overlap()
    
    def basic_metric_mean(self) -> Dict[str, float]:
      output_dict = {column_name: self.basic_metric[column_name].mean() for column_name in self.basic_metric.columns}
      return output_dict  


class BaseFinetuneOverlap():
  def set_dataframes(self) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    df= super().set_dataframes()
    df = self.query.apply_query(df)
    df["train_instances"] = df["train_instances"].astype(str)
    return df


  def compute_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
    hidden_states = HiddenStates(self.df, self.path_result)
    return hidden_states.layer_point_overlap()
  


  

  
class DatasetMetrics():
  """
  Class to compute metrics across all the runs of a dataset. Generally used to compare between different n-shots
  """

  def __init__(self, instances_result: List[InstanceResult] ) -> None:
    """
    Parameters
    ----------
    nearest_neig : List[Dict]
        List of dictionaries containing the nearest neighbours of each layer of the two runs
        Dict[k-method, Array(num_layers, num_instances, k_neighbours)]
    """
    self.instances_results = instances_result 

  def get_last_layer_id_diff(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute difference in last layer ID among 0-shot and few-shot
    Output
    ----------
    Dict[layer, Dict[metric, Array(neigh_order)]]
    """
    metric_id = {}
    zero_shot = next((x for x in self.instances_results if x.train_instances=='0'), None)
    
    for method in [Layer.LAST.value, Layer.SUM.value]:
        metric_id[method] = {"id_diff": [], "metric_diff": {key: [] for key in self.instances_results[0].metrics.keys()}} 
        for i_result in self.instances_results:
          if i_result.train_instances == '0':
            continue
          zero_shot_id = zero_shot.intrinsic_dim[Match.ALL.value][method]
          i_result_id = i_result.intrinsic_dim[Match.ALL.value][method]
          # the intrinisc dimension is calculated up to the nearest neighbour order that change between runs
          nn_order = min(zero_shot_id.shape[1], i_result_id.shape[1])
          id_diff = np.abs(zero_shot_id[-5][:nn_order] - i_result_id[-5][:nn_order])
          metric_id[method]["id_diff"].append(id_diff)
          for metric in i_result.metrics.keys():
            metric_id[method]["metric_diff"][metric].append(np.abs(zero_shot.metrics[metric] - i_result.metrics[metric]))
    return metric_id

  def get_all_overlaps(self) -> Dict:
    """
    Compute the overlap between all the runs
    Output
    ----------
    Dict[k-method, np.ndarray[i,j, Array(num_layers, num_layers)]]
    """
    overlaps = {}
    num_runs = len(self.nearest_neig)
    for method in ["last", "sum"]:
      overlaps[method] = {}
      desc = "Computing overlap for method " + method
      for i in tqdm.tqdm(range(num_runs-1), desc = desc):
        for j in range(i+1,num_runs):
          # indexes are not 0-based because it's easier to mange with plolty sub_trace
          overlaps[method][(i+1,j+1)] = self._instances_overlap(i,j,method)
    return overlaps

    
  
  def _instances_overlap(self, i,j,method) -> np.ndarray:
    nn1 = self.nearest_neig[i][method]
    nn2 = self.nearest_neig[j][method]
    assert nn1.shape == nn2.shape, "The two nearest neighbour matrix must have the same shape" 
    layers_len = nn1.shape[0]
    overlaps = np.empty([layers_len, layers_len])
    for i in range(layers_len):
      for j in range(layers_len):
        # WARNING : the overlap is computed with K=K THE OTHER OCC IS IN RUNGEOMETRY
        overlaps[i][j] = neig_overlap(nn1[i], nn2[j])
    return overlaps
  
  
  
