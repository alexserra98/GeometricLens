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
    def __init__(self, db: MetadataDB, query: DataFrameQuery, path_result: Path) -> None:
      self.db = db
      self.query = query
      self.df = self.set_dataframes()
      self.path_result = path_result
    
    @abstractmethod
    def set_dataframes(self) -> pd.DataFrame:
      """
      Aggregate in a dataframe the hidden states of all instances
      ----------
      hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
      """
      pass
    def evaluate(self) -> InstanceResult:
      pass
class ShotMetrics(Metrics):
  """
  Class to compute the metrics of a run. It takes a list of request results computed by generate.py
  and compute the per instance metrics 
  """
  def __init__(self, scenario_result: ScenarioResult):
    self.requests_results = scenario_result.requests_results
    self.train_instances = scenario_result.train_instances
    self.dataset = scenario_result.dataset
    self.model = scenario_result.model_name
    self.basic_metric, self.hidden_states = self.set_dataframes()

  def set_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate in two different dataframes the data for basic metrics and the hidden states of all instances
    Output
    ----------
    basic_metrics: pd.DataFrame(num_instances, num_metrics)
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    basic_metrics_dict = {"loss":[], "perplexity":[], "std_exact_match":[],"std_quasi_exact_match":[], "ref_exact_match":[]}
    hidden_states_dict = {"hidden_states": [],"layer": [], "match": [], "answered_letter": [], "gold_letter": []}
    for request_result in self.requests_results:
      basic_metrics_dict["loss"].append(request_result.loss)
      basic_metrics_dict["perplexity"].append(np.exp(request_result.loss))
      basic_metrics_dict["std_exact_match"].append(exact_match(request_result.preds["std_pred"]["letter"], request_result.gold["letter"]))
      basic_metrics_dict["ref_exact_match"].append(exact_match(request_result.preds["only_ref_pred"]["letter"],request_result.gold["letter"]))
      basic_metrics_dict["std_quasi_exact_match"].append(quasi_exact_match(request_result.preds["std_pred"]["letter"], request_result.gold["letter"]))
      for layer in ["last","sum"]:
        hidden_states_dict["hidden_states"].append(request_result.hidden_states[layer])
        hidden_states_dict["layer"].append(layer)
        match = "correct" if basic_metrics_dict["std_exact_match"][-1] else "wrong"
        hidden_states_dict["match"].append(match)
        hidden_states_dict["answered_letter"].append(request_result.preds["only_ref_pred"]["letter"])
        hidden_states_dict["gold_letter"].append(request_result.gold["letter"])
    basic_metrics = pd.DataFrame(basic_metrics_dict)
    hidden_states = pd.DataFrame(hidden_states_dict)
    return basic_metrics, hidden_states
  
  def evaluate(self) -> InstanceResult: 
    """
    Compute all the implemented metrics
    Output
    ----------
    InstanceResult object
    """
    basic_metric = self.basic_metric_mean()
    hidden_states = HiddenStates(self.hidden_states)
    intrinsic_dim = hidden_states.get_instances_id()
    return InstanceResult(self.dataset, self.train_instances, intrinsic_dim, basic_metric)
  
  def compute_nn(self,k = 20) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute the nearest neighbours of each instance in the run per layer
    using the provided methodv
    Output
    ----------
    Dict[k-method, Array(num_layers, num_instances, k_neighbours)]
    """
    hidden_states = HiddenStates(self.hidden_states)
    warn(f'Computing nearest neighbours with k={k}')
    return hidden_states.get_nearest_neighbour(k) 
  
  #TODO use property decorator
  def basic_metric_mean(self) -> Dict[str, float]:
    output_dict = {column_name: self.basic_metric[column_name].mean() for column_name in self.basic_metric.columns}
    return output_dict  



class Overlap(ABC):
  """
  Abstract Class for compute different kinds of overlap 
  """  
  def __init__(self, db: MetadataDB, query: DataFrameQuery, path_result: Path) -> None:
    self.db = db
    self.query = query
    self.df = self.set_dataframes()
    self.path_result = path_result
  
  @abstractmethod
  def set_dataframes(self) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    pass
  
  @abstractmethod
  def compute_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
    pass
  
class LabelOverlap(Overlap):
  def set_dataframes(self) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    df = pd.read_sql("SELECT * FROM metadata", self.db.conn)
    columns = ['id_instance', 
               'dataset', 
               'train_instances', 
               'model_name',
               'only_ref_pred', 
               'method']
    df = df[columns]  # Keep only the columns in the list "columns"
    return df
  def _compute_overlap(self, label) -> Dict[str, Dict[str, np.ndarray]]:
    hidden_states = HiddenStates(self.df, self.path_result)
    return hidden_states.layer_overlap_label(label)

class SubjectOverlap(LabelOverlap):
  
  def set_dataframes(self) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    df = super().set_dataframes()
    df = self.query.apply_query(df)
    df = df.rename(columns={'dataset': 'subject'})
    df["train_instances"] = df["train_instances"].astype(str)
    return df 
  
  def compute_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
    return self._compute_overlap("subject")


  
class LetterOverlap(LabelOverlap):
  def set_dataframes(self) -> DataFrame:
    df= super().set_dataframes()
    df = self.query.apply_query(df)
    df["train_instances"] = df["train_instances"].astype(str)
    return df 
  
  def compute_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
    return self._compute_overlap("letter")

class BaseFinetuneOverlap(Overlap):
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
  
  
  
