import numpy as np
from pandas.core.api import DataFrame as DataFrame
import torch

from dataclasses import dataclass
from typing import List, Type
from .utils import  Match

import tqdm

import pandas as pd
from collections import namedtuple

from common.metadata_db import MetadataDB
from dataclasses import asdict

from .hidden_states import HiddenStates
from enum import Enum
from .query import DataFrameQuery
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        
    def evaluate(self, results_path) -> List[pd.DataFrame]:
      """
      Compute all the implemented metrics
      Output
      ----------
      pandas DataFrame
      """
      df_out = {}
      for metric in tqdm.tqdm(self.metrics_list, desc = "Computing metrics"):
        logging.info(f'Computing {metric}...')
        out = self._compute_metric(metric)
        out.to_pickle(Path(results_path,f'{metric}.pkl'))
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
      elif metric == "intrinsic_dim":
        return self._compute_intrinsic_dim()
      else:
        raise NotImplementedError
    
    def _compute_overlap(self, df,label) -> pd.DataFrame:
      hidden_states = HiddenStates(df, self.path_result)
      return hidden_states.label_overlap(label)
    
    def _compute_letter_overlap(self) -> pd.DataFrame:
      return self._compute_overlap(self.df,"only_ref_pred")
    
    def _compute_subject_overlap(self) -> pd.DataFrame:
      return self._compute_overlap(self.df,"dataset")
    
    def _compute_intrinsic_dim(self) -> pd.DataFrame:
      hidden_states = HiddenStates(self.df, self.path_result)
      return hidden_states.intrinsic_dim()

    def shot_metrics(self):
      """
      Compute all the implemented metrics
      Output
      ----------
      InstanceResult object
      """
      hidden_states = HiddenStates(self.df, self.path_result)
      return hidden_states.shot_metrics()
    
    def _compute_base_finetune_overlap(self) -> pd.DataFrame:
      hidden_states = HiddenStates(self.df, self.path_result)
      return hidden_states.point_overlap()
    
    
    def get_last_layer_id_diff(self, id_df, shot_metrics_df) -> pd.DataFrame:
      """
      Compute the difference in ID between the last layer of two runs
      Output
      ----------
      Dict[layer: List[Array(num_layers, num_layers)]]
      """
      # The last token is always the same, thus its first layer activation (embedding) is always the same
      rows = []
      for model in self.df["model_name"].unique().tolist():
        for method in self.df["method"].unique().tolist():
          query_template = {"match": Match.ALL.value,
                    "method": method,
                    "model_name": model,
                    "train_instances": '0'}
          query = DataFrameQuery(query_template)
          id_metrics_iter = id_df.apply_query(query)
          shot_metrics_df_iter = shot_metrics_df.apply_query(query)

          df_iter = id_metrics_iter.join(shot_metrics_df_iter, how='outer')
          df_iter["accuracy"] = df.groupby("train_instances")["accuracy"].transform(lambda x: x - x.iloc[0])
          df_iter["id"] = df.groupby("train_instances")["id_per_layer"].transform(lambda x: x - x.iloc[0])
          rows.append([model, method, df["accuracy"].tolist(), df["id"].tolist()])              
      df = pd.DataFrame(rows, columns = ["k","model","method","train_instances","id_per_layer"])
      return df
      
  

  
  
