  import numpy as np
from pandas.core.api import DataFrame as DataFrame
import torch
from typing import List, Type
import tqdm
import pandas as pd
from common.metadata_db import MetadataDB

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
    def __init__(self, db: MetadataDB, metrics_list: List, path_result: Path, variations: dict) -> None:
      self.db = db
      self.metrics_list = metrics_list
      self.df = self.set_dataframes()
      self.path_result = path_result
      self.tensor_path = Path(path_result, "tensor_files")
      self.variations = variations
    
    def set_dataframes(self) -> pd.DataFrame:
      """
      Aggregate in a dataframe the hidden states of all instances
      ----------
      hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
      """
      df = pd.read_sql("SELECT * FROM metadata", self.db.conn)
      df["train_instances"] = df["train_instances"].astype(str)
      df.drop(columns=["id"],inplace = True)
      #import pdb; pdb.set_trace()
      df.drop_duplicates(subset = ["id_instance"],inplace = True, ignore_index = True) # why there are duplicates???
      return df
        
    def evaluate(self) -> List[pd.DataFrame]:
      """
      Compute all the implemented metrics
      Output
      ----------
      pandas DataFrame
      """
      df_out = {}
      
      
      logging.info(f"Metrics will be computed with the following variations:\n{self.variations=}")
      
      hidden_states = HiddenStates(self.df, self.tensor_path, self.variations)
      
      #hidden_states = HiddenStates(self.df, self.tensor_path)
      for metric in tqdm.tqdm(self.metrics_list, desc = "Computing metrics"):
        logging.info(f'Computing {metric}...')
        out = self._compute_metric(metric, hidden_states)
        
        if metric == "letter_overlap" or metric == "subject_overlap":
          variation_key = "label_overlap"
        else:
          variation_key = metric
    
        variation = self.variations.get(variation_key, None)
        
        
        
        name = "_".join([metric.replace(":","_"),variation]) if variation else metric.replace(":","_")
        
        out.to_pickle(Path(self.path_result,f'{name}.pkl'))
        
      return df_out
    
    def _compute_metric(self, metric, hidden_states) -> pd.DataFrame:
      
      if metric == "shot_metric":
        return self.shot_metrics()
      
      elif metric == "letter_overlap":
        return hidden_states.label_overlap(label = "only_ref_pred")
      
      elif metric == "subject_overlap":
        return hidden_states.label_overlap(label = "dataset")
      
      elif metric == "point_overlap":
        return hidden_states.point_overlap()
      
      elif metric == "base_finetune_cluster":
        return hidden_states.point_cluster()
      
      elif metric == "intrinsic_dim":
        return hidden_states.intrinsic_dim()
      
      elif metric == "last_layer_id_diff":
        return self._compute_layer_id_diff()
      
      elif metric == "cka":
        return hidden_states.cka()
      else:
        raise NotImplementedError
    

    
    def _compute_layer_id_diff(self):
      shot_metric_path = Path(self.path_result, 'shot_metric.pkl')
      id_path = Path(self.path_result, 'intrinsic_dim.pkl')
      try:
        shot_metrics_df = pd.read_pickle(shot_metric_path)
        id_df = pd.read_pickle(id_path)
      except FileNotFoundError:
        print("You need to computer intrinisic dimension and shot metrics first")
      return self.get_last_layer_id_diff(id_df, shot_metrics_df)

    

    def shot_metrics(self):
      """
      Compute all the implemented metrics
      Output
      ----------
      InstanceResult object
      """
      hidden_states = HiddenStates(self.df, self.tensor_path)
      return hidden_states.shot_metrics()
    
    def _compute_base_finetune_overlap(self) -> pd.DataFrame:
      hidden_states = HiddenStates(self.df, self.tensor_path)
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
      shot_metrics_df.drop(["dataset"], axis=1, inplace=True)
      shot_metrics_df= shot_metrics_df.groupby(['train_instances','model']).mean().reset_index()
      query_template = {"match": "all",
                  "method": "last"}
      query = DataFrameQuery(query_template)
      id_df = query.apply_query(id_df)
      df = pd.merge(id_df,shot_metrics_df, on=['train_instances','model'], how='inner')
      return df
      
  

  
  
