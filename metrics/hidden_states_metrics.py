import numpy as np
from pandas.core.api import DataFrame as DataFrame
from pathlib import Path
from abc import ABC, abstractmethod
from common.tensor_storage import TensorStorage
import sys


class HiddenStatesMetrics(ABC):

  def __init__(self, df: DataFrame(), tensor_storage, variations: dict = None, storage_logic: str = "h5"):
    self.df = df
    self.tensor_storage = tensor_storage
    self.storage_logic = storage_logic
    self.variations = variations
    
  
  def main(self, hidden_states: np.ndarray ,  algorithm = "gride") -> np.ndarray:
    raise NotImplementedError
  
  
  def parallel_compute(self, hidden_states: np.ndarray ,  algorithm = "gride") -> np.ndarray:
    raise NotImplementedError
  
  def process_layer(self, hidden_states: np.ndarray ,  algorithm = "gride") -> np.ndarray:
    raise NotImplementedError
