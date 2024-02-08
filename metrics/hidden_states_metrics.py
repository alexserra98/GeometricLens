import numpy as np
from pandas.core.api import DataFrame as DataFrame
from pathlib import Path
from abc import ABC, abstractmethod
from common.tensor_storage import TensorStorage


class HiddenStatesMetrics(ABC):

  def __init__(self, df: DataFrame(), tensor_storage):
    self.df = df
    self.tensor_storage = tensor_storage
    
  @abstractmethod
  def main(self, hidden_states: np.ndarray ,  algorithm = "gride") -> np.ndarray:
    raise NotImplementedError
  
  @abstractmethod
  def parallel_compute(self, hidden_states: np.ndarray ,  algorithm = "gride") -> np.ndarray:
    raise NotImplementedError
  
  def process_layer(self, hidden_states: np.ndarray ,  algorithm = "gride") -> np.ndarray:
    raise NotImplementedError