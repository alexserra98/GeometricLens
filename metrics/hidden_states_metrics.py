import numpy as np
from pandas.core.api import DataFrame as DataFrame
from pathlib import Path
from abc import ABC, abstractmethod
from common.tensor_storage import TensorStorage

from common.globals_vars import Array
from jaxtyping import Float, Int
from typing import Dict
import sys


class HiddenStatesMetrics(ABC):
    def __init__(
        self,
        queries: dict,
        df: DataFrame() = None,
        tensor_storage=None,
        variations: dict = None,
        storage_logic: str = "npy",
        parallel: bool = True,
    ):
        self.queries = queries
        self.df = None
        self.tensor_storage = tensor_storage
        self.storage_logic = storage_logic
        self.variations = variations
        self.parallel = parallel

    def main(self, hidden_states: np.ndarray, algorithm="gride") -> np.ndarray:
        raise NotImplementedError

    def parallel_compute(
        self, hidden_states: np.ndarray, algorithm="gride"
    ) -> np.ndarray:
        raise NotImplementedError

    def process_layer(self, hidden_states: np.ndarray, algorithm="gride") -> np.ndarray:
        raise NotImplementedError

    def concatenate_layers(self, 
                           input_array: Float[Array, "num_instances \
                                                     num_layers model_dim"],
                           window_size: Int = 2):
        """
        Concatenate the hidden states of successive layers in a window of size 
        `window_size`.
        For example for window_size=2:
            output[0] will be the concatenation of layer [0,1],
            output[1] will be the concatenation of layer [1,2],
            ...
        Inputs:
            input_Float: Float[Array, "num_instances, num_layers, model_dim"]
            window_size: Int
        Returns:
            Float[Array, "num_instances, num_layers, window_size*model_dim"]
        """
        # Ensure input_Float is a numpy Float
        if not isinstance(input_array, np.ndarray):
            input_array = np.ndarray(input_array)
        
        # Prepare output Float
        num_windows = input_array.shape[1] - window_size + 1
        output_shape = (input_array.shape[0], input_array.shape[1],
                        window_size * input_array.shape[2])
        output = np.zeros(output_shape, dtype=input_array.dtype)
        
        # Create all windows for each position that fits the full window size
        window_shape = (1, window_size, input_array.shape[2])
        windows = np.lib.stride_tricks.sliding_window_view(input_array, 
                                                           window_shape)
        windows = windows.reshape(input_array.shape[0], num_windows, -1)
        
        # Assign these windows to the output
        output[:, :num_windows] = windows
        
        # Handling the last layers by concatenating backwards
        # We need to handle the case where the indices fall out of the bounds 
        # normally handled by the first loop
        if window_size > 1:
            for i in range(num_windows, input_array.shape[1]):
                input_array_sliced = input_array[:, i - window_size + 1:i + 1]
                output[:, i, :] = input_array_sliced.reshape(
                    input_array.shape[0], -1)
        
        return output
