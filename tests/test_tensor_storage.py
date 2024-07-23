from src.metrics.utils import TensorStorageManager
import unittest
import numpy as np
from sklearn.model_selection import StratifiedKFold
from unittest.mock import Mock
from pathlib import Path
import pandas as pd

class TestTensorStorage(unittest.TestCase):
    def setUp(self):
        self.tsm = TensorStorageManager(tensor_storage_location="npy")
        self.mock_logger = Mock()

    def test_retrieve_from_path(self):
        path = Path("/orfeo/cephfs/scratch/area/ddoimo/"
                    "open/geometric_lens/repo/results/"
                    "finetuned_dev_val_balanced_20samples/"
                    "evaluated_dev+validation/llama-3-8b/"
                    "4epochs/10ckpts/step_1")
        tensor, logits, df = self.tsm.retrieve_from_storage_npy_path("llama-3-8b",
                                                                     0,
                                                                     path)
        self.assertTrue(isinstance(tensor, np.ndarray))
        self.assertTrue(isinstance(logits, np.ndarray))
        self.assertTrue(isinstance(df, pd.DataFrame))


if __name__ == '__main__':
    unittest.main()
