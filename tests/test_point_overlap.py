from src.metrics.overlap import PointOverlap
import unittest
import numpy as np
from sklearn.model_selection import StratifiedKFold
from unittest.mock import Mock

class TestProbe(unittest.TestCase):
    def setUp(self):
        self.query_dict = {"model_name": "fake", "method": "fake", "train_instances": "fake"}
        self.my_class = PointOverlap(queries=self.query_dict,
                                    df = "",
                                    tensor_storage = "",
                                    variations = {"point_overlap": "None"},
                                    parallel = False)
        self.mock_logger = Mock()

    def test_paralell_compute(self):
        hidden_states_1 = np.random.rand(100, 10, 768).astype(np.float32)  # Fake input
        hidden_states_2 = np.random.rand(100, 10, 768).astype(np.float32)  # Fake input
        
        actual_output = self.my_class.parallel_compute(hidden_states_1, hidden_states_2, k=30)
        print(f"type = {type(actual_output)},\n{actual_output=}")
        # self.mock_logger.info.assert_called_with("Processing data in compute_fold")
        self.assertTrue(isinstance(actual_output, np.ndarray))
        self.assertTrue(actual_output.shape == (10,))
        


if __name__ == '__main__':
    unittest.main()
