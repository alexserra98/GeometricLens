from metrics.clustering import LabelClustering
import unittest
import numpy as np
from sklearn.model_selection import StratifiedKFold
from unittest.mock import Mock

class TestProbe(unittest.TestCase):
    def setUp(self):
        self.query_dict = {"model_name": "fake", "method": "fake", "train_instances": "fake"}
        self.my_class = LabelClustering(queries=self.query_dict,
                                    df = "",
                                    tensor_storage = "",
                                    variations = {"label_clustering": "None"},
                                    parallel = False)
        self.mock_logger = Mock()

    def test_paralell_compute(self):
            
        hidden_states = np.random.rand(100, 2, 768).astype(np.float32)  # Fake input
        label = np.random.randint(0, 10, (100)).astype(np.float32)  # Fake target
        
        actual_output = self.my_class.parallel_compute(hidden_states, label, z=0.1)
        print(f"type = {type(actual_output)},\n{actual_output=}")
        # self.mock_logger.info.assert_called_with("Processing data in compute_fold")
        self.assertTrue(isinstance(actual_output, dict))
        


if __name__ == '__main__':
    unittest.main()
