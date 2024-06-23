from metrics.probe import LinearProbe
import unittest
import numpy as np
from sklearn.model_selection import StratifiedKFold
from unittest.mock import Mock

class TestProbe(unittest.TestCase):
    def setUp(self):
        self.query_dict = {"model_name": "fake", "method": "fake", "train_instances": "fake"}
        self.my_class = LinearProbe(queries=self.query_dict,
                                    df = "",
                                    tensor_storage = "",
                                    variations = {"probe": None},
                                    parallel = False)
        self.mock_logger = Mock()

    def test_compute_fold(self):
        hidden_states = np.random.rand(100, 2, 768).astype(np.float32)  # Fake input
        target = np.random.randint(0, 10, (100)).astype(np.float32)  # Fake target
        n_folds = 3
        skf = StratifiedKFold(n_splits=n_folds,
                            shuffle=True,
                            random_state=42)
        
        

        actual_output = self.my_class.compute_fold(hidden_states, target, skf, n_folds, module_logger=self.mock_logger, query_dict=self.query_dict)
        print(f"type = {type(actual_output)},\n{actual_output=}")
        # self.mock_logger.info.assert_called_with("Processing data in compute_fold")
        self.assertTrue(isinstance(actual_output[-1], np.ndarray))
        self.assertTrue(np.allclose(actual_output, 0.1, atol=1e-1))


if __name__ == '__main__':
    unittest.main()
