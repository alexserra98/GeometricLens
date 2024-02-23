from metrics.hidden_states_metrics import HiddenStatesMetrics
from .utils import hidden_states_collapse, HiddenPrints, exact_match, quasi_exact_match
from metrics.query import DataFrameQuery
from common.globals_vars import _NUM_PROC

from dadapy.data import Data

import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from functools import partial

class IntrinsicDimension(HiddenStatesMetrics):
    
    def main(self) -> pd.DataFrame:
        rows = []
        for model in tqdm.tqdm(self.df["model_name"].unique().tolist()):
            for method in ["last"]: #self.df["method"].unique().tolist():
                for train_instances in ["0","2","5"]:#self.df["train_instances"].unique().tolist():
                    for match in ["correct", "incorrect", "all"]:
                        query = DataFrameQuery({"method":method,
                                                "model_name":model, 
                                                "dataset": "mmlu:professional_law",
                                                "train_instances": train_instances})

                        if match == "correct":
                            df = self.df[self.df.apply(lambda r: exact_match(r["only_ref_pred"], r["letter_gold"]), axis=1)]
                        elif match == "incorrect":
                            df = self.df[self.df.apply(lambda r: not exact_match(r["only_ref_pred"], r["letter_gold"]), axis=1)]
                        else:
                            df = self.df
                            #import pdb;pdb.set_trace()
                            hidden_states,_, _= hidden_states_collapse(df,query, self.tensor_storage)
                            #random_integers = np.random.randint(1, 14001, size=2500)
                            #hidden_states = hidden_states[random_integers]
                            id_per_layer_gride, id_per_layer_lpca, id_per_layer_danco = self.parallel_compute(hidden_states)
                            rows.append([model, 
                                        method,
                                        match,
                                        train_instances,
                                        id_per_layer_gride,
                                        id_per_layer_lpca,
                                        id_per_layer_danco])
        
        return pd.DataFrame(rows, columns = ["model",
                                             "method",
                                             "match",  
                                             "train_instances",
                                             "id_per_layer_gride",
                                             "id_per_layer_lpca",
                                             "id_per_layer_danco"]) 
    
    def parallel_compute(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Collect hidden states of all instances and compute ID
        we employ two different approaches: the one of the last token, the sum of all tokens
        Parameters
        ----------
        hidden_states: np.array(num_instances, num_layers, model_dim)
        algorithm: 2nn or gride --> what algorithm to use to compute ID

        Output
        ---------- 
        Dict np.array(num_layers)
        """

        id_per_layer_gride = []
        id_per_layer_lpca = []
        id_per_layer_danco = []
        num_layers = hidden_states.shape[1]
        process_layer = partial(self.process_layer, hidden_states=hidden_states, algorithm="gride")
        with Parallel(n_jobs=_NUM_PROC) as parallel:
            id_per_layer_gride = parallel(delayed(process_layer)(i) for i in range(1, num_layers))
            #id_per_layer_lpca = parallel(delayed(process_layer)(i, hidden_states, "lpca") for i in range(1, num_layers)) 
            #id_per_layer_danco = parallel(delayed(process_layer)(i, hidden_states, "DANco") for i in range(1, num_layers))
        return np.stack(id_per_layer_gride[1:]), id_per_layer_lpca, id_per_layer_danco
    
    def process_layer(self,i, hidden_states: np.array, algorithm: str):
        # Function to replace the loop body
        data = Data(hidden_states[:, i, :])
        with HiddenPrints():
            data.remove_identical_points()

        if algorithm == "2nn":
            raise NotImplementedError
        elif algorithm == "gride":
            return data.return_id_scaling_gride(range_max=1000)[0] 
        elif algorithm == "DANco":
            return skdim.id.DANCo().fit(hidden_states[:, i, :])
        elif algorithm == "lPCA":
            return skdim.id.lPCA().fit_pw(hidden_states[:, i, :], n_neighbors = 100, n_jobs = 4)
