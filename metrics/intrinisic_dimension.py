from metrics.hidden_states_metrics import HiddenStatesMetrics
from .utils import exact_match, TensorStorageManager
from metrics.query import DataFrameQuery
from common.globals_vars import _NUM_PROC, _OUTPUT_DIR

from dadapy.data import Data

import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
import logging


class IntrinsicDimension(HiddenStatesMetrics):
    def main(self) -> pd.DataFrame:
        module_logger = logging.getLogger("my_app.id")
        module_logger.info("Computing ID")

        rows = []

        # Directory to save checkpoints
        check_point_dir = Path(_OUTPUT_DIR, "checkpoints")
        check_point_dir.mkdir(exist_ok=True, parents=True)

        tsm = TensorStorageManager()
        for query_dict in tqdm.tqdm(self.queries, desc="Processing queries"):
            module_logger.debug(f"Processing query {query_dict}")
            for match in ["correct", "incorrect", "all"]:
                if not self.variations or not self.variations.get("intrisic_dimension"):
                    query = DataFrameQuery(query_dict)
                elif self.variations["intrinsic_dimension"] == "misc":
                    query_dict["dataset"] = "mmlu:miscellaneous"
                    query = DataFrameQuery(query_dict)
                else:
                    raise ValueError(
                        "Unknown variation. It must be either None or 'norm'"
                    )

                hidden_states, _, hidden_states_df = tsm.retrieve_tensor(
                    query, self.storage_logic
                )

                if match == "correct":
                    hidden_states_df = hidden_states_df[
                        hidden_states_df.apply(
                            lambda r: exact_match(r["std_pred"], r["letter_gold"]),
                            axis=1,
                        )
                    ]
                    hidden_states = hidden_states[hidden_states_df.index]
                elif match == "incorrect":
                    hidden_states_df = hidden_states_df[
                        hidden_states_df.apply(
                            lambda r: not exact_match(r["std_pred"], r["letter_gold"]),
                            axis=1,
                        )
                    ]
                    hidden_states = hidden_states[hidden_states_df.index]

                try:
                    id_per_layer_gride, id_per_layer_lpca, id_per_layer_danco = (
                        self.parallel_compute(hidden_states)
                    )
                except Exception as e:
                    module_logger.error(
                        f"Error computing ID for {query_dict} with match {match}. Error: {e}"
                    )
                    continue

                rows.append(
                    [
                        query_dict["model_name"],
                        query_dict["method"],
                        match,
                        query_dict["train_instances"],
                        id_per_layer_gride,
                        id_per_layer_lpca,
                        id_per_layer_danco,
                    ]
                )
            # Save checkpoint
            df_temp = pd.DataFrame(
                rows,
                columns=[
                    "model",
                    "method",
                    "match",
                    "train_instances",
                    "id_per_layer_gride",
                    "id_per_layer_lpca",
                    "id_per_layer_danco",
                ],
            )

            df_temp.to_pickle(check_point_dir / f"checkpoint_id.pkl")

        return pd.DataFrame(
            rows,
            columns=[
                "model",
                "method",
                "match",
                "train_instances",
                "id_per_layer_gride",
                "id_per_layer_lpca",
                "id_per_layer_danco",
            ],
        )

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
        process_layer = partial(
            self.process_layer, hidden_states=hidden_states, algorithm="gride"
        )

        with Parallel(n_jobs=_NUM_PROC) as parallel:
           id_per_layer_gride = parallel(
               delayed(process_layer)(i)
               for i in tqdm.tqdm(range(1,num_layers), desc="Processing layers")
           )
           # id_per_layer_lpca = parallel(delayed(process_layer)(i, hidden_states, "lpca") for i in range(1, num_layers))
           # id_per_layer_danco = parallel(delayed(process_layer)(i, hidden_states, "DANco") for i in range(1, num_layers))

        # Sequential version
        #id_per_layer_gride = []

        #for layer in range(1, hidden_states.shape[1]):
        #    id_per_layer_gride.append(process_layer(layer))
        ##
        id_per_layer_gride.insert(0, np.ones(id_per_layer_gride[-1].shape[0]))
        return np.stack(id_per_layer_gride), id_per_layer_lpca, id_per_layer_danco

    def process_layer(self, i, hidden_states: np.array, algorithm: str):
        # Function to replace the loop body

        data = Data(hidden_states[:, i, :])

        # with HiddenPrints():
        data.remove_identical_points()

        if algorithm == "2nn":
            raise NotImplementedError
        elif algorithm == "gride":
            return data.return_id_scaling_gride(range_max=1000)[0]
        elif algorithm == "DANco":
            return skdim.id.DANCo().fit(hidden_states[:, i, :])
        elif algorithm == "lPCA":
            return skdim.id.lPCA().fit_pw(
                hidden_states[:, i, :], n_neighbors=100, n_jobs=4
            )


# datasets = ['mmlu:clinical_knowledge',
#             'mmlu:astronomy',
#             'mmlu:computer_security',
#             'mmlu:econometrics',
#             'mmlu:electrical_engineering',
#             'mmlu:elementary_mathematics',
#             'mmlu:formal_logic',
#             'mmlu:global_facts',
#             'mmlu:high_school_biology,high_school_chemistry',
#             'mmlu:high_school_computer_science',
#             'mmlu:high_school_geography',
#             'mmlu:high_school_government_and_politics',
#             'mmlu:high_school_psychology',
#             'mmlu:high_school_us_history',
#             'mmlu:international_law',
#             'mmlu:jurisprudence',
#             'mmlu:management',
#             'mmlu:marketing',
#             'mmlu:medical_genetics',
#             'mmlu:miscellaneous',
#             'mmlu:nutrition',
#             'mmlu:prehistory',
#             'mmlu:public_relations']
# datasets = ['mmlu:clinical_knowledge',
#             'mmlu:astronomy']
