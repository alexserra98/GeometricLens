from metrics.hidden_states_metrics import HiddenStatesMetrics
from .utils import exact_match
from metrics.query import DataFrameQuery
from common.globals_vars import _NUM_PROC, _OUTPUT_DIR, Array
from common.error import DataNotFoundError, UnknownError

from dadapy.data import Data

import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
import logging
from jaxtyping import Float, Int


class IntrinsicDimension(HiddenStatesMetrics):
    def main(self) -> pd.DataFrame:
        """
        Compute the intrinsic dimension of the hidden states of a model
        Returns
            pd.DataFrame
                DataFrame with the intrinsic dimension of the hidden states
        """
        module_logger = logging.getLogger("my_app.id")
        module_logger.info("Computing ID")

        rows = []

        # Directory to save checkpoints
        check_point_dir = Path(_OUTPUT_DIR, "checkpoints")
        check_point_dir.mkdir(exist_ok=True, parents=True)

        tsm = self.tensor_storage
        for query_dict in tqdm.tqdm(self.queries, desc="Processing queries"):
            module_logger.debug(f"Processing query {query_dict}")
            for match in ["incorrect", "correct", "all"]:
                if not self.variations\
                   or not self.variations.get("intrisic_dimension"):
                    query = DataFrameQuery(query_dict)
                elif self.variations["intrinsic_dimension"] == "misc":
                    query_dict["dataset"] = "mmlu:miscellaneous"
                    query = DataFrameQuery(query_dict)
                else:
                    raise ValueError(
                        "Unknown variation. It must be either None or 'norm'"
                    )
                try:
                    hidden_states, _, hidden_states_df = tsm.retrieve_tensor(
                        query, self.storage_logic
                    )
                except DataNotFoundError as e:
                    module_logger.error(f"Error processing query {query_dict}"
                                        f"data not found: {e}")
                    continue
                except UnknownError as e:
                    module_logger.error(f"Error processing query {query_dict}:"
                                        f"{e}")
                    raise
                if match == "correct":
                    hidden_states_df = hidden_states_df[
                        hidden_states_df.apply(
                            lambda r: exact_match(r["std_pred"],
                                                  r["letter_gold"]),
                            axis=1,
                        )
                    ]
                    hidden_states = hidden_states[hidden_states_df.index]
                elif match == "incorrect":
                    hidden_states_df = hidden_states_df[
                        hidden_states_df.apply(
                            lambda r: not exact_match(r["std_pred"],
                                                      r["letter_gold"]),
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
                        f"Error computing ID for {query_dict}"
                        f"with match {match}. Error: {e}"
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

            df_temp.to_pickle(check_point_dir / "checkpoint_id.pkl")

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

    def parallel_compute(
            self, 
            hidden_states: Float[Array, "num_instances num_layers model_dim"]
    ) -> Float[Array, "order_of_nearest_neighbour num_layers"]:
        """
        Collect hidden states of all instances and compute ID
        we employ two different approaches: the one of the last token, 
        the sum of all tokens
        
        Inputs
            hidden_states: Float[Float, "num_instances, num_layers, model_dim"]
        Returns
            Float[Array, "order of nearest neighbour, num_layers"]
                Array with the ID of each layer,
                for each order of nearest neighbour
        """

        id_per_layer_gride = []

        num_layers = hidden_states.shape[1]
        process_layer = partial(
            self.process_layer, hidden_states=hidden_states, algorithm="gride"
        )
        if "concat" in self.variations["intrinsic_dimension"]:
            print("I'm working")
            hidden_states = self.concatenate_layers(hidden_states,
                                                    window_size=2)
        if self.parallel:
            with Parallel(n_jobs=_NUM_PROC) as parallel:
                id_per_layer_gride = parallel(
                    delayed(process_layer)(i)
                    for i in tqdm.tqdm(range(1, num_layers),
                                       desc="Processing layers")
                )
                # id_per_layer_lpca = parallel(delayed(process_layer)
                # (i, hidden_states, "lpca") for i in range(1, num_layers))
                # id_per_layer_danco = parallel(delayed(process_layer)
                # (i, hidden_states, "DANco") for i in range(1, num_layers))
        else:
            # Sequential version
            for layer in range(1, num_layers):
                id_per_layer_gride.append(process_layer(layer))
        id_per_layer_gride.insert(0, np.ones(id_per_layer_gride[-1].shape[0]))
        # TODO fix output
        return np.stack(id_per_layer_gride), [], []
               
    def process_layer(
            self, 
            layer: Int, 
            hidden_states: Float[Array, "num_instances num_layers model_dim"],
            algorithm: str = "gride"
    ) -> Float[Array, "order_of_nearest neighbour"]:
        """
        Process a single layer
        Inputs
            layer: Int
                Layer to process
            hidden_states: Float[Float, "num_instances, num_layers, model_dim"]
                Hidden states of the model
            algorithm: str
                Algorithm to compute the ID
        Returns
        """
        data = Data(hidden_states[:, layer, :])

        # with HiddenPrints():
        data.remove_identical_points()

        if algorithm == "2nn":
            raise NotImplementedError
        elif algorithm == "gride":
            return data.return_id_scaling_gride(range_max=1000)[0]
        elif algorithm == "DANco":
            raise NotImplementedError
        elif algorithm == "lPCA":
            raise NotImplementedError


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
