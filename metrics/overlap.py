from metrics.hidden_states_metrics import HiddenStatesMetrics
from .utils import (
    exact_match,
    angular_distance,
    TensorStorageManager,
)
from metrics.query import DataFrameQuery
from common.globals_vars import _NUM_PROC, _OUTPUT_DIR
from common.error import DataNotFoundError, UnknownError

from dadapy import Data


import tqdm
import pandas as pd
import numpy as np
import time
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed
import logging
import pdb


class PointOverlap(HiddenStatesMetrics):
    def main(self) -> pd.DataFrame:
        """
        Compute the overlap between same dataset, same train instances, different models (pretrained and finetuned)

        Parameters
        ----------
        data: Dict[model, Dict[dataset, Dict[train_instances, Dict[method, Dict[match, Dict[layer, np.ndarray]]]]]]
        Output
        df: pd.DataFrame (k,dataset,method,train_instances_i,train_instances_j,overlap)
        """
        module_logger = logging.getLogger("my_app.point_overlap")
        module_logger.info("Computing point overlap")

        iter_list = [100, 500, 1000]

        # Directory to save checkpoints
        check_point_dir = Path(_OUTPUT_DIR, "checkpoints")
        check_point_dir.mkdir(exist_ok=True, parents=True)
        rows = []

        couples_list = [
            ("llama-3-70b", "llama-3-70b"),
            ("llama-2-70b", "llama-2-70b"),
            ("llama-2-70b", "llama-2-70b-chat"),
            ("llama-3-70b", "llama-3-70b-chat"),
            ("llama-2-13b", "llama-2-13b-hf"),
            ("llama-2-13b", "llama-2-13b-chat"),
            ("llama-2-13b", "llama-2-13b"),
            ("llama-2-13b", "llama-2-13b-ft"),
            ("llama-2-7b", "llama-2-7b"),
            ("llama-2-7b", "llama-2-7b-chat"),
            ("llama-3-8b", "llama-3-8b"),
            ("llama-3-8b", "llama-3-8b-chat"),
            ("llama-3-8b", "llama-3-8b-ft"),
        ]

        # couples_list = [
        #    ("meta-llama-Llama-3-8b-hf", "meta-llama-Llama-3-8b-chat-hf"),
        # ]

        tsm = self.tensor_storage
        for k in tqdm.tqdm(iter_list, desc="Computing overlaps k"):
            for couples in couples_list:
                for method in ["last"]:  # self.df["method"].unique().tolist():ù
                    module_logger.debug(f"Processing {couples} with k {k}")
                    shot_number = "4" if "70" in couples[0] else "5"
                    if couples[0] == couples[1]:
                        iterlist = [("0", shot_number)]
                    else:
                        iterlist = [(shot_number, "0"), ("0", "0")]

                    for shots in iterlist:
                        shot_i, shot_j = shots

                        query_i = DataFrameQuery(
                            {
                                "method": method,
                                "model_name": couples[0],
                                "train_instances": shot_i,
                            }
                        )

                        query_j = DataFrameQuery(
                            {
                                "method": method,
                                "model_name": couples[1],
                                "train_instances": shot_j,
                            }
                        )
                        try:
                            # Hidden states to compare
                            hidden_states_i, _, df_i = tsm.retrieve_tensor(
                                query_i, self.storage_logic
                            )
                            hidden_states_j, _, df_j = tsm.retrieve_tensor(
                                query_j, self.storage_logic
                            )
                        except DataNotFoundError as e:
                            module_logger.error(
                                f"Data not found for {query_i} or {query_j}. Error: {e}"
                            )
                            continue
                        except UnknownError as e:
                            module_logger.error(
                                f"Unknown error for {query_i} or {query_j}. Error: {e}"
                            )
                            raise e

                        df_i.reset_index(inplace=True)
                        df_j.reset_index(inplace=True)

                        if (
                            self.variations["point_overlap"] == "cosine"
                            or self.variations["point_overlap"] == "norm"
                            or self.variations["point_overlap"] == "shared_answers"
                        ):
                            df_i["exact_match"] = df_i.apply(
                                lambda r: exact_match(r["std_pred"], r["letter_gold"]),
                                axis=1,
                            )
                            df_j["exact_match"] = df_j.apply(
                                lambda r: exact_match(r["std_pred"], r["letter_gold"]),
                                axis=1,
                            )
                            # find the index of rows that have "exact_match" True in both df_i and df_j
                            indices_i = df_i[df_i["exact_match"] == True].index
                            indices_j = df_j[df_j["exact_match"] == True].index
                            # find the intersection of the two sets of indices
                            indices = indices_i.intersection(indices_j)
                            hidden_states_i = hidden_states_i[indices]
                            hidden_states_j = hidden_states_j[indices]

                        try:
                            # pdb.set_trace()
                            overlap = self.parallel_compute(
                                hidden_states_i, hidden_states_j, k
                            )
                        except Exception as e:
                            module_logger.error(
                                f"Error computing overlap for {couples} with k {k}. Error: {e}"
                            )
                            raise e
                        rows.append(
                            [
                                k,
                                couples,
                                method,
                                shot_i,
                                shot_j,
                                overlap,
                            ]
                        )
                        if len(rows) % 3 == 0:
                            # Save checkpoint
                            df_temp = pd.DataFrame(
                                rows,
                                columns=[
                                    "k",
                                    "couple",
                                    "method",
                                    "shot_i",
                                    "shot_j",
                                    "point_overlap",
                                ],
                            )
                            df_temp.to_pickle(
                                check_point_dir / f"checkpoint_point_overlap.pkl"
                            )

        df = pd.DataFrame(
            rows,
            columns=[
                "k",
                "couple",
                "method",
                "shot_i",
                "shot_j",
                "point_overlap",
            ],
        )
        return df

    def pair_names(self, names_list):
        """
        Pairs base names with their corresponding 'chat' versions.

        Args:
        names_list (list): A list of strings containing names.

        Returns:
        list: A list of tuples, each containing a base name and its 'chat' version.
        """
        # Separating base names and 'chat' names
        difference = "chat"
        base_names = [name for name in names_list if difference not in name]
        chat_names = [name for name in names_list if difference in name]
        base_names.sort()
        chat_names.sort()
        # Pairing base names with their corresponding 'chat' versions
        pairs = []
        for base_name, chat_name in zip(base_names, chat_names):
            pairs.append((base_name, base_name))
            pairs.append((chat_name, chat_name))
            pairs.append((base_name, chat_name))
        return pairs

    def parallel_compute(
        self, data_i: np.ndarray, data_j: np.ndarray, k: int
    ) -> np.ndarray:
        """
        Compute the overlap between two runs

        Parameters
        ----------
        data_i: np.ndarray(num_instances, num_layers, model_dim)
        data_j: np.ndarray(num_instances, num_layers, model_dim)
        k: int

        Output
        ----------
        Array(num_layers)
        """
        assert (
            data_i.shape[1] == data_j.shape[1]
        ), "The two runs must have the same number of layers"
        number_of_layers = data_i.shape[1]

        process_layer = partial(self.process_layer, data_i=data_i, data_j=data_j, k=k)

        if self.parallel:
            with Parallel(n_jobs=_NUM_PROC) as parallel:
                results = parallel(
                    delayed(process_layer)(layer)
                    for layer in tqdm.tqdm(
                        range(number_of_layers), desc="Processing layers"
                    )
                )
        else:
            results = []
            for layer in range(number_of_layers):
                results.append(process_layer(layer))

        overlaps = list(results)

        return np.stack(overlaps)

    def process_layer(self, layer, data_i, data_j, k) -> np.ndarray:
        """
        Process a single layer.
        """

        data_i = data_i[:, layer, :]
        data_j = data_j[:, layer, :]

        if self.variations["point_overlap"] == "norm":
            data_i = data_i / np.linalg.norm(data_i, axis=1, keepdims=True)
            data_j = data_j / np.linalg.norm(data_j, axis=1, keepdims=True)
            data = Data(coordinates=data_i, maxk=k)
            overlap = data.return_data_overlap(data_j, k=k)
        elif self.variations["point_overlap"] == "cosine":
            distances_i = angular_distance(data_i)
            distances_j = angular_distance(data_j)
            data = Data(coordinates=data_i, distances=distances_i, maxk=k)
            overlap = data.return_data_overlap(data_j, distances=distances_j, k=k)
        else:
            overlap = data.return_data_overlap(data_j, k=k)
        return overlap


class LabelOverlap(HiddenStatesMetrics):
    def main(self, label) -> pd.DataFrame:
        """
        Compute the overlap between the layers of instances in which the model answered with the same letter
        Output
        ----------
        Dict[layer: List[Array(num_layers, num_layers)]]
        """

        module_logger = logging.getLogger("my_app.label_overlap")
        module_logger.info(f"Computing label overlap with label {label}")
        self.label = label
        # The last token is always the same, thus its first layer activation (embedding) is always the same
        iter_list = [0.05, 0.10, 0.20, 0.50]

        iter_list = [0.10, 0.20]
        rows = []

        check_point_dir = Path(_OUTPUT_DIR, "checkpoints")
        check_point_dir.mkdir(exist_ok=True, parents=True)

        tsm = self.tensor_storage
        for class_fraction in iter_list:
            for n, query_dict in tqdm.tqdm(
                enumerate(self.queries), desc="Processing queries"
            ):
                module_logger.debug(f"Processing query {query_dict}")
                query = DataFrameQuery(query_dict)
                try:
                    hidden_states, _, hidden_states_df = tsm.retrieve_tensor(
                        query, self.storage_logic
                    )
                except DataNotFoundError as e:
                    module_logger.error(f"Data not found for {query}. Error: {e}")
                    continue
                except UnknownError as e:
                    module_logger.error(f"Unknown error for {query}. Error: {e}")
                    raise e
                if self.variations["label_overlap"] == "balanced_letter":
                    hidden_states_df.reset_index(inplace=True)
                    hidden_states_df, index = balance_by_label_within_groups(
                        hidden_states_df, "dataset", "letter_gold"
                    )
                    hidden_states = hidden_states[hidden_states_df["index"]]
                    hidden_states_df.reset_index(inplace=True)

                row = [
                    query_dict["model_name"],
                    query_dict["method"],
                    query_dict["train_instances"],
                    class_fraction,
                ]
                label_per_row = self.constructing_labels(
                    hidden_states_df, hidden_states
                )

                try:
                    overlap = self.parallel_compute(
                        hidden_states, label_per_row, class_fraction=class_fraction
                    )
                except Exception as e:
                    module_logger.error(
                        f"Error computing overlap for {query_dict} with class_fraction {class_fraction}. Error: {e}"
                    )
                    raise e

                row.append(overlap)
                rows.append(row)
                if n % 3 == 0:
                    df_temp = pd.DataFrame(
                        rows,
                        columns=[
                            "model",
                            "method",
                            "shot",
                            "class_fraction",
                            "overlap",
                        ],
                    )
                    df_temp.to_pickle(check_point_dir / f"checkpoint_label_overlap.pkl")
        df = pd.DataFrame(
            rows,
            columns=["model", "method", "shot", "class_fraction", "overlap"],
        )
        return df

    def constructing_labels(
        self, hidden_states_df: pd.DataFrame, hidden_states: np.ndarray
    ) -> np.ndarray:
        labels_literals = hidden_states_df[self.label].unique()
        labels_literals.sort()

        map_labels = {class_name: n for n, class_name in enumerate(labels_literals)}

        label_per_row = hidden_states_df[self.label].reset_index(drop=True)
        label_per_row = np.array(
            [map_labels[class_name] for class_name in label_per_row]
        )[: hidden_states.shape[0]]

        return label_per_row

    def parallel_compute(
        self, hidden_states: np.ndarray, labels: np.array, class_fraction: float
    ) -> np.ndarray:
        overlaps = []
        if class_fraction is None:
            raise ValueError("You must provide either k or class_fraction")
        start_time = time.time()
        process_layer = partial(
            self.process_layer,
            hidden_states=hidden_states,
            labels=labels,
            class_fraction=class_fraction,
        )

        number_of_layers = hidden_states.shape[1]
        if self.parallel:
            with Parallel(n_jobs=_NUM_PROC) as parallel:
                results = parallel(
                    delayed(process_layer)(num_layer)
                    for num_layer in tqdm.tqdm(
                        range(number_of_layers), desc="Processing layers"
                    )
                )
        else:
            results = []
            for layer in range(number_of_layers):
                results.append(process_layer(layer))

        end_time = time.time()
        print(f"Label overlap over batch of data took: {end_time-start_time}")
        overlaps = list(results)

        return np.stack(overlaps)

    def process_layer(self, num_layer, hidden_states, labels, class_fraction):
        """
        Process a single layer.
        """

        if self.variations["label_overlap"] == "norm":
            hidden_states = hidden_states[:, num_layer, :] / np.linalg.norm(
                hidden_states[:, num_layer, :], axis=1, keepdims=True
            )
        else:
            hidden_states = hidden_states[:, num_layer, :]
        maxk = int(np.bincount(labels).max() * class_fraction)

        data = Data(hidden_states, maxk=maxk)

        overlap = data.return_label_overlap(labels, class_fraction=class_fraction)

        return overlap


def balance_by_label_within_groups(df, group_field, label_field):
    """
    Balance the number of elements for each value of `label_field` within each group defined by `group_field`.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to balance.
    group_field : str
        The column name to group by.
    label_field : str
        The column name whose values need to be balanced within each group.

    Returns
    -------
    pd.DataFrame
        A new dataframe where each group defined by `group_field` is balanced according to `label_field`.
    """

    # Function to balance each group
    def balance_group(group):
        # Count instances of each label within the group
        class_counts = group[label_field].value_counts()
        min_count = class_counts.min()  # Find the minimum count
        # Sample each subset to have the same number of instances as the minimum count
        return (
            group.groupby(label_field)
            .apply(lambda x: x.sample(min_count))
            .reset_index(drop=True)
        )

    # Group the dataframe by `group_field` and apply the balancing function
    balanced_df = df.groupby(group_field).apply(balance_group)
    index = balanced_df.index
    index = [r[1] for r in list(index)]
    return balanced_df.reset_index(drop=True), index


# for row_i, row_j in zip(df_i.iterrows(), df_j.iterrows()):
#    assert (
#        row_i[1]["id_instance"].replace("chat-", "")[:-1]
#        == row_j[1]["id_instance"].replace("chat-", "")[:-1]
#    ), "The two runs must have the same instances"

# import pdb; pdb.set_trace()

# id_instance_i = list(map(lambda k: k[:64],df_i["id_instance"].tolist()))
# id_instance_j = list(map(lambda k: k[:64],df_j["id_instance"].tolist()))
#
# if id_instance_i != id_instance_j:
#    id_instance_i = sorted(id_instance_i)
#    id_instance_j = sorted(id_instance_j)
#    print(f'{couples}--{shot}--{k}')
#    assert id_instance_i == id_instance_j, "The two runs must have the same instances"

# id_instance_i = list(
#    map(lambda k: k[:64], df_i["id_instance"].tolist())
# )
# id_instance_j = list(
#    map(lambda k: k[:64], df_j["id_instance"].tolist())
# )
## import pdb; pdb.set_trace()
# if id_instance_i != id_instance_j:
#    indices = [
#        id_instance_j.index(id_instance_i[k])
#        for k in range(len(id_instance_i))
#    ]
#    hidden_states_i = hidden_states_i[np.array(indices)]
# df_i.reset_index(inplace=True)
# df_j.reset_indiex(inplace=True):q

# df_i = df_i.where(df_j.only_ref_pred == df_i.only_ref_pred)
# df_j = df_j.where(df_j.only_ref_pred == df_i.only_ref_pred)
# df_i.dropna(inplace=True)
# df_j.dropna(inplace=True)
# hidden_states_j, _, df_j = hidden_states_collapse(df_j,query_j, self.tensor_storage)
