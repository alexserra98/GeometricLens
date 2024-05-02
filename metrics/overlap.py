from metrics.hidden_states_metrics import HiddenStatesMetrics
from .utils import (
    hidden_states_collapse,
    exact_match,
    angular_distance,
    TensorStorageManager,
)
from metrics.query import DataFrameQuery
from common.globals_vars import _NUM_PROC

from .dadapy_handler import DataAdapter
from dadapy import Data
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

import tqdm
import pandas as pd
import numpy as np
import time
from functools import partial
from joblib import Parallel, delayed


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
        # warn("Computing overlap using k with  2 -25- 500")
        comparison_metrics = {
            "adjusted_rand_score": adjusted_rand_score,
            "adjusted_mutual_info_score": adjusted_mutual_info_score,
            "mutual_info_score": mutual_info_score,
        }

        iter_list = [5, 10, 30, 100, 500]
        iter_list = [5, 10, 30, 100]
        models = [
            "meta-llama-Llama-2-7b-hf",
            "meta-llama-Llama-2-7b-chat-hf",
            "meta-llama-Llama-2-13b-hf",
            "meta-llama-Llama-2-13b-chat-hf",
            "meta-llama-Llama-2-70b-hf",
            "meta-llama-Llama-2-70b-chat-hf",
            "meta-llama-Llama-3-8b-hf",
            "meta-llama-Llama-3-8b-chat-hf",
            "meta-llama-Llama-3-70b-hf",
            "meta-llama-Llama-3-70b-chat-hf",
        ]

        rows = []
        couples_list = [
            ("meta-llama-Llama-2-70b-hf", "meta-llama-Llama-2-70b-hf"),
            ("meta-llama-Llama-2-70b-hf", "meta-llama-Llama-2-70b-chat-hf"),
            ("meta-llama-Llama-3-70b-hf", "meta-llama-Llama-3-70b-hf"),
            ("meta-llama-Llama-3-70b-hf", "meta-llama-Llama-3-70b-chat-hf"),
            ("meta-llama-Llama-2-13b-hf", "meta-llama-Llama-2-13b-hf"),
            ("meta-llama-Llama-2-13b-hf", "meta-llama-Llama-2-13b-chat-hf"),
            ("meta-llama-Llama-2-7b-hf", "meta-llama-Llama-2-7b-hf"),
            ("meta-llama-Llama-2-7b-hf", "meta-llama-Llama-2-7b-chat-hf"),
        ]
        for k in tqdm.tqdm(iter_list, desc="Computing overlaps k"):
            tsm = TensorStorageManager()
            for couples in couples_list:
                for method in ["last"]:  # self.df["method"].unique().tolist():
                    if couples[0] == couples[1]:
                        iterlist = [("0", "4")]
                    else:
                        iterlist = [("4", "0"), ("0", "0")]
                    for shot in iterlist:
                        train_instances_i, train_instances_j = shot
                        query_i = DataFrameQuery(
                            {
                                "method": method,
                                "model_name": couples[0],
                                "train_instances": train_instances_i,
                            }
                        )
                        # "dataset": "mmlu:miscellaneous"})
                        query_j = DataFrameQuery(
                            {
                                "method": method,
                                "model_name": couples[1],
                                "train_instances": train_instances_j,
                            }
                        )
                        # "dataset": "mmlu:miscellaneous"})
                        hidden_states_i, _, df_i = tsm.retrieve_tensor(query_i, "npy")
                        hidden_states_j, _, df_j = tsm.retrieve_tensor(query_j, "npy")

                        # hidden_states_i, _, df_i = hidden_states_collapse(self.df,query_i, self.tensor_storage)
                        # hidden_states_j, _, df_j = hidden_states_collapse(self.df,query_j, self.tensor_storage)

                        df_i.reset_index(inplace=True)
                        df_j.reset_index(inplace=True)

                        # for row_i, row_j in zip(df_i.iterrows(), df_j.iterrows()):
                        #    assert (
                        #        row_i[1]["id_instance"].replace("chat-", "")[:-1]
                        #        == row_j[1]["id_instance"].replace("chat-", "")[:-1]
                        #    ), "The two runs must have the same instances"

                        # import pdb; pdb.set_trace()
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
                        # hidden_states_i, _, df_i = hidden_states_collapse(df_i,query_i, self.tensor_storage)
                        # hidden_states_j, _, df_j = hidden_states_collapse(df_j,query_j, self.tensor_storage)

                        rows.append(
                            [
                                k,
                                couples,
                                method,
                                train_instances_i,
                                train_instances_j,
                                self.parallel_compute(
                                    hidden_states_i, hidden_states_j, k
                                ),
                            ]
                        )
                df_temp = pd.DataFrame(
                    rows,
                    columns=[
                        "k",
                        "couple",
                        "method",
                        "train_instances_i",
                        "train_instances_j",
                        "point_overlap",
                    ],
                )
                df_temp.to_pickle("checkpoint_point.pkl")

        df = pd.DataFrame(
            rows,
            columns=[
                "k",
                "couple",
                "method",
                "train_instances_i",
                "train_instances_j",
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
        with Parallel(n_jobs=_NUM_PROC) as parallel:
            results = parallel(
                delayed(process_layer)(layer) for layer in range(number_of_layers)
            )
        # results = []
        # for layer in range(number_of_layers):
        #    results.append(process_layer(layer))
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
        # data = DataAdapter(data_i, variation=self.variations["point_overlap"], maxk=k)
        # warnings.filterwarnings("ignore")
        # data.compute_distances(maxk=k)
        # print(f'{k} -- {data_j[:,layer,:]}')
        return overlap


class LabelOverlap(HiddenStatesMetrics):
    def main(self, label) -> pd.DataFrame:
        """
        Compute the overlap between the layers of instances in which the model answered with the same letter
        Output
        ----------
        Dict[layer: List[Array(num_layers, num_layers)]]
        """
        self.label = label
        # The last token is always the same, thus its first layer activation (embedding) is always the same
        iter_list = [0.05, 0.10, 0.20, 0.50]

        iter_list = [0.10]
        rows = []
        models = [
            "meta-llama-Llama-2-7b-hf",
            "meta-llama-Llama-2-7b-chat-hf",
            "meta-llama-Llama-2-13b-hf",
            "meta-llama-Llama-2-13b-chat-hf",
            "meta-llama-Llama-2-70b-hf",
            "meta-llama-Llama-2-70b-chat-hf",
            "meta-llama-Llama-3-8b-hf",
            "meta-llama-Llama-3-8b-chat-hf",
            "meta-llama-Llama-3-70b-hf",
            "meta-llama-Llama-3-70b-chat-hf",
            "meta-llama-Llama-3-8b-ft-hf",
        ]
        for class_fraction in tqdm.tqdm(iter_list, desc="Computing overlap"):
            for model in models:  # self.df["model_name"].unique().tolist():
                tsm = TensorStorageManager()
                for method in ["last"]:  # self.df["method"].unique().tolist():
                    for train_instances in [
                        "0",
                        "2",
                        "5",
                    ]:  # self.df["train_instances"].unique().tolist():
                        if "70" in model and train_instances == "5" and "chat" not in model:
                            train_instances = "4"
                        if "chat" in model and train_instances == "5" and "chat" in model:
                            continue
                        query = DataFrameQuery(
                            {
                                "method": method,
                                "model_name": model,
                                "train_instances": train_instances,
                            }
                        )
                        # hidden_states, logits, hidden_states_df= hidden_states_collapse(self.df,query, self.tensor_storage)
                        hidden_states, _, hidden_states_df = tsm.retrieve_tensor(
                            query, self.storage_logic
                        )
                        if self.variations["label_overlap"] == "balanced_letter":
                            hidden_states_df.reset_index(inplace=True)
                            hidden_states_df, index = balance_by_label_within_groups(
                                hidden_states_df, "dataset", "letter_gold"
                            )
                            # import pdb; pdb.set_trace()
                            hidden_states = hidden_states[hidden_states_df["index"]]
                            hidden_states_df.reset_index(inplace=True)

                        row = [model, method, train_instances, class_fraction]
                        label_per_row = self.constructing_labels(
                            hidden_states_df, hidden_states
                        )

                        overlap = self.parallel_compute(
                            hidden_states, label_per_row, class_fraction=class_fraction
                        )
                        row.append(overlap)
                        rows.append(row)
                df_temp = pd.DataFrame(
                    rows,
                    columns=[
                        "model",
                        "method",
                        "train_instances",
                        "class_fraction",
                        "overlap",
                    ],
                )
                df_temp.to_pickle("checkpoint.pkl")
        df = pd.DataFrame(
            rows,
            columns=["model", "method", "train_instances", "class_fraction", "overlap"],
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

        # print(f"Is Variation: {self.variations['overlap']} active? {self.variations['overlap'] == 'norm'}")
        with Parallel(n_jobs=_NUM_PROC) as parallel:
            results = parallel(
                delayed(process_layer)(num_layer)
                for num_layer in range(hidden_states.shape[1])
            )

        # results = []
        # for layer in range(number_of_layers):
        #    results.append(process_layer(layer))

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

        # data = DataAdapter(
        #    hidden_states, variation=self.variations["label_overlap"], maxk=maxk
        # )
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


# Example usage:
# Assuming `dataframe` is your initial dataframe
