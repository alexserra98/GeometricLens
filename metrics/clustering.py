from metrics.hidden_states_metrics import HiddenStatesMetrics
from .utils import (
    exact_match,
    angular_distance,
    TensorStorageManager,
)
from metrics.query import DataFrameQuery
from common.globals_vars import _NUM_PROC, _OUTPUT_DIR
from common.error import DataNotFoundError, UnknownError
from dadapy.data import Data
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import f1_score

import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from functools import partial
import logging
import einops
from pathlib import Path


f1_score_micro = partial(f1_score, average="micro")
_COMPARISON_METRICS = {
    "adjusted_rand_score": adjusted_rand_score,
    "adjusted_mutual_info_score": adjusted_mutual_info_score,
    "mutual_info_score": mutual_info_score,
    "f1_score": f1_score_micro,
}


class LabelClustering(HiddenStatesMetrics):
    def main(self, label) -> pd.DataFrame:
        """
        Compute the overlap between the layers of instances in which the model answered with the same letter
        Output
        ----------
        Dict[layer: List[Array(num_layers, num_layers)]]
        """
        module_logger = logging.getLogger("my_app.label_cluster")
        module_logger.info(f"Computing label cluster with label {label}")

        self.label = label

        check_point_dir = Path(_OUTPUT_DIR, "checkpoints")
        check_point_dir.mkdir(exist_ok=True, parents=True)

        iter_list = [0, 0.5, 1, 1.6, 2.1]

        #iter_list = [1,1.6]

        rows = []
        tsm = self.tensor_storage

        for z in tqdm.tqdm(iter_list, desc="Computing overlap"):
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
                    module_logger.error(f"Error processing query {query_dict}: {e}")
                    continue
                except UnknownError as e:
                    module_logger.error(f"Error processing query {query_dict}: {e}")
                    raise

                if self.variations["label_clustering"] == "balanced_letter":
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
                    z,
                ]
                label_per_row = self.constructing_labels(
                    hidden_states_df, hidden_states
                )

                try:
                    clustering_dict = self.parallel_compute(
                        hidden_states, label_per_row, z
                    )
                except Exception as e:
                    module_logger.error(
                        f"Error computing clustering for query {query_dict}: {e}"
                    )
                    raise e

                row.extend(
                    [
                        clustering_dict["bincount"],
                        clustering_dict["adjusted_rand_score"],
                        clustering_dict["adjusted_mutual_info_score"],
                        clustering_dict["mutual_info_score"],
                        clustering_dict["clusters_assignment"],
                        clustering_dict["labels"],
                    ]
                )
                rows.append(row)
                # Save checkpoint
                if n % 3 == 0:
                    df_temp = pd.DataFrame(
                        rows,
                        columns=[
                            "model",
                            "method",
                            "train_instances",
                            "z",
                            "clustering_bincount",
                            "adjusted_rand_score",
                            "adjusted_mutual_info_score",
                            "mutual_info_score",
                            "clusters_assignment",
                            "labels",
                        ],
                    )
                    df_temp.to_pickle(
                        check_point_dir / f"checkpoint_{self.label}_cluster.pkl"
                    )

        df = pd.DataFrame(
            rows,
            columns=[
                "model",
                "method",
                "train_instances",
                "z",
                "clustering_bincount",
                "adjusted_rand_score",
                "adjusted_mutual_info_score",
                "mutual_info_score",
                "clusters_assignment",
                "labels",
            ],
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
        self, hidden_states: np.ndarray, label: np.array, z: float
    ) -> dict:
        assert (
            hidden_states.shape[0] == label.shape[0]
        ), "Label lenght don't mactch the number of instances"
        number_of_layers = hidden_states.shape[1]
        if self.variations["label_clustering"] == "concat":
            hidden_states = self.concatenate_layers(hidden_states,
                                                    window_size=3)
        process_layer = partial(
            self.process_layer, hidden_states=hidden_states, label=label, z=z
        )
        results = []
        if self.parallel:
            # Parallelize the computation of the metrics
            with Parallel(n_jobs=_NUM_PROC) as parallel:
                results = parallel(
                    delayed(process_layer)(layer)
                    for layer in range(1, number_of_layers)
                )
        else:
            for layer in range(1, number_of_layers):
                results.append(process_layer(layer))
        keys = list(results[0].keys())

        output = {key: [] for key in keys}

        # Merge the results
        for layer_result in results:
            for key in output:
                output[key].append(layer_result[key])
        return output

    def process_layer(self, layer, hidden_states, label, z) -> dict:
        layer_results = {}
        hidden_states = hidden_states[:, layer, :]
        base_unique, base_idx, base_inverse = np.unique(
            hidden_states, axis=0, return_index=True, return_inverse=True
        )
        indices = np.sort(base_idx)
        base_repr = hidden_states[indices]
        subjects = label[indices]

        # do clustering
        data = Data(coordinates=base_repr)
        ids, _, _ = data.return_id_scaling_gride(range_max=100)
        data.set_id(ids[3])
        data.compute_density_kNN(k=16)

        halo = True if self.variations["label_clustering"] == "halo" else False
        clusters_assignment = data.compute_clustering_ADP(Z=z, halo=halo)

        layer_results["bincount"] = []
        # Comparison metrics
        for key, func in _COMPARISON_METRICS.items():
            layer_results[key] = func(clusters_assignment, subjects)

        layer_results["clusters_assignment"] = clusters_assignment
        layer_results["labels"] = subjects

        # data = Data(hidden_states[:, layer, :])
        # data.remove_identical_points()
        # data.compute_distances(maxk=100)
        # clusters_assignement = data.compute_clustering_ADP(Z=z)

        # #Bincount
        # unique_clusters, cluster_counts = np.unique(clusters_assignement, return_counts=True)
        # bincount = np.zeros((len(unique_clusters), len(np.unique(label))))
        # for unique_cluster in unique_clusters:
        #     bincount[unique_cluster] = np.bincount(label[clusters_assignement == unique_cluster], minlength=len(np.unique(label)))
        # layer_results["bincount"] = bincount
        # layer_results["bincount"] = []
        # #Comparison metrics
        # for key, func in _COMPARISON_METRICS.items():
        #     layer_results[key] = func(clusters_assignment, label)
        return layer_results

    def concatenate_layers(self, input_array, window_size=2):
        # Ensure input_array is a numpy array
        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)
        
        # Prepare output array
        num_windows = input_array.shape[1] - window_size + 1
        output_shape = (input_array.shape[0], input_array.shape[1], window_size * input_array.shape[2])
        output = np.zeros(output_shape, dtype=input_array.dtype)
        
        # Create all windows for each position that fits the full window size
        windows = np.lib.stride_tricks.sliding_window_view(input_array, (1, window_size, input_array.shape[2]))
        windows = windows.reshape(input_array.shape[0], num_windows, -1)
        
        # Assign these windows to the output
        output[:, :num_windows] = windows
        
        # Handling the last layers by concatenating backwards
        # We need to handle the case where the indices fall out of the bounds normally handled by the first loop
        if window_size > 1:
            for i in range(num_windows, input_array.shape[1]):
                output[:, i, :] = input_array[:, i - window_size + 1:i + 1].reshape(input_array.shape[0], -1)
        
        return output


class PointClustering(HiddenStatesMetrics):
    def main(self) -> pd.DataFrame:
        """
        Compute the overlap between same dataset, same train instances, different models (pretrained and finetuned)

        Parameters
        ----------
        data: Dict[model, Dict[dataset, Dict[train_instances, Dict[method, Dict[match, Dict[layer, np.ndarray]]]]]]
        Output
        df: pd.DataFrame (k,dataset,method,train_instances_i,train_instances_j,overlap)
        """
        module_logger = logging.getLogger("my_app.point_clustering")
        module_logger.info("Computing point clustering")

        iter_list = [0.2, 0.5, 1, 1.68]

        # Directory to save checkpoints
        check_point_dir = Path(_OUTPUT_DIR, "checkpoints")
        check_point_dir.mkdir(exist_ok=True, parents=True)
        rows = []

        tsm = self.tensor_storage
        for z in iter_list:
            for n, query_dict in tqdm.tqdm(
                enumerate(self.queries), desc="Processing queries"
            ):
                couple = query_dict["couple"]
                shot_i = query_dict["shot_i"]
                shot_j = query_dict["shot_j"]
                method = query_dict["method"]
                module_logger.debug(f"Processing query {query_dict}")

                query_i = DataFrameQuery(
                    {
                        "method": method,
                        "model_name": couple[0],
                        "train_instances": shot_i,
                    }
                )

                query_j = DataFrameQuery(
                    {
                        "method": method,
                        "model_name": couple[1],
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

                    # if (
                    #    self.variations["point_clustering"] == "cosine"
                    #    or self.variations["point_clustering"] == "norm"
                    #    or self.variations["point_clustering"] == "shared_answers"
                    # ):
                    #    df_i["exact_match"] = df_i.apply(
                    #        lambda r: exact_match(r["std_pred"], r["letter_gold"]),
                    #        axis=1,
                    #    )
                    #    df_j["exact_match"] = df_j.apply(
                    #        lambda r: exact_match(r["std_pred"], r["letter_gold"]),
                    #        axis=1,
                    #    )
                    #    # find the index of rows that have "exact_match" True in both df_i and df_j
                    #    indices_i = df_i[df_i["exact_match"] == True].index
                    #    indices_j = df_j[df_j["exact_match"] == True].index
                    #    # find the intersection of the two sets of indices
                    #    indices = indices_i.intersection(indices_j)
                    #    hidden_states_i = hidden_states_i[indices]
                    #    hidden_states_j = hidden_states_j[indices]

                    try:
                        clustering_out = self.parallel_compute(
                            hidden_states_i, hidden_states_j, z
                        )

                    except Exception as e:
                        module_logger.error(
                            f"Error computing overlap for {couple} with k {z}. Error: {e}"
                        )
                        raise e
                    rows.append(
                        [
                            z,
                            couple,
                            method,
                            shot_i,
                            shot_j,
                            clustering_out["adjusted_rand_score"],
                            clustering_out["adjusted_mutual_info_score"],
                            clustering_out["mutual_info_score"],
                            clustering_out["f1_score"],
                        ]
                    )
                    if len(rows) % 3 == 0:
                        # Save checkpoint
                        df_temp = pd.DataFrame(
                            rows,
                            columns=[
                                "z",
                                "couple",
                                "method",
                                "train_instances_i",
                                "train_instances_j",
                                "adjusted_rand_score",
                                "adjusted_mutual_info_score",
                                "mutual_info_score",
                                "f1_score",
                            ],
                        )
                        df_temp.to_pickle(
                            check_point_dir / f"checkpoint_point_overlap.pkl"
                        )

        df = pd.DataFrame(
            rows,
            columns=[
                "z",
                "couple",
                "method",
                "train_instances_i",
                "train_instances_j",
                "adjusted_rand_score",
                "adjusted_mutual_info_score",
                "mutual_info_score",
                "f1_score",
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
        self, input_i: np.ndarray, input_j: np.ndarray, z: int
    ) -> np.ndarray:
        assert (
            input_i.shape[1] == input_j.shape[1]
        ), "The two runs must have the same number of layers"
        number_of_layers = input_i.shape[1]

        comparison_output = {key: [] for key in _COMPARISON_METRICS.keys()}

        process_layer = partial(
            self.process_layer, input_i=input_i, input_j=input_j, z=z
        )
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
            for layer in range(1, number_of_layers):
                results.append(process_layer(layer))

        # Organize the results
        comparison_output = {key: [] for key in _COMPARISON_METRICS}
        for layer_result in results:
            for key in comparison_output:
                comparison_output[key].append(layer_result[key])
        return comparison_output

    def process_layer(self, layer, input_i, input_j, z):
        """
        Process a single layer.
        """
        data_i = input_i[:, layer, :]
        data_j = input_j[:, layer, :]

        if self.variations["point_clustering"] == "norm":
            data_i = data_i / np.linalg.norm(data_i, axis=1, keepdims=True)
            clusters_i = self.compute_cluster_assignment(data_i, z)

            data_j = data_j / np.linalg.norm(data_j, axis=1, keepdims=True)
            clusters_j = self.compute_cluster_assignment(data_j, z)

        else:
            clusters_i = self.compute_cluster_assignment(data_i, z)
            clusters_j = self.compute_cluster_assignment(data_j, z)
        layer_results = {}

        for key, func in _COMPARISON_METRICS.items():
            layer_results[key] = func(clusters_i, clusters_j)

        return layer_results

    def compute_cluster_assignment(self, base_repr, z):
        data = Data(coordinates=base_repr, maxk=100)
        ids, _, _ = data.return_id_scaling_gride(range_max=100)
        data.set_id(ids[3])
        data.compute_density_kNN(k=16)
        clusters_assignment = data.compute_clustering_ADP(Z=z)
        return clusters_assignment


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
