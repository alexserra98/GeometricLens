from metrics.hidden_states_metrics import HiddenStatesMetrics
from .utils import (
    TensorStorageManager,
)
from metrics.query import DataFrameQuery
from common.globals_vars import _NUM_PROC, _OUTPUT_DIR


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression

import tqdm
import pandas as pd
import numpy as np
import time
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed
import logging


class LinearProbe(HiddenStatesMetrics):
    def main(self, label) -> pd.DataFrame:
        """
        Compute the overlap between the layers of instances in which the model answered with the same letter
        Output
        ----------
        Dict[layer: List[Array(num_layers, num_layers)]]
        """
        module_logger = logging.getLogger("my_app.probe")
        module_logger.info(f"Computing linear probe with label {label}")

        self.label = label
        n_folds = 5
        rows = []

        # Directory to save checkpoints
        check_point_dir = Path(_OUTPUT_DIR, "checkpoints")
        check_point_dir.mkdir(exist_ok=True, parents=True)
        tsm = self.tensor_storage
        for n, query_dict in tqdm.tqdm(
            enumerate(self.queries), desc="Processing queries"
        ):
            module_logger.debug(f"Processing query {query_dict}")
            query = DataFrameQuery(query_dict)

            # Retrieve hidden states and labels
            hidden_states, _, hidden_states_df = tsm.retrieve_tensor(
                query, self.storage_logic
            )
            target = np.asarray(hidden_states_df[label].tolist())

            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            train_folds = []
            test_folds = []
            class_weight_folds = []

            # Split the data into train and test sets for each fold
            for train_index, test_index in skf.split(hidden_states, target):
                X_train, y_train = hidden_states[train_index], target[train_index]
                X_test, y_test = hidden_states[test_index], target[test_index]

                class_weight = compute_class_weight(
                    "balanced", classes=np.unique(y_train), y=y_train
                )

                train_folds.append((X_train, y_train))
                test_folds.append((X_test, y_test))
                class_weight_folds.append(dict(zip(np.unique(y_train), class_weight)))

            try:
                accuracies = self.parallel_compute(
                    train_folds, test_folds, class_weight_folds, n_folds
                )
            except Exception as e:
                module_logger.error(f"Error processing query {query_dict}: {e}")
                accuracies = np.nan

            row = [
                query_dict["model_name"],
                query_dict["method"],
                query_dict["train_instances"],
                accuracies,
            ]
            rows.append(row)

            if n % 3 == 0:
                # Save checkpoint
                df_temp = pd.DataFrame(
                    rows,
                    columns=["model", "method", "shot", "accuracies"],
                )
                df_temp.to_pickle(check_point_dir / f"checkpoint_probe.pkl")

        df = pd.DataFrame(
            rows,
            columns=["model", "method", "shot", "accuracies"],
        )
        return df

    def parallel_compute(
        self, train_folds, test_folds, class_weights_folds, n_folds=5
    ) -> np.ndarray:
        start_time = time.time()

        process_layer = partial(
            self.process_layer,
            train_folds=train_folds,
            test_folds=test_folds,
            class_weights_folds=class_weights_folds,
            n_folds=n_folds,
        )
        number_of_layers = train_folds[0][0].shape[1]
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
        accuracies = list(results)

        # Sort results by layer. Parallel processing shuffle the results
        accuracies.sort()
        accuracies = [acc for _, acc, _ in results]
        return np.stack(accuracies)

    def process_layer(
        self, num_layer, train_folds, test_folds, class_weights_folds, n_folds=5
    ):
        """
        Process a single layer.
        """

        scores = []
        for i in range(n_folds):
            X_train, y_train = train_folds[i]
            X_train = X_train[:, num_layer, :]
            X_test, y_test = test_folds[i]
            X_test = X_test[:, num_layer, :]
            model = LogisticRegression(
                max_iter=3500, class_weight=class_weights_folds[i], n_jobs=-1
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            scores.append(accuracy_score(y_test, predictions))

        return num_layer, np.mean(scores), scores


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


def stratified_split(df_hiddenstates):
    # Splitting the dataset into train and test sets for each subject
    train_frames = []
    test_frames = []

    for subject in df_hiddenstates["dataset"].unique():
        subject_data = df_hiddenstates[df_hiddenstates["dataset"] == subject]
        train, test = train_test_split(subject_data, test_size=0.2, random_state=42)
        train_frames.append(train)
        test_frames.append(test)

    train_df = pd.concat(train_frames)
    test_df = pd.concat(test_frames)

    return train_df, test_df


def cross_validate_model(
    train_folds, test_folds, class_weights_folds, layer, n_folds=5
):
    scores = []
    for i in range(n_folds):
        # import pdb; pdb.set_trace()
        X_train, y_train = train_folds[i]
        X_train = X_train[:, layer, :]
        X_test, y_test = test_folds[i]
        X_test = X_test[:, layer, :]
        model = LogisticRegression(max_iter=3000, class_weight=class_weights_folds[i])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores.append(accuracy_score(y_test, predictions))

    return np.mean(scores), scores

    # models = [
    #     "meta-llama-Llama-2-7b-hf",
    #     "meta-llama-Llama-2-7b-chat-hf",
    #     "meta-llama-Llama-2-13b-hf",
    #     "meta-llama-Llama-2-13b-chat-hf",
    #     "meta-llama-Llama-2-70b-hf",
    #     "meta-llama-Llama-2-70b-chat-hf",
    #     "meta-llama-Llama-3-8b-hf",
    #     "meta-llama-Llama-3-8b-chat-hf",
    #     "meta-llama-Llama-3-70b-hf",
    #     "meta-llama-Llama-3-70b-chat-hf",
    #     "meta-llama-Llama-3-8b-ft-hf",
    # ]
