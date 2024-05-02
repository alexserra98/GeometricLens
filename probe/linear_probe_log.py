import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from utils_log import cross_validate_model
from utils import _PATH, retrieve_df

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def perform_cross_validation(train_folds, test_folds, class_weights_folds, layer):
    average_accuracy, scores = cross_validate_model(
        train_folds=train_folds,
        test_folds=test_folds,
        class_weights_folds=class_weights_folds,
        layer=layer,
        n_folds=5,
    )
    return layer, average_accuracy, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", help="Label to probe", required=True)
    args = parser.parse_args()

    result_path = Path(_PATH, "log_reg")
    result_path.mkdir(exist_ok=True, parents=True)
    n_folds = 5

    accuracies_dict = {}
    models = [
        "meta-llama/Llama-3-8b-hf",
        "meta-llama/Llama-3-70b-hf",
        "meta-llama/Llama-3-8b-chat-hf",
        "meta-llama/Llama-3-70b-chat-hf",
    ]
    models = ["meta-llama/Llama-3-70b-chat-hf"]

    for model in models:
        for shot in [0, 2]:
            if "70" in model and shot == 5 and "chat" not in model:
                shot = 4
            if "70" in model and "chat" in "model" and shot == 5:
                continue
            X, y = retrieve_df(args.label, model, train_instances=shot)
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            print(f"Model: {model}; shot: {shot}")
            train_folds = []
            test_folds = []
            class_weight_folds = []
            for train_index, test_index in skf.split(X, y):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]
                class_weight = compute_class_weight(
                    "balanced", classes=np.unique(y_train), y=y_train
                )
                train_folds.append((X_train, y_train))
                test_folds.append((X_test, y_test))
                class_weight_folds.append(dict(zip(np.unique(y_train), class_weight)))

            results = Parallel(n_jobs=-1)(
                delayed(perform_cross_validation)(
                    train_folds, test_folds, class_weight_folds, layer
                )
                for layer in range(33)
            )
            results.sort()  # Sort results by layer as they may not be in order due to parallel processing
            accuracies = [acc for _, acc, _ in results]
            accuracies_dict[shot] = accuracies
            for layer, acc, scores in results:
                logging.info(
                    f"Layer {layer}: Mean CV Accuracy = {acc:.2f}, Scores = {scores}"
                )

        df = pd.DataFrame(accuracies_dict)
        df.to_pickle(
            result_path / f"cross_val_accuracies_{args.label}_{model.split('/')[1]}.pkl"
        )


if __name__ == "__main__":
    main()
