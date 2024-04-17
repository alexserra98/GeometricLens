from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from utils_log import cross_validate_model
from common.utils import _PATH, retrieve_df, retrieve_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')

def cross_validate_layer(args):
    train_folds, test_folds, class_weights_folds, layer = args
    average_accuracy, scores = cross_validate_model(
        train_folds=train_folds,
        test_folds=test_folds,
        class_weights_folds=class_weights_folds,
        layer=layer,
        n_folds=5
    )
    return layer, average_accuracy, scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', help='Label to probe', required=True)
    args = parser.parse_args()

    result_path = Path(_PATH, "log_reg")
    result_path.mkdir(exist_ok=True, parents=True)
    n_folds = 5

    accuracies_dict = {}

    for shot in [0, 2, 5]:
        df_hiddenstates = retrieve_df(train_instances=shot)
        X, y = retrieve_dataset(df_hiddenstates, args.label)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        train_folds = []
        test_folds = []
        class_weight_folds = []
        for train, test in skf.split(X, y):
            X_train, y_train = X[train], y[train]
            X_test, y_test = X[test], y[test]
            class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            train_folds.append((X_train, y_train))
            test_folds.append((X_test, y_test))
            class_weight_folds.append(dict(zip(np.unique(y_train), class_weight)))

        args_list = [(train_folds, test_folds, class_weight_folds, layer) for layer in range(33)]
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(cross_validate_layer, args) for args in args_list]
            accuracies = []
            for future in as_completed(futures):
                layer, average_accuracy, scores = future.result()
                logging.info(f'Layer {layer}: Mean CV Accuracy = {average_accuracy:.2f}, Scores = {scores}')
                accuracies.append(average_accuracy)
        
        accuracies_dict[shot] = accuracies

    df = pd.DataFrame(accuracies_dict)
    df.to_pickle(result_path / f"cross_val_accuracies_{args.label}.pkl")

if __name__ == "__main__":
    main()
