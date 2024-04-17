from common.utils import *
from utils import  _PATH, \
                   retrieve_df, \
                   retrieve_dataset
from utils_log import cross_validate_model

from pathlib  import Path

import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold


from pathlib import Path
import pickle
import joblib
import logging
import tqdm
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--label', help='Label to probe', required=True)
args, remaining_argv = parser.parse_known_args()
label = args.label
def main():
    result_path = Path(_PATH,"log_reg")
    result_path.mkdir(exist_ok=True,parents=True)    
    
    n_folds = 5
    
    
    train_folds = []
    test_folds = []
    class_weight_folds = []
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    rows = {}
    for shot in [0,2,5]:
        df_hiddenstates = retrieve_df(train_instances=0) 
        X,y  = retrieve_dataset(df_hiddenstates, label)
        
        for train, test in skf.split(df_hiddenstates, df_hiddenstates["dataset"]):
            X_train, y_train = X[train],y[train]
            X_test, y_test = X[test],y[test]
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight = dict(zip(classes, weights))
            train_folds.append((X_train, y_train))
            test_folds.append((X_test, y_test))
            class_weight_folds.append(class_weight)
        accuracies = []
        for layer in tqdm.tqdm(range(33), desc="Computing probe..."):
            # Cross-validate the model
            average_accuracy, scores = cross_validate_model(train_folds=train_folds,
                                                            test_folds=test_folds,
                                                            class_weights_folds=class_weight_folds,
                                                            layer=layer,
                                                            n_folds=5)

            accuracies.append(average_accuracy)
            
            logging.info(f'Layer {layer}: Mean CV Accuracy = {average_accuracy:.2f}, Scores = {scores}')
        rows[shot]=accuracies
        
    df = pd.DataFrame(rows)
    df.to_pickle(result_path / f"cross_val_accuracies_{label}.pkl")

  
if __name__ == "__main__":
    main()


