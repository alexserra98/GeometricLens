from metrics.utils import hidden_states_collapse
from metrics.query import DataFrameQuery
from common.tensor_storage import TensorStorage
from common.metadata_db import MetadataDB
from common.utils import *
from utils_mlp import MLP, create_data_loaders, EarlyStopping, train_model, evaluate_model
from utils import set_dataframes, retrieve_dataset

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from pathlib import Path
import pickle
import logging
import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')

def main():
    train, test = retrieve_dataset(Path("/orfeo/scratch/dssc/zenocosini/mmlu_result/"))
    X_train, y_train,_ = train
    X_test, y_test,_ = test
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_path = Path('models')
    result_path.mkdir(exist_ok=True)

    accuracies = []

    for layer in tqdm.tqdm(range(X_train.shape[1]), desc="Computing probe..."):
        model = MLP(input_size=X_train.shape[2], hidden_size=100, num_classes=len(np.unique(y_train)))
        model.to(device)

        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters())

        # Split data into training and validation
        X_train_layer, X_val_layer, y_train_layer, y_val_layer = train_test_split(
            X_train[:, layer, :], y_train, test_size=0.2, random_state=42)
        
        train_loader, val_loader = create_data_loaders(X_train_layer, y_train_layer, X_val_layer, y_val_layer)

        # Adding test data loader for evaluation
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test[:, layer, :]).float(),
                                               torch.from_numpy(y_test).long()),
                                 batch_size=64, shuffle=False)

        # Implementing early stopping
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        # Training with early stopping
        for epoch in range(50):  # Set a maximum number of epochs
            stop = train_model(model, train_loader, val_loader, criterion, optimizer, device, early_stopping)
            if stop:
                break

        accuracy = evaluate_model(model, test_loader, device)
        accuracies.append(accuracy)

        torch.save(model.state_dict(), result_path / f'model_layer{layer}.pth')

    with open(result_path / "accuracies.pkl", "wb") as f:
        pickle.dump(accuracies, f)

    logging.info(f'The accuracy of the MLP model at each layer is: {accuracies}')

if __name__ == "__main__":
    main()
