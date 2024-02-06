import torch
import h5py
import os
from .utils import _generate_hash, retry_on_failure
import numpy as np


class TensorStorage:
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    @retry_on_failure(3)
    def create_dataset(h5f, dataset_name, tensor):
        h5f.create_dataset(dataset_name, data=tensor)

    def save_tensors(self, tensors, ids, file_name,):
        file_path = os.path.join(self.storage_dir, file_name + '.h5')
        mode = 'a' if os.path.exists(file_path) else 'w'
        with h5py.File(file_path, mode) as h5f:
            existing_datasets = set(h5f.keys())
            for id,tensor in zip(ids,tensors):
                dataset_name = f'tensor_{id}'
                if dataset_name not in existing_datasets:
                    h5f.create_dataset(dataset_name, data=tensor)
                

    def load_tensor(self, file_name, id):
        file_path = os.path.join(self.storage_dir, file_name + '.h5')
        with h5py.File(file_path, 'r') as h5f:
            return torch.from_numpy(np.array(h5f[f'tensor_{id}'][:]))
    
    def load_tensors(self, file_name, key_user=None):
        file_path = os.path.join(self.storage_dir, file_name + '.h5')
        tensors = []
        with h5py.File(file_path, 'r') as h5f:
            for key in h5f.keys():
                if key.split("_")[1] in key_user:
                    tensors.append(torch.from_numpy(np.array(h5f[key][:])))
        return tensors
