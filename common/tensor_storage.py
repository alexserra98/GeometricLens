import torch
import h5py
import os
from .utils import _generate_hash
import numpy as np

class TensorStorage:
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)



    def save_tensors(self, tensors, file_name):
        file_path = os.path.join(self.storage_dir, file_name + '.h5')
        with h5py.File(file_path, 'w') as h5f:
            for tensor in tensors:
                hash_code = _generate_hash(tensor)
                h5f.create_dataset(f'tensor_{hash_code}', data=tensor)

    def load_tensors(self, file_name, key_user=None):
        file_path = os.path.join(self.storage_dir, file_name + '.h5')
        tensors = []
        with h5py.File(file_path, 'r') as h5f:
            for key in h5f.keys():
                if key.split("_")[1] in key_user:
                    tensors.append(torch.from_numpy(np.array(h5f[key][:])))
        return tensors
