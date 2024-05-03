import os
import h5py
from safetensors.numpy import save_file
from safetensors import safe_open


def read_h5_files(directory):
    """Reads all .h5 files in a directory and returns their contents."""
    data = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "hidden_states.h5":
                filepath = os.path.join(root, file)
                with h5py.File(filepath, 'r') as file:
                    for key in file.keys():
                        data[key] = file[key][:]
    return data

def main():
    
    # directory = "/home/alexserra98/helm-suite/MCQA_Benchmark/output_test_result" #input("Enter the directory path containing .h5 files: ")
    # h5_data = read_h5_files(directory)

    # save_file(h5_data, "model.safetensors")

    tensors = {}
    with safe_open("model.safetensors", framework="np") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

if __name__ == "__main__":
    main()
