from dadapy import data
import pickle
import numpy as np

path = "/home/diego/area_science/ricerca/geometric_lens/chat"

with open(f"{path}/llama-7b-chat-5shot-dist-matrix.pkl", "rb") as file:
    # Load the object from the file
    dataset = pickle.load(file)

subject = np.load(f"{path}/subjects-chat.npy")
letter = np.load(f"{path}/letter-chat.npy")


dist, dist_indices = dataset[6]

d = data.Data(distances = (dist, dist_indices))


d.compute_clusrt