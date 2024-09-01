import numpy as np
import torch
import pickle
from dadapy import Data
from collections import Counter
from sklearn.metrics import adjusted_rand_score


base_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo"
# path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/results"

# test llama 70
path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/results"  # /test_llama3-70"

model = "llama-3-8b"
mode = "finetuned"


post_path = {
    "0shot": "evaluated_test/random_order",
    "5shot": "evaluated_test/random_order",
    "4shot": "evaluated_test/random_order",
    "finetuned": "finetuned_dev_val_balanced_20sample/evaluated_test",
}

post_model = {
    "0shot": "/0shot",
    "5shot": "/5shot",
    "4shot": "/4shot",
    "finetuned": "",
}
#     f"{path}/{post_path[mode]}/{model}{post_model[mode]}/statistics_target_sorted_sample42.pkl",
# #
with open(
    f"{path}/{post_path[mode]}/{model}{post_model[mode]}/statistics_target.pkl",
    "rb",
) as f:
    stats = pickle.load(f)

print(stats["accuracy"]["macro"])


layer = 30
# layer = 73  # llama70b
mask_path = f"{base_dir}/diego/analysis"
mask = np.load(f"{mask_path}/test_mask_200.npy")


X = torch.load(f"{path}/{post_path[mode]}/{model}{post_model[mode]}/l{layer}_target.pt")
# mask = np.arange(len(X))
y = stats["answers"]
y = y[mask]


X = X.to(torch.float64).numpy()[mask]
# check we do not have opverlapping datapoints in case ramove tham from the relevant arrays
X_, indx, inverse = np.unique(X, axis=0, return_inverse=True, return_index=True)
sorted_indx = np.sort(indx)
X_sub = X[sorted_indx]
y = y[sorted_indx]


d = Data(coordinates=X_sub, maxk=300)
ids, _, _ = d.return_id_scaling_gride(range_max=300)
print(ids)
d.set_id(ids[1])
d.compute_density_kNN(k=8)
cluster_assignment = d.compute_clustering_ADP(Z=1.6, halo=False)


print(adjusted_rand_score(cluster_assignment, y))


ground_truth_labels = y
mc_fraction = []
clust_stats = {}

for index in np.unique(d.cluster_assignment):
    ind_ = np.nonzero(d.cluster_assignment == index)[0]

    mc = Counter(ground_truth_labels[ind_]).most_common()[0][1]
    mc_fraction.append(mc / len(ind_))
    clust_stats[index] = (len(ind_), mc / len(ind_))

clust_stats
np.sum(np.array(mc_fraction) > 0.80)
