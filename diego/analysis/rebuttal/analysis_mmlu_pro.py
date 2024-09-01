import numpy as np
import pickle
import torch
import dadapy
from collections import Counter
from datasets import load_from_disk

from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    adjusted_mutual_info_score,
)

dataset_path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/mmlu_pro_race"
dataset = load_from_disk(f"{dataset_path}/test")


base_path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/results/mmlu_pro_race"
subset = dataset
sub_to_int = {sub: i for i, sub in enumerate(np.unique(subset["subject"]))}
int_to_sub = {i: sub for sub, i in sub_to_int.items()}
int_labels = np.array([sub_to_int[sub] for sub in subset["subject"]])


act = torch.load(f"{base_path}/0shot/l5_target.pt")
act = act.to(torch.float64).numpy()
d = dadapy.Data(act)
ov_gt = d.return_label_overlap(int_labels, avg=False, class_fraction=0.95)
strat_overlaps_0shot = {}
for i in np.unique(int_labels):
    indices = np.nonzero(int_labels == i)[0]
    strat_overlaps_0shot[int_to_sub[i]] = (np.mean(ov_gt[indices]), len(indices))

act = torch.load(f"{base_path}/5shot/l5_target.pt")
act = act.to(torch.float64).numpy()
d = dadapy.Data(act)
ov_gt = d.return_label_overlap(int_labels, avg=False, class_fraction=0.95)
strat_overlaps_5shot = {}
for i in np.unique(int_labels):
    indices = np.nonzero(int_labels == i)[0]
    strat_overlaps_5shot[int_to_sub[i]] = (np.mean(ov_gt[indices]), len(indices))


pruned = {}
subj_to_select = []
for sub, value in strat_overlaps_5shot.items():
    ov_gt, n_points = value
    ov_gt0, _ = strat_overlaps_0shot[sub]
    delta = ov_gt - ov_gt0
    if ov_gt > 0.82 and n_points > 30 and ov_gt - ov_gt0 > 0.1:
        pruned[sub] = (ov_gt, n_points, delta)
        subj_to_select.append(sub)


#######################################################

indices = []
for topic in subj_to_select:
    ind = np.nonzero(np.array(dataset["subject"]) == topic)[0]
    assert len(ind) > 0, topic
    indices.extend(list(ind[:1000]))

indices = np.arange(len(dataset["subject"]))


sub_to_int = {sub: i for i, sub in enumerate(np.unique(dataset["subject"]))}
int_to_sub = {i: sub for sub, i in sub_to_int.items()}
int_labels = np.array([sub_to_int[sub] for sub in dataset["subject"]])


act = torch.load(f"{base_path}/0shot/l5_target.pt")
act = act.to(torch.float64).numpy()
act = act[indices]
d = dadapy.Data(act)
print("computing id")
ids, _, _ = d.return_id_scaling_gride(range_max=100)
d.set_id(ids[3])
print("computing clustering")
d.compute_density_kNN(k=16)
# d.compute_density_PAk()
assignment = d.compute_clustering_ADP(Z=1.6, halo=False)
d.N_clusters
print(adjusted_rand_score(assignment, int_labels[indices]))
print(homogeneity_score(assignment, int_labels[indices]))
