from dadapy.data import Data
from dadapy.plot import get_dendrogram

import matplotlib.pyplot as plt
import scipy as sp
from helper_dendogram import (
    get_subject_array,
    get_dataset_mask,
    get_dissimilarity,
    get_xy_coords,
    get_clusters,
    plot_with_labels,
)
from collections import defaultdict
import umap
import numpy as np
import torch

results_path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/plots/final/dendrograms"
base_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo"
subjects = get_subject_array(base_dir)
mask, ground_truth_labels = get_dataset_mask(
    base_dir, nsample_subjects=200, subjects=subjects
)
subjects = subjects[mask]

selected_subjetcs = [
    "anatomy",
    "college_biology",
    "medical_genetics",
    "clinical_knowledge",
    "philosophy",
    "moral_disputes",
    "high_school_physics",
    "high_school_chemistry",
    "conceptual_physics",
    "college_computer_science",
    "high_school_statistics",
    "machine_learning",
]

colors = {
    "anatomy": "C0",
    "college_biology": "C0",
    "medical_genetics": "C0",
    "clinical_knowledge": "C0",
    "philosophy": "C1",
    "moral_disputes": "C1",
    "high_school_physics": "C2",
    "high_school_chemistry": "C2",
    "conceptual_physics": "C2",
    "college_computer_science": "C3",
    "high_school_statistics": "C3",
    "machine_learning": "C3",
}

np.unique(subjects)
mmask = []
col = []
for sub in selected_subjetcs:
    if sub not in subjects:
        assert False, sub
    else:
        ind = list(np.nonzero(sub == subjects)[0])
        mmask.extend(ind)
        col.extend([colors[sub] for _ in range(len(ind))])
mmask = np.array(mmask)

# ********************************************************************


################################################À

# 5 SHOT

####################################################À

print("finding clusters 5 shot")
model_path = f"{base_dir}/results/evaluated_test/random_order/llama3-8b/5shot"

X = torch.load(f"{model_path}/l3_target.pt")
X = X.to(torch.float64).numpy()[mask[mmask]]

fit = umap.UMAP(n_neighbors=100, min_dist=1, metric="euclidean")
u = fit.fit_transform(X)
u.shape
plt.scatter(u[:, 0], u[:, 1], s=4, c=col)
