from dadapy.data import Data
import torch
import numpy as np
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import scipy as sp


dirpath = "./results/mmlu/llama-3-8b/5shot"
x = torch.load(f"{dirpath}/l6_target.pt")


with open(f"{dirpath}/statistics_target.pkl", "rb") as f:
    stats = pickle.load(f)

stats["subjects"]

X = x.to(torch.float64).numpy()
X, indx, inverse = np.unique(X, axis=0, return_inverse=True, return_index=True)
d = Data(coordinates=X)
d.return_id_scaling_gride(range_max=100)
d.compute_density_PAk()

cluster_assignment = d.compute_clustering_ADP(Z=1.6)


np.sum(np.array([len(ind) for ind in d.cluster_indices]) > 30)

np.where(cluster_assignment == 0)

len(Counter(cluster_assignment))


# *****************************************************


Fmax = max(d.log_den)
Rho_bord_m = d.log_den_bord
nclus = d.N_clusters
# **********************************************

Dis = []
# nd = int((nclus * nclus - nclus) / 2)
for i in range(nclus - 1):
    for j in range(i + 1, nclus - 1):
        Dis.append(Fmax - Rho_bord_m[i][j])

# similar clusters are those whic are closer "small distance".
# we subtract the saddle point density to the max density peak hiest saddles --> close categories
# why not simply take the negative of the saddle density?

Dis = np.array(Dis)

# methods: 'single', 'complete', 'average', 'weighted', 'centroid'
DD = sp.cluster.hierarchy.linkage(Dis, method="weighted")


# ****************************************************


thr = 32  # color threshold

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))  # create figure & 1 axis
# truncate_mode: 'lastp', 'level', None
# labels = lab
dn = sp.cluster.hierarchy.dendrogram(
    DD,
    p=32,
    truncate_mode=None,
    color_threshold=thr,
    get_leaves=True,
    orientation="bottom",
    above_threshold_color="b",
)

plt.tight_layout()
# fig.savefig('dendrogram.png', dpi = 500)
plt.show()
