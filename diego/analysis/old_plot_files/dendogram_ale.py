from dadapy.data import Data
import torch
import numpy as np
import pickle
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import scipy as sp
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def get_composition(cluster_indices, index_to_subject, subject_relevance):
    # subject integer identifier: frequency of the subjects
    counts = Counter(np.array(cluster_indices) // 100)
    # total_number of points in the cluster
    population = len(cluster_indices)

    # all subjects contained in a cluster
    clust_subjects = [index_to_subject[t[0]] for t in counts.most_common()]

    clust_percent = [t[1] / population for t in counts.most_common()]

    assert np.sum(clust_percent) > 0.9999, (clust_percent, np.sum(clust_percent))

    class_to_count = []
    current_perc = 0

    #
    for i in range(len(clust_subjects)):
        current_perc += clust_percent[i]
        # we consider a subject relevant only iif its presence is >0.2 the maximum popolated class
        if clust_percent[i] / clust_percent[0] > subject_relevance:
            class_to_count.append(clust_subjects[i])
        if current_perc > 0.9:
            break

    if clust_percent[0] < 0.3 or len(class_to_count) > 3:
        class_to_count = [f"mix of {len(class_to_count)} subjects"]

    return clust_subjects, clust_percent, class_to_count


# *******************************************************************************************************

base_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/results"

mask = np.load("test_mask.npy")
nshots = "5shot"
dirpath = f"{base_dir}/mmlu/llama-3-8b/{nshots}"

with open(f"{dirpath}/statistics_target.pkl", "rb") as f:
    stats = pickle.load(f)


# we need to take just 100 samples per class (for simplicity)
nsamples_per_subject = 100
subjects = np.array(stats["subjects"])
frequences = Counter(np.array(stats["subjects"])[mask]).values()
assert len(np.unique(list(frequences))) == 1
assert np.unique(list(frequences))[0] == nsamples_per_subject, np.unique(
    list(frequences)
)[0]


subjects = np.array(stats["subjects"])[mask]
# ground truth labels array where an integer corresponds to each subject
# this ground truth label array should be (carefully!) adapted for cases
# in which we do not have an equal number of samples per class

gtl = np.repeat(np.arange(57), nsamples_per_subject)


# *********************************************************************************

X = torch.load(f"{dirpath}/l6_target.pt")
X = X.to(torch.float64).numpy()[mask]

# check we do not have opverlapping datapoints in case ramove tham from the relevant arrays
X_, indx, inverse = np.unique(X, axis=0, return_inverse=True, return_index=True)
sorted_indx = np.sort(indx)
X_sub = X[sorted_indx]
gtl = gtl[sorted_indx]
subjects = subjects[sorted_indx]


# mapping indices to subjects
index_to_subject = {}
for i in range(len(gtl)):
    if gtl[i] in index_to_subject:
        assert index_to_subject[gtl[i]] == subjects[i], (i, gtl[i], subjects[i])
    else:
        index_to_subject[gtl[i]] = subjects[i]

# ************************************************************************
# here we should try also halo = True
d = Data(coordinates=X_sub, maxk=300)
ids, _, _ = d.return_id_scaling_gride(range_max=300)
d.set_id(ids[4])
# d.return_id_scaling_gride(range_max=100)
# d.compute_density_PAk()
d.compute_density_kNN(k=32)
cluster_assignment = d.compute_clustering_ADP(Z=1, halo=False)
is_core = cluster_assignment != -1

# **************************************************************
# we consider only clusters with at least 30 points (this can be relaxed or decreased)
min_population = 30

cluster_mask = []
final_clusters_tmp = []
for cluster_indices in d.cluster_indices:
    if len(cluster_indices) > min_population:
        cluster_mask.append(True)
        final_clusters_tmp.append(cluster_indices)
    else:
        cluster_mask.append(False)

cluster_mask = np.array(cluster_mask)
assert cluster_mask.shape[0] == d.N_clusters


final_clusters = final_clusters_tmp
nclus = len(final_clusters)
assert nclus == np.sum(cluster_mask)
saddle_densities = d.log_den_bord[cluster_mask]
saddle_densities = saddle_densities[:, cluster_mask]
density_peak_indices = np.array(d.cluster_centers)[cluster_mask]


# *****************************************************
# pivotal similrity matrices of the clusters with a population > min_population

# key quantity for the dendogram is here: similarity matrix given by the density
# default from the advanced density peak paper. Other similarities can be tried in a second phase.
density_peaks = d.log_den[density_peak_indices]
Fmax = max(density_peaks)
Dis = []
for i in range(nclus):
    for j in range(i + 1, nclus):
        Dis.append(Fmax - saddle_densities[i][j])


# similar clusters are those whic are closer "small distance".
# we subtract the saddle point density to the max density peak highest saddles --> close categories
Dis = np.array(Dis)
# methods: 'single', 'complete', 'average', 'weighted', 'centroid'
DD = sp.cluster.hierarchy.linkage(Dis, method="weighted")
# ************************************************************************
# get subjects in the final clusters

# we consider a subject relevant in a cluster only if it has a frequancy 0.2*the most present class
subject_relevance = 0.2
clust_subjects = defaultdict(list)
clust_approx_subjects = defaultdict(list)
clust_percent = defaultdict(list)

for i in range(len(final_clusters)):
    sub, comp, approx = get_composition(
        final_clusters[i], index_to_subject, subject_relevance
    )
    clust_subjects[i] = sub
    clust_approx_subjects[i] = approx
    clust_percent[i] = comp


labels = []
for i, (clust, subjects) in enumerate(clust_approx_subjects.items()):
    name = ""
    for sub in subjects:
        name += f" {sub},"
    labels.append(name.strip()[:-1])

# ****************************************************
thr = 10  # color threshold

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))  # create figure & 1 axis
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
    labels=labels,
)

plt.tight_layout()
fig.savefig(f"./plots/dendogram_llama3-8b_layer_6_{nshots}.png", dpi=150)
