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

# dirpath = f"{base_dir}/mmlu/llama-2-7b-shuffled"

# dist = np.load(f"{dirpath}/distances-6.npy")
# indices = np.load(f"{dirpath}/dist_indices-6.npy")
# gtl = np.load(f"{dirpath}/subjects-labels.pkl.npy")

# d = Data(distances=(dist, indices), maxk=300)

# with open(f"{dirpath}/subjects-map", "rb") as f:
#     stats = pickle.load(f)

# index_to_subject = {val: key[5:] for key, val in stats.items()}


mask = np.load("test_mask.npy")
nshots = "5shot"


dirpath = f"{base_dir}/evaluated_test/random_order/llama3-70b/4shot"

# finetuned_model
# dirpath = f"{base_dir}/finetuned_dev/evaluated_test/llama-3-8b/4epochs/epoch_4"
# dirpath = (
#     f"{base_dir}/finetuned_test_balanced/evaluated_test/llama-3-8b/4epochs/epoch_4"
# )


# "wrong answers"
options = ["wrong_answers", "dummy", "random_subject"]
dirpath_sub = f"{base_dir}/evaluated_test/random_subject/llama-3-8b/5shot"

with open(f"{dirpath_sub}/statistics_target.pkl", "rb") as f:
    stats = pickle.load(f)


# we need to take just 100 samples per class (for simplicity)
nsamples_per_subject = 100


subjects = np.array(stats["subjects"])
base_path_mask = (
    "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis"
)
mask = np.load(f"{base_path_mask}/test_mask_100.npy")

frequences = Counter(np.array(stats["subjects"])[mask]).values()
assert len(np.unique(list(frequences))) == 1
assert np.unique(list(frequences))[0] == nsamples_per_subject, np.unique(
    list(frequences)
)[0]


subjects = np.array(stats["subjects"])[mask]
gtl = np.repeat(np.arange(57), nsamples_per_subject)

print(dirpath)

X = torch.load(f"{dirpath}/l10_target.pt")
X = X.to(torch.float64).numpy()[mask]


X_, indx, inverse = np.unique(X, axis=0, return_inverse=True, return_index=True)
sorted_indx = np.sort(indx)
X_sub = X[sorted_indx]
gtl = gtl[sorted_indx]
subjects = subjects[sorted_indx]


index_to_subject = {}
for i in range(len(gtl)):
    if gtl[i] in index_to_subject:
        assert index_to_subject[gtl[i]] == subjects[i], (i, gtl[i], subjects[i])
    else:
        index_to_subject[gtl[i]] = subjects[i]


# ************************************************************************

# Z = 1.6 was just fine
d = Data(coordinates=X_sub, maxk=300)
ids, _, _ = d.return_id_scaling_gride(range_max=300)
d.set_id(ids[3])
# d.return_id_scaling_gride(range_max=100)
# d.compute_density_PAk()
d.compute_density_kNN(k=16)
cluster_assignment = d.compute_clustering_ADP(Z=1, halo=False)

is_core = cluster_assignment != -1


print("n_clusters", d.N_clusters)

adjusted_rand_score(gtl[is_core], cluster_assignment[is_core])

# adjusted_mutual_info_score(gtl[is_core], cluster_assignment[is_core])

# **************************************************************


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

# *****************************************************

# pivotal quantities
# old code
# Fmax = max(d.log_den)
# Rho_bord_m = d.log_den_bord
# nclus = d.N_clusters
# nd = int((nclus * nclus - nclus) / 2)
# for i in range(nclus - 1):
#     for j in range(i + 1, nclus - 1):
#         Dis.append(Fmax - Rho_bord_m[i][j])


# remove small_clusters
final_clusters = final_clusters_tmp
nclus = len(final_clusters)
assert nclus == np.sum(cluster_mask)

saddle_densities = d.log_den_bord[cluster_mask]
saddle_densities = saddle_densities[:, cluster_mask]
density_peak_indices = np.array(d.cluster_centers)[cluster_mask]
len(density_peak_indices)

# key quantity for the dendogram is here: try to come up with the best

density_peaks = d.log_den[density_peak_indices]
Fmax = max(density_peaks)
Dis = []
for i in range(nclus):
    for j in range(i + 1, nclus):
        Dis.append(Fmax - saddle_densities[i][j])


# Dis = []
# for i in range(nclus):
#     for j in range(i + 1, nclus):
#         Fmax = max(density_peaks[i], density_peaks[j])
#         Dis.append(Fmax - saddle_densities[i][j])


# similar clusters are those whic are closer "small distance".
# we subtract the saddle point density to the max density peak hiest saddles --> close categories
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
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(np.sort(list(Counter(cluster_assignment).values())), marker=".")
# ax.set_ylabel("cluster size", fontsize=11)
# ax.set_xlabel("clusters", fontsize=11)
# ax.set_yscale("log")
# plt.savefig("./plots/cluster_population_llama3_layer6_5shot_z1_0shot.png")


thr = 20  # color threshold

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
