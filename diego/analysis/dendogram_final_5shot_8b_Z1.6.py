from dadapy.data import Data
from dadapy.plot import get_dendrogram

import torch
import numpy as np
import pickle
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import scipy as sp
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import seaborn as sns


def get_composition_imbalanced(
    gtl, final_clusters, index_to_subject, subject_relevance=0.2
):
    cluster_subject_samples = Counter(gtl[np.array(final_clusters)])

    total_subject_samples = Counter(gtl)

    freq_in_cluster = {
        key: val / total_subject_samples[key]
        for key, val in cluster_subject_samples.items()
    }

    subject_fraction = {
        k: v
        for k, v in sorted(
            freq_in_cluster.items(), key=lambda item: item[1], reverse=True
        )
    }
    population_fraction = {
        k: v / len(cluster_indices)
        for k, v in sorted(
            cluster_subject_samples.items(), key=lambda item: item[1], reverse=True
        )
    }
    clust_subjects = [index_to_subject[key] for key in subject_fraction.keys()]

    class_to_count = []
    highest_fraction = max(list(subject_fraction.values()))
    # dumb_check
    for key, val in subject_fraction.items():
        assert highest_fraction == val
        break

    for i, (index, fraction) in enumerate(subject_fraction.items()):
        # current_perc += clust_percent[i]
        # we consider a subject relevant only iif its presence is >0.2 the maximum popolated class
        if (
            fraction / highest_fraction > subject_relevance
            or population_fraction[index] > 0.5
        ):
            class_to_count.append(clust_subjects[i])

    if len(class_to_count) > 4:
        class_to_count = [f"mix of {len(class_to_count)} subjects"]

    return clust_subjects, subject_fraction, class_to_count


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


def get_subject_array(base_dir):
    subject_path = f"{base_dir}/results/mmlu/llama-3-8b/5shot"
    with open(f"{subject_path}/statistics_target.pkl", "rb") as f:
        stats = pickle.load(f)
    return np.array(stats["subjects"])


def get_dataset_mask(base_dir, subjects, nsample_subjects=100):

    if nsample_subjects == 100:
        mask_path = f"{base_dir}/diego/analysis"
        mask = np.load(f"{mask_path}/test_mask_100.npy")
        gtl = np.repeat(np.arange(57), nsample_subjects)

        # double check
        frequences = Counter(subjects[mask]).values()
        assert len(np.unique(list(frequences))) == 1
        assert np.unique(list(frequences))[0] == nsample_subjects, np.unique(
            list(frequences)
        )[0]
    if nsample_subjects == 200:
        mask_path = f"{base_dir}/diego/analysis"
        mask = np.load(f"{mask_path}/test_mask_200.npy")
        subjects = subjects[mask]
        sub_to_int = {sub: i for i, sub in enumerate(np.unique(subjects))}
        gtl = [sub_to_int[sub] for sub in subjects]

    else:

        mask = []
        gtl = []
        for i, sub in enumerate(np.unique(subjects)):
            ind = np.where(sub == subjects)[0]
            nsamples = min(nsample_subjects, len(ind))
            chosen = np.random.choice(ind, nsamples, replace=False)
            if sub == "world_religions":
                chosen = np.where(subjects == "world_religions")[0][
                    : (min(nsamples, 169))
                ]
            mask.extend(list(np.sort(chosen)))
            gtl.extend([i for _ in range(len(chosen))])

    return np.array(mask), np.array(gtl)


# *******************************************************************************************************

base_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo"


subjects = get_subject_array(base_dir)

mask, gtl = get_dataset_mask(base_dir, nsample_subjects=200, subjects=subjects)
subjects = subjects[mask]


# **************************************************************************


model_path = f"{base_dir}/results/evaluated_test/random_order/llama3-8b/5shot"
# model_path = f"{base_dir}/results/evaluated_test/questions_sampled13/llama3-70b/4shot"

X = torch.load(f"{model_path}/l3_target.pt")
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
d.set_id(ids[3])


# d.return_id_scaling_gride(range_max=100)
# d.compute_density_PAk()
d.compute_density_kNN(k=16)
cluster_assignment = d.compute_clustering_ADP(Z=1.6, halo=False)
is_core = cluster_assignment != -1

print("n_clusters", d.N_clusters)

print(adjusted_rand_score(gtl[is_core], cluster_assignment[is_core]))

F = [d.log_den[c] for c in d.cluster_centers]

# we filter out the maximum which seems an outplier in density

# **************************************************************
# we consider only clusters with at least 30 points (this can be relaxed or decreased)
min_population = 30

cluster_mask = []
final_clusters_tmp = []
for c, cluster_indices in zip(d.cluster_centers, d.cluster_indices):
    if len(cluster_indices) > min_population and d.log_den[c] < max(F):
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
# the highest density peak is an outlier in density we filter it out
Fmax = np.sort(F)[-2]

# F = [d.log_den[c] for c in d.cluster_centers]
peak_y = density_peaks - Fmax


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

# **********************************************************************************************


for i in range(len(final_clusters)):
    # sub, comp, approx = get_composition_imbalanced(
    #     final_clusters[i], index_to_subject, subject_relevance
    # )

    sub, comp, approx = get_composition_imbalanced(
        gtl, final_clusters[i], index_to_subject, subject_relevance=0.2
    )

    clust_subjects[i] = sub
    clust_approx_subjects[i] = approx
    clust_percent[i] = comp


labels = []
for i, (clust, subjects) in enumerate(clust_approx_subjects.items()):
    name = ""
    for i, sub in enumerate(subjects):
        if i < 2:
            name += f" {sub},"
    labels.append(name.strip()[:-1])

# ****************************************************
# thr = 20  # color threshold

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 13))  # create figure & 1 axis
# truncate_mode: 'lastp', 'level', None
# labels = lab
# sns.set_style("whitegrid")
dn = sp.cluster.hierarchy.dendrogram(
    DD,
    p=30,
    truncate_mode=None,
    color_threshold=5,
    get_leaves=True,
    orientation="left",
    above_threshold_color="C5",
    labels=labels,
    leaf_font_size=10,
)
ax.set_xticklabels([])
ax.set_xticks([])
plt.tight_layout()

# xcoords = np.array(dn["icoord"]) / 10 - 0.5


# np.max(xcoords)


# dn["icoord"][0]
# dn["dcoord"][0]
# fig.savefig(f"./plots/dendogram_llama3-8b_layer_6_{nshots}.png", dpi=150)

# *************************************************************


# without labels
dn = sp.cluster.hierarchy.dendrogram(
    DD,
    p=35,
    truncate_mode=None,
    color_threshold=24,
    get_leaves=True,
    orientation="bottom",
    labels=None,
    above_threshold_color="b",
)


xcoords = np.array(dn["icoord"]) / 10 - 0.5

# integer label of the leaf nodes (representing the clusters)
order = np.array([int(dn["ivl"][i]) for i in range(len(dn["ivl"]))])

# density of the peaks
reordered_peaks = peak_y[order]
# with this rescoling the integer coordinated are in one to
# one correspondece with the integer label of the clusters above
xcoords = np.array(dn["icoord"]) / 10 - 0.5
ycoords = -np.array(dn["dcoord"])


# left_leaf_rows = np.where(ycoords[:, 3] == 0)

# left_leaf_rows = np.where(ycoords[:, 0] == 0)
# len(left_leaf_rows[0])

# left_leaf_clus_index = xcoords[left_leaf_rows, 3][0]
# len(left_leaf_clus_index)

# right_leaf_rows = np.where(ycoords[:, 3] == 0)
# right_leaf_clus_index = xcoords[right_leaf_rows, 3][0]
# ycoords[left_leaf_rows, 0] = reordered_peaks[(left_leaf_clus_index).astype(int)]
# ycoords[right_leaf_rows, 3] = reordered_peaks[(right_leaf_clus_index).astype(int)]

ycoords[np.where(ycoords[:, 0] == 0), 0] = reordered_peaks[
    (xcoords[np.where(ycoords[:, 0] == 0), 0][0]).astype(int)
]
ycoords[np.where(ycoords[:, 3] == 0), 3] = reordered_peaks[
    (xcoords[np.where(ycoords[:, 3] == 0), 3][0]).astype(int)
]


# **************************************************************************
import matplotlib

gs = fig.add_gridspec(nrows=2, ncols=6)

fig = plt.figure(figsize=(6, 3.5))

ax = fig.add_subplot(gs[1, :])

nlinks = len(final_clusters) - 1
for i, link in enumerate(range(nlinks)):
    for i in range(3):
        x0, y0 = xcoords[link, i], ycoords[link, i]
        x1, y1 = xcoords[link, i + 1], ycoords[link, i + 1]
        ax.plot([x0, x1], [y0, y1], color="C5", lw=1.5)
ax.set_xlim(-1, 65)
plt.xticks(
    [
        1,
        6,
        13.75,
        23,
        27.5,
        32,
        38,
        45,
        50.5,
        56,
        60.5,
    ],
    [
        "history",
        "accounting,\nlaw",
        "philosophy,\nlogic",
        "physics",
        "STEM",
        "biology",
        "machine\nlearning",
        "medicine",
        "social\nsciences",
        "politics",
        "geography,\nmacroeconomics",
    ],
    rotation=30,
    fontsize=7,
)

# plt.scatter(xcoords[2, 0], ycoords[2, 0])
# plt.xticks(np.arange(last_animal), lab_, rotation="vertical", fontsize=6)

plt.yticks([-18, -12, -6, 0])
plt.ylabel("log density")

plt.xlim(-1, 65)


### FIRST INSET ###

ax = fig.add_subplot(gs[0, :4])

for link in range(nlinks):
    for i in range(3):
        x0, y0 = xcoords[link, i], ycoords[link, i]
        x1, y1 = xcoords[link, i + 1], ycoords[link, i + 1]
        ax.plot([x0, x1], [y0, y1], color="C3", lw=1.5)  # color = cl[c_nums[link]]


axT = ax.twiny()
axT.tick_params()  # direction = 'out')
axT.set_xticklabels([])
# axT.set_xticks(
#     [0, 3, 5, 8],
# )
xticks = [0.07, 0.23, 0.35, 0.45, 0.55, 0.75, 0.85, 0.95]
xticklabels = [
    "conceptual\nphysics",
    "high school\nphysics",
    "statistics",
    "computer\nscience",
    "math",
    "biology",
    "chemistry",
    "genetics\nphysics",
]


for x, xlabel in zip(xticks, xticklabels):
    axT.text(
        x=x,
        y=0,
        s=f"{xlabel}",
        va="center",
        ha="center",
        rotation=40,
        fontsize=7,
    )


# axT.set_xticklabels(xticklabels, fontsize=7, rotation=90)
axT.tick_params(axis="both", which="both", length=0)

axT.spines["top"].set_visible(False)
axT.spines["bottom"].set_visible(False)
axT.spines["left"].set_visible(False)
axT.spines["right"].set_visible(False)

ax.set_xlim(21.5, 36.5)
ax.set_ylim(-17.2, 0.5)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticks([])
ax.axis("off")
gs.tight_layout(fig)
plt.savefig("final_dendogram_70b.png", dpi=400)


# **********************************************************
