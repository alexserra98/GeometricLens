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
    get_composition_imbalanced,
)
from collections import defaultdict


base_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo"

subjects = get_subject_array(base_dir)
mask, ground_truth_labels = get_dataset_mask(
    base_dir, nsample_subjects=200, subjects=subjects
)
subjects = subjects[mask]

model_path = f"{base_dir}/results/evaluated_test/random_order/llama3-8b/5shot"

d, index_to_subject = get_clusters(
    model_path=model_path,
    layer=3,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
)


F = [d.log_den[c] for c in d.cluster_centers]
plt.plot(F, marker=".")

dis, peak_y, final_clusters = get_dissimilarity(d, threshold=-10, min_population=0)


plt.plot(peak_y)

# ****************************************************************************************************************
DD = sp.cluster.hierarchy.linkage(dis, method="weighted")
# we consider a subject relevant in a cluster only if it has a frequancy 0.2*the most present class
subject_relevance = 0.2
clust_subjects = defaultdict(list)
clust_approx_subjects = defaultdict(list)
clust_percent = defaultdict(list)

for i in range(len(final_clusters)):
    sub, comp, approx = get_composition_imbalanced(
        ground_truth_labels,
        final_clusters[i],
        index_to_subject,
        subject_relevance=0.2,
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

# ************************************
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
dn = sp.cluster.hierarchy.dendrogram(
    DD,
    p=30,
    truncate_mode=None,
    color_threshold=10,
    get_leaves=True,
    orientation="bottom",
    above_threshold_color="C5",
    labels=labels,
    leaf_font_size=10,
)
plt.tight_layout()
# **********************************************************************

xcoords_5shot, ycoords_5shot = get_xy_coords(
    dis,
    final_clusters,
    index_to_subject,
    ground_truth_labels,
    peak_y,
    max_displayed_names=2,
)


# **********************************************************************
# **********************************************************************


fig = plt.figure(figsize=(12, 10))
gs0 = fig.add_gridspec(nrows=1, ncols=6)
ax = fig.add_subplot(gs0[0, :])

nlinks = len(final_clusters) - 1
for i, link in enumerate(range(nlinks)):
    for i in range(3):
        x0, y0 = xcoords_5shot[link, i], ycoords_5shot[link, i]
        x1, y1 = xcoords_5shot[link, i + 1], ycoords_5shot[link, i + 1]
        ax.plot([x0, x1], [y0, y1], color="C5", lw=1.5)
plt.xticks(
    [
        3.5,
        7,
        14,
        21,
        26.5,
        33,
        36.5,
        41,
        45,
        52,
        57.5,
        61,
        69,
    ],
    [
        "accounting,\nlaw",
        "history",
        "medicine",
        "marketing\nmanagement\npsychology",
        "law\npolicy",
        "geography\nglobal facts",
        "economics",
        "phylosophy\njurisprudence",
        "moral disputes\nsexuality",
        "math\nphysics\nchemistry",
        "logic",
        "machine learning\ncomputer science\neconometrics",
        "abstract algebra",
    ],
    rotation=30,
    fontsize=7,
)

gs0.tight_layout(fig, rect=[0, 0.33, 1, 0.67])
plt.ylabel("log density")


# ************************************************************************

### FIRST INSET ###
gs1 = fig.add_gridspec(nrows=1, ncols=1)
ax = fig.add_subplot(gs1[0])

for link in range(nlinks):
    for i in range(3):
        x0, y0 = xcoords_5shot[link, i], ycoords_5shot[link, i]
        x1, y1 = xcoords_5shot[link, i + 1], ycoords_5shot[link, i + 1]
        ax.plot([x0, x1], [y0, y1], color="C3", lw=1.5)

axT = ax.twiny()
axT.tick_params()  # direction = 'out')
axT.set_xticklabels([])
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
axT.tick_params(axis="both", which="both", length=0)
axT.spines["top"].set_visible(False)
axT.spines["bottom"].set_visible(False)
axT.spines["left"].set_visible(False)
axT.spines["right"].set_visible(False)

ax.set_xlim(8.5, 19.5)
ax.set_ylim(-15, 0.5)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticks([])
ax.axis("off")
gs1.tight_layout(fig, rect=[0.05, 0.7, 0.3, 0.95])


# **********************************************************
####à INSET

gs2 = fig.add_gridspec(nrows=1, ncols=1)
ax = fig.add_subplot(gs2[0])

for link in range(nlinks):
    for i in range(3):
        x0, y0 = xcoords_5shot[link, i], ycoords_5shot[link, i]
        x1, y1 = xcoords_5shot[link, i + 1], ycoords_5shot[link, i + 1]
        ax.plot([x0, x1], [y0, y1], color="C3", lw=1.5)

axT = ax.twiny()
axT.tick_params()  # direction = 'out')
axT.set_xticklabels([])
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
axT.tick_params(axis="both", which="both", length=0)
axT.spines["top"].set_visible(False)
axT.spines["bottom"].set_visible(False)
axT.spines["left"].set_visible(False)
axT.spines["right"].set_visible(False)

ax.set_xlim(60, 68)
ax.set_ylim(-15, 0.5)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticks([])
ax.axis("off")
gs2.tight_layout(fig, rect=[0.7, 0.7, 0.9, 0.95])


plt.savefig("final_dendogram_70b.png", dpi=400)


# ****************************************************************
# ****************************************************************
