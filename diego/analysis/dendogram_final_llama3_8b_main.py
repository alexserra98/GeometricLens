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

results_path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/plots/final/dendrograms"
base_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo"
subjects = get_subject_array(base_dir)
mask, ground_truth_labels = get_dataset_mask(
    base_dir, nsample_subjects=200, subjects=subjects
)
subjects = subjects[mask]

# ********************************************************************


################################################À

# 5 SHOT

####################################################À

print("finding clusters 5 shot")
model_path = f"{base_dir}/results/evaluated_test/random_order/llama3-8b/5shot"

d, index_to_subject = get_clusters(
    model_path=model_path,
    layer=3,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=1.6,
)


# F = [d.log_den[c] for c in d.cluster_centers]
# plt.plot(F, marker=".")
dis, peak_y, final_clusters = get_dissimilarity(d, threshold=-10)

plot_with_labels(
    dis,
    final_clusters,
    ground_truth_labels,
    index_to_subject,
    ori="left",
    # path=f"{results_path}/llama-3-8b-5shot-Z=4.png",
)


xcoords_5shot, ycoords_5shot = get_xy_coords(
    dis,
    final_clusters,
    index_to_subject,
    ground_truth_labels,
    peak_y,
    max_displayed_names=2,
)


################################################################

# 0 SHOT

################################################################

print("finding clusters 0 shot")
model_path = f"{base_dir}/results/evaluated_test/random_order/llama3-8b/0shot"

d_0s, index_to_subject_0s = get_clusters(
    model_path=model_path,
    layer=9,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=1.6,
)


# F = [d_0s.log_den[c] for c in d_0s.cluster_centers]
# plt.plot(F, marker=".")
dis_0s, peak_y_0s, final_clusters_0s = get_dissimilarity(d_0s, threshold=-14)

plot_with_labels(
    dis_0s,
    final_clusters_0s,
    ground_truth_labels,
    index_to_subject_0s,
    # path=f"{results_path}/llama-3-8b-0shot-Z=1.6.png",
)
# **********************************************************************

xcoords_0shot, ycoords_0shot = get_xy_coords(
    dis_0s,
    final_clusters_0s,
    index_to_subject_0s,
    ground_truth_labels,
    peak_y_0s,
    max_displayed_names=2,
)

############################################################

# FINETUNED

#######################################################
print("finding clusters finetuned")


model_path = f"{base_dir}/results/finetuned_dev_val_balanced_20sample/evaluated_test/llama-3-8b/4epochs/epoch_4"

d_ft, index_to_subject_ft = get_clusters(
    model_path=model_path,
    layer=5,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=1.6,
)


# F = [d_ft.log_den[c] for c in d_ft.cluster_centers]
# plt.plot(F, marker=".")
dis_ft, peak_y_ft, final_clusters_ft = get_dissimilarity(d_ft, threshold=-14)

plot_with_labels(
    dis_ft,
    final_clusters_ft,
    ground_truth_labels,
    index_to_subject_ft,
    # path=f"{results_path}/llama-3-8b-ft-Z=1.6.png",
)


xcoords_ft, ycoords_ft = get_xy_coords(
    dis_ft,
    final_clusters_ft,
    index_to_subject_ft,
    ground_truth_labels,
    peak_y_ft,
    max_displayed_names=2,
)


##################################
##################################
##################################

# DENDOGRAM

##################################
##################################
##################################


fig = plt.figure(figsize=(12, 7))
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
        43,
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
        "geography",
        "economics",
        "phylosophy\njurisprudence\nmoral disputes",
        "math\nphysics\nchemistry",
        "logic",
        "machine learning\ncomputer science\neconometrics",
        "abstract algebra",
    ],
    rotation=30,
    fontsize=10,
)

gs0.tight_layout(fig, rect=[0.03, 0.35, 0.97, 0.69])
plt.ylabel("log density")


# ************************************************************************
################################################

# MEDICINE

################################################À

gs1 = fig.add_gridspec(nrows=1, ncols=1)
ax = fig.add_subplot(gs1[0])

for link in range(nlinks):
    for i in range(3):
        x0, y0 = xcoords_5shot[link, i], ycoords_5shot[link, i]
        x1, y1 = xcoords_5shot[link, i + 1], ycoords_5shot[link, i + 1]
        ax.plot([x0, x1], [y0, y1], color="C1", lw=1.5)

axT = ax.twiny()
axT.tick_params()  # direction = 'out')
axT.set_xticklabels([])
xticks = [0.05, 0.15, 0.27, 0.41, 0.55, 0.68, 0.78, 0.91]
xticklabels = [
    "nutrition",
    "professional\nmedicine",
    "anatomy",
    "college\nbiology",
    "high school\nbiology",
    "medical\ngenetics",
    "virology",
    "clinical\nknowledge",
]
for x, xlabel in zip(xticks, xticklabels):
    axT.text(
        x=x,
        y=0,
        s=f"{xlabel}",
        va="bottom",
        ha="center",
        rotation=90,
        fontsize=9,
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
gs1.tight_layout(fig, rect=[0.05, 0.7, 0.3, 0.98])


#####################################################

# PHILOSOPHY

#####################################################


gs2 = fig.add_gridspec(nrows=1, ncols=1)
ax = fig.add_subplot(gs2[0])

for link in range(nlinks):
    for i in range(3):
        x0, y0 = xcoords_5shot[link, i], ycoords_5shot[link, i]
        x1, y1 = xcoords_5shot[link, i + 1], ycoords_5shot[link, i + 1]
        ax.plot([x0, x1], [y0, y1], color="C0", lw=1.5)

axT = ax.twiny()
axT.tick_params()  # direction = 'out')
axT.set_xticklabels([])
xticks = [0.07, 0.27, 0.52, 0.75, 0.95]
xticklabels = [
    "philosophy",
    "jurisprudence",
    "prehistory",
    "human\nsexuality",
    "moral\ndisputes",
]
for x, xlabel in zip(xticks, xticklabels):
    axT.text(
        x=x,
        y=-0.5,
        s=f"{xlabel}",
        va="bottom",
        ha="center",
        rotation=90,
        fontsize=9,
    )
axT.tick_params(axis="both", which="both", length=0)
axT.spines["top"].set_visible(False)
axT.spines["bottom"].set_visible(False)
axT.spines["left"].set_visible(False)
axT.spines["right"].set_visible(False)

ax.set_xlim(39.5, 47)
ax.set_ylim(-14.8, 0.5)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticks([])
ax.axis("off")


gs2.tight_layout(fig, rect=[0.3, 0.7, 0.47, 0.98])


#####################################################

# MATH PHYSISCS

#####################################################

gs3 = fig.add_gridspec(nrows=1, ncols=1)
ax = fig.add_subplot(gs3[0])

for link in range(nlinks):
    for i in range(3):
        x0, y0 = xcoords_5shot[link, i], ycoords_5shot[link, i]
        x1, y1 = xcoords_5shot[link, i + 1], ycoords_5shot[link, i + 1]
        ax.plot([x0, x1], [y0, y1], color="C3", lw=1.5)

axT = ax.twiny()
axT.tick_params()  # direction = 'out')
axT.set_xticklabels([])
xticks = [0.04, 0.17, 0.32, 0.55, 0.75, 0.92]
xticklabels = [
    "high school\nphysics",
    "high school\nchemistry",
    "mathematics",
    "conceptual\nphisics",
    "college\nphysics",
    "college\nchemistry",
]
for x, xlabel in zip(xticks, xticklabels):
    axT.text(
        x=x,
        y=0,
        s=f"{xlabel}",
        va="bottom",
        ha="center",
        rotation=90,
        fontsize=9,
    )
axT.tick_params(axis="both", which="both", length=0)
axT.spines["top"].set_visible(False)
axT.spines["bottom"].set_visible(False)
axT.spines["left"].set_visible(False)
axT.spines["right"].set_visible(False)

ax.set_xlim(47.5, 56.5)
ax.set_ylim(-15, 0.5)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticks([])
ax.axis("off")


gs3.tight_layout(fig, rect=[0.5, 0.7, 0.73, 0.98])


############################################

# COMPUTER SCIENCE

#######################################

gs4 = fig.add_gridspec(nrows=1, ncols=1)
ax = fig.add_subplot(gs4[0])

for link in range(nlinks):
    for i in range(3):
        x0, y0 = xcoords_5shot[link, i], ycoords_5shot[link, i]
        x1, y1 = xcoords_5shot[link, i + 1], ycoords_5shot[link, i + 1]
        ax.plot([x0, x1], [y0, y1], color="C2", lw=1.5)

axT = ax.twiny()
axT.tick_params()  # direction = 'out')
axT.set_xticklabels([])
xticks = [0.06, 0.17, 0.32, 0.48, 0.62, 0.88]
xticklabels = [
    "mathematics",
    "computer\nsecurity",
    "machine\nlearning",
    "computer\nscience",
    "statistics",
    "econometrics",
]
for x, xlabel in zip(xticks, xticklabels):
    axT.text(
        x=x,
        y=0,
        s=f"{xlabel}",
        va="bottom",
        ha="center",
        rotation=90,
        fontsize=9,
    )
axT.tick_params(axis="both", which="both", length=0)
axT.spines["top"].set_visible(False)
axT.spines["bottom"].set_visible(False)
axT.spines["left"].set_visible(False)
axT.spines["right"].set_visible(False)

ax.set_xlim(58.5, 66.5)
ax.set_ylim(-15.2, 0.5)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticks([])
ax.axis("off")


gs4.tight_layout(fig, rect=[0.75, 0.7, 0.98, 0.98])


fig.text(0.5, 0.35, "5 shot", fontsize=14, fontweight="bold")
##########################################################à

# 0 SHOT

############################################################
gs5 = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs5[0])

nlinks = len(final_clusters_0s) - 1
for i, link in enumerate(range(nlinks)):
    for i in range(3):
        x0, y0 = xcoords_0shot[link, i], ycoords_0shot[link, i]
        x1, y1 = xcoords_0shot[link, i + 1], ycoords_0shot[link, i + 1]
        ax.plot([x0, x1], [y0, y1], color="C5", lw=1.5)

ax.scatter(30, -11, s=50, color="C3")
ax.scatter(26, -7, s=700, color="C3")
ax.scatter(25, -11, s=50, color="C3")
ax.scatter(24, -11, s=100, color="C3")
ax.scatter(20, -11, s=50, color="C3")
ax.scatter(18.8, -10, s=300, color="C3")
ax.scatter(15, -10, s=80, color="C3")

ax.text(26, 4, "mix 26", rotation=90, fontsize=10, fontweight="bold")
ax.text(18, 4, "mix 11", rotation=90, fontsize=10, fontweight="bold")
ax.text(24, 4, "mix 7", rotation=90, fontsize=10, fontweight="bold")
ax.text(30, 4, "mix 5", rotation=90, fontsize=10, fontweight="bold")

ax.set_xticks([])
ax.set_xticklabels([])
ax.set_ylabel("log density")
gs5.tight_layout(fig, rect=[0.02, 0.05, 0.48, 0.42])
fig.text(0.25, 0.03, "0 shot", fontsize=14, fontweight="bold")


##########################################################

# FINETUNED

##########################################################


gs6 = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs6[0])

nlinks = len(final_clusters_ft) - 1
for i, link in enumerate(range(nlinks)):
    for i in range(3):
        x0, y0 = xcoords_ft[link, i], ycoords_ft[link, i]
        x1, y1 = xcoords_ft[link, i + 1], ycoords_ft[link, i + 1]
        ax.plot([x0, x1], [y0, y1], color="C5", lw=1.5)


ax.set_xticks([])
ax.set_xticklabels([])
ax.set_ylim(-20.5, 2)
ax.set_ylabel("log density")


ax.scatter(28, -4, s=50, color="C3")
ax.scatter(26, -3, s=100, color="C3")
ax.scatter(24, -1, s=600, color="C3")
ax.scatter(22, -2, s=200, color="C3")
ax.scatter(20, -3, s=50, color="C3")
# ax.scatter(18.8, -10, s=300, color="C3")
# ax.scatter(15, -10, s=80, color="C3")

ax.text(26, 4, "mix 6", rotation=90, fontsize=10, fontweight="bold")
ax.text(22, 4, "mix 12", rotation=90, fontsize=10, fontweight="bold")
ax.text(24, 4, "mix 19", rotation=90, fontsize=10, fontweight="bold")
ax.text(20, 4, "mix 7", rotation=90, fontsize=10, fontweight="bold")
ax.text(28, 4, "mix 6", rotation=90, fontsize=10, fontweight="bold")

gs6.tight_layout(fig, rect=[0.52, 0.05, 0.98, 0.42])
fig.text(0.75, 0.03, "fine-tuned", fontsize=14, fontweight="bold")
plt.savefig(f"{results_path}/final_dendogram_llama3-8b.png")
plt.savefig(f"{results_path}/final_dendogram_llama3-8b.pdf")
