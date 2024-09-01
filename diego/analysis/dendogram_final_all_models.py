from dadapy.data import Data
from dadapy.plot import get_dendrogram

import matplotlib.pyplot as plt
import scipy as sp
from helper_dendogram import (
    get_subject_array,
    get_dataset_mask,
    get_dissimilarity,
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

import numpy as np
from collections import Counter

# ********************************************************************


################################################À

# LLAMA 2 7B 5 SHOT

####################################################À

print("finding clusters 5 shot")
model_path = f"{base_dir}/results/evaluated_test/random_order/llama-2-7b/5shot"

d, index_to_subject = get_clusters(
    model_path=model_path,
    layer=6,
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
    path=f"{results_path}/llama-2-7b-5shot-Z=1.6",
    width=10,
    color_threshold=13,
)


################################################################

# LLAMA 2 13B 5 SHOT

################################################################

print("finding clusters 5 shot")
model_path = f"{base_dir}/results/evaluated_test/random_order/llama-2-13b/5shot"

d_0s, index_to_subject_0s = get_clusters(
    model_path=model_path,
    layer=5,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=1.6,
)


# F = [d_0s.log_den[c] for c in d_0s.cluster_centers]
# plt.plot(F, marker=".")
dis_0s, peak_y_0s, final_clusters_0s = get_dissimilarity(d_0s, threshold=-10)

plot_with_labels(
    dis_0s,
    final_clusters_0s,
    ground_truth_labels,
    index_to_subject_0s,
    path=f"{results_path}/llama-2-13b-5shot-Z=1.6",
    width=9,
    color_threshold=11,
)


############################################################

# LLAMA 2 70B

#######################################################
print("finding clusters llama 2 70")

model_path = f"{base_dir}/results/evaluated_test/random_order/llama-2-70b/5shot"

d_ft, index_to_subject_ft = get_clusters(
    model_path=model_path,
    layer=10,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=1.6,
)

# F = [d_ft.log_den[c] for c in d_ft.cluster_centers]
# plt.plot(F, marker=".")
dis_ft, peak_y_ft, final_clusters_ft = get_dissimilarity(d_ft, threshold=-10)

plot_with_labels(
    dis_ft,
    final_clusters_ft,
    ground_truth_labels,
    index_to_subject_ft,
    path=f"{results_path}/llama-2-70b-5shot-Z=1.6",
    width=8,
    color_threshold=11,
)


############################################################

# MISTRAL

#######################################################
print("finding clusters mistral")

model_path = f"{base_dir}/results/evaluated_test/random_order/mistral/5shot"

d_ft, index_to_subject_ft = get_clusters(
    model_path=model_path,
    layer=4,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=1.6,
)

# F = [d_ft.log_den[c] for c in d_ft.cluster_centers]
# plt.plot(F, marker=".")
dis_ft, peak_y_ft, final_clusters_ft = get_dissimilarity(d_ft, threshold=-10)

plot_with_labels(
    dis_ft,
    final_clusters_ft,
    ground_truth_labels,
    index_to_subject_ft,
    path=f"{results_path}/mistral-5shot-Z=1.6",
    width=11,
    color_threshold=18,
)


############################################################

# LLAMA 3 70B

#######################################################


model_path = f"{base_dir}/results/evaluated_test/random_order/llama-3-70b/4shot"

d_ft, index_to_subject_ft = get_clusters(
    model_path=model_path,
    layer=7,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=1,
)
d = d_ft
mc_fraction = []
for index in np.unique(d.cluster_assignment):
    ind_ = np.nonzero(d.cluster_assignment == index)[0]
    mc = Counter(ground_truth_labels[ind_]).most_common()[0][1]
    mc_fraction.append(mc / len(ind_))

np.sum(np.array(mc_fraction) > 0.80)


# F = [d_ft.log_den[c] for c in d_ft.cluster_centers]
# plt.plot(F, marker=".")
dis_ft, peak_y_ft, final_clusters_ft = get_dissimilarity(d_ft, threshold=-10)

plot_with_labels(
    dis_ft,
    final_clusters_ft,
    ground_truth_labels,
    index_to_subject_ft,
    path=f"{results_path}/llama-3-70b-5shot-Z=1.6",
    width=9,
    color_threshold=10,
)


######################################################################
# ********************************************************************

#  LLAMA 3 8B


######################

# 5 SHOT Z = 1.6

######################

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
    path=f"{results_path}/llama-3-8b-5shot-Z=1.6",
    width=7,
    color_threshold=15,
)

######################

# 5 SHOT Z = 0

######################


d, index_to_subject = get_clusters(
    model_path=model_path,
    layer=3,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=0,
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
    path=f"{results_path}/llama-3-8b-5shot-Z=0",
    width=6.5,
    color_threshold=14,
)


######################

# 5 SHOT Z = 4

######################


d, index_to_subject = get_clusters(
    model_path=model_path,
    layer=3,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=4,
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
    path=f"{results_path}/llama-3-8b-5shot-Z=4",
    width=9,
    color_threshold=15,
)


################################################################

# 0 SHOT Z = 1.6

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
    path=f"{results_path}/llama-3-8b-0shot-Z=1.6",
    width=12,
    color_threshold=18,
)


################################################################

# 0 SHOT Z = 0

################################################################

print("finding clusters 0 shot")
model_path = f"{base_dir}/results/evaluated_test/random_order/llama3-8b/0shot"

d_0s, index_to_subject_0s = get_clusters(
    model_path=model_path,
    layer=9,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=0,
)


# F = [d_0s.log_den[c] for c in d_0s.cluster_centers]
# plt.plot(F, marker=".")
dis_0s, peak_y_0s, final_clusters_0s = get_dissimilarity(d_0s, threshold=-14)

plot_with_labels(
    dis_0s,
    final_clusters_0s,
    ground_truth_labels,
    index_to_subject_0s,
    path=f"{results_path}/llama-3-8b-0shot-Z=0",
    width=12,
    color_threshold=18,
)


################################################################

# 0 SHOT Z = 4

################################################################

print("finding clusters 0 shot")
model_path = f"{base_dir}/results/evaluated_test/random_order/llama3-8b/0shot"

d_0s, index_to_subject_0s = get_clusters(
    model_path=model_path,
    layer=9,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=4,
)


# F = [d_0s.log_den[c] for c in d_0s.cluster_centers]
# plt.plot(F, marker=".")
dis_0s, peak_y_0s, final_clusters_0s = get_dissimilarity(d_0s, threshold=-14)

plot_with_labels(
    dis_0s,
    final_clusters_0s,
    ground_truth_labels,
    index_to_subject_0s,
    path=f"{results_path}/llama-3-8b-0shot-Z=4",
    width=9,
    color_threshold=18,
)


##################################################

# FINETUNED Z = 1.6

###################################################
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
    path=f"{results_path}/llama-3-8b-ft-Z=1.6",
    width=13,
    color_threshold=13,
)


##################################################

# FINETUNED Z = 0

###################################################
print("finding clusters finetuned")


model_path = f"{base_dir}/results/finetuned_dev_val_balanced_20sample/evaluated_test/llama-3-8b/4epochs/epoch_4"

d_ft, index_to_subject_ft = get_clusters(
    model_path=model_path,
    layer=5,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=0,
)


# F = [d_ft.log_den[c] for c in d_ft.cluster_centers]
# plt.plot(F, marker=".")
dis_ft, peak_y_ft, final_clusters_ft = get_dissimilarity(d_ft, threshold=-14)

plot_with_labels(
    dis_ft,
    final_clusters_ft,
    ground_truth_labels,
    index_to_subject_ft,
    path=f"{results_path}/llama-3-8b-ft-Z=0",
    width=11,
    color_threshold=15,
)


##################################################

# FINETUNED Z = 4

###################################################
print("finding clusters finetuned")


model_path = f"{base_dir}/results/finetuned_dev_val_balanced_20sample/evaluated_test/llama-3-8b/4epochs/epoch_4"

d_ft, index_to_subject_ft = get_clusters(
    model_path=model_path,
    layer=5,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=4,
)


# F = [d_ft.log_den[c] for c in d_ft.cluster_centers]
# plt.plot(F, marker=".")
dis_ft, peak_y_ft, final_clusters_ft = get_dissimilarity(d_ft, threshold=-14)

plot_with_labels(
    dis_ft,
    final_clusters_ft,
    ground_truth_labels,
    index_to_subject_ft,
    path=f"{results_path}/llama-3-8b-ft-Z=4",
    width=9,
    color_threshold=12,
)
