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
    path=f"{results_path}/llama-2-7b-5shot-Z=1.6.png",
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
    path=f"{results_path}/llama-2-13b-5shot-Z=1.6.png",
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
    path=f"{results_path}/llama-2-70b-5shot-Z=1.6.png",
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
    path=f"{results_path}/mistral-5shot-Z=1.6.png",
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

# F = [d_ft.log_den[c] for c in d_ft.cluster_centers]
# plt.plot(F, marker=".")
dis_ft, peak_y_ft, final_clusters_ft = get_dissimilarity(d_ft, threshold=-10)

plot_with_labels(
    dis_ft,
    final_clusters_ft,
    ground_truth_labels,
    index_to_subject_ft,
    path=f"{results_path}/llama-3-70b-5shot-Z=1.6.png",
)
