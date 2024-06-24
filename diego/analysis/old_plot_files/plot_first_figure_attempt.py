import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from datasets import load_dataset


def get_repr_for_test(results_dir, folder, mode, model, dataset, spec=""):

    name = f"cluster_{model}_{mode}{spec}_eval_{dataset}_0shot.pkl"
    with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
        clus_train = pickle.load(f)

    name = f"overlap_{model}_{mode}{spec}_eval_{dataset}_0shot.pkl"
    with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
        ov_train = pickle.load(f)

    return clus_train, ov_train


def compute_distr(seq, q=[25, 50, 75]):
    stats = np.array([np.percentile(elem, q=q) for elem in seq])

    first_quartile = stats[:, 0]
    median = stats[:, 1]
    second_quartile = stats[:, 2]
    return first_quartile, median, second_quartile


# ***********************************************************************************
plots_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/plots/final"
results_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/results"
dataset = "test_balanced200"


clus_test_ll3_ft, ov_test_ll3_ft = get_repr_for_test(
    results_dir=results_dir,
    folder="finetuned",
    mode="finetuned_dev_val_balanced_20samples",
    model="llama-3-8b",
    dataset="test_balanced200",
    spec="_epoch4",
)

clus_test_ll3_pt, ov_test_ll3_pt = get_repr_for_test(
    results_dir=results_dir,
    folder="pretrained",
    mode="random_order",
    model="llama-3-8b",
    dataset="test_balanced200",
)

clus_test_ll2_ft, ov_test_ll2_ft = get_repr_for_test(
    results_dir=results_dir,
    folder="finetuned",
    mode="finetuned_dev_val_balanced_20samples",
    model="llama-2-13b",
    dataset="test_balanced200",
    spec="_epoch4",
)

clus_test_ll2_pt, ov_test_ll2_pt = get_repr_for_test(
    results_dir=results_dir,
    folder="pretrained",
    mode="random_order",
    model="llama-2-13b",
    dataset="test_balanced200",
)

clus_test_mis_ft, ov_test_mis_ft = get_repr_for_test(
    results_dir=results_dir,
    folder="finetuned",
    mode="finetuned_dev_val_balanced_20samples",
    model="mistral-1-7b",
    dataset="test_balanced200",
    spec="_epoch4",
)

clus_test_mis_pt, ov_test_mis_pt = get_repr_for_test(
    results_dir=results_dir,
    folder="pretrained",
    mode="random_order",
    model="mistral-1-7b",
    dataset="test_balanced200",
)


clus_test_ll3_pt.keys()


############################à
#############################

# FIGURE

##############################
##############################

sns.set_style("whitegrid")
fig = plt.figure(figsize=(13, 3.7))
gs1 = GridSpec(1, 3)

######################

# NUMBER CLUSTERS

#######################

ax = fig.add_subplot(gs1[0])
sns.lineplot(
    clus_test_ll3_pt["nclus-0shot-z1.6"],
    marker=".",
    label="0-shot",
)
sns.lineplot(
    clus_test_ll3_pt["nclus-5shot-z1.6"],
    marker=".",
    label="5-shot",
)
sns.lineplot(
    clus_test_ll3_ft["nclus-ep-4-z1.6"],
    marker=".",
    label="finteuned",
)

ax.legend()
ax.set_ylabel("N clusters")
ax.set_xlabel("layers")
ax.set_title("llama-3-8b")
nlayers = len(clus_test_ll3_pt["nclus-5shot-z1.6"])
ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))

# *****************************************************

ax = fig.add_subplot(gs1[1])
sns.lineplot(
    clus_test_ll2_pt["nclus-0shot-z1.6"],
    marker=".",
    label="0-shot",
)
sns.lineplot(
    clus_test_ll2_pt["nclus-5shot-z1.6"],
    marker=".",
    label="5-shot",
)
sns.lineplot(
    clus_test_ll2_ft["nclus-ep-4-z1.6"],
    marker=".",
    label="finteuned",
)

ax.legend()
ax.set_ylabel("N clusters")
ax.set_xlabel("layers")
ax.set_title("llama-2-13b")
nlayers = len(clus_test_ll2_pt["nclus-5shot-z1.6"])
ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))

# ******************************************************

ax = fig.add_subplot(gs1[2])
sns.lineplot(
    clus_test_mis_pt["nclus-0shot-z1.6"],
    marker=".",
    label="0-shot",
)
sns.lineplot(
    clus_test_mis_pt["nclus-5shot-z1.6"],
    marker=".",
    label="5-shot",
)
sns.lineplot(
    clus_test_mis_ft["nclus-ep-4-z1.6"],
    marker=".",
    label="finteuned",
)

ax.legend()
ax.set_ylabel("N clusters")
ax.set_xlabel("layers")
ax.set_title("mistral-7b")
nlayers = len(clus_test_mis_pt["nclus-5shot-z1.6"])
ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))
gs1.tight_layout(fig)
plt.savefig(f"{plots_dir}/fig1.png", dpi=200)

# ***********************************************************************
