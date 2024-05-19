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


clus_test_ll3_pt["nclus-5shot-z1.6"]
len(clus_test_ll3_ft["population-ep-4-z1.6"])
len(clus_test_ll3_ft["population-ep-4-z1.6"][0])


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


############################

# DISTRIBUTION

############################


ax = fig.add_subplot(gs1[1])

x = np.arange(len(clus_test_ll3_ft["population-ep-4-z1.6"])) + 1
y1, y, y2 = compute_distr(clus_test_ll3_ft["population-ep-4-z1.6"], q=[40, 50, 60])
ax.fill_between(x, y1, y2, alpha=0.2)
sns.lineplot(
    x=x,
    y=y,
    marker=".",
    label="5-shot",
)


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


######################################à


ax = fig.add_subplot(gs1[1])

sns.lineplot(
    ov_test_ll3_ft["letters-ep-4-0.1"],
    marker=".",
    label="fine-tuned",
)

sns.lineplot(
    ov_test_ll3_pt["letters-0shot-0.1"],
    marker=".",
    label="5shot",
)
ax.legend()
ax.set_ylabel("overlap answers")
ax.set_xlabel("layers")
ax.set_title("llama-2-13b")

ax.set_xticks(np.arange(1, 42, 4))
ax.set_xticklabels(np.arange(1, 42, 4))

gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])


# *****************************************************************
gs2 = GridSpec(1, 2)
ax = fig.add_subplot(gs2[0])
sns.lineplot(
    clus_test_ll2_ft["letters-ari-ep-4-z1.6"],
    marker=".",
    label="fine-tuned",
)

sns.lineplot(
    clus_test_ll2_pt["letters-ari-5shot-z1.6"],
    marker=".",
    label="5shot",
)
ax.legend()
ax.set_ylabel("ARI answers")
ax.set_xlabel("layers")
ax.set_title("llama-2-13b")
ax.set_xticks(np.arange(1, 42, 4))
ax.set_xticklabels(np.arange(1, 42, 4))


ax = fig.add_subplot(gs2[1])

sns.lineplot(
    ov_test_ll2_ft["letters-ep-4-0.1"],
    marker=".",
    label="fine-tuned",
)

sns.lineplot(
    ov_test_ll2_pt["letters-0shot-0.1"],
    marker=".",
    label="5shot",
)
ax.legend()
ax.set_ylabel("overlap answers")
ax.set_xlabel("layers")
ax.set_title("llama-2-13b")

ax.set_xticks(np.arange(1, 42, 4))
ax.set_xticklabels(np.arange(1, 42, 4))


gs2.tight_layout(fig, rect=[0.5, 0, 1, 1])
plt.savefig(f"{plots_dir}/overlap_labels.png", dpi=200)


# *******************************************************

# ax.plot(ov_train["letters-ep-4-0.1"], marker=".", label="epoch 4", color="C3", alpha=1)
# ax.set_title("llama-2-13b")
# ax.set_ylabel("overlap labels")
# ax.set_xticks(np.arange(1, 42, 4))
# ax.set_xticklabels(np.arange(1, 42, 4))
# ax.legend()


# ax = fig.add_subplot(gs[1])
# ax.plot(comparison_ov["ep0_k30"], marker=".", label="epoch 0", color="C0", alpha=0.2)
# ax.plot(comparison_ov["ep1_k30"], marker=".", label="epoch 1", color="C0", alpha=0.35)
# ax.plot(comparison_ov["ep2_k30"], marker=".", label="epoch 2", color="C0", alpha=0.55)
# ax.plot(comparison_ov["ep4_k30"], marker=".", label="epoch 4", color="C0", alpha=1)

# ax.legend()
# ax.set_xticks(np.arange(1, 42, 4))
# ax.set_xticklabels(np.arange(1, 42, 4))
# ax.set_title("llama-2-13b")
# ax.set_ylabel("overlap pretrained")
# gs.tight_layout(fig)
# plt.savefig(f"{plots_dir}/finetuned_dynamics", dpi=200)


# plt.plot(clus_test_ft["letters-ari-ep-4-z1.6"])
# plt.plot(clus_test_pt["letters-ari-5shot-z1.6"])


# plt.plot(clus_test_ft["letters-entropies-ep-4"])
# plt.plot(clus_test_pt["letters-ari-5shot-z1.6"])


# clus_test_ft.keys()
# len(clus_test_ft["population-ep-4-z1.6"])
# clus_test_ft["letters-entropies-ep-4-z1.6"]
# clus_test_ft.keys()


# plt.plot(ov_test_ft["letters-ep-4-0.1"])
# plt.plot(ov_test_pt["letters-5shot-0.1"])


# comparison_clusters.keys()
