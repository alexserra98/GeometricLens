import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from datasets import load_dataset


def get_repr_for_dynmics(results_dir, folder, mode, model, dataset, ckpts=""):

    name = f"cluster_{model}_{mode}_epoch4{ckpts}_eval_{dataset}_0shot.pkl"
    with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
        clus_train = pickle.load(f)

    name = f"overlap_{model}_{mode}_epoch4{ckpts}_eval_{dataset}_0shot.pkl"
    with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
        ov_train = pickle.load(f)

    try:
        name = f"overlap_repr_{model}_{mode}_epoch4{ckpts}_eval_dev+validation_Noneshot.pkl"
        with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
            comparison_ov = pickle.load(f)

        name = f"cluster_repr_{model}_{mode}_epoch4{ckpts}_eval_dev+validation_Noneshot.pkl"
        with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
            comparison_clust = pickle.load(f)
    except:
        name = (
            f"overlap_repr_{model}_{mode}_epoch4{ckpts}_eval_dev+validation_0shot.pkl"
        )
        with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
            comparison_ov = pickle.load(f)

        name = (
            f"cluster_repr_{model}_{mode}_epoch4{ckpts}_eval_dev+validation_0shot.pkl"
        )
        with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
            comparison_clust = pickle.load(f)

    return clus_train, ov_train, comparison_ov, comparison_clust


# ***********************************************************************************
plots_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/plots/final"


results_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/results"


# *******************************************


clus_train_ll2, ov_train_ll2, comparison_ov_ll2, comparison_clust_ll2 = (
    get_repr_for_dynmics(
        results_dir=results_dir,
        folder="finetuned",
        mode="finetuned_dev_val_balanced_20samples",
        model="llama-2-13b",
        dataset="dev+validation",
        ckpts="_10ckpts",
    )
)

clus_train_ll3, ov_train_ll3, comparison_ov_ll3, comparison_clust_ll3 = (
    get_repr_for_dynmics(
        results_dir=results_dir,
        folder="finetuned",
        mode="finetuned_dev_val_balanced_20samples",
        model="llama-3-8b",
        dataset="dev+validation",
        ckpts="_10ckpts",
    )
)


clus_train_mis, ov_train_mis, comparison_ov_mis, comparison_clust_mis = (
    get_repr_for_dynmics(
        results_dir=results_dir,
        folder="finetuned",
        mode="finetuned_dev_val_balanced_20samples",
        model="mistral-1-7b",
        dataset="dev+validation",
        ckpts="_10ckpts",
    )
)


ckpts = [1, 2, 3, 6, 12, 22, 42, 77, 144, 268]
ckpts = [6, 12, 42, 77, 144, 268]
alphas = [0.2, 0.3, 0.4, 0.5, 0.7, 1]

#################################################

# ONLY SIMILARITY

#################################################

# comparison_ov_ll3[f"letters-step-{cpt}_k30"][:-1]
# comparison_ov_ll3
# comparison_ov_ll3[f"letters-step-{cpt}-0.1"][:-1]
sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

fig = plt.figure(figsize=(13, 4))

gs1 = GridSpec(1, 3)


ax = fig.add_subplot(gs1[0])
for cpt, alpha in zip(ckpts, alphas):

    ax.plot(
        np.arange(1, 33),
        comparison_ov_ll3[f"step-{cpt}_k30"][:-1],
        marker=".",
        label=f"step {cpt}",
        color="C0",
        alpha=alpha,
    )

nlayers = len(comparison_ov_ll3[f"step-{cpt}_k30"])
ax.set_title("llama-3-8b")
ax.set_ylabel("layer similarity", fontsize=11)
ax.set_xlabel("layer", fontsize=11)
ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))
ax.legend(fontsize=9)

# ********************************************************
fig = plt.figure(figsize=(13, 4))
gs1 = GridSpec(1, 3)
ax = fig.add_subplot(gs1[1])
for cpt, alpha in zip(ckpts, alphas):

    ax.plot(
        np.arange(1, 41),
        comparison_ov_ll2[f"step-{cpt}_k30"][:-1],
        marker=".",
        label=f"step {cpt}",
        color="C0",
        alpha=alpha,
    )

nlayers = len(comparison_ov_ll2[f"step-{cpt}_k30"])
ax.set_title("llama-2-13b")
ax.set_xlabel("layer", fontsize=11)
ax.set_ylabel("layer similarity", fontsize=11)
ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))
ax.legend(fontsize=9)


ax = fig.add_subplot(gs1[2])
for cpt, alpha in zip(ckpts, alphas):

    ax.plot(
        np.arange(1, 33),
        comparison_ov_mis[f"step-{cpt}_k30"][:-1],
        marker=".",
        label=f"step {cpt}",
        color="C0",
        alpha=alpha,
    )

nlayers = len(comparison_ov_mis[f"step-{cpt}_k30"])
ax.set_title("mistral-7b")
ax.set_xlabel("layer", fontsize=11)
ax.set_ylabel("layer similarity", fontsize=11)
ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))
ax.legend(fontsize=9)
gs1.tight_layout(fig, rect=[0, 0, 1, 1])

plt.savefig(f"{plots_dir}/dynamics_only_similarity.png")
plt.savefig(f"{plots_dir}/dynamics_only_similarity.pdf")


# *********************************
# **********************************

plot_config = {
    #'font.size': 12,
    "axes.titlesize": 30,
    "axes.labelsize": 29,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 23,
    "figure.figsize": (10, 8),
    "lines.linewidth": 2.5,
    "lines.markersize": 10,
}


# dynamics
sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

fig = plt.figure(figsize=(16, 4.5))


#################

# LLAMA 3 8B

################

gs1 = GridSpec(1, 2)
ax = fig.add_subplot(gs1[0])
for cpt, alpha in zip(ckpts, alphas):

    ax.plot(
        np.arange(1, 33),
        ov_train_ll3[f"letters-step-{cpt}-0.1"][:-1],
        marker=".",
        label=f"step {cpt}",
        color="C3",
        alpha=alpha,
    )

nlayers = len(ov_train_ll3[f"letters-step-{cpt}-0.1"])
ax.set_title("llama-3-8b")
ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))
ax.set_ylabel("overlap answers", fontsize=11)
ax.set_xlabel("layer", fontsize=11)
ax.legend(fontsize=10)


# ********************************************************

ax = fig.add_subplot(gs1[1])
for cpt, alpha in zip(ckpts, alphas):

    ax.plot(
        np.arange(1, 33),
        comparison_ov_ll3[f"step-{cpt}_k30"][:-1],
        marker=".",
        label=f"step {cpt}",
        color="C0",
        alpha=alpha,
    )

nlayers = len(ov_train_ll3[f"letters-step-{cpt}-0.1"])
ax.set_title("llama-3-8b")

ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))
ax.set_ylabel("layer similarity", fontsize=11)
ax.set_xlabel("layer", fontsize=11)
ax.legend(fontsize=10)
gs1.tight_layout(fig, rect=[0, 0, 0.49, 1])


###########################

# MISTRAL

##############################

# fig = plt.figure(figsize=(6, 3.7))

gs1 = GridSpec(1, 2)
ax = fig.add_subplot(gs1[0])
for cpt, alpha in zip(ckpts, alphas):

    ax.plot(
        np.arange(1, 33),
        ov_train_mis[f"letters-step-{cpt}-0.1"][:-1],
        marker=".",
        label=f"step {cpt}",
        color="C3",
        alpha=alpha,
    )

nlayers = len(ov_train_mis[f"letters-step-{cpt}-0.1"])
ax.set_title("mistral-7b")
ax.set_ylabel("overlap answers", fontsize=11)
ax.set_xlabel("layer", fontsize=11)
ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))
ax.legend(fontsize=10)


# ********************************************************

ax = fig.add_subplot(gs1[1])
for cpt, alpha in zip(ckpts, alphas):

    ax.plot(
        np.arange(1, 33),
        comparison_ov_mis[f"step-{cpt}_k30"][:-1],
        marker=".",
        label=f"step {cpt}",
        color="C0",
        alpha=alpha,
    )

nlayers = len(ov_train_mis[f"letters-step-{cpt}-0.1"])
ax.set_title("mistral-7b")
ax.set_ylabel("layer similarity", fontsize=11)
ax.set_xlabel("layer", fontsize=11)
ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))
ax.legend(fontsize=10)
gs1.tight_layout(fig, rect=[0.51, 0, 1, 1])
plt.savefig(f"{plots_dir}/finetuned_dynamics.png", dpi=300)


#################

# LLAMA 2 13B

################


sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

fig = plt.figure(figsize=(8, 4))

gs1 = GridSpec(1, 2)
ax = fig.add_subplot(gs1[0])
for cpt, alpha in zip(ckpts, alphas):

    ax.plot(
        np.arange(1, 41),
        ov_train_ll2[f"letters-step-{cpt}-0.1"][:-1],
        marker=".",
        label=f"step {cpt}",
        color="C3",
        alpha=alpha,
    )

nlayers = len(ov_train_ll2[f"letters-step-{cpt}-0.1"])
ax.set_title("llama-2-13b")
ax.set_ylabel("overlap answers", fontsize=11)
ax.set_xlabel("layer", fontsize=11)
ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))
ax.legend(fontsize=9)


# ********************************************************

ax = fig.add_subplot(gs1[1])
for cpt, alpha in zip(ckpts, alphas):

    ax.plot(
        np.arange(1, 41),
        comparison_ov_ll2[f"step-{cpt}_k30"][:-1],
        marker=".",
        label=f"step {cpt}",
        color="C0",
        alpha=alpha,
    )

nlayers = len(ov_train_ll2[f"letters-step-{cpt}-0.1"])
ax.set_title("llama-2-13b")
ax.set_xlabel("layer", fontsize=11)
ax.set_ylabel("layer similarity", fontsize=11)
ax.set_xticks(np.arange(1, nlayers, 4))
ax.set_xticklabels(np.arange(1, nlayers, 4))
ax.legend(fontsize=1)
gs1.tight_layout(fig, rect=[0, 0, 1, 1])

# plt.savefig(f"{plots_dir}/dynamics_llama2-13.png", dpi=200)


# ******************************************************
# **********************************************************


def get_repr_for_test(results_dir, folder, mode, model, dataset, spec=""):

    name = f"cluster_{model}_{mode}{spec}_eval_{dataset}_0shot.pkl"
    with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
        clus_train = pickle.load(f)

    name = f"overlap_{model}_{mode}{spec}_eval_{dataset}_0shot.pkl"
    with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
        ov_train = pickle.load(f)

    return clus_train, ov_train


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


# name = f"cluster_llama-2-13b_{mode}_epoch4_eval_{dataset}_0shot.pkl"
# with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
#     clus_test_ft = pickle.load(f)

# name = f"overlap_llama-2-13b_{mode}_epoch4_eval_{dataset}_0shot.pkl"
# with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
#     ov_test_ft = pickle.load(f)


# folder = "pretrained"
# mode = "random_order"

# name = f"cluster_llama-2-13b_{mode}_eval_{dataset}_0shot.pkl"
# with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
#     clus_test_pt = pickle.load(f)

# name = f"overlap_llama-2-13b_{mode}_eval_{dataset}_0shot.pkl"
# with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
#     ov_test_pt = pickle.load(f)


fig = plt.figure(figsize=(13, 3.7))

gs1 = GridSpec(1, 2)
ax = fig.add_subplot(gs1[0])
sns.lineplot(
    clus_test_ll3_ft["letters-ari-ep-4-z1.6"],
    marker=".",
    label="fine-tuned",
)

sns.lineplot(
    clus_test_ll3_pt["letters-ari-5shot-z1.6"],
    marker=".",
    label="5shot",
)
ax.legend()
ax.set_ylabel("ARI answers")
ax.set_xlabel("layers")
ax.set_title("llama-2-13b")
ax.set_xticks(np.arange(1, 42, 4))
ax.set_xticklabels(np.arange(1, 42, 4))


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
# plt.savefig(f"{plots_dir}/overlap_labels.png", dpi=200)
