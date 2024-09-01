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


def remove_outliers(datas, w_size=3):
    new_list = []
    for i in range(len(datas) - w_size):
        x = datas[i : i + w_size]
        ave = np.mean(x)
        std = np.std(x)
        for j, val in enumerate(x):
            if val < ave - std:
                del x[j]
        new_list.append(np.mean(x))
    return new_list


# ***********************************************************************************

results_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/results"


# *******************************************


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


ckpts = [1, 2, 3, 6, 12, 22, 42, 77, 144, 268]
ckpts = [12, 42, 77, 144, 268]
alphas = [0.3, 0.4, 0.5, 0.7, 1]


#################################################

# ONLY SIMILARITY

#################################################

plots_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/plots/rebuttal"

sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

fig = plt.figure(figsize=(10, 8))

gs = GridSpec(1, 1)

ax = fig.add_subplot(gs[0])
for cpt, alpha in zip(ckpts, alphas):
    datas = clus_train_ll3[f"letters-ari-step-{cpt}-z2.1"]
    datas = remove_outliers(datas, w_size=4)
    ax.plot(
        np.arange(len(datas)),
        datas,
        label=f"iter {cpt}",
        alpha=alpha,
        color="C3",
        marker=".",
        markersize=10,
        linewidth=2.5,
    )
ax.legend(fontsize=20)
ax.set_xlabel("layer", fontsize=29)
ax.set_ylabel("ARI letters", fontsize=29)
ax.tick_params(labelsize=20)

# ax = fig.add_subplot(gs[1])
# for cpt, alpha in zip(ckpts, alphas):
#     datas = ov_train_ll3[f"letters-step-{cpt}-0.3"]
#     datas = remove_outliers(datas, w_size=4)
#     ax.plot(np.arange(len(datas)), datas, label=cpt)
# ax.legend(title="train step")
# ax.set_ylabel("overlap letters")
# ax.set_xlabel("layer")
gs.tight_layout(fig)
plt.savefig(f"{plots_dir}/dynamics_letter.png", dpi=120)
