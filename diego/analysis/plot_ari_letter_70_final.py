import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from datasets import load_dataset


def get_repr_for_test(results_dir, folder, mode, model, dataset, spec="", shot=0):

    name = f"cluster_{model}_{mode}{spec}_eval_{dataset}_{shot}shot.pkl"
    with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
        clus_train = pickle.load(f)

    name = f"overlap_{model}_{mode}{spec}_eval_{dataset}_{shot}shot.pkl"
    with open(f"{results_dir}/{folder}/{model}/{name}", "rb") as f:
        ov_train = pickle.load(f)

    return clus_train, ov_train


# ***********************************************************************************
plots_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/plots/final"
results_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/results"


dataset = "test_balanced200"
model = "llama-3-70b"

clus_test_ll3_ft, ov_test_ll3_ft = get_repr_for_test(
    results_dir=results_dir,
    folder="finetuned",
    mode="finetuned_dev_val_balanced_40samples",
    model=f"{model}",
    dataset="test_balanced200",
    spec="_epoch4",
)

clus_test_pt = {}
for shot in [0, 1, 2, 5]:
    if shot == 5 and model == "llama-3-70b":
        shot = 4
    clus_test_ll3_pt, ov_test_ll3_pt = get_repr_for_test(
        results_dir=results_dir,
        folder="pretrained",
        mode="random_order",
        model=f"{model}",
        dataset="test_balanced200",
        shot=shot,
    )
    for key, val in clus_test_ll3_pt.items():
        clus_test_pt[key] = val


clus_test_ll3_pt.keys()


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


def run_avg(x):
    res = []
    for i in range(0, len(x) - 1):
        res.append(np.mean([x[i], x[i + 1]]))
    return res


plt.rcParams.update(plot_config)

fig = plt.figure()
# Set the style
sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
for shot in [0, 1, 2, 5]:
    label_shot = shot
    if shot == 5 and model == "llama-3-70b":
        shot = 4
    # sns.lineplot(
    #     x=np.arange(1, 81),
    #     y=clus_test_pt[f"letters-ari-{shot}shot-z1.6"][:80],
    #     marker=".",
    #     label=f"{label_shot} shot pt",
    # )
    ax.plot(
        np.arange(1, 80),
        run_avg(clus_test_pt[f"letters-ari-{shot}shot-z1.6"])[:79],
        marker=".",
        label=f"{label_shot} shot pt",
    )
ax.plot(
    np.arange(1, 80),
    run_avg(clus_test_ll3_ft["letters-ari-ep-4-z1.6"])[:79],
    marker=".",
    label=f"0 shot ft",
)

ax.legend()
ax.set_ylabel("ARI")
ax.set_xlabel("layers")
ax.set_title("llama-3-70b")
ax.set_xticks(np.arange(1, 81, 4))
ax.set_yticks(np.arange(0, 71, 7) / 100)
ax.set_xticklabels(np.arange(1, 81, 4))
ax.set_ylim(-0.03, 0.7)
gs.tight_layout(fig)
plt.savefig(f"./plots/final/{model}-ari-letters.pdf")


len(clus_test_pt[f"letters-ari-0shot-z1.6"])
