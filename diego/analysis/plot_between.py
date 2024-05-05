import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from datasets import load_dataset


# ***********************************************************************************
results_dir = "./plots/comparison_finetuned"
base_dir = "./results"
model = "llama-3-8b"
finetuned_mode = "test_balanced"
eval_dataset = "test_balanced"
epochs = "4"

shot = "0"
with open(
    f"{base_dir}/overlaps_{model}_finetuned_{finetuned_mode}_eval_{eval_dataset}_epoch{epochs}_{shot}.pkl",
    "rb",
) as f:

    ov_0shot = pickle.load(f)

shot = "5"
with open(
    f"{base_dir}/overlaps_{model}_finetuned_{finetuned_mode}_eval_{eval_dataset}_epoch{epochs}_{shot}.pkl",
    "rb",
) as f:

    ov_5shot = pickle.load(f)


# overlaps 0 vs 5 shot
sns.set_style("whitegrid")
fig = plt.figure(figsize=(10, 3))
gs = GridSpec(1, 3)
for i, k in enumerate([30, 100, 300]):
    ax = fig.add_subplot(gs[i])
    sns.lineplot(
        ax=ax,
        x=np.arange(len(ov_0shot[f"ep4_k{k}"])),
        y=ov_0shot[f"ep4_k{k}"],
        label="0shot",
    )
    sns.lineplot(
        ax=ax,
        x=np.arange(len(ov_5shot[f"ep4_k{k}"])),
        y=ov_5shot[f"ep4_k{k}"],
        label="5shot",
    )
    ax.set_title(f"{model}-k{k}")
    ax.set_ylim(0.1, 1)
gs.tight_layout(fig)
plt.savefig(f"{results_dir}/{model}_ov_few_shot_finetuned_ks.png", dpi=150)


sns.set_style("whitegrid")
fig = plt.figure(figsize=(4, 4))
gs = GridSpec(1, 1)

ax = fig.add_subplot(gs[0])
for i in range(5):
    sns.lineplot(
        ax=ax,
        x=np.arange(len(ov_0shot[f"ep4_k300"])),
        y=ov_0shot[f"ep{i}_k300"],
        label=f"ep {i}",
        alpha=(i + 1) / 5,
        color="C0",
        marker=".",
    )
    sns.lineplot(
        ax=ax,
        x=np.arange(len(ov_5shot[f"ep4_k300"])),
        y=ov_5shot[f"ep{i}_k300"],
        label=f"ep {i}",
        alpha=(i + 1) / 5,
        color="C1",
        marker=".",
    )
ax.set_title(f"{model}-k300")
ax.set_ylim(0.1, 1.02)
gs.tight_layout(fig)
plt.savefig(f"{results_dir}/{model}_ov_few_shot_finetuned_epochs.png", dpi=150)


# *****************************************************************************
# subjects
dataset = load_dataset("cais/mmlu", "all", split="dev")
sns.set_style("whitegrid")
fig = plt.figure(figsize=(12, 24))
gs = GridSpec(6, 3)

for i in range(18):
    ax = fig.add_subplot(gs[i])
    for j, subject in enumerate(np.unique(dataset["subject"])[i * 3 : (i + 1) * 3]):
        sns.lineplot(
            ax=ax,
            x=np.arange(len(ov_0shot[f"ep4_k300"])),
            y=ov_0shot[f"ep4_{subject}_k30"],
            label=subject,
            color=f"C{j}",
            marker="o",
        )
        sns.lineplot(
            ax=ax,
            x=np.arange(len(ov_0shot[f"ep4_k300"])),
            y=ov_5shot[f"ep4_{subject}_k30"],
            color=f"C{j}",
            marker="^",
        )
    ax.legend(fontsize=9)
    ax.set_title(f"{model}-k30")
    ax.set_ylim(0.0, 1.02)
gs.tight_layout(fig)
plt.savefig(f"{results_dir}/{model}_ov_few_shot_finetuned_subjetcs.png", dpi=150)


# ************************************************************************
# clusters
base_dir = "./results"
model = "llama-3-8b"
finetuned_mode = "test_balanced"
eval_dataset = "test_balanced"
epochs = "4"

shot = "0"
with open(
    f"{base_dir}/cluster_comparison_{model}_finetuned_{finetuned_mode}_eval_{eval_dataset}_epoch{epochs}_{shot}.pkl",
    "rb",
) as f:

    clu_0shot = pickle.load(f)


shot = "5"
with open(
    f"{base_dir}/cluster_comparison_{model}_finetuned_{finetuned_mode}_eval_{eval_dataset}_epoch{epochs}_{shot}.pkl",
    "rb",
) as f:

    clu_5shot = pickle.load(f)


# overlaps 0 vs 5 shot
sns.set_style("whitegrid")
fig = plt.figure(figsize=(12, 3))
gs = GridSpec(1, 4)
for i, z in enumerate([0.5, 1, 1.6, 2.1]):
    ax = fig.add_subplot(gs[i])
    sns.lineplot(
        ax=ax,
        x=np.arange(len(clu_0shot[f"ari-ep4-z{z}"])),
        y=clu_0shot[f"ari-ep4-z{z}-halo"],
        label="0shot",
    )
    sns.lineplot(
        ax=ax,
        x=np.arange(len(clu_5shot[f"ari-ep4-z{z}"])),
        y=clu_5shot[f"ari-ep4-z{z}-halo"],
        label="5shot",
    )
    ax.set_title(f"{model}-z={z}")
    ax.set_ylim(0.0, 1.1)
gs.tight_layout(fig)
plt.savefig(
    f"{results_dir}/{model}_cluster_ari_few_shot_finetuned_zs_halo.png", dpi=150
)
