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
eval_dataset = "test_balanced"

with open(
    f"{base_dir}/cluster_subjetcs_llama-3-8b_base_question_sampled_5_epoch4.pkl",
    "rb",
) as f:

    clu = pickle.load(f)


clu.keys()


# overlaps 0 vs 5 shot
sns.set_style("whitegrid")
fig = plt.figure(figsize=(12, 3))
gs = GridSpec(1, 4)
for i, z in enumerate([0.5, 1, 1.6, 2.1]):
    ax = fig.add_subplot(gs[i])
    sns.lineplot(
        ax=ax,
        x=np.arange(len(clu[f"ari-ep4-z{z}"])),
        y=clu[f"ari-ep4-z{z}"],
        label="5shot",
    )
    ax.set_title(f"{model}-z={z}")
    ax.set_ylim(0.0, 1.1)
gs.tight_layout(fig)
plt.savefig(
    f"{results_dir}/{model}_cluster_ari_few_shot_finetuned_zs_halo.png", dpi=150
)
