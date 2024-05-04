import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from collections import defaultdict
from collections import Counter

from datasets import load_dataset


base_dir = "./results"
model = "llama-3-8b"
finetuned_mode = "dev_val_balanced"
eval_dataset = "test_balanced"
epochs = "4"
ckpt = "0"


overlaps = defaultdict(list)
for shot in ["0shot", "5shot"]:
    with open(
        f"{base_dir}/overlaps_{model}_finetuned_{finetuned_mode}_eval_{eval_dataset}_epoch{epochs}_{ckpt}_{shot}.pkl",
        "rb",
    ) as f:

        stats = pickle.load(f)
        # overlaps[shot + "-k30"] = [np.mean(ovs) for ovs in stats["k_30"]]
        overlaps[shot + "-k100"] = [np.mean(ovs) for ovs in stats["k_100"]]


overlaps["5shot-k100"]


fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(
    ax=ax,
    x=np.arange(len(overlaps["0shot-k100"])),
    y=overlaps["0shot-k100"],
    label="0shot",
)
sns.lineplot(
    ax=ax,
    x=np.arange(len(overlaps["5shot-k100"])),
    y=overlaps["5shot-k100"],
    label="5shot",
)
