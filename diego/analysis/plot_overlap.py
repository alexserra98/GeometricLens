import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from collections import defaultdict
from collections import Counter

from datasets import load_dataset



dataset = load_dataset("cais/mmlu", 'all', split="validation")

datasetCounter(dataset["subject"])
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(dataset.keys(), dataset.values())
ax.plot()












base_dir = "./results"


overlaps = defaultdict(list)
for shot in ["0shot", "5shot"]:
    with open(
        f"{base_dir}/overlaps_llama-2-7b_finetuned_dev_eval_validation_epoch4_4_{shot}.pkl",
        "rb",
    ) as f:

        stats = pickle.load(f)
        overlaps[shot + "-k10"] = [np.mean(ovs) for ovs in stats["10"]]
        overlaps[shot + "-k100"] = [np.mean(ovs) for ovs in stats["100"]]


fig = plt.figure()
ax = fig.add_subplot()
sns.lineplot(ax=ax, x=np.arange(len(overlaps["0shot-k10"])), y=overlaps["0shot-k100"])
sns.lineplot(ax=ax, x=np.arange(len(overlaps["5shot-k10"])), y=overlaps["5shot-k100"])
