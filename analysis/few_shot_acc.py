import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns


base_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/analysis/results/mmlu"


models = [
    "llama-2-7b",
    "llama-2-7b-chat",
    "llama-2-13b",
    "llama-2-13b-chat",
    "llama-2-70b",
    "llama-2-70b-chat",
]


accuracy = {}
constrained = {}
for model in models:
    for shot in range(6):
        try:
            with open(
                f"{base_dir}/{model}/{shot}shot/statistics_target.pkl", "rb"
            ) as f:
                stats = pickle.load(f)
                accuracy[f"{model[8:]}-{shot}"] = stats["accuracy"]
                constrained[f"{model[8:]}-{shot}"] = stats["constrained_accuracy"]
        except:
            print(f"{model} {shot} not found")


len(accuracy)
x = np.arange(len(accuracy))
y = list(accuracy.values())
y_constrained = list(constrained.values())
xticklabels = list(accuracy.keys())
xticks = x

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot()
sns.lineplot(ax=ax, x=x, y=y, marker="o", linestyle="None", label="exact match")
sns.lineplot(
    ax=ax, x=x, y=y_constrained, marker="o", linestyle="None", label="constrained"
)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=60)
ax.set_ylabel("accuracy", fontsize=13)
ax.set_title("MMLU few shot accuracies")
plt.tight_layout()
plt.savefig("./plots/mmlu_few_shot.png", dpi=150)
