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
    "llama-3-8b",
    "llama-3-8b-chat",
    "llama-3-70b",
    "llama-3-70b-chat",
]


with open(f"{base_dir}/{models[0]}/0shot/statistics_target.pkl", "rb") as f:
    stats = pickle.load(f)

num = []
for subject in np.unique(stats["subjects"]):
    mask = subject == np.array(stats["subjects"])
    print(subject, np.sum(mask))
    num.append(np.sum(mask))

np.sum(num)
mask

accuracy = {}
constrained = {}
colors = []
for i, model in enumerate(models):
    tmp = {}
    constr_tmp = {}
    for shot in range(6):
        try:
            with open(
                f"{base_dir}/{model}/{shot}shot/statistics_target.pkl", "rb"
            ) as f:
                stats = pickle.load(f)
                tmp[f"{shot}"] = stats["accuracy"]
                constr_tmp[f"{shot}"] = stats["constrained_accuracy"]
        except:
            print(f"{model} {shot} not found")

    accuracy[f"{model}"] = tmp
    constrained[f"{model}"] = constr_tmp


accuracy
# len(accuracy)
# x = np.arange(len(accuracy))
# y = list(accuracy.values())
# y_constrained = list(constrained.values())
# xticklabels = list(accuracy.keys())
# xticks = x

x = []
y = []
xticklabels = []
xticks = []
count = 0
for model, shots in accuracy.items():
    y.append(list(shots.values()))
    xticklabels.extend([f"{model[6:]}-{shot}" for shot in shots.keys()])
    x_tmp = count + np.arange(len(shots.keys()))
    x.append(x_tmp)
    xticks.extend(list(x_tmp))
    count += len(shots.keys())

colors = [f"C{i}" for i in range(10)]
sns.set_style(style="whitegrid")
fig = plt.figure(figsize=(9.5, 4))
ax = fig.add_subplot()
ax.axvspan(-2, 26, color="C0", alpha=0.08, label="Llama 2")
ax.axvspan(26, 46, color="C1", alpha=0.08, label="Llama 3")
for i, model in enumerate(models):
    ax.scatter(x[i], y[i], marker="o", linestyle="None", color=colors[i])
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=60, fontsize=9)

ax.set_ylim(0.22, 0.83)
ax.set_xlim(-1, 42)
ax.axhline(0.25, label="random baseline", color="black", linewidth="0.8")
ax.legend(fontsize=11)

ax.set_ylabel("accuracy", fontsize=13)
ax.set_title("MMLU few shot accuracies")
plt.tight_layout()
plt.savefig("./plots/mmlu_few_shot.png", dpi=150)
