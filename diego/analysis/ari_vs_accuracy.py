import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt


num_samples = 30

dirpath = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/results/evaluated_test/questions_sampled_same/llama-3-8b"
filename = "statistics_target_seed0.pkl"


acc_subjects = defaultdict(list)

accuracy = defaultdict(list)
metrics = defaultdict(list)
for num_shots in [1, 2]:
    acc_macro = []
    acc_micro = []
    metrics_ = []
    for seed in range(num_samples):
        with open(
            f"{dirpath}/{num_shots}shot/statistics_target_sorted_sample{seed}.pkl",
            "rb",
        ) as f:
            file = pickle.load(f)
        acc_macro.append(file["accuracy"]["macro"])
        acc_micro.append(file["accuracy"]["micro"])
        metrics_.append(file["metrics"])

    accuracy[f"{num_shots}shot"] = acc_macro
    metrics[f"{num_shots}shot"] = metrics_


metrics[f"{num_shots}shot"]["ari"]


def moving_avg(arr, step=1):
    return [np.mean(arr[i : i + step]) for i in range(len(arr) - step)]


file["metrics"]
fig = plt.figure()
ax = fig.add_subplot()
for num_shot in [1, 2]:
    clus_metric = [
        max(moving_avg(metrics[f"{num_shots}shot"][i]["completeness"], step=1))
        for i in range(len(metrics[f"{num_shots}shot"]))
    ]
    ax.scatter(accuracy[f"{num_shot}shot"], clus_metric)

plt.scatter([max(metric["ari"]) for metric in aris], acc_macro)


with open(f"{dirpath}/statistics_target_seed0.pkl", "rb") as f:
    file = pickle.load(f)

# *****************************************************************************************


acc_subjects = defaultdict(list)
indices_subjects = defaultdict(list)
acc_macro = []
acc_micro = []
for seed in range(10):
    with open(f"{dirpath}/statistics_target_seed{seed}.pkl", "rb") as f:
        file = pickle.load(f)
    for sub, acc in file["accuracy"]["subjects"].items():
        acc_subjects[sub].append(acc)
    for sub, indices in file["few_shot_indices"].items():
        indices_subjects[sub].append(indices)
    acc_macro.append(file["accuracy"]["macro"])
    acc_micro.append(file["accuracy"]["micro"])


sorted_acc = np.zeros((len(acc_subjects), 10))
sorted_indices = np.zeros((len(acc_subjects), 10, 5), dtype=int)

dict_list = [{} for _ in range(10)]

for i, sub in enumerate(acc_subjects.keys()):

    indices = np.argsort(acc_subjects[sub])
    sorted_acc[i] = np.array(acc_subjects[sub])[indices]
    sorted_indices = np.array(indices_subjects[sub])[indices]
    for j in range(len(sorted_indices)):
        dict_list[j][sub] = sorted_indices[j]

accuracies = np.mean(sorted_acc, axis=0)

for i, acc in enumerate(accuracies):
    dict_list[i]["acc_macro"] = acc


path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego"
for i, data in enumerate(dict_list):
    with open(f"{path}/sorted_indices_{i}.pkl", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
