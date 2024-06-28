import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import os

num_samples = 30
num_few_shots = 5
dirpath = f"/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/results/evaluated_test/questions_sampled_same/llama-3-8b/{num_few_shots}shot"


acc_subjects = defaultdict(list)
indices_subjects = defaultdict(list)
acc_macro = []
acc_micro = []
for seed in range(num_samples):
    with open(f"{dirpath}/statistics_target_seed{seed}.pkl", "rb") as f:
        file = pickle.load(f)
    for sub, acc in file["accuracy"]["subjects"].items():
        acc_subjects[sub].append(acc)
    for sub, indices in file["few_shot_indices"].items():
        indices_subjects[sub].append(indices)
    acc_macro.append(file["accuracy"]["macro"])
    acc_micro.append(file["accuracy"]["micro"])


sorted_acc = np.zeros((len(acc_subjects), num_samples))
sorted_indices = np.zeros((len(acc_subjects), num_samples, 5), dtype=int)
dict_list = [{} for _ in range(num_samples)]


for i, sub in enumerate(acc_subjects.keys()):

    indices = np.argsort(acc_subjects[sub])
    sorted_acc[i] = np.array(acc_subjects[sub])[indices]
    sorted_indices = np.array(indices_subjects[sub])[indices]
    for j in range(len(sorted_indices)):
        dict_list[j][sub] = sorted_indices[j]

accuracies = np.mean(sorted_acc, axis=0)

for i, acc in enumerate(accuracies):
    dict_list[i]["acc_macro"] = acc


path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/sorted_indices_subjects"
for i, data in enumerate(dict_list):
    os.makedirs(f"{path}/{num_few_shots}shot", exist_ok=True)
    with open(f"{path}/{num_few_shots}shot/sorted_indices_{i}.pkl", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
