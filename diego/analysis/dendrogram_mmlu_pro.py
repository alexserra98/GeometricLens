import numpy as np
import pickle
import torch
import dadapy
from collections import Counter
from datasets import load_from_disk

from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    adjusted_mutual_info_score,
)

from helper_dendogram import (
    get_subject_array,
    get_dataset_mask,
    get_dissimilarity,
    get_clusters,
    plot_with_labels,
)


dataset_path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/mmlu_pro_race"
dataset = load_from_disk(f"{dataset_path}/best/test")

results_path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/plots/rebuttal"
model_path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/results/mmlu_pro_race/5shot"

with open(f"{model_path}/statistics_target.pkl", "rb") as f:
    stats = pickle.load(f)


stats.keys()
len(stats["subjects"])

sub_to_int = {sub: i for i, sub in enumerate(np.unique(stats["subjects"]))}
int_to_sub = {i: sub for sub, i in sub_to_int.items()}
int_labels = np.array([sub_to_int[sub] for sub in stats["subjects"]])


ground_truth_labels = int_labels

subjects = stats["subjects"]
for i in range(len(subjects)):
    if subjects[i] == "miscellaneous":
        subjects[i] = "RACE"

subjects = np.array(subjects)
mask = np.arange(len(subjects))

d, index_to_subject = get_clusters(
    model_path=model_path,
    layer=5,
    mask=mask,
    gtl=ground_truth_labels,
    subjects=subjects,
    Z=1.6,
)


dis, peak_y, final_clusters = get_dissimilarity(d, threshold=-10)

plot_with_labels(
    dis,
    final_clusters,
    ground_truth_labels,
    index_to_subject,
    ori="left",
    path=f"{results_path}/llama-3-8b-5shot-Z=1.6-mmlu-pro",
    width=3,
    color_threshold=15,
    height=3.5,
)
