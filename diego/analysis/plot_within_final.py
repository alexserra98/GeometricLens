import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from datasets import load_dataset


# ***********************************************************************************
results_dir = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/results/finetuned/llama-3-8b"
name = "cluster_subject_llama-3-8b_finetuned_dev_val_balanced_20samples_epoch4_eval_test_0shot.pkl"


with open(f"{results_dir}/{name}", "rb") as f:
    stats = pickle.load(f)


plt.plot(stats["letters-ari-ep4_z1.6"])
plt.plot(stats["letters-ep1_0.03"])
plt.plot(stats["letters-ep2_0.3"])
plt.plot(stats["letters-ep4_0.1"])

stats.keys()


stats["letters-ari-ep4_z1.6"]
