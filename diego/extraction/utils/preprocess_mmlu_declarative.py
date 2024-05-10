from datasets import load_dataset
import torch
from functools import partial
from datasets.utils.logging import disable_progress_bar
import numpy as np

# from dataset_utils.utils import MMLU_Dataset
import sys
import random
import json


disable_progress_bar()

dataset = load_dataset("cais/mmlu", "all", split="dev")

path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/extraction/utils/mmul_declarative_processed1.txt"

with open(f"{path}", "r") as f:
    text = f.read()

lines = text.split("\n\n")


mmlu_declarative = {subject: [] for subject in np.unique(dataset["subject"])}
for i, subject in enumerate(dataset["subject"]):

    line = lines[i]
    assert line.startswith(f"{i+1}."), i + 1
    num_to_remove = len(f"{i+1}.")
    sentence = line[num_to_remove:].strip()
    mmlu_declarative[subject].append(sentence)


for key, val in mmlu_declarative.items():
    assert len(val) == 5


import json

with open("mmlu_declarative.json", "w", encoding="utf-8") as f:
    json.dump(mmlu_declarative, f, ensure_ascii=False, indent=4)
