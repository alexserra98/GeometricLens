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


# *******************************************************************


# other junk

# def format_subject(subject):
#     l = subject.split("_")
#     s = ""
#     for entry in l:
#         s += " " + entry
#     return s


# def construct_question(question, choices, answer, include_answer=False):
#     answers = np.array(["A", "B", "C", "D"])
#     # added strip
#     prompt = f"{question.strip()}\n"
#     for i, choice in enumerate(choices):
#         # added strip
#         prompt += f"{answers[i]}. {choice.strip()}\n"
#     # added space to final answers
#     prompt += "Answer:"
#     if include_answer:
#         prompt += f" {answers[answer]}\n\n"
#     return prompt


# dev_set = load_dataset("cais/mmlu", "all", split="dev")
# subjects = np.unique(dev_set["subject"])[3:5]


# local_dev_set = {}
# for subject in subjects:
#     local_dev_set[subject] = dev_set.filter(
#         lambda dev_example: dev_example["subject"] == subject,
#     )

# for i, question in enumerate(subjects):
#     # prompt = f"The following are multiple choice questions (with answers) about{format_subject(subjects[i])}.\n\n"
#     prompt = ""
#     current_subject = subjects[i]
#     for j in range(5):
#         shot = local_dev_set[current_subject][j]
#         prompt += construct_question(
#             shot["question"],
#             shot["choices"],
#             shot["answer"],
#             include_answer=True,
#         )

# with open("./subjects", "w") as f:
#     for subject in np.unique(dataset["subject"]):
#         f.write(f"'{subject}',\n")


# area_to_subjects = {"stem": [], "not_stem": []}

# area_to_subjects["stem"] = [
#     "abstract_algebra",
#     "anatomy",
#     "astronomy",
#     "college_biology",
#     "college_chemistry",
#     "college_computer_science",
#     "college_mathematics",
#     "college_physics",
#     "computer_security",
#     "conceptual_physics",
#     "electrical_engineering",
#     "elementary_mathematics",
#     "high_school_biology",
#     "high_school_chemistry",
#     "high_school_computer_science",
#     "high_school_mathematics",
#     "high_school_physics",
#     "high_school_statistics",
#     "machine_learning",
# ]


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


with open("mmlu_declarative.txt", "w") as f:
    for i, example in enumerate(dataset):
        f.write(f"{i+1}. {example['question']}\n")
        f.write(f"{example['choices'][example['answer']]}\n\n")

with open("mmlu_declarative_qa.txt", "w") as f:
    for i, example in enumerate(dataset):
        f.write(f"{i+1}. {example['question']}\n")
        for j in range(len(example["choices"])):
            f.write(f"{j}. {example['choices'][j]}\n")
        f.write(f"{example['answer']}\n\n")


# for subject in np.unique(dataset["subject"]):
#     if subject not in area_to_subjects["stem"]:
#         area_to_subjects["not_stem"].append(subject)

# with open("./asset/mmlu_macro_areas.json", "w") as f:
#     json.dump(area_to_subjects, f)


few_shot_dataset = load_dataset("cais/mmlu", "all", split="dev+validation")

from collections import Counter

Counter(few_shot_dataset["subject"])


# path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/generation/utils/few_shots.txt"

# with open(f"{path}", "r", encoding="utf-8") as f:
#     lines = f.read()

# lines = [line.replace("\\n", "\n")[:-1] for line in lines]


# print(lines.split("\n\n")[10])

# prompt = ""
# for line in lines:
#     prompt += line
# prompt += "\n"

# base_path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/extraction/utils"
# path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/extraction/utils/mmlu_affirmative_ale.txt"

# dataset = load_dataset("cais/mmlu", "all", split="dev")

# with open(f"{base_path}/question_answers.txt", "w", encoding="utf-8") as f:
#     f.write()
