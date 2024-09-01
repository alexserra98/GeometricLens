from datasets import load_dataset, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
import numpy as np
from datasets import load_dataset
from collections import Counter

rng = np.random.default_rng(42)
disable_progress_bar()

dataset = load_dataset("derek-thomas/ScienceQA", split="train")


subject_field = "topic"

dataset = dataset.filter(lambda example: example["image"] == None)

selected_categories = []
mask = []
train_indices = []
test_indices = []

for cat in np.unique(dataset[subject_field]):
    indices = np.nonzero(cat == np.array(dataset[subject_field]))[0]
    if len(indices) > 10:
        # "we keep only categories with more than 10 examples"
        mask.extend(indices)
        selected_categories.append(cat)
        # we select a training set with at most 40 examples per category
        # and at leas 5 for 5shot learning
        if subject_field == "category":
            num_train_indices = int(min(40, max(5, 0.2 * len(indices))))
        elif subject_field == "topic":
            num_train_indices = int(min(60, max(5, 0.2 * len(indices))))
        train_indices.extend(list(indices[:num_train_indices]))
        test_indices.extend(list(indices[num_train_indices:]))


train = dataset.select(train_indices)
train_for_test = dataset.select(test_indices)


train = dataset.select(train_indices)
validation = load_dataset("derek-thomas/ScienceQA", split="validation")
validation = validation.filter(lambda example: example["image"] == None)
validation = validation.filter(
    lambda example: example[subject_field] in selected_categories
)


test = load_dataset("derek-thomas/ScienceQA", split="test")
test = test.filter(lambda example: example["image"] == None)
test = test.filter(lambda example: example[subject_field] in selected_categories)


test_set = concatenate_datasets([train_for_test, validation, test])


test_set["choices"]
test_set["answer"]

# *************************************
# double checks


Counter(train["topic"])

data = train.filter(lambda example: example["topic"] == "")
data["question"]

for subject in np.unique(train_for_test[subject_field]):
    if len(train.filter(lambda example: example[subject_field] == subject)) < 5:
        print(subject)
        assert False

# ********************


test_set.save_to_disk(
    f"/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/science_qa/{subject_field}_partition/test"
)
train.save_to_disk(
    f"/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/science_qa/{subject_field}_partition/train"
)
