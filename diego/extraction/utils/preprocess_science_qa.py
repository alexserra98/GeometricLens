from datasets import load_dataset, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
import numpy as np
from datasets import load_dataset
from collections import Counter

rng = np.random.default_rng(42)
disable_progress_bar()

dataset = load_dataset("derek-thomas/ScienceQA", split="train")
dataset = dataset.filter(lambda example: example["image"] == None)

selected_categories = []
mask = []
train_indices = []
test_indices = []

for cat in np.unique(dataset["category"]):
    indices = np.nonzero(cat == np.array(dataset["category"]))[0]
    if len(indices) > 10:
        # "we keep only categories with more than 10 examples"
        mask.extend(indices)
        selected_categories.append(cat)
        # we select a training set with at most 40 examples per category
        # and at leas 5 for 5shot learning
        num_train_indices = int(min(40, max(5, 0.2 * len(indices))))
        train_indices.extend(list(indices[:num_train_indices]))
        test_indices.extend(list(indices[num_train_indices:]))

        # dataset.select(list(indices[:num_train_indices]))
        # assert dataset["category"] in selected_categories, (
        #     "train",
        #     dataset["category"],
        # )
        # dataset.select(list(indices[num_train_indices:]))
        # assert dataset["category"] in selected_categories, (
        #     "test",
        #     dataset["category"],
        # )

train = dataset.select(train_indices)
train_for_test = dataset.select(test_indices)

# for subject in np.unique(train_for_test["category"]):
#     if len(train.filter(lambda example: example["category"] == subject)) < 5:
#         print(subject)
#         assert False
# selected_categories


train = dataset.select(train_indices)
validation = load_dataset("derek-thomas/ScienceQA", split="validation")
validation = validation.filter(lambda example: example["image"] == None)
validation = validation.filter(
    lambda example: example["category"] in selected_categories
)


test = load_dataset("derek-thomas/ScienceQA", split="test")
test = test.filter(lambda example: example["image"] == None)
test = test.filter(lambda example: example["category"] in selected_categories)


test_set = concatenate_datasets([train_for_test, validation, test])

for subject in np.unique(train_for_test["category"]):
    if len(train.filter(lambda example: example["category"] == subject)) < 5:
        print(subject)
        assert False


test_set.save_to_disk(
    "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/science_qa/test"
)
train.save_to_disk(
    "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/science_qa/train"
)
