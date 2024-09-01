from datasets import load_dataset, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
import numpy as np
from datasets import load_dataset
from collections import Counter, defaultdict
from datasets import Dataset


rng = np.random.default_rng(42)


rng.choice(4, 1, replace=False)[0]

################

#  EXAMPLE STRUCTURE

##################


mmlu = load_dataset("cais/mmlu", "all")

mmlu["test"]
# fields must be "question", "subject", "choices", "answer"


def update_mmlu_pro_dataset(data_):
    final = defaultdict(list)
    for example in data_:
        choices = example["options"]
        index = example["answer_index"]
        ans = choices[index]
        if index > 3:
            index_new = rng.choice(4, 1)[0]
            choices[index_new] = choices[index]
            index = index_new
            assert choices[index] == ans

        final["question"].append(example["question"])
        final["subject"].append(example["category"])
        final["choices"].append(choices[:4])
        final["answer"].append(index)
        final["src"].append(example["src"])
    assert max(final["answer"]) < 4

    return Dataset.from_dict(final)


def update_mmlu_pro_dev(data_):
    final = defaultdict(list)

    for sub in np.unique(data_["category"]):
        subset = data_.filter(lambda example: example["category"] == sub)

        index_new = rng.choice(4, 1)[0]
        answ_indices = list(range(4))
        answ_indices.append(index_new)
        rng.shuffle(answ_indices)

        for index_new, example in zip(answ_indices, subset):
            choices = example["options"]
            index = example["answer_index"]
            ans = choices[index]

            if index > 3:
                choices[index_new] = choices[index]

            elif index != index_new:
                choices[index_new], choices[index] = choices[index], choices[index_new]

            assert choices[index_new] == ans
            for i, c in enumerate(choices):
                if i != index_new and i < 4:
                    assert c != ans, (c, ans, sub, choices)

            final["question"].append(example["question"])
            final["subject"].append(example["category"])
            final["choices"].append(choices[:4])
            final["answer"].append(index_new)
            final["src"].append(example["src"])
        assert max(final["answer"]) < 4

    return Dataset.from_dict(final)


def update_race_dataset(data_):
    final = defaultdict(list)
    letter_to_int = {"A": 0, "B": 1, "C": 2, "D": 3}
    for example in data_:
        choices = example["options"]
        index = letter_to_int[example["answer"]]

        question = example["article"].strip() + "\n" + example["question"].strip()
        final["question"].append(question)
        final["subject"].append("miscellaneous")
        final["choices"].append(choices[:4])
        final["answer"].append(index)
        final["src"].append("race")
    assert max(final["answer"]) < 4
    return Dataset.from_dict(final)


##################

# MMLU PRO PREPROCESSING

##################


mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro")

theoremqa = mmlu_pro.filter(lambda example: "theorem" in example["src"])

scibench = mmlu_pro.filter(lambda example: "scibench" in example["src"])


stemez = mmlu_pro.filter(lambda example: "stemez" in example["src"])

scibench["test"]["src"]
np.unique(stemez["test"]["category"])
scibench
theoremqa
stemez
np.unique(theoremqa["test"]["category"])
#############

# SPLIT TEST IN TAIN AND TEST

#################

indices = []
for i, name in enumerate(mmlu_pro["test"]["src"]):
    if "theorem" in name or "scibench" in name or "stemez" in name:
        indices.append(i)
mmlu_pro_test = mmlu_pro["test"]
mmlu_pro_test = mmlu_pro_test.select(indices)


"splitting the dataset in a part devoted to training and another to test"
train_indices = []
test_indices = []
for sub in np.unique(mmlu_pro_test["category"]):
    indices = np.nonzero(np.array(mmlu_pro_test["category"]) == sub)[0]
    train_indices.extend(list(indices[: int((0.2 * len(indices)))]))
    test_indices.extend(list(indices[int((0.2 * len(indices))) :]))
mmlu_pro_test_ = mmlu_pro_test.select(test_indices)
mmlu_pro_train_ = mmlu_pro_test.select(train_indices)

###########

# dev set

###########

indices = []
for i, name in enumerate(mmlu_pro["validation"]["category"]):
    if name in np.unique(mmlu_pro_test["category"]):
        indices.append(i)
# here we also have chain of thought
mmlu_pro_dev = mmlu_pro["validation"].select(indices)

mmlu_pro_dev_dataset = update_mmlu_pro_dev(mmlu_pro_dev)
mmlu_pro_train_dataset = update_mmlu_pro_dataset(mmlu_pro_train_)
mmlu_pro_test_dataset = update_mmlu_pro_dataset(mmlu_pro_test)

assert np.all(
    np.unique(mmlu_pro_dev_dataset["subject"])
    == np.unique(mmlu_pro_train_dataset["subject"])
)
#######################

# RACE PREPROCESSING

########################


##########

# TEST SET

############

race = load_dataset("ehovy/race", "middle")

race_test_dataset = update_race_dataset(race["test"])


##########

# SPLIT VALIDATION IN TRAIN+DEV

"splitting the dataset in a part devoted to training and another to dev"
race_val = race["validation"]

dev_indices = [11, 100, 500, 800, 1200]
train_indices = [i for i in range(len(race_val)) if i not in dev_indices]

race_dev_dataset = race_val.select(dev_indices)
race_train_dataset = race_val.select(train_indices)


race_dev_dataset = update_race_dataset(race_dev_dataset)
race_train_dataset = update_race_dataset(race_train_dataset)


####################


test_dataset = concatenate_datasets([mmlu_pro_test_dataset, race_test_dataset])
dev_dataset = concatenate_datasets([mmlu_pro_dev_dataset, race_dev_dataset])
train_dataset = concatenate_datasets([mmlu_pro_train_dataset, race_train_dataset])

path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego"
test_dataset.save_to_disk(f"{path}/mmlu_pro_race/test")
train_dataset.save_to_disk(f"{path}/mmlu_pro_race/train")
dev_dataset.save_to_disk(f"{path}/mmlu_pro_race/dev")


###########

# SST

##########


sst = load_dataset("sst")
sst["train"]["sentence"]

very_positive = np.nonzero(np.array(sst["train"]["label"]) > 0.9)[0]

positive = np.nonzero(
    np.logical_and(
        np.array(sst["train"]["label"]) < 0.7, np.array(sst["train"]["label"]) > 0.68
    )
)[0]


negative = np.nonzero(
    np.logical_and(
        np.array(sst["train"]["label"]) < 0.4, np.array(sst["train"]["label"]) > 0.37
    )
)[0]

very_negative = np.nonzero(np.array(sst["train"]["label"]) < 0.1)[0]


################################################


from datasets import load_from_disk
import numpy as np


rng = np.random.default_rng(42)
path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/mmlu_pro_race"

train = load_from_disk(f"{path}/new/train")


samples_per_subject = 20


subjects = np.array(train["subject"])
mask = []
for sub in np.unique(subjects):
    ind = np.nonzero(sub == subjects)[0]
    nsamples = min(samples_per_subject, len(ind))
    chosen = rng.choice(ind, nsamples, replace=False)
    mask.extend(list(np.sort(chosen)))

mask = np.array(mask)

mask.shape
np.save(f"{path}/mask_20.npy", mask)


# ********************************************************************
import numpy as np
from collections import defaultdict
from datasets import Dataset

path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/mmlu_pro_race"

old_dev = load_from_disk(f"{path}/old/dev")


def shuffle_field(dev_set, data_, new_indices):
    for field in data_.features:
        field_examples = data_[field]
        for index in new_indices:
            dev_set[field].append(field_examples[index])
    return dev_set


rng = np.random.default_rng(42)

for iter in range(5):
    dev_set = defaultdict(list)
    new_indices = rng.choice(5, 5, replace=False)
    for sub in np.unique(old_dev["subject"]):
        examples = old_dev.filter(lambda example: example["subject"] == sub)
        dev_set = shuffle_field(dev_set, data_=examples, new_indices=new_indices)
    dev_dataset = Dataset.from_dict(dev_set)
    dev_dataset.save_to_disk(f"{path}/mmlu_pro_race/dev_{iter}")


####################

# construct the dataset with the best of new and original

########################
from datasets import load_from_disk

path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/mmlu_pro_race"


new_dev = load_from_disk(f"{path}/new/train")
new_dev = load_from_disk(f"{path}/new/test")
new_dev
np.unique(new_dev["subject"])
len(np.load(f"{path}/mask_50.npy"))

new_dev = load_from_disk(f"{path}/new/dev")
old_dev = load_from_disk(f"{path}/old/dev")
dev_1 = load_from_disk(f"{path}/mmlu_pro_race/dev_1")
dev_2 = load_from_disk(f"{path}/mmlu_pro_race/dev_2")


best = defaultdict(list)
for sub in np.unique(old_dev["subject"]):
    old_sub = old_dev.filter(lambda example: example["subject"] == sub)
    new_sub = new_dev.filter(lambda example: example["subject"] == sub)
    dev_1_sub = dev_1.filter(lambda example: example["subject"] == sub)
    dev_2_sub = dev_2.filter(lambda example: example["subject"] == sub)

    current_data = old_sub
    if sub in ["biology"]:
        current_data = dev_1_sub
    elif sub in ["business"]:
        current_data = dev_2_sub
    elif sub in ["engineering"]:
        current_data = new_sub

    for example in current_data:
        best["question"].append(example["question"])
        best["subject"].append(example["subject"])
        best["choices"].append(example["choices"])
        best["answer"].append(example["answer"])


mmlu_dev_best = Dataset.from_dict(best)
mmlu_dev_best.save_to_disk(f"{path}/dev_best_0")
mmlu_dev_best
