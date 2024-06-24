import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib.gridspec import GridSpec

from datasets import load_dataset, concatenate_datasets

rng = np.random.default_rng(42)


dataset = load_dataset("cais/mmlu", "all", split="test")

arr = list(Counter(dataset["subject"]).values())
np.sort(arr)[23]


dataset = load_dataset("cais/mmlu", "all", split="dev+validation")


subjects = np.array(dataset["subject"])
mask = []
for sub in np.unique(subjects):
    ind = np.nonzero(sub == subjects)[0]
    nsamples = min(20, len(ind))
    chosen = rng.choice(ind, nsamples, replace=False)
    mask.extend(list(np.sort(chosen)))

mask = np.array(mask)
np.save("dev+validation_mask_20.npy", mask)
len(np.load("test_mask_200.npy"))


# -------------------------------------------------


dev_set = load_dataset("cais/mmlu", "all", split="dev")

val_set = load_dataset("cais/mmlu", "all", split="validation")
subjects = np.array(val_set["subject"])
mask = []
for sub in np.unique(subjects):
    ind = np.nonzero(sub == subjects)[0]
    nsamples = min(8, len(ind))
    chosen = rng.choice(ind, nsamples, replace=False)
    mask.extend(list(np.sort(chosen)))
mask = np.array(mask)
val_set_balanced = val_set.select(mask)
final = concatenate_datasets([dev_set, val_set_balanced])

# just double check that all is fine
counts = Counter(final["subject"])
assert len(np.unique(list(counts.values()))) == 1
assert np.unique(list(counts.values()))[0] == 13


with open("mmlu_decalrative.txt", "w") as f:
    for i, example in enumerate(final):
        f.write(
            f"Question {i}: {example['question']}\nAnswer: {example['choices'][example['answer']]}\n\n"
        )


# ***************************************************
dataset = load_dataset("cais/mmlu", "all", split="test")
subjects = np.array(dataset["subject"])


mask = []
for i, sub in enumerate(np.unique(subjects)):
    ind = np.nonzero(sub == subjects)[0]
    nsamples = min(200, len(ind))
    chosen = rng.choice(ind, nsamples, replace=False)
    if sub == "world_religions":
        chosen = np.where(subjects == "world_religions")[0][: (min(nsamples, 169))]
    mask.extend(list(np.sort(chosen)))


np.save("test_mask_200", np.array(mask))


mask = []
for sub in np.unique(subjects):
    ind = np.nonzero(sub == subjects)[0]
    chosen = rng.choice(ind, 100, replace=False)
    mask.extend(list(np.sort(chosen)))

mask = np.array(mask)

final = dataset.select(mask)
frequences = Counter(final["subject"]).values()

assert len(np.unique(list(frequences))) == 1
assert np.unique(list(frequences))[0] == 100
assert mask.shape[0] == 5700
np.save("test_mask_100.npy", np.array(mask))


# for i, subject in enumerate(subjects):
#     tmp_data = dataset.filter(lambda example: example["subject"] == subject)
#     num_samples = len(tmp_data)
#     print(num_samples)
#     to_select = np.arange(num_samples)
#     if num_samples > 15:
#         to_select = rng.choice(num_samples, size=15, replace=False)
#         to_select = np.sort(to_select)
#     final_tmp = tmp_data.select(list(to_select))
#     print(len(final_tmp))
#     assert len(final_tmp) <= 15
#     if i == 0:
#         final = final_tmp
#     else:
#         final = concatenate_datasets([final, final_tmp])

dataset_test = Counter(dataset["subject"])
dataset_test = dict(sorted(dataset_test.items(), key=lambda item: item[1]))
# **************************************************


dataset = load_dataset("cais/mmlu", "all", split="validation")
dataset_val = Counter(dataset["subject"])

subjects = np.unique(dataset["subject"])


for i, subject in enumerate(subjects):
    tmp_data = dataset.filter(lambda example: example["subject"] == subject)
    num_samples = len(tmp_data)
    print(num_samples)
    to_select = np.arange(num_samples)
    if num_samples > 15:
        to_select = rng.choice(num_samples, size=15, replace=False)
        to_select = np.sort(to_select)
    final_tmp = tmp_data.select(list(to_select))
    print(len(final_tmp))
    assert len(final_tmp) <= 15
    if i == 0:
        final = final_tmp
    else:
        final = concatenate_datasets([final, final_tmp])


dataset = load_dataset("cais/mmlu", "all", split="dev")
for i, subject in enumerate(subjects):
    tmp_data = dataset.filter(lambda example: example["subject"] == subject)
    assert len(tmp_data) == 5
    final = concatenate_datasets([final, tmp_data])

for i, subject in enumerate(subjects):
    tmp_data = final.filter(lambda example: example["subject"] == subject)
    assert len(tmp_data) <= 20
    assert len(tmp_data) >= 13, (subject, len(tmp_data))


dataset_val = Counter(final["subject"])


dataset_val = dict(sorted(dataset_val.items(), key=lambda item: item[1]))


# ******************************************************************************************


sns.set_style("whitegrid")
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(1, 1)

# y_val = np.array(list(dataset_val.values()))  # / np.max(list(dataset_val.values()))
y_test = np.array(list(dataset_test.values()))  # / np.max(list(dataset_test.values()))


ax = fig.add_subplot(gs[0])
# ax.plot(y_val, label="val", marker=".")
ax.plot(y_test, label="test", marker=".")
ax.legend()
ax.set_xticks(np.arange(len(y_test)))

gs.tight_layout(fig)
