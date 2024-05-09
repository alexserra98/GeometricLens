import pickle
import numpy as np


with open("./results/statistics_1gpu_test_target.pkl", "rb") as f:
    stat1 = pickle.load(f)

with open("./results/statistics_2gpu_test_target.pkl", "rb") as f:
    stat2 = pickle.load(f)

with open("./results/statistics_mmlu.pkl", "rb") as f:
    stat = pickle.load(f)


np.all(stat1["predictions"] == stat2["predictions"])
np.all(stat["answers"] == stat1["answers"])
np.all(np.array(stat["subjects"]) == np.array(stat1["subjects"]))

letters = np.array(["A", "B", "C", "D"])


stat["input_ids"]
stat["logits"]

set(stat["constrained_predictions"])

np.where(stat1["contrained_predictions"] != letters[stat["constrained_predictions"]])
