import pickle


model_size = "llama-3-70b"
num_samples = 40
seed = 0
dir_path = f"/home/diego/area_science/ricerca/finetuning_llm/open-instruct/results/{model_size}/dev_val_balanced_{num_samples}samples"


acc = {}
for seed in [0, 1, 2, 3]:
    with open(f"{dir_path}/4epochs_seed{seed}/train_statistics_epoch4.pkl", "rb") as f:
        stats = pickle.load(f)

    stats["train_stats"]["mmlu_test_micro"]
    stats["train_stats"]["mmlu_test_macro"]
    stats["train_stats"]["epoch"]

    acc[f"seed{seed}_micro"] = stats["train_stats"]["mmlu_test_micro"]
    acc[f"seed{seed}_macro"] = stats["train_stats"]["mmlu_test_macro"]


acc
