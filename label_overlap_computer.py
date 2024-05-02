import pandas as pd
from pathlib import Path
from metrics.overlap import LabelOverlap
from metrics.utils import TensorStorageManager, DataFrameQuery
from tqdm import tqdm


def load_and_process_data(shot, model_name, method):
    tsm = TensorStorageManager()
    dict_query = {
        "method": "last",
        "model_name": model_name,
        "train_instances": shot,
    }
    query = DataFrameQuery(dict_query)
    hidden_states, logits, df = tsm.retrieve_tensor(query, "npy")
    return df


def main():
    base_dir = Path("/orfeo/scratch/dssc/zenocosini/mmlu_result/transposed_dataset")
    models = [
        "meta-llama/Llama-3-8b-hf",
    ]
    method = "last"

    dataframes = []
    for model in models:
        for shot in [0]:#[0, 2, 5]:
            if "70" in model and shot == 5 and "chat" not in model:
                shot = 4
            elif "70" in model and "chat" in model and shot == 5:
                continue
            df = load_and_process_data(shot, model, method)
            df.to_pickle(base_dir / f"df_{shot}.pkl")
            dataframes.append(df)

        print(f'Iteration model: {model}') 
        df_combined = pd.concat(dataframes)
        variations = {"intrinsic_dimension": "None", "point_overlap": "cosine", "label_overlap": "balanced_letter", "cka": "rbf"}
        label_overlap = LabelOverlap(
            df=df_combined, tensor_storage=None, variations=variations, storage_logic="npy"
        )
        result = label_overlap.main(label="only_ref_pred")
        result_path = base_dir / "result" / f"{model.split('/')[1]}_letter_overlap.pkl"
        result.to_pickle(result_path)


if __name__ == "__main__":
    main()
