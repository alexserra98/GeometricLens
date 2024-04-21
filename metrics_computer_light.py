import pandas as pd
import pickle
from pathlib import Path
from common.tensor_storage import TensorStorage
from metrics.intrinisic_dimension import IntrinsicDimension

def load_and_process_data(file_path, train_instances, model_name, method):
    with open(file_path, "rb") as file:
        statistics = pickle.load(file)

    df = pd.DataFrame(statistics)
    df = df.rename(columns={
        "subjects": "dataset",
        "predictions": "std_pred",
        "answers": "letter_gold",
        "contrained_predictions": "only_ref_pred",
    })
    df["train_instances"] = str(train_instances)
    df["model_name"] = model_name
    df["method"] = method
    return df

def main():
    base_dir = Path("/orfeo/scratch/dssc/zenocosini/mmlu_result/transposed_dataset")
    shots = {
        "0shot": 0,
        "2shot": 2,
        "5shot": 5,
    }
    model_name = "meta-llama/Llama-2-7b-hf"
    method = "last"

    dataframes = []
    for shot, instances in shots.items():
        file_path = f"/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens/repo/results/mmlu/llama-2-7b/{shot}/statistics_target.pkl"
        df = load_and_process_data(file_path, instances, model_name, method)
        df.to_pickle(base_dir / f"df_{instances}.pkl")
        dataframes.append(df)

    df_combined = pd.concat(dataframes)
    #import pdb; pdb.set_trace()
    intrinsic_dim = IntrinsicDimension(df=df_combined, tensor_storage=None, variations=None, storage_logic="npy")
    result = intrinsic_dim.main()
    result_path = base_dir / "result" / "intrinsic_dim.pkl"
    result.to_pickle(result_path)

if __name__ == "__main__":
    main()
