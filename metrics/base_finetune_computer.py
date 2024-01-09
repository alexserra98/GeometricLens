from inference_id.metrics.metrics import *
import os
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
import sys

metric = sys.argv[1]
assert metric in ["final_metric", "nn"], "Metric not supported"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
working_path = "/orfeo/scratch/dssc/zenocosini" 
results_path = os.path.join(working_path, "inference_result")
models = os.listdir(results_path)
data = {}
for model in models:
    datasets = os.listdir(os.path.join(results_path, model))
    if "Llama" not in model:
        continue
    data[model] = {}
    for dataset in datasets:
        instances_id = []
        instances_metrics = []
        max_train_instances = os.listdir(os.path.join(results_path, model, dataset))
        data[model][dataset] = {}
        for max_train_instance in tqdm(max_train_instances, desc="Computing per instance result"):
            logging.info(f'Computing results for {model}, {dataset}, {max_train_instance}')
            instance_path = Path(os.path.join(results_path, model, dataset, max_train_instance))
            with open(instance_path /"scenario_results.pkl", 'rb') as f:
                    scenario_results = pickle.load(f)
            shot_metrics: ShotMetrics = ShotMetrics(scenario_results)
            nn = shot_metrics.compute_nn()
            data[model][dataset][max_train_instance] = nn
        
with open(Path(results_path,"nn.pkl"),"wb") as f:
    pickle.dump(data,f)   
base_finetuned_overlap: BaseFinetuneOverlap = BaseFinetuneOverlap(data)
overlap = base_finetuned_overlap.compute_overlap()     
with open(Path(results_path,"overlap_base_finetuned.pkl"),"wb") as f:
    pickle.dump(overlap,f)