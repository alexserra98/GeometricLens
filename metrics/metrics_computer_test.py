from inference_id.metrics.metrics import *
import os
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
import sys
import argparse


#Getting commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset-folder', type=str, help='The name of the dataset folder')
parser.add_argument('--label', choices=["letter", "subject", "base_finetune"], type=str, help='Label for label overlap')
parser.add_argument('--metric-type', choices=["per_instance", "across_instances"], type=str, help='Metrics on the single instance or Metrics computed across different instances')
args = parser.parse_args()

dataset_folder = args.dataset_folder
label = args.label
metric_type = args.metric_type


class OverlapClasses(Enum):
    letter = LetterOverlap
    subject = SubjectOverlap
    base_finetune = BaseFinetuneOverlap
    

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#working_path = "/orfeo/LTS/LADE/LT_storage/zenocosini" 
working_path = "/orfeo/scratch/dssc/zenocosini/"
results_path = os.path.join(working_path, dataset_folder)
models = os.listdir(results_path)
list_scenario_results = []
for model in models:
    if "Llama" not in model:
        continue 
    datasets = os.listdir(os.path.join(results_path, model))
    datasets.sort()
    for dataset in datasets:
        if dataset != "commonsenseqa" or dataset=="result":
            continue
        max_train_instances = os.listdir(os.path.join(results_path, model, dataset))
        logging.info(f'Collecting results for {model}, {dataset}')
        for max_train_instance in max_train_instances:
            instance_path = Path(os.path.join(results_path, model, dataset, max_train_instance))
            with open(instance_path /"scenario_results.pkl", 'rb') as f:
                    scenario_results = pickle.load(f)
            list_scenario_results.append(scenario_results)
            if metric_type == "per_instance":
                shot_metrics: ShotMetrics = ShotMetrics(scenario_results)
                final_metrics = shot_metrics.evaluate()
                with open(instance_path /"final_metrics.pkl", 'wb') as f:
                    pickle.dump(final_metrics,f)
                nn = shot_metrics.compute_nn()
                with open(instance_path /"nn.pkl", 'wb') as f:
                    pickle.dump(nn,f)

if metric_type == "across_instances":
    overlap_instance = OverlapClasses[label].value(list_scenario_results)
    overlaps = overlap_instance.compute_overlap()
    logging.info(f'Saving results...')
    file_name = f"{label}_overlaps.pkl"
    final_path = Path(results_path,"result") 
    final_path.mkdir(parents=True, exist_ok=True)
    overlaps.to_pickle(Path(final_path,file_name))





    
