from dataset_utils.utils import ScenarioBuilder
from generation.generation import Huggingface_client
from common.metadata_db import MetadataDB
from common.tensor_storage import TensorStorage
from metrics.metrics import ShotMetrics, LetterOverlap, SubjectOverlap, BaseFinetuneOverlap
import common.globals as g
from common.utils import *
import os
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
import sys
import argparse
from enum import Enum
from metrics.query import DataFrameQuery


#Getting commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--conf-path', help='Path to configuration file', required=False)
args, remaining_argv = parser.parse_known_args()

if args.conf_path:
    # Load arguments from config file
    config_args = read_config(args.conf_path)

    # Update the parser with new arguments from config file
    parser.set_defaults(**config_args)   
    

parser.add_argument('--dataset-folder', type=str, help='The name of the dataset folder')
parser.add_argument('--label', choices=["letter", "subject", "base_finetune"], type=str, help='Label for label overlap')
parser.add_argument('--metric-type', choices=["per_instance", "across_instances"], type=str, help='Metrics on the single instance or Metrics computed across different instances')
parser.add_argument('--dataset',  nargs='+', action='append', type=str, help='The name of the dataset')
parser.add_argument('--model-name', type=str, help='The name of the model')

args = parser.parse_args(remaining_argv)


dataset_folder = args.dataset_folder
label = args.label
metric_type = args.metric_type
dataset = args.dataset
model_name = args.model_name


class OverlapClasses(Enum):
    letter = LetterOverlap
    subject = SubjectOverlap
    base_finetune = BaseFinetuneOverlap
    

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

results_path = os.path.join(g._OUTPUT_DIR, dataset_folder)

if metric_type == "across_instances":
    db = MetadataDB(Path(results_path,'metadata.db'))
    query = DataFrameQuery({'dataset':dataset, 'model_name':model_name})
    overlap_instance = OverlapClasses[label].value(db, query, Path(results_path,"tensor_files"))
    overlaps = overlap_instance.compute_overlap()
    logging.info(f'Saving results...')
    file_name = f"{label}_overlaps.pkl"
    final_path = Path(results_path,"result") 
    final_path.mkdir(parents=True, exist_ok=True)
    overlaps.to_pickle(Path(final_path,file_name))



# for model in models:
#     if "Llama" not in model:
#         continue 
#     datasets = os.listdir(os.path.join(results_path, model))
#     datasets.sort()
#     for dataset in datasets:
#         if dataset=="result":
#             continue
#         max_train_instances = os.listdir(os.path.join(results_path, model, dataset))
#         logging.info(f'Collecting results for {model}, {dataset}')
#         for max_train_instance in max_train_instances:
#             instance_path = Path(os.path.join(results_path, model, dataset, max_train_instance))
#             with open(instance_path /"scenario_results.pkl", 'rb') as f:
#                     scenario_results = pickle.load(f)
#             list_scenario_results.append(scenario_results)
#             if metric_type == "per_instance":
#                 shot_metrics: ShotMetrics = ShotMetrics(scenario_results)
#                 final_metrics = shot_metrics.evaluate()
#                 with open(instance_path /"final_metrics.pkl", 'wb') as f:
#                     pickle.dump(final_metrics,f)
#                 nn = shot_metrics.compute_nn()
#                 with open(instance_path /"nn.pkl", 'wb') as f:
#                     pickle.dump(nn,f)

    
