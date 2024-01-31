from common.metadata_db import MetadataDB
import common.globals as g
from metrics.metrics import Metrics
from common.utils import *
import os
import logging
from tqdm import tqdm
from pathlib import Path
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
parser.add_argument('--metrics',  nargs='+', action='append',type=str, help='Metrics to compute')
args = parser.parse_args(remaining_argv)

dataset_folder = args.dataset_folder
metrics_list = args.metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

results_path = os.path.join(g._OUTPUT_DIR, dataset_folder)
db = MetadataDB(Path(results_path,'metadata.db'))

# Evaluate metrics
metrics = Metrics(db, metrics_list,Path(results_path,"tensor_files") )
out = metrics.evaluate()
logging.info(f'Saving results...')
for metric in out.keys():
    with open(Path(results_path,f"{metric}.pkl"), 'wb') as f:
        out[metric].to_pickle(f)
    



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

    
