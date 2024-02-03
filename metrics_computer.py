from common.metadata_db import MetadataDB
import common.globals as g
from metrics.metrics import Metrics
from common.utils import *
import os
import logging
from pathlib import Path
import argparse

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
metrics = Metrics(db, metrics_list,Path(results_path))
out = metrics.evaluate()
logging.info(f'Saving results...')
#for metric in out.keys():
#    out[metric].to_pickle(Path(results_path,f'{metric}.pkl'))
    

    
