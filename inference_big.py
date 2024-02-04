import os
from pathlib import Path
import argparse
import logging
from dataset_utils.utils import ScenarioBuilder
from generation.generation import Huggingface_client
from common.metadata_db import MetadataDB
from common.tensor_storage import TensorStorage
import MCQA_Benchmark.common.globals_vars as g
from common.utils import *
from safetensors.numpy import save_file
import gc
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')


working_path = Path(os.getcwd())

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
parser.add_argument('--model-name', type=str, help='The name of the model')
parser.add_argument('--dataset',  nargs='+', action='append', type=str, help='The name of the dataset')
parser.add_argument('--max-train-instances', type=int, nargs='+',  action='append',help='The name of the output directory')

args = parser.parse_args(remaining_argv)
# Now you can use args.model_name to get the model name
dataset_folder = args.dataset_folder
model_name = args.model_name
datasets = args.dataset
max_train_instances = args.max_train_instances
print(f'{datasets=}')


# Getting the dataset
# find files inside datasets that contain the name of the dataset
logging.info("Getting the datasets...")

# Instantiating model and tokenizer
logging.info("Loading model and tokenizer...")
client = Huggingface_client(model_name)



result_path = Path(g._OUTPUT_DIR, f'{dataset_folder}_result')
result_path.mkdir(parents=True, exist_ok=True)
metadata_db = MetadataDB(Path(result_path,'metadata.db'))
tensor_storage = TensorStorage(Path(result_path,'tensor_files'))

for dataset in datasets:
    hidden_states_rows = {}
    logits_rows = {}
    db_rows = []
    for train_instances in max_train_instances:
        logging.info("Starting inference on %s with %s train instances...",
                     dataset, train_instances)
        scenario_builder = ScenarioBuilder(dataset,train_instances,model_name,-1)
        scenario = scenario_builder.build()
        hidden_states_rows_i, logits_rows_i, db_rows_i = client.make_request(scenario,metadata_db)
        if not hidden_states_rows_i or not logits_rows_i  or not db_rows_i:
            logging.info("Skipping inference on %s with %s train instances...",
                     dataset, train_instances)
            continue

        logging.info("Saving tensors..")
        tensor_storage.save_tensors(logits_rows_i.values(),logits_rows.keys(),'logits')
        tensor_storage.save_tensors(hidden_states_rows_i.values(),hidden_states_rows.keys(), 'hidden_states')
        del hidden_states_rows
        del logits_rows
        gc.collect()
        db_rows.extend(db_rows_i)
    logging.info("Saving run results..")
    metadata_db.add_metadata(db_rows)
logging.info("Closing...")

metadata_db.close()
