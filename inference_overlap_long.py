import os
from pathlib import Path
import pickle
import argparse
import logging
from inference_id.datasets.utils import Scenario
from inference_id.generation.generation import Huggingface_client, ScenarioResult
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')

    
working_path = Path(os.getcwd())
#working_path = Path("/home/alexserra98/helm-suite/no-helm")
#working_path = os.getcwd()

#Getting commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, help='The name of the model')
parser.add_argument('--dataset',  nargs='+', action='append', type=str, help='The name of the dataset')
parser.add_argument('--max-train-instances', nargs='+',  action='append',help='The name of the output directory')
args = parser.parse_args()

# Now you can use args.model_name to get the model name
model_name = args.model_name
datasets = args.dataset
max_train_instances = args.max_train_instances

# Getting the dataset
# find files inside datasets that contain the name of the dataset
logging.info("Getting the datasets...")

# Instantiating model and tokenizer
logging.info("Loading model and tokenizer...")
client = Huggingface_client(model_name)

for dataset in datasets[0]:
    for train_instances in max_train_instances[0]:
        logging.info("Starting inference on %s with %s train instances...",
                     dataset, train_instances)
        scenario = Scenario(dataset, train_instances, model_name, 200)
        requests_results = client.make_request(scenario)
        logging.info("Saving the results...")
        scratch_path = "/orfeo/scratch/dssc/zenocosini"
        result_path = Path(scratch_path, "overlap_long_result", 
                           model_name.split('/')[1], dataset, train_instances)
        result_path.mkdir(parents=True, exist_ok=True)
        scenario_state = ScenarioResult(dataset, train_instances, model_name,
                                        requests_results)
        with open(Path(result_path, "scenario_results.pkl"), "wb") as f:
            pickle.dump(scenario_state, f)


