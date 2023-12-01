import os
from pathlib import Path
import pickle
import argparse
from utils_nobatch import *
import logging
from dataset_utils import *
from generation import *
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#working_path = Path(os.getcwd())
#working_path = Path("/home/alexserra98/helm-suite/no-helm")
working_path = os.getcwd()

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

#Getting the dataset
#find files inside datasets that contain the name of the dataset
logging.info("Getting the datasets...")

#Instantiating model and tokenizer
logging.info("Loading model and tokenizer...")
client = Huggingface_client(model_name)

for dataset in datasets[0]:
    dataset = dataset
    for train_instances in max_train_instances[0]:
        logging.info(f"Starting inference on {dataset} with {train_instances} train instances...")
        scenario = Scenario(dataset,train_instances,model_name)
        requests_results = client.make_request(scenario)
        logging.info("Saving the results...")
        run_name = dataset + f",max_train_instances={train_instances}"
        #SET LTS PATH FOR ORFEO!
        result_path = Path("inference_result", model_name.split('/')[1],dataset,train_instances).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(result_path,"request_results.pkl"),"wb") as f:
            pickle.dump(requests_results,f)


