import os
from pathlib import Path
import json
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Any, Dict, List
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
import copy
from einops import reduce
import numpy as np
import argparse
from utils_nobatch import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

#working_path = Path(os.getcwd())
#working_path = Path("/home/alexserra98/helm-suite/no-helm")
working_path = os.getcwd()

scratch_path = Path("/orfeo/scratch/dssc/zenocosini/")
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
model_kwargs = {}

model_kwargs["output_hidden_states"] = True
if os.path.exists(scratch_path):
    model_kwargs["cache_dir"]="/orfeo/scratch/dssc/zenocosini/"
if "llama" in model_name:
    model_kwargs["torch_dtype"]="auto"
    model_kwargs["device_map"]="auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **model_kwargs)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **model_kwargs).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
#Setting up generation config
request_config = {'temperature': 1e-07,
                  'num_return_sequences': 1,
                  'max_new_tokens': 1,
                  'top_p': 1,
                  'output_hidden_states': True,
                  "do_sample":True,
                  "return_dict_in_generate":True,
                  "output_scores":True,
                  "pad_token_id":tokenizer.eos_token_id}



for dataset in datasets[0]:
    dataset = dataset
    for train_instances in max_train_instances[0]:
        logging.info(f"Starting inference on {dataset} with {train_instances} train instances...")
        #Getting the dataset
        file = [f for f in os.listdir(os.path.join(working_path, "datasets")) if dataset in f and f'max_train_instances={train_instances}' in f]
        dataset_path = os.path.join(working_path, "datasets", file[0])
        with open(os.path.join(dataset_path,"request_states.pkl"),"rb") as f:
            request_states = pickle.load(f)
        
        request_states = request_states[:2500]    
        output_mapping = request_states[0].output_mapping
        tokens_answers = [tokenizer.encode(letter)[0] for letter in list(output_mapping.keys())]
        hidden_states_results = []
        metrics_results = []
        for request_state in tqdm(request_states, desc="Inference..."):
            prompt, instance, len_tokens_question, letter_gold, token_gold = aggregate(model_name,request_state,tokenizer, output_mapping)
            if prompt is None:
                continue
            encoded_input = tokenizer(prompt,return_tensors="pt", padding=True,return_token_type_ids=False).to(
            device
            )
            #generation
            output = inference(model,encoded_input)
            scores = output.logits[:,-1].detach().cpu()
            probs = torch.nn.functional.softmax(scores/float(request_config["temperature"]),dim=-1)
            pred =  torch.multinomial(probs, num_samples=1)
            # generation with only letters
            #import pdb; pdb.set_trace()
            probs_letters = torch.nn.functional.softmax(scores.index_select(-1,torch.tensor(tokens_answers))/float(request_config["temperature"]),dim=-1)
            pred_letter = torch.multinomial(probs_letters, num_samples=1)
            sequences = torch.concat([encoded_input["input_ids"].cpu(),pred],dim=1)
            hidden_states_results.append(hidden_states(output.hidden_states,len_tokens_question))
            metrics_results.append(metrics(scores,probs, pred, pred_letter, token_gold, letter_gold, tokenizer,tokens_answers))
            del output.hidden_states
        # Saving the results
        logging.info("Saving the results...")
        run_name = dataset + f",max_train_instances={train_instances}"
        result_path = os.path.join(scratch_path, "inference_result", model_name.split('/')[1],dataset,train_instances)
        ensure_directory_exists(result_path)

        with open(os.path.join(result_path,"hidden_states.pkl"),"wb") as f:
            pickle.dump(hidden_states_results,f)

        with open(os.path.join(result_path,"metrics.pkl"),"wb") as f:
            pickle.dump(metrics_results,f)

