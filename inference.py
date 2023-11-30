import os
from helm.benchmark.hidden_geometry.geometry import RunGeometry, InstanceHiddenSates
from pathlib import Path
import json
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Any, Dict, List

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from helm.proxy.clients.client import Client, wrap_request_time, truncate_sequence, cleanup_tokens
from helm.proxy.clients.huggingface_tokenizer import HuggingFaceTokenizers
from helm.proxy.clients.huggingface_model_registry import (
    get_huggingface_model_config,
    HuggingFaceModelConfig,
    HuggingFaceHubModelConfig,
    HuggingFaceLocalModelConfig,
)
import copy
from einops import reduce
import numpy as np
import argparse
from utils import *
import logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
batch_size = 2

working_path = Path(os.getcwd())
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
if os.path.exists("/orfeo/scratch/dssc/zenocosini/"):
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
        hidden_states_results = []
        metrics_results = []
        for i in tqdm(range(0,len(request_states),batch_size), desc="Inference on batches"):
            prompts, instances, len_tokens_question, letters_gold, tokens_gold = aggregate(request_states,i,batch_size,tokenizer, output_mapping)
            encoded_input = tokenizer(prompts,return_tensors="pt", padding=True,return_token_type_ids=False).to(
            device
            )
            #generation
            output = inference(model,encoded_input)
            scores = output.logits[:,-1].detach().cpu()
            probs = torch.nn.functional.softmax(scores/float(request_config["temperature"]),dim=-1)
            preds =  torch.multinomial(probs, num_samples=1)
            sequences = torch.concat([encoded_input["input_ids"].cpu(),preds],dim=1)
            hidden_states_results.append(hidden_states(output.hidden_states,len_tokens_question))
            metrics_results.append(metrics(probs, preds, tokens_gold, letters_gold, tokenizer))
            del output.hidden_states
        # Saving the results
        logging.info("Saving the results...")
        run_name = dataset + f",max_train_instances={train_instances}"
        result_path = os.path.join(working_path, "results",model_name.split("/")[1],run_name)
        ensure_directory_exists(result_path)

        with open(os.path.join(result_path,"hidden_states.pkl"),"wb") as f:
            pickle.dump(hidden_states_results,f)

        with open(os.path.join(result_path,"metrics.pkl"),"wb") as f:
            pickle.dump(metrics_results,f)


# for i in tqdm(range(0,len(request_states),batch_size), desc="Inference on batches"):
#     prompts, instances, len_tokens_question, letters_gold, tokens_gold = aggregate(request_states,i,batch_size,tokenizer, output_mapping)
#     encoded_input = tokenizer(prompts,return_tensors="pt", padding=True,return_token_type_ids=False).to(
#     device
#     )
#     #generation
#     output = inference(model,encoded_input)
#     scores = output.logits[:,-1].detach().cpu()
#     probs = torch.nn.functional.softmax(scores/float(request_config["temperature"]),dim=-1)
#     preds =  torch.multinomial(probs, num_samples=1)
#     sequences = torch.concat([encoded_input["input_ids"].cpu(),preds],dim=1)
#     hidden_states_results.append(hidden_states(output.hidden_states,len_tokens_question))
#     metrics_results.append(metrics(probs, preds, tokens_gold, letters_gold, tokenizer))


# # Saving the results
# logging.info("Saving the results...")
# result_path = os.path.join(working_path, "results", dataset, model_name)
# ensure_directory_exists(result_path)

# with open(os.path.join(result_path,"hidden_states.pkl"),"wb") as f:
#     pickle.dump(hidden_states_results,f)

# with open(os.path.join(result_path,"metrics.pkl"),"wb") as f:
#     pickle.dump(metrics_results,f)
