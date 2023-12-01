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
import time
from dataclasses import dataclass, field
from dataset_utils import *
from generation import *

def test_scenario():
    scenario = Scenario("commonsenseqa",0,"llama",10)
    print(f'{scenario=}')
    return scenario

def test_generation():
    scenario = test_scenario()
    client = Huggingface_client("gpt2")
    encoded_input = client.tokenizer(scenario.requests_instances[0].prompt,return_tensors="pt", padding=True,return_token_type_ids=False).to("cuda")
    request_result=client.inference(encoded_input)
    print(f'{request_result[0].loss}')
    return request_result

def test_prediction():
    scenario = test_scenario()
    client = Huggingface_client("gpt2")
    encoded_input = client.tokenizer(scenario.requests_instances[0].prompt,return_tensors="pt", padding=True,return_token_type_ids=False).to("cuda")
    request_result=client.inference(encoded_input)
    request_config = {'temperature': 1e-07,
                        'num_return_sequences': 1,
                        'max_new_tokens': 1,
                        'top_p': 1,
                        'output_hidden_states': True,
                        "do_sample":True,
                        "return_dict_in_generate":True,
                        "output_scores":True,
                        "pad_token_id":client.tokenizer.eos_token_id}
    tokens_answers = [client.tokenizer.encode(letter)[0] for letter in list(scenario.output_mapping.keys())]
    request_result.logits = request_result.logits[:,-1].detach().cpu()
    preds=client.prediction(request_result, request_config,tokens_answers)
    print(f'{preds=}')
    return preds    

def test_make_request():
    scenario = test_scenario()
    client = Huggingface_client("gpt2")
    requests_results = client.make_request(scenario)
    print(f'{requests_results=}')
    return requests_results
def test_tokenizer():
    scenario = test_scenario()
    client = Huggingface_client("gpt2")
    encoded_input = client.tokenizer(scenario.requests_instances[0].prompt,return_tensors="pt", padding=True,return_token_type_ids=False).to("cuda")
    print(f'{encoded_input=} \n {encoded_input.shape=}')
    
    
if __name__ == "__main__":
    #test_scenario()
    #test_generation()
    #test_prediction()
    test_make_request()
    test_prediction()