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

def retry_on_failure(max_retries, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying...")
                    time.sleep(delay)
                    delay *= 1.5
            raise Exception("Maximum retries exceeded. Function failed.")
        return wrapper
    return decorator

def index_last_question(prompt, tokenizer):
    index_in_prompt = prompt.rfind("Question")
    tokens_question = tokenizer(prompt[index_in_prompt:], return_tensors="pt", return_token_type_ids=False)
    len_tokens_question = tokens_question["input_ids"].shape[1]
    return len_tokens_question

def hidden_states(hidden_states,len_tokens_question):
    hs = torch.stack(hidden_states)[:,0,-len_tokens_question:,:].clone().detach().cpu().numpy()
    return {"last": hs[:,-1],"sum":reduce(hs[:,-len_tokens_question:], "l s d -> l d", "mean")}

def exact_match(answers, letters_gold):
    return answers == letters_gold
def quasi_exact_match(answers, letters_gold):
    is_in_string = answers.lower() in letters_gold.lower()
    #print(f"Is {answers} in {letters_gold}? {is_in_string}")
    return is_in_string
    

def metrics(probs,preds, token_gold, letters_gold,tokenizer):
    #import pdb; pdb.set_trace()
    loss = torch.nn.functional.cross_entropy(probs,token_gold.unsqueeze(0))
    perp = torch.exp(loss)
    answers = tokenizer.decode(preds[0]).strip()
    exact_match_result = exact_match(answers, letters_gold) 
    quasi_exact_match_result = quasi_exact_match(answers, letters_gold) 
    return {"perp":perp, "loss":loss, "exact_match":exact_match_result, "quasi_exact_match":quasi_exact_match_result}   

def aggregate(model_name,request_state,tokenizer,output_mapping):
    prompt  = request_state.request.prompt
    if tokenizer.encode(request_state.request.prompt, return_tensors="pt").shape[1] >= tokenizer.max_len_single_sentence:
        (f"Prompt too long: {request_state.request.prompt}. Skipping this sequence")
        return -1
    len_tokens_question = index_last_question(request_state.request.prompt, tokenizer)
    instance = request_state.instance
    index = [ref.tags == ['correct'] for ref in request_state.instance.references].index(True)
    letter_gold = list(output_mapping.keys())[index]
    if "llama" in model_name or "facebook" in model_name:
        token_gold = tokenizer(letter_gold[-1], return_tensors="pt", return_token_type_ids=False)["input_ids"][0,1]
    else:
        token_gold = tokenizer(letter_gold[-1], return_tensors="pt", return_token_type_ids=False)["input_ids"]
    return prompt, instance, len_tokens_question, letter_gold, token_gold

@retry_on_failure(5, delay=5)
def inference(model,encoded_input):
    with torch.no_grad():
        output = model(**encoded_input, output_hidden_states=True)
    return output
