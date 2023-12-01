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
    out = []
    for batch in range(hidden_states[0].shape[0]):
        hs = torch.stack(hidden_states[-len_tokens_question[batch]:])[:,batch].clone().detach().cpu().numpy()
        out.append({"last": copy.deepcopy(hs[-1,:]),"sum":reduce(copy.deepcopy(hs[-len_tokens_question[batch]:,:]), "l s d -> l d", "mean")})
    return out

def exact_match(answers, letters_gold):
    return sum([a == b for a, b in zip(answers, letters_gold)])

def metrics(probs,preds, token_gold, letters_gold,tokenizer):
    loss = torch.nn.functional.cross_entropy(probs,torch.tensor(token_gold).to(torch.long))
    perp = torch.exp(loss)
    answers = list(map(lambda k: k.strip(), tokenizer.batch_decode(preds)))
    exact_match_result = exact_match(answers, letters_gold) 
    return {"perp":perp, "loss":loss, "exact_match":exact_match_result}   

def aggregate(request_states,i,batch_size,tokenizer,output_mapping):
    prompts = []
    instances = []
    len_tokens_question = []
    letters_gold = []
    tokens_gold = []
    top = min(i+batch_size,len(request_states))
    for j in range(i,top):
        prompts.append(request_states[j].request.prompt)
        if tokenizer.encode(request_states[j].request.prompt, return_tensors="pt").shape[1] >= tokenizer.max_len_single_sentence:
            print(f"Prompt too long: {request_states[j].request.prompt}. Skipping this sequence")
            continue
        len_tokens_question.append(index_last_question(request_states[j].request.prompt, tokenizer))
        instances.append(request_states[j].instance)
        index = [ref.tags == ['correct'] for ref in request_states[j].instance.references].index(True)
        letters_gold.append(list(output_mapping.keys())[index])
        tokens_gold.append(tokenizer(letters_gold[-1], return_tensors="pt", return_token_type_ids=False)["input_ids"][0,1])
    return prompts, instances, len_tokens_question, letters_gold, tokens_gold

@retry_on_failure(5, delay=5)
def inference(model,encoded_input):
    with torch.no_grad():
        output = model(**encoded_input, output_hidden_states=True)
    return output
