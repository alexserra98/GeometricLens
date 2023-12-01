import os
from pathlib import Path
import json
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from typing import Any, Dict, List
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token

import copy
from einops import reduce
import numpy as np
import argparse
import time
from dataclasses import dataclass, field
from dataset_utils import *
from utils_nobatch import retry_on_failure
from abc import ABC, abstractmethod
from utils_nobatch import *

@dataclass
class RequestResult():
    loss: float
    logits: torch.Tensor
    hidden_states: torch.Tensor
    preds: Dict[str, torch.Tensor]
    gold: torch.Tensor
    
    
class Huggingface_client():
    def __init__(self,model_name) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {}
        model_kwargs["output_hidden_states"] = True
        if (cache_path:=Path("/orfeo/scratch/dssc/zenocosini/")).exists() :
            model_kwargs["cache_dir"]=cache_path
        if "llama" in model_name:
            model_kwargs["torch_dtype"]="auto"
            model_kwargs["device_map"]="auto"
            self.model =LlamaForCausalLM.from_pretrained(model_name, trust_remote_code=True, **model_kwargs)
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name, **model_kwargs)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **model_kwargs).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def make_request(self, scenario):
        request_config = {'temperature': 1e-07,
                        'num_return_sequences': 1,
                        'max_new_tokens': 1,
                        'top_p': 1,
                        'output_hidden_states': True,
                        "do_sample":True,
                        "return_dict_in_generate":True,
                        "output_scores":True,
                        "pad_token_id":self.tokenizer.eos_token_id}

        tokens_answers = [self.tokenizer.encode(letter)[0] for letter in list(scenario.output_mapping.keys())]
        requests_results = []
        for request_instance in tqdm(scenario.requests_instances, desc="Generating requests"):
            encoded_input = self.tokenizer(request_instance.prompt,return_tensors="pt", padding=True,return_token_type_ids=False).to(
            self.device
            )
            request_result=self.inference(encoded_input)
            request_result.logits = request_result.logits[:,-1].detach().cpu()
            predictions = self.prediction(request_result, request_config,tokens_answers)
            token_gold = torch.tensor(self.tokenizer.encode(request_instance.letter_gold)[0]).unsqueeze(0)
            request_instance.token_gold = token_gold[0].item()
            loss = torch.nn.functional.cross_entropy(request_result.logits,token_gold)
            hidden_states = HiddenStates(request_result.hidden_states)
            del request_result.hidden_states
            result = RequestResult(loss, request_result.logits,hidden_states.preprocess(request_instance,self.tokenizer), predictions, request_instance.token_gold)
            requests_results.append(result)
            
        return requests_results
    
        
    def prediction(self, request_result, request_config, tokens_answers):
        std_pred_strat: StandardPrediction = StandardPrediction(request_result.logits, request_config)
        only_ref_pred_stat: OnlyReferencePrediction = OnlyReferencePrediction(request_result.logits, request_config, tokens_answers)
        std_pred = self.tokenizer.encode(self.tokenizer.decode(std_pred_strat.predict()).strip())[0]
        pred_dict = {"std_pred":std_pred, "only_ref_pred":only_ref_pred_stat.predict()}
        return pred_dict

    @retry_on_failure(3)
    def inference(self,encoded_input):
        self.model.eval()
        with torch.no_grad():
            output = self.model(**encoded_input, output_hidden_states=True)
        return output 
    


class PredictionStrategy(ABC):
    def __init__(self,logits, request_config) -> None:
        self.logits=logits
        self.request_config=request_config
    
    @abstractmethod
    def predict(self, request_instances):
        pass
    
class StandardPrediction(PredictionStrategy):
    
    def predict(self):
        #scores = request_result.logits[:,-1].detach().cpu()
        probs = torch.nn.functional.softmax(self.logits/float(self.request_config["temperature"]),dim=-1)
        pred =  torch.multinomial(probs, num_samples=1)
        return pred.item()
    
class OnlyReferencePrediction(PredictionStrategy):
    def __init__(self,logits,request_config,tokens_answers) -> None:
        super().__init__(logits,request_config)
        self.tokens_answers=tokens_answers
    
    def predict(self):
        #scores = request_result.logits[:,-1].detach().cpu()
        stripped_logits = self.logits.index_select(-1,torch.tensor(self.tokens_answers))
        probs = torch.nn.functional.softmax(stripped_logits/float(self.request_config["temperature"]),dim=-1)
        pred = torch.multinomial(probs, num_samples=1)
        return self.tokens_answers[pred]
   