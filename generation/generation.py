from pathlib import Path
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from typing import Any, Dict, List
from helm.common.request import RequestResult
from dataclasses import dataclass, field
from inference_id.datasets.utils import *
from inference_id.generation.utils import retry_on_failure
from abc import ABC, abstractmethod
from inference_id.generation.utils import *
import pandas as pd
@dataclass
class RequestResult():
    loss: float
    logits: torch.Tensor
    hidden_states: torch.Tensor
    preds: Dict[str, torch.Tensor]
    gold: torch.Tensor

@dataclass
class ScenarioResult():
    dataset: str
    train_instances: int
    model_name: str
    requests_results: List[RequestResult] = field(default_factory=list)
    
class Huggingface_client():
    """
    Client for Huggingface models. It instantiate the model and the tokenizer, and provides method to make inference over a dataset
    """
    def __init__(self,model_name) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
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
    
    def make_request(self, scenario: Scenario) -> List[RequestResult]:
        request_config = {'temperature': 1e-07,
                        'num_return_sequences': 1,
                        'max_new_tokens': 1,
                        'top_p': 1,
                        'output_hidden_states': True,
                        "do_sample":True,
                        "return_dict_in_generate":True,
                        "output_scores":True,
                        "pad_token_id":self.tokenizer.eos_token_id}

        tokens_answers = [self.encode(letter)[0] for letter in list(scenario.output_mapping.keys())]
        
        requests_results = []
        for request_instance in tqdm(scenario.requests_instances, desc="Generating requests"):
            encoded_input = self.tokenizer(request_instance.prompt,return_tensors="pt", padding=True,return_token_type_ids=False).to(
            self.device
            )
            request_result=self.inference(encoded_input)
            request_result.logits = request_result.logits[:,-1].detach().cpu().to(torch.float32)
            predictions = self.prediction(request_result, request_config,tokens_answers)
            token_gold = torch.tensor(self.encode(request_instance.letter_gold)[0]).unsqueeze(0)
            request_instance.token_gold = token_gold[0].item()
            loss = torch.nn.functional.cross_entropy(request_result.logits,token_gold)
            Warning("Llama class return hidden states as a torch tensor while Auto class return it as a tuple of torch tensor")
            hidden_states = HiddenStatesHandler(request_result.hidden_states)
            result = [scenario.dataset, scenario.train_instances, 
                      scenario.model_name, loss, 
                      request_result.logits, 
                      hidden_states.preprocess(request_instance,self.tokenizer), 
                      predictions,
                      {"token":request_instance.token_gold,"letter":request_instance.letter_gold}]
            requests_results.append(result)
        return ScenarioResult(scenario.dataset, 
                              scenario.train_instances, 
                              scenario.model_name, 
                              requests_results)
    
        
    def prediction(self, request_result: RequestResult, request_config: Dict, tokens_answers: List[int]):
        std_pred_strat: StandardPrediction = StandardPrediction(request_result.logits, request_config)
        only_ref_pred_stat: OnlyReferencePrediction = OnlyReferencePrediction(request_result.logits, request_config, tokens_answers)
        std_pred = std_pred_strat.predict()
        only_ref = only_ref_pred_stat.predict()
        pred_dict = {"std_pred":{"token":std_pred,"letter":self.tokenizer.decode(std_pred)}, "only_ref_pred":{"token":only_ref,"letter":self.tokenizer.decode(only_ref)}}

        return pred_dict

    @retry_on_failure(3)
    def inference(self,encoded_input):
        self.model.eval()
        with torch.no_grad():
            output = self.model(**encoded_input, output_hidden_states=True)
        return output
 
    def encode(self,input: str)-> List[int]:
        if "llama" in self.model_name or "facebook" in self.model_name:
            encoded_input=self.tokenizer.encode(input)
            #import pdb; pdb.set_trace()
            encoded_input=encoded_input[1:]
            return encoded_input
        else:
            return self.tokenizer.encode(input)
        
    


class PredictionStrategy(ABC):
    """
    Abstract class for prediction strategies
    """
    def __init__(self,logits, request_config) -> None:
        self.logits=logits
        self.request_config=request_config
    
    @abstractmethod
    def predict(self, request_instances):
        pass
    
class StandardPrediction(PredictionStrategy):
    """
    Standard Inference with sampling from logits
    """
    def predict(self) -> int:
        #scores = request_result.logits[:,-1].detach().cpu()
        rescaled_logit=self.logits/float(self.request_config["temperature"])
        #import pdb; pdb.set_trace()
        probs = torch.nn.functional.softmax(rescaled_logit,dim=-1)
        pred =  torch.multinomial(probs, num_samples=1)
        return pred.item()
    
class OnlyReferencePrediction(PredictionStrategy):
    """
    Inference with sampling from logits, but only from the reference tokens
    """
    def __init__(self,logits,request_config,tokens_answers) -> None:
        super().__init__(logits,request_config)
        self.tokens_answers=tokens_answers
    
    def predict(self) -> int:
        stripped_logits = self.logits.index_select(-1,torch.tensor(self.tokens_answers))
        rescaled_logits=stripped_logits/float(self.request_config["temperature"])
        probs = torch.nn.functional.softmax(rescaled_logits,dim=-1)
        pred = probs.max(1)[1]
        return self.tokens_answers[pred]
   
