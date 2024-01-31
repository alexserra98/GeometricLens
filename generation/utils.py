import torch
import gc

from einops import reduce
import time
from dataclasses import dataclass, field
            
def hidden_states_preprocess(hidden_states,len_tokens_question):
    hs = torch.stack(hidden_states)[:,0,-len_tokens_question:,:].clone().detach().cpu().numpy()
    return {"last": hs[:,-1],"sum":reduce(hs[:,-len_tokens_question:], "l s d -> l d", "mean")}


class HiddenStatesHandler():
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
    
    def preprocess(self,request_instance, tokenizer):
        len_tokens_question = self.index_last_question(request_instance.prompt, tokenizer)
        hs = torch.stack(self.hidden_states)[:,0,-len_tokens_question:,:].clone().detach().cpu().numpy()
        del self.hidden_states
        gc.collect()
        return {"last": hs[:,-1],"sum":reduce(hs[:,-len_tokens_question:], "l s d -> l d", "mean")}
   
    def index_last_question(self,prompt, tokenizer):
        index_in_prompt = prompt.rfind("Question")
        tokens_question = tokenizer(prompt[index_in_prompt:], return_tensors="pt", return_token_type_ids=False)
        len_tokens_question = tokens_question["input_ids"].shape[1]
        return len_tokens_question

