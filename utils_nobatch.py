import torch

from einops import reduce
import time
from dataclasses import dataclass, field
            
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



def hidden_states_preprocess(hidden_states,len_tokens_question):
    hs = torch.stack(hidden_states)[:,0,-len_tokens_question:,:].clone().detach().cpu().numpy()
    return {"last": hs[:,-1],"sum":reduce(hs[:,-len_tokens_question:], "l s d -> l d", "mean")}

def exact_match(answers, letters_gold):
    return answers == letters_gold
def quasi_exact_match(answers, letters_gold):
    is_in_string = answers.lower() in letters_gold.lower()
    #print(f"Is {answers} in {letters_gold}? {is_in_string}")
    return is_in_string
    

def metrics(scores,probs,pred,pred_letter, token_gold, letters_gold,tokenizer,tokens_answers):
    #import pdb; pdb.set_trace()
    loss = torch.nn.functional.cross_entropy(scores,token_gold.unsqueeze(0))
    #loss = torch.nn.functional.cross_entropy(scores,token_gold[0])
    perp = torch.exp(loss)
    answer = tokenizer.decode(pred[0]).strip()
    answer_letter = tokenizer.decode(tokens_answers[pred_letter[0]]).strip()
    exact_match_letter_result = exact_match(answer_letter, letters_gold)
    exact_match_result = exact_match(answer, letters_gold) 
    quasi_exact_match_result = quasi_exact_match(answer, letters_gold) 
    return {"perp":perp, 
            "loss":loss, 
            "exact_match":exact_match_result, 
            "quasi_exact_match":quasi_exact_match_result, 
            "exact_match_letter":exact_match_letter_result}   

class HiddenStates():
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
    
    def preprocess(self,request_instance, tokenizer):
        len_tokens_question = self.index_last_question(request_instance.prompt, tokenizer)
        hs = torch.stack(self.hidden_states)[:,0,-len_tokens_question:,:].clone().detach().cpu().numpy()
        return {"last": hs[:,-1],"sum":reduce(hs[:,-len_tokens_question:], "l s d -> l d", "mean")}
   
    def index_last_question(self,prompt, tokenizer):
        index_in_prompt = prompt.rfind("Question")
        tokens_question = tokenizer(prompt[index_in_prompt:], return_tensors="pt", return_token_type_ids=False)
        len_tokens_question = tokens_question["input_ids"].shape[1]
        return len_tokens_question

