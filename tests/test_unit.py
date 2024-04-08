import sys
from pathlib import Path
import os
from dataset_utils.scenario_adapter import ScenarioAdapter
import pickle


# Set working directory as the parent of the file
#os.chdir(Path(__file__).resolve().parent.parent)





    
def test_scenario():
    scenario = ScenarioAdapter("commonsenseqa_wrong","commonsenseqa",5,"gpt2",100)
    scenario = scenario.build()
    # with open("tests/assets/unit/scenario.pkl", "wb") as f:
    #     pickle.dump(scenario,f)
           
    # def correct_letter(letter):
    #     out = []
    #     for request in scenario.requests_instances:
    #         condition = request.letter_gold in ["A","B","C","D","E"]    
    #         out.append(condition)
    #     return all(out)
    # assert correct_letter(scenario.requests_instances[0].letter_gold)
    #print(f'{scenario=}')
    return scenario

def test_generation():
    with open("tests/assets/unit/scenario.pkl", "rb") as f:
       scenario = pickle.load(f)    
       
    client = Huggingface_client("gpt2")
    encoded_input = client.tokenizer(scenario.requests_instances[0].prompt,return_tensors="pt", padding=True,return_token_type_ids=False).to("cuda")
    request_result=client.inference(encoded_input)
    assert len(request_result.hidden_states) == 13 , "The number of hidden states is not correct"
    assert request_result.hidden_states[0].shape == torch.Size([1, 63, 768]), "The hidden states shape is not correct"
    assert request_result.logits.shape == torch.Size([1, 63, 50257]), "The logits shape is not correct"

    print(f'{request_result}')
    return request_result

def test_prediction():
    with open("tests/assets/unit/scenario.pkl", "rb") as f:
         scenario = pickle.load(f)
    client = Huggingface_client("gpt2")
    with open("tests/assets/unit/request_result.pkl", "rb") as f:
         request_result = pickle.load(f)
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
    assert preds["std_pred"]["token"] in range(50257) and preds["only_ref_pred"]["token"] in range(50257) , "The token prediction is not inside the range"
    assert preds["only_ref_pred"]["letter"] in scenario.output_mapping.keys() , "The letter prediction is not val>id"
    #print(f'{preds=}')
    #return preds    

def test_make_request():
    with open("tests/assets/unit/scenario.pkl", "rb") as f:
         scenario = pickle.load(f)
    client = Huggingface_client("gpt2")
    requests_results = client.make_request(scenario)
    assert len(requests_results) == len(scenario.requests_instances), "The number of requests is not correct"
    assert requests_results[0].hidden_states["last"].shape == requests_results[0].hidden_states["sum"],  "The shape of the hidden states last sum are different"
    assert requests_results[0].hidden_states["last"].shape[0] == 13, "The shape of the hidden states last is not correct"
    #print(f'{requests_results=}')
    #return requests_results

def test_tokenizer():
    client = Huggingface_client("gpt2")
    #encoded_input = client.tokenizer(scenario.requests_instances[0].prompt,return_tensors="pt", padding=True,return_token_type_ids=False).to("cuda")
    encoded_input = client.encode("A")
    assert type(encoded_input) == list, "The encoded input is not a list"
    assert len(encoded_input) == 1, "The encoded input is not a list of length 1"
    #print(f'{type(encoded_input)}')  
    
def test_basic_metrics():
    with open("tests/assets/unit/requests_results.pkl", "rb") as f:
       requests_results = pickle.load(f)
    scenario_result = ScenarioResult("commonsenseqa",0,"gpt2",requests_results)
    metrics = ShotMetrics(scenario_result)
    basic_metric = metrics.basic_metric_mean()
    assert all(1>=val and val>=0 for key, val in basic_metric.items() if "match" in key), "The metrics are not in the range [0,1]"
    assert basic_metric["ref_exact_match"] >= 0.2, "The reference exact match is smaller than random baseline"
    #print(f'{metrics.basic_metric_mean()=}')
    #return metrics
    
def test_intrinsic_dim():
    with open("tests/assets/unit/requests_results.pkl", "rb") as f:
       requests_results = pickle.load(f)
    scenario_result = ScenarioResult("commonsenseqa",0,"gpt2",requests_results)
    metrics = ShotMetrics(scenario_result)
    hidden_states = HiddenStates(metrics.hidden_states)
    id = hidden_states.get_instances_id()
    condition = []
    for key, val in id.items():
        condition.append(val["last"].shape[0] == 11 and val["sum"].shape[0] == 11)
    assert all(condition), "The shape of the hidden states last sum are different"
    #print(f'{id=}')
    #return metrics

def old_test_letter_overlap():
    with open("tests/assets/unit/requests_results.pkl", "rb") as f:
       requests_results = pickle.load(f)
    scenario_result = ScenarioResult("commonsenseqa",0,"gpt2",requests_results)
    metrics = ShotMetrics(scenario_result)
    hidden_states = HiddenStates(metrics.hidden_states)
    letter_overlap = hidden_states.layer_overlap_label("answered_letter")
    assert len(letter_overlap["last"]) == 13 and len(letter_overlap["sum"]) == 13
    assert letter_overlap["last"][0].shape == (4,4), "The shape of the hidden states last sum are different"
    with open("letter_overlap.pkl", "wb") as f:
        pickle.dump(letter_overlap,f)
    #print(f'{letter_overlap=}')
    #return metrics
    
def test_letter_overlap():
    with open("tests/assets/unit/Llama-2-7b-commonsenseqa-0/0/scenario_results.pkl", "rb") as f:
       scenario_result = pickle.load(f)
    letter_overlap: LetterOverlap = LetterOverlap([scenario_result])
    overlaps = letter_overlap.compute_overlap()  
    print(f'{overlaps}')
    assert overlaps[overlaps["method"]=="last"].iloc[2]["overlap"].shape[0] == 33, "The number of layers is not correct"
    

def test_subject_overlap():
    subjects = os.listdir("tests/assets/unit/subjects")
    scenario_results = []
    for s in subjects:
        with open(Path(f'tests/assets/unit/subjects/{s}/0/scenario_results.pkl'),"rb") as f:
            scenario_results.append(pickle.load(f))
    subject_overlap: SubjectOverlap = SubjectOverlap(scenario_results)
    overlaps=subject_overlap.compute_overlap()  
    print(f'{overlaps}')
    assert overlaps[overlaps["method"]=="last"].iloc[2]["overlap"].shape[0] == 33, "The number of layers is not correct"

def test_chat_base_overlap():
    with open("tests/assets/unit/Llama-2-7b-commonsenseqa-0/0/scenario_results.pkl", "rb") as f:
       scenario_result_base = pickle.load(f)
    with open("tests/assets/unit/Llama-2-7b-chat-commonsenseqa-0/0/scenario_results.pkl", "rb") as f:
       scenario_result_chat = pickle.load(f)
    bf_overlap = BaseFinetuneOverlap([scenario_result_base,scenario_result_chat])
    overlaps = bf_overlap.compute_overlap()
    print(f'{overlaps}')

if __name__ == "__main__":
     test_scenario()
#     test_generation()
#     test_prediction()
#     test_make_request()
#     test_prediction()
#     test_tokenizer()
#     test_basic_metrics()
#     test_intrinsic_dim()
#     test_letter_overlap()
#     test_subject_overlap()
#     test_chat_base_overlap()
    