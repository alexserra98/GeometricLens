import os
from helm.benchmark.hidden_geometry.geometry import RunGeometry, InstanceHiddenSates
from pathlib import Path
import json
import pickle
import torch
from tqdm import tqdm
import transformers

working_dir = Path(os.getcwd())
version_path = os.path.join(working_dir, 'runs/v7/')
def ensure_dir(directory):
    device = "cpu"


# with open("relevant_raw_request.pkl","rb") as f:
#     request_config = pickle.load(f)
request_config = {'temperature': 1e-07, 'num_return_sequences': 1, 'max_new_tokens': 1, 'top_p': 1, 'output_hidden_states': True}
with open("scenario_state.pkl","rb") as f:
    scenario_state = pickle.load(f)
request_config["do_sample"] = True
request_config["return_dict_in_generate"] = True
request_config["output_scores"] = True 
model_kwargs["output_hidden_states"] = True
model_kwargs["device_map"]="auto"
model_kwargs["return_dict"]=True
#model_kwargs["cache_dir"]="/orfeo/scratch/dssc/zenocosini/"
model_name = "gpt2"

print(f'Importing model...\n')
def ensure_dir(directory):
#import pdb; pdb.set_trace()
instances_hiddenstates = []
prompts = prompts[:10]
correct = 0
for request_state in tqdm(scenario_state.request_states, desc="Generating"):
    prompt = scenario_state.request_states[0].request.prompt
    encoded_input = tokenizer(prompt,return_tensors="pt", return_token_type_ids=False).to(
    device
    )
	@@ -71,12 +77,24 @@ def ensure_dir(directory):

    # generation
    output = model.generate(**encoded_input, **request_config)
    answer = tokenizer.decode(output.sequences[:,-1]).strip()
    reference_index = request_state.output_mapping.get(answer, "incorrect")
    if reference_index != "incorrect":
        result = list(filter(lambda a: a.output.text == reference_index, request_state.instance.references))
        if result and result[0].tags and result[0].tags[0] == "correct":
            correct += 1


#     # getting hidden states
#     hs = torch.cat(output.hidden_states[0][-len_tokens_question:]).detach().cpu().numpy()
#     hidden_states = {"last": copy.deepcopy(hs[:,-1,:]), "sum":reduce(copy.deepcopy(hs[:,-len_tokens_question:,:]), "l s d -> l d", "mean")}
#     instance_hiddenstate = InstanceHiddenSates(0, hidden_states)
#     instances_hiddenstates.append(instance_hiddenstate)

# accuracy = correct/len(scenario_state.request_states)   
# json_dict = [{'name': {'name':'exact_match', 'split':'test'},'mean': accuracy}]
# with open(os.path.join(run_path, "stats.json",), "w") as f:
#     json.dump(json_dict, f) 
# # Write hidden states of the run
# hidden_geometry = RunGeometry(instaces_hiddenstates=instances_hiddenstates)
# # Write ID of the instance
# print(f"Writing ID of the instances to {run_path}intrinsic_dim.pkl")
# path = os.path.join(run_path, "intrinsic_dim.pkl")
# with open(path, 'wb') as f:  
#     pickle.dump(hidden_geometry.instances_id,f)
# # Write nearest neighbours matrices of the run
# print(f"Writing nearest neighbours matrices of the run to {run_path}nearest_neigh.pkl")
# path = os.path.join(run_path, "nearest_neigh.pkl")
# print("Warning: I am setting the number of nearest neighbours to 80/100 of instances")
# with open(path, 'wb') as f:  
#     pickle.dump(hidden_geometry.dict_nn,f) 