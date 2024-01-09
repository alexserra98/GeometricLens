import os
from pathlib import Path
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class RequestInstance():
    prompt: str
    letter_gold: str
    token_gold: int = None
    def __init__(self, request_state):
        # TODO not all dataset my have the word "Question" in the prompt
        self.prompt = request_state.request.prompt
        self.letter_gold = self.letter_gold_setter(request_state)
    #TODO use property
    def letter_gold_setter(self, request_state):
        index = [ref.tags == ['correct'] for ref in request_state.instance.references].index(True)
        letter_gold = list(request_state.output_mapping.keys())[index]
        return letter_gold

@dataclass
class Scenario():
    dataset: str
    train_instances: int
    model_name: str
    truncation: int = None
    requests_instances: List[RequestInstance] = field(default_factory=list)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    #TODO probably output_mapping not needed just use letter_answers 
    #Why am I using post init?
    def __post_init__(self):
        helm_requests_states = self.retrieve_requests_states()
        self.output_mapping = helm_requests_states[0].output_mapping
        self.letter_answers = [letter for letter in list(self.output_mapping.keys())]
        self.requests_instances = [RequestInstance(request_state) for request_state in helm_requests_states]
        self.requests_instances = self.requests_instances[:self.truncation] if self.truncation else self.requests_instances
        #self.tokens_answers = [tokenizer.encode(letter)[0] for letter in list(self.output_mapping.keys())]
    
    def retrieve_requests_states(self):
        if not (path := Path(os.getcwd(), "datasets_overlap_long")).exists():
            raise OSError("Invalid path try maybe you didn't launch the script from the inference_id folder")
        file = next(path.glob(f"*{self.dataset}*max_train_instances={self.train_instances}*"), None)
        if file is None:
            raise Exception("The dataset is not present in the datasets folder.")
        dataset_path = path / file / "request_states.pkl"
        with open(dataset_path, "rb") as f:
            helm_request_state = pickle.load(f) 
        return helm_request_state
