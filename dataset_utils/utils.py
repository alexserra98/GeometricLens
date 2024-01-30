import os
from pathlib import Path
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List
from datasets import load_dataset
import random
from tqdm import tqdm
import string

_KNOWN_DATASET_ALIASES: Dict[str, str] = {
    "mmlu": "Stevross/mmlu"
}
def map_aliases(dataset):
    if len(dataset.split(":"))==2:
        dataset = f'{_KNOWN_DATASET_ALIASES.get(dataset.split(":")[0],dataset.split(":")[0])}:{dataset.split(":")[1]}'
    return dataset
def subject_retriever(dataset):
    if len(dataset.split(":"))==2:
        return dataset.split(":")[1]
    else:
        return dataset

@dataclass
class RequestInstance():
    prompt: str
    letter_gold: str
    token_gold: int = None


@dataclass
class Scenario():
    dataset: str
    train_instances: int
    model_name: str
    requests_instances: List[RequestInstance] = field(default_factory=list)
    output_mapping: List[str] = field(default_factory=dict)
    
class ScenarioBuilder():
    def __init__(self, dataset, train_instances, model_name, number_of_instances = -1):
        self.dataset = dataset
        self.train_instances = train_instances
        self.model_name = model_name
        self.number_of_instances = number_of_instances
    
    def retrieve_dataset(self):
        dataset_name  = map_aliases(self.dataset)
        try:
            dataset = load_dataset(*dataset_name.split(":"))
        except Exception as e:
            error : str = f'Huggingface error {e}, peraphs you didnt specify the subdataset?'
            raise ValueError(error)
        return dataset
    
    def construct_request_instance(self) -> List[RequestInstance]:
        """
        Construct the request instances for the scenario
        """
        dataset = self.retrieve_dataset()
        output_mapping = [letter 
                          for letter in string.ascii_uppercase[:len(dataset["test"][0]["choices"])]]
        
        ri = []
        def construct_question(row,shot=False):
            prompt = f'Question: {row["question"]}\n'
            for n, choice in enumerate(row["choices"]):
                prompt += f'{output_mapping[n]}. {choice}\n'
            prompt += f'Answer: {output_mapping[row["answer"]]}\n\n' if shot else  f'Answer:' 
            return prompt 
        dataset_test = dataset["test"].select(range(self.number_of_instances)) \
                                  if self.number_of_instances != -1 else dataset["test"]
        for row in tqdm(dataset_test, desc="Constructing Prompts"):
            prompt = f'The following are multiple choice questions (with answers) about {subject_retriever(self.dataset)}.\n\n'
            for i in range(self.train_instances):
                random_row = dataset["dev"][i]
                prompt += construct_question(random_row,shot=True)
            prompt += construct_question(row)
            ri.append(RequestInstance(prompt, output_mapping[row["answer"]]))
        return ri, output_mapping
    def build(self) -> Scenario:
        """
        Build the scenario
        """
        self.requests_instances, output_mapping = self.construct_request_instance()
  
        return Scenario(self.dataset, 
                        self.train_instances, 
                        self.model_name, 
                        self.requests_instances,
                        output_mapping 
                        )
        
