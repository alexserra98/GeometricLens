import os
from pathlib import Path
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List
from datasets import load_dataset
import random
from tqdm import tqdm

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
    
class ScenarioBuilder():
    def __init__(self, dataset, train_instances, model_name, number_of_instances = -1):
        self.dataset = dataset
        self.train_instances = train_instances
        self.model_name = model_name
        self.number_of_instances = number_of_instances
    
    def retrieve_dataset(self):
        dataset = load_dataset(*self.dataset.split(":"))
        return dataset
    
    def construct_request_instance(self):
        dataset = self.retrieve_dataset()
        mapping = ['A','B','C','D','E']
        ri = []
        def construct_question(row,shot=False):
            prompt = f'Question: {row["question"]}\n'
            for n, choice in enumerate(row["choices"]):
                prompt += f'{mapping[n]}. {choice}\n'
            prompt += f'Answer: {mapping[row["answer"]]}\n\n' if shot else  f'Answer:' 
            return prompt 
        
        for row in tqdm(dataset["test"], desc="Constructing Prompts"):
            prompt = f'The following are multiple choice questions (with answers) about {row["subject"]}.\n\n'
            for i in range(self.train_instances):
                random_row = random.choice(dataset["auxiliary_train"])
                prompt += construct_question(random_row,shot=True)
            prompt += construct_question(row)
            ri.append(RequestInstance(prompt, mapping[row["answer"]]))
        return ri
    def build(self):
        self.requests_instances = self.construct_request_instance()
        self.requests_instances = self.requests_instances[:self.number_of_instances] \
                                  if self.number_of_instances != -1 else self.requests_instances
        return Scenario(self.dataset, 
                        self.train_instances, 
                        self.model_name, 
                        self.requests_instances, 
                        )
        