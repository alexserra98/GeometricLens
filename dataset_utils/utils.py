import os
from dataclasses import dataclass, field
from typing import Any, Dict, List
from datasets import load_dataset
import random
from tqdm import tqdm
import string
from abc import ABC, abstractmethod

_TMP = True

_KNOWN_DATASET_ALIASES: Dict[str, str] = {
    "mmlu": "Stevross/mmlu",
    "commonsenseqa": "tau/commonsense_qa"
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
    

        
class ScenarioBuilder(ABC):
    def __init__(self, train_instances, model_name, number_of_instances = -1):
        self.train_instances = train_instances
        self.model_name = model_name
        self.number_of_instances = number_of_instances
        
    @abstractmethod
    def retrieve_dataset(self):
        raise NotImplementedError
    
    @abstractmethod
    def construct_request_instance(self) -> List[RequestInstance]:
        raise NotImplementedError
    
    @abstractmethod
    def build(self) -> Scenario:
        raise NotImplementedError
    

class MMLU_ScenarioBuilder(ScenarioBuilder):
    def __init__(self, subject, train_instances, model_name, number_of_instances = -1):
        super().__init__(train_instances, model_name, number_of_instances)
        self.dataset = f'mmlu:{subject}'
        self.subject = subject
    def retrieve_dataset(self):
        try:
            dataset = load_dataset("cais/mmlu", self.subject, trust_remote_code=True)
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
        if _TMP:
            dataset_dev = load_dataset('cais/mmlu','all',trust_remote_code=True)
        else:
            dataset_dev = dataset
        
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
                random_row = dataset_dev["dev"][i]
                prompt += construct_question(random_row,shot=True)
            prompt += construct_question(row)
            ri.append(RequestInstance(prompt, output_mapping[row["answer"]]))
        return ri, output_mapping
    def build(self) -> Scenario:
        """
        Build the scenario
        """
        self.requests_instances, output_mapping = self.construct_request_instance()
        print(f'Example of prompt: {self.request_instances[0].prompt}')
  
        return Scenario(self.dataset, 
                        self.train_instances, 
                        self.model_name, 
                        self.requests_instances,
                        output_mapping 
                        )
        
class OpenbookQA_ScenarioBuilder(ScenarioBuilder):
    def retrieve_dataset(self):
        try:
            dataset = load_dataset("openbookqa")
        except Exception as e:
            error : str = f'Huggingface error {e}, peraphs you didnt specify the subdataset?'
            raise ValueError(error)
        return dataset
    def construct_question(self,row,shot=False):
        prompt = f'Question: {row["question_stem"]}\n'
        for letter, choice in zip(row["choices"]["label"],row["choices"]["text"]):
            prompt += f'{letter}. {choice}\n'
        #prompt += f'Answer: {row["choices"]["text"][row["choices"]["label"].index(row["answerKey"])]}\n\n' if shot else  f'Answer:' 
        prompt += f'Answer: {row["answerKey"]}\n\n' if shot else  f'Answer:' 

        return prompt 
    def construct_request_instance(self) -> List[RequestInstance]:
        """
        Construct the request instances for the scenario
        """
        dataset = self.retrieve_dataset()

        
        ri = []

        dataset_test = dataset["test"].select(range(self.number_of_instances)) \
                                  if self.number_of_instances != -1 else dataset["test"]
        for row in tqdm(dataset_test, desc="Constructing Prompts"):
            prompt = f'The following are multiple choice questions (with answers) about science.\n\n'
            for i in range(self.train_instances):
                random_row = dataset["validation"][i]
                prompt += self.construct_question(random_row,shot=True)
            prompt += self.construct_question(row)
            ri.append(RequestInstance(prompt, row["answerKey"]))
        return ri, dataset["test"][0]["choices"]["label"]
    def build(self) -> Scenario:
        """
        Build the scenario
        """
        self.requests_instances, output_mapping = self.construct_request_instance()
  
        return Scenario("openbookqa", 
                        self.train_instances, 
                        self.model_name, 
                        self.requests_instances,
                        output_mapping 
                        )

class CommonsenseQA_ScenarioBuilder(ScenarioBuilder):
    def retrieve_dataset(self):
        try:
            dataset = load_dataset("tau/commonsense_qa")
        except Exception as e:
            error : str = f'Huggingface error {e}'
            raise ValueError(error)
        return dataset
    def construct_question(self,row,shot=False):
        prompt = f'Question: {row["question"]}\n'
        for letter, choice in zip(row["choices"]["label"],row["choices"]["text"]):
            prompt += f'{letter}. {choice}\n'
        #prompt += f'Answer: {row["choices"]["text"][row["choices"]["label"].index(row["answerKey"])]}\n\n' if shot else  f'Answer:' 
        prompt += f'Answer: {row["answerKey"]}\n\n' if shot else  f'Answer:' 
        return prompt 
    def construct_request_instance(self) -> List[RequestInstance]:
        """
        Construct the request instances for the scenario
        """
        dataset = self.retrieve_dataset()

        
        ri = []

        dataset_test = dataset["validation"].select(range(self.number_of_instances)) \
                                  if self.number_of_instances != -1 else dataset["validation"]
        for row in tqdm(dataset_test, desc="Constructing Prompts"):
            prompt = f'The following are multiple choice questions (with answers).\n\n'
            for i in range(self.train_instances):
                random_row = dataset["train"][i]
                prompt += self.construct_question(random_row,shot=True)
            prompt += self.construct_question(row)
            ri.append(RequestInstance(prompt, row["answerKey"]))
        return ri, dataset["validation"][0]["choices"]["label"]
    def build(self) -> Scenario:
        """
        Build the scenario
        """
        self.requests_instances, output_mapping = self.construct_request_instance()
        
        print(f'Example of prompt: {self.request_instances[0].prompt}')
        return Scenario("commonsenseqa", 
                        self.train_instances, 
                        self.model_name, 
                        self.requests_instances,
                        output_mapping 
                        )
class ScenarioAdapter:
    def __init__(self, dataset, train_instances, model_name, number_of_instances = -1):
        self.dataset = dataset
        self.train_instances = train_instances
        self.model_name = model_name
        self.number_of_instances = number_of_instances
    def build(self):
        if self.dataset.split(":")[0] == "mmlu":
            subject = self.dataset.split(":")[1]
            return MMLU_ScenarioBuilder(subject, self.train_instances, self.model_name, self.number_of_instances).build()
        elif self.dataset.split(":")[0] == "openbookqa":
            return OpenbookQA_ScenarioBuilder( self.train_instances, self.model_name, self.number_of_instances).build()
        elif self.dataset.split(":")[0] == "commonsenseqa":
            return CommonsenseQA_ScenarioBuilder(self.train_instances, self.model_name, self.number_of_instances).build()
        else:
            raise ValueError("Unknown dataset")
    
