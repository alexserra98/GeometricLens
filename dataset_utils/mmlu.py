import os
from dataclasses import dataclass, field
from typing import Any, Dict, List
from datasets import load_dataset
import random
from tqdm import tqdm
import string
from abc import ABC, abstractmethod
import random
from .scenario_builder import RequestInstance, Scenario, ScenarioBuilder, _KNOWN_DATASET_ALIASES
import pickle

def subject_retriever(dataset):
    if len(dataset.split(":"))==2:
        return dataset.split(":")[1]
    else:
        return dataset

class MMLU_ScenarioBuilder(ScenarioBuilder):
    def __init__(self, subject, shots, model_name, number_of_instances = -1):
        super().__init__(shots, model_name, number_of_instances)
        self.dataset = f'mmlu:{subject}'
        self.subject = subject
    
    def construct_request_instance(self) -> List[RequestInstance]:
        """
        Construct the request instances for the scenario
        """
        dataset = self.retrieve_dataset()
        output_mapping = [letter 
                          for letter in string.ascii_uppercase[:len(dataset["test"][0]["choices"])]]

        ri = []
        dataset_test = dataset["test"].select(range(self.number_of_instances)) \
                                  if self.number_of_instances != -1 else dataset["test"]
                                
        ri = self.construct_prompt(dataset_test, dataset["dev"])
        return ri, output_mapping
    
    def construct_question(self,row,shot=False):
        prompt = f'Question: {row["question"]}\n'
        for n, choice in enumerate(row["choices"]):
            prompt += f'{string.ascii_uppercase[n]}. {choice}\n'
        prompt += f'Answer: {string.ascii_uppercase[row["answer"]]}\n\n' if shot else  f'Answer:' 
        return prompt
    
    def construct_prompt(self,dataset_test, dataset_dev):
        ri = []
        for row in tqdm(dataset_test, desc="Constructing Prompts"):
            prompt = f'The following are multiple choice questions (with answers) about {subject_retriever(self.dataset)}.\n\n'
            for i in range(self.shots):
                random_row = dataset_dev[i]
                prompt += self.construct_question(random_row,shot=True)
            question = self.construct_question(row) 
            prompt += question
            ri.append(RequestInstance(question, prompt, string.ascii_uppercase[row["answer"]]))
        return ri

class MMLU_Corruption_ScenarioBuilder(MMLU_ScenarioBuilder):
    def __init__(self, subject, shots, model_name, number_of_instances = -1, type_of_corruption="gibberish"):
        super().__init__(subject, shots, model_name, number_of_instances)
        self.type_of_corruption = type_of_corruption
    
    def construct_prompt(self, dataset_test, dataset_dev):
        file_name = 'dataset_utils/asset/gibberish_list.pkl' if self.type_of_corruption=="gib" else 'dataset_utils/asset/dummy_list.pkl'
        with open(file_name, 'rb') as f:
            dataset_dev = pickle.load(f)    
        return super().construct_prompt(dataset_test, dataset_dev)
    
class MMLU_Invariant_ScenarioBuilder(MMLU_ScenarioBuilder):
    
    def construct_prompt(self, dataset_test, dataset_dev):
        dataset_dev = load_dataset('cais/mmlu','all',trust_remote_code=True)
        dataset_dev = dataset_dev["dev"].select([0,25,50,100,125])
        return super().construct_prompt(dataset_test, dataset_dev)

class MMLU_Shuffled_Subject_ScenarioBuilder(MMLU_ScenarioBuilder):
    
    def construct_prompt(self, dataset_test, dataset_dev):       
        with open('dataset_utils/asset/subjects.txt', 'r') as f:
            subjects = f.readlines()
        subjects = set([subject.split(",")[0].split(":")[1][:-1] for subject in subjects])
        subjects -= set([self.subject])
        subject = random.choice(list(subjects))
        
        dataset_dev = load_dataset(_KNOWN_DATASET_ALIASES["mmlu"],subject,trust_remote_code=True)["dev"]
 
        return super().construct_prompt(dataset_test, dataset_dev)