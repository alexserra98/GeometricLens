import os
from dataclasses import dataclass, field
from typing import Any, Dict, List
from datasets import load_dataset
import random
from tqdm import tqdm
import string
from abc import ABC, abstractmethod
import random
from .scenario_builder import RequestInstance, Scenario, ScenarioBuilder

class CommonsenseQA_ScenarioBuilder(ScenarioBuilder):
    def __init__(self, shots, model_name, number_of_instances = -1, answer= "letter"):
        super().__init__(shots, model_name, number_of_instances)
        self.answer = answer

    def construct_question(self,row,shot=False):
        prompt = f'Question: {row["question"]}\n'
        for letter, choice in zip(row["choices"]["label"],row["choices"]["text"]):
            prompt += f'{letter}. {choice}\n'
    
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
            for i in range(self.shots):
                random_row = dataset["train"][i]
                prompt += self.construct_question(random_row,shot=True)
            question = self.construct_question(row)
            prompt += question
            ri.append(RequestInstance(question, prompt, row["answerKey"]))
        return ri, dataset["validation"][0]["choices"]["label"]
    
class CommonsenseQA_Ref_ScenarioBuilder(CommonsenseQA_ScenarioBuilder):
    def construct_question(self, row, shot=False):
        prompt = f'Question: {row["question"]}\n'
        for letter, choice in zip(row["choices"]["label"],row["choices"]["text"]):
            prompt += f'{letter}. {choice}\n'
        prompt += f'Answer: {row["choices"]["text"][row["choices"]["label"].index(row["answerKey"])]}\n\n' if shot else  f'Answer:' 
                
        return prompt
    
class CommonsenseQA_Wrong_ScenarioBuilder(CommonsenseQA_ScenarioBuilder):
    def construct_question(self, row, shot=False):
        prompt = f'Question: {row["question"]}\n'
        
        for letter, choice in zip(row["choices"]["label"],row["choices"]["text"]):
            prompt += f'{letter}. {choice}\n'

        valid_answers = {"A","B","C","D"}-set(row["answerKey"])
        wrong_answer = random.choice(list(valid_answers))
        prompt += f'Answer: {wrong_answer}\n\n' if shot else  f'Answer:'
        
        return prompt 