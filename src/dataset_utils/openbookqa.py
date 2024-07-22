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

class OpenbookQA_ScenarioBuilder(ScenarioBuilder):
    def __init__(self, shots, model_name, number_of_instances = -1):
        super().__init__(shots, model_name, number_of_instances)
        self.dataset = "openbookqa"        
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
            for i in range(self.shots):
                random_row = dataset["validation"][i]
                prompt += self.construct_question(random_row,shot=True)
            question = self.construct_question(row)
            prompt += question
            ri.append(RequestInstance(question, prompt, row["answerKey"]))
        return ri, dataset["test"][0]["choices"]["label"]
    
class OpenbookQA_Wrong_ScenarioBuilder(OpenbookQA_ScenarioBuilder):
    def construct_question(self, row, shot=False):
        prompt = f'Question: {row["question_stem"]}\n'
        for letter, choice in zip(row["choices"]["label"],row["choices"]["text"]):
            prompt += f'{letter}. {choice}\n'
        
        valid_answers = ["A", "B", "C", "D"] 
        index_answ = valid_answers.index(row["answerKey"])
        wrong_answer = valid_answers[index_answ+1] if index_answ < 3 else valid_answers[0]
        prompt += f'Answer: {wrong_answer}\n\n' if shot else  f'Answer:'
                
        return prompt
