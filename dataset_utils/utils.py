import os
from dataclasses import dataclass, field
from typing import Any, Dict, List
from datasets import load_dataset
import random
from tqdm import tqdm
import string
from abc import ABC, abstractmethod
import random



dataset = load_dataset("cais/mmlu", 'all', split = 'test')


output_mapping = [letter 
                    for letter in string.ascii_uppercase[:len(dataset[0]["choices"])]]

dataset['subject']
output_mapping

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
    question: str
    prompt: str
    letter_gold: str
    token_gold: int = None


@dataclass
class Scenario():
    dataset: str
    num_few_shots: int
    model_name: str
    requests_instances: List[RequestInstance] = field(default_factory=list)
    output_mapping: List[str] = field(default_factory=dict)

        
class ScenarioBuilder(ABC):
    def __init__(self, num_few_shots, model_name, number_of_instances = -1):
        self.num_few_shots = num_few_shots
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
    



# #prompt builder
# class MMLU_ScenarioBuilder(ScenarioBuilder):
#     #num_few_shots = # shots
#     def __init__(self, subject, num_few_shots, model_name, number_of_instances = -1):
#         super().__init__(num_few_shots, model_name, number_of_instances)
#         self.dataset = f'mmlu:{subject}'
#         self.subject = subject

#     def retrieve_dataset(self):
#         try:
#             dataset = load_dataset("cais/mmlu", self.subject, trust_remote_code=True)
#         except Exception as e:
#             error : str = f'Huggingface error {e}, peraphs you didnt specify the subdataset?'
#             raise ValueError(error)
#         return dataset
    

#     def construct_request_instance(self) -> List[RequestInstance]:
#         """
#         Construct the request instances for the scenario
#         """
#         dataset = self.retrieve_dataset()
#         output_mapping = [letter 
#                           for letter in string.ascii_uppercase[:len(dataset["test"][0]["choices"])]]
#         if _TMP:
#             dataset_dev = load_dataset('cais/mmlu','all',trust_remote_code=True)
#         else:
#             dataset_dev = dataset
        
#         ri = []

#         def construct_question(row,shot=False):
#             prompt = f'Question: {row["question"]}\n'
#             for n, choice in enumerate(row["choices"]):
#                 prompt += f'{output_mapping[n]}. {choice}\n'
#             prompt += f'Answer: {output_mapping[row["answer"]]}\n\n' if shot else  f'Answer:' 
#             return prompt 
        
#         #init dataset
#         dataset_test = dataset["test"].select(range(self.number_of_instances)) \
#                                   if self.number_of_instances != -1 else dataset["test"]
        
#         #prompt contruction
#         for row in tqdm(dataset_test, desc="Constructing Prompts"):
#             prompt = f'The following are multiple choice questions (with answers) about {subject_retriever(self.dataset)}.\n\n'

#             for i in range(self.num_few_shots):
#                 if _TMP:
#                   # otherwise all the shot would belong to the same subject
#                   index = i+i*10
#                   random_row = dataset_dev["dev"][index]
#                 else:
#                   random_row = dataset_dev["dev"][i]
#                 prompt += construct_question(random_row,shot=True)
#             question = construct_question(row) 
#             prompt += question
#             ri.append(RequestInstance(question, prompt, output_mapping[row["answer"]]))
#         return ri, output_mapping
    

#     def build(self) -> Scenario:
#         """
#         Build the scenario
#         """
#         self.requests_instances, output_mapping = self.construct_request_instance()
#         print(f'Example of prompt: {self.requests_instances[0].prompt}')
  
#         return Scenario(self.dataset, 
#                         self.num_few_shots, 
#                         self.model_name, 
#                         self.requests_instances,
#                         output_mapping 
#                         )



# def generate_question(question, choices, answer=None, shot=False):

#     choices = "".join(
#         [
#             f"{key}. {choice.strip()}\n"
#             for key, choice in zip(string.ascii_uppercase, choices)
#         ]
#     )
#     if shot:
#         return f"{question.strip()}\n{choices}Answer: {answer}"
#     return f"{question.strip()}\n{choices}Answer: "


# def generate_prompt(tokenizer, batch, mmlu_dev, num_few_shots=0):
    
#     for question, choices, subject in zip(batch["question"], batch["choices"]):
#         input_premise = []
#         for i in range(num_few_shots):
#             input_premise+=[generate_question(question, choices, answer)
#             input_premise+=[mmlu_dev[subject][i]["question"]

#         input_premise += [
#             generate_question(question, choices, subject)
#         ]
#     output_response = [
#         f"{string.ascii_uppercase[answer]}{tokenizer.eos_token}"
#         for answer in batch["answer"]
#     ]

#     return dict(intput=input_premise, output=output_response)





#prompt builder
class MMLU_ScenarioBuilder(ScenarioBuilder):
    #num_few_shots = # shots
    def __init__(self, subject, num_few_shots, model_name, number_of_instances = -1):
        super().__init__(num_few_shots, model_name, number_of_instances)
        self.subject = subject
        self.dataset = 'mmlu'
        if subject is not None:
            self.dataset= f'mmlu:{self.subject}'

    def construct_request_instance(self) -> List[RequestInstance]:
        """
        Construct the request instances for the scenario
        """
        if self.subject is not None:
            dataset = load_dataset("cais/mmlu", self.subject, split = 'test', trust_remote_code=True)
        else:
            dataset = load_dataset('cais/mmlu','all', split = 'test', trust_remote_code=True)
        
        few_shot_dataset = None
        if self.num_few_shots > 0 and self.num_few_shots <=5:
            few_shot_dataset = load_dataset("cais/mmlu", 'all', split = 'dev', trust_remote_code=True)
        elif self.num_few_shots > 5:
            few_shot_dataset = load_dataset("cais/mmlu", 'all', split = 'dev+val', trust_remote_code=True)

        output_mapping = ['A', 'B', 'C', 'D']
        prompts = []

        def construct_question(row,is_shot=False):
            #added strip
            prompt = f'{row["question"].strip()}\n'
            for i, choice in enumerate(row["choices"]):
                #added strip
                prompt += f'{output_mapping[i]}. {choice.strip()}\n'
            #added space to final answers
            prompt += f'Answer: {output_mapping[row["answer"]]}\n\n' if is_shot else  f'Answer: ' 
            return prompt 
        
        #prompt contruction
        def construct_prompt(example, few_shot_dataset, num_few_shots):
            full_prompt = f'The following are multiple choice questions (with answers) about {row['subject']}.\n\n'
            for i in range(num_few_shots):
                shot = few_shot_dataset[example['subject']][i]
                full_prompt += construct_question(shot,is_shot=True)
            question = construct_question(example) 
            full_prompt += question
            prompts.append(RequestInstance(question, full_prompt, output_mapping[row["answer"]]))
            return prompts

        dataset.map(construct_prompt, )

        for row in tqdm(dataset, desc="Constructing Prompts"):
            full_prompt = f'The following are multiple choice questions (with answers) about {row['subject']}.\n\n'

            for i in range(self.num_few_shots):
                shot = few_shot_dataset[row['subject']][i]
                full_prompt += construct_question(shot,is_shot=True)

            question = construct_question(row) 
            
            full_prompt += question

            prompts.append(RequestInstance(question, full_prompt, output_mapping[row["answer"]]))
        
        return prompts, output_mapping
    

    def build(self) -> Scenario:
        """
        Build the scenario
        """
        self.requests_instances, output_mapping = self.construct_request_instance()
        print(f'Example of prompt: {self.requests_instances[0].prompt}')
  
        return Scenario(self.dataset, 
                        self.num_few_shots, 
                        self.model_name, 
                        self.requests_instances,
                        output_mapping 
                        )







class MMLU_Gib_ScenarioBuilder(ScenarioBuilder):
    def __init__(self, subject, num_few_shots, model_name, number_of_instances = -1, gib="gib"):
        super().__init__(num_few_shots, model_name, number_of_instances)
        self.dataset = f'mmlu:{subject}'
        self.subject = subject
        self.gib = gib

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
            file_name = 'dataset_utils/gibberish.txt' if self.gib=="gib" else 'dataset_utils/dummy.txt'
            with open(file_name) as f:
                random_row = ''
                lines = f.readlines()      
                for i in range(self.num_few_shots):
                    prompt += lines[i][:-1].replace('\\n','\n')
            question = construct_question(row) 
            prompt += question
            ri.append(RequestInstance(question, prompt, output_mapping[row["answer"]]))
        return ri, output_mapping
    
    def build(self) -> Scenario:
        """
        Build the scenario
        """
        self.requests_instances, output_mapping = self.construct_request_instance()
        print(f'Example of prompt: {self.requests_instances[0].prompt}')
  
        return Scenario(self.dataset,
                        self.num_few_shots, 
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
            for i in range(self.num_few_shots):
                random_row = dataset["validation"][i]
                prompt += self.construct_question(random_row,shot=True)
            question = self.construct_question(row)
            prompt += question
            ri.append(RequestInstance(question, prompt, row["answerKey"]))
        return ri, dataset["test"][0]["choices"]["label"]
    def build(self) -> Scenario:
        """
        Build the scenario
        """
        self.requests_instances, output_mapping = self.construct_request_instance()
        print(f'Example of prompt: {self.requests_instances[0].prompt}')

        return Scenario("openbookqa", 
                        self.num_few_shots, 
                        self.model_name, 
                        self.requests_instances,
                        output_mapping 
                        )

class CommonsenseQA_ScenarioBuilder(ScenarioBuilder):
    def __init__(self, num_few_shots, model_name, number_of_instances = -1, answer= "letter"):
        super().__init__(num_few_shots, model_name, number_of_instances)
        self.answer = answer
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
        if self.answer == "letter":
            prompt += f'Answer: {row["answerKey"]}\n\n' if shot else  f'Answer:' 
        elif self.answer == "ref":
            prompt += f'Answer: {row["choices"]["text"][row["choices"]["label"].index(row["answerKey"])]}\n\n' if shot else  f'Answer:' 
        elif self.answer == "wrong":
            valid_answers = {"A","B","C","D"}-set(row["answerKey"])
            wrong_answer = random.choice(list(valid_answers))
            prompt += f'Answer: {wrong_answer}\n\n' if shot else  f'Answer:'
        
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
            for i in range(self.num_few_shots):
                random_row = dataset["train"][i]
                prompt += self.construct_question(random_row,shot=True)
            question = self.construct_question(row)
            prompt += question
            ri.append(RequestInstance(question, prompt, row["answerKey"]))
        return ri, dataset["validation"][0]["choices"]["label"]
    def build(self) -> Scenario:
        """
        Build the scenario
        """
        self.requests_instances, output_mapping = self.construct_request_instance()
        
        print(f'Example of prompt: {self.requests_instances[0].prompt}')
        return Scenario("commonsenseqa", 
                        self.num_few_shots, 
                        self.model_name, 
                        self.requests_instances,
                        output_mapping 
                        )
class ScenarioAdapter:
    def __init__(self, dataset_folder, dataset, num_few_shots, model_name, number_of_instances = -1):
        self.dataset_folder = dataset_folder
        self.dataset = dataset
        self.num_few_shots = num_few_shots
        self.model_name = model_name
        self.number_of_instances = number_of_instances

    def build(self):
        
        if self.dataset.split(":")[0] == "mmlu":
            subject = self.dataset.split(":")[1]
            if self.dataset_folder == "mmlu_gibberish":
              print("PROMPT CONFIGURATION: GIBBERISH")
              return MMLU_Gib_ScenarioBuilder(subject, self.num_few_shots, self.model_name, self.number_of_instances,"gib").build()
            elif self.dataset_folder == "mmlu_dummy":
              print("PROMPT CONFIGURATION: DUMMY")
              return MMLU_Gib_ScenarioBuilder(subject, self.num_few_shots, self.model_name, self.number_of_instances,"dummy").build()
            else:
              return MMLU_ScenarioBuilder(subject, self.num_few_shots, self.model_name, self.number_of_instances).build()
        
        elif self.dataset.split(":")[0] == "openbookqa":
            return OpenbookQA_ScenarioBuilder( self.num_few_shots, self.model_name, self.number_of_instances).build()
        
        elif self.dataset.split(":")[0] == "commonsenseqa":
            if self.dataset_folder == "commonsenseqa_ref": 
              return CommonsenseQA_ScenarioBuilder(self.num_few_shots, self.model_name, self.number_of_instances,"ref").build()
            elif self.dataset_folder == "commonsenseqa_letter": 
              return CommonsenseQA_ScenarioBuilder(self.num_few_shots, self.model_name, self.number_of_instances, "letter").build()
            elif self.dataset_folder == "commonsenseqa_wrong":
              return CommonsenseQA_ScenarioBuilder(self.num_few_shots, self.model_name, self.number_of_instances, "wrong").build()

        else:
            raise ValueError("Unknown dataset")
    



