import os
from dataclasses import dataclass, field
from typing import Any, Dict, List
from datasets import load_dataset
import random
from tqdm import tqdm
import string
from abc import ABC, abstractmethod
import random


class ScenarioAdapter:
    def __init__(self, dataset_folder, dataset, shots, model_name, number_of_instances = -1):
        self.dataset_folder = dataset_folder
        self.dataset = dataset
        self.shots = shots
        self.model_name = model_name
        self.number_of_instances = number_of_instances
        
    def build(self):
        
        if self.dataset.split(":")[0] == "mmlu":
            subject = self.dataset.split(":")[1]
        
            if self.dataset_folder == "mmlu_gibberish":
              print("PROMPT CONFIGURATION: GIBBERISH")
              from src.dataset_utils.mmlu import MMLU_Corruption_ScenarioBuilder
              return MMLU_Corruption_ScenarioBuilder(subject, self.shots, self.model_name, self.number_of_instances,"gib").build()
        
            elif self.dataset_folder == "mmlu_dummy":
              print("PROMPT CONFIGURATION: DUMMY")
              from src.dataset_utils.mmlu import MMLU_Corruption_ScenarioBuilder
              return MMLU_Corruption_ScenarioBuilder(subject, self.shots, self.model_name, self.number_of_instances,"dummy").build()
            elif self.dataset_folder == "mmlu_invariant":
              from src.dataset_utils.mmlu import MMLU_Invariant_ScenarioBuilder
              return MMLU_Invariant_ScenarioBuilder(subject, self.shots, self.model_name, self.number_of_instances).build()
            elif self.dataset_folder == "mmlu_shuffled_sub":
              from src.dataset_utils.mmlu import MMLU_Shuffled_Subject_ScenarioBuilder
              return MMLU_Shuffled_Subject_ScenarioBuilder(subject, self.shots, self.model_name, self.number_of_instances).build()
            elif self.dataset_folder == "mmlu_train":
              from src.dataset_utils.mmlu import MMLU_train_ScenarioBuilder
              return MMLU_train_ScenarioBuilder(subject, self.shots, self.model_name, self.number_of_instances).build()
            else:
              from src.dataset_utils.mmlu import MMLU_ScenarioBuilder
              return MMLU_ScenarioBuilder(subject, self.shots, self.model_name, self.number_of_instances).build()
          
        elif self.dataset.split(":")[0] == "openbookqa":
            
            if self.dataset_folder == "openbookqa_wrong":
              from src.dataset_utils.openbookqa import OpenbookQA_Wrong_ScenarioBuilder
              return OpenbookQA_Wrong_ScenarioBuilder( self.shots, self.model_name, self.number_of_instances).build()
            else:
              from src.dataset_utils.openbookqa import OpenbookQA_ScenarioBuilder
              return OpenbookQA_ScenarioBuilder( self.shots, self.model_name, self.number_of_instances).build()
            
  
        elif self.dataset.split(":")[0] == "commonsenseqa":
        
            if self.dataset_folder == "commonsenseqa_ref":
              from src.dataset_utils.commonsenseqa import CommonsenseQA_Ref_ScenarioBuilder 
              return CommonsenseQA_Ref_ScenarioBuilder(self.shots, self.model_name, self.number_of_instances).build()

            elif self.dataset_folder == "commonsenseqa_letter": 
              from src.dataset_utils.commonsenseqa import CommonsenseQA_ScenarioBuilder
              return CommonsenseQA_ScenarioBuilder(self.shots, self.model_name, self.number_of_instances, "letter").build()
        
            elif self.dataset_folder == "commonsenseqa_wrong":
              from src.dataset_utils.commonsenseqa import CommonsenseQA_Wrong_ScenarioBuilder
              return CommonsenseQA_Wrong_ScenarioBuilder(self.shots, self.model_name, self.number_of_instances, "wrong").build()

        else:
            raise ValueError("Unknown dataset")
        
        
        
        
    # def retrieve_dataset(self):
    #     try:
    #         dataset = load_dataset("cais/mmlu", self.subject, trust_remote_code=True)
    #     except Exception as e:
    #         error : str = f'Huggingface error {e}, peraphs you didnt specify the subdataset?'
    #         raise ValueError(error)
    #     return dataset
    
