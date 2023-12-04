import torch
import dadapy
import numpy
from inference_id.metrics.metrics import *
import os
import pickle
import logging
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
working_path = "/orfeo/scratch/dssc/zenocosini" 
results_path = os.path.join(working_path, "inference_result")
models = os.listdir(results_path)
for model in models:
    datasets = os.listdir(os.path.join(results_path, model))
    for dataset in datasets:
        instances_id = []
        instances_metrics = []
        max_train_instances = os.listdir(os.path.join(results_path, model, dataset))
        for max_train_instance in tqdm(max_train_instances, desc="Computing per instance result"):
            logging.info(f'Computing results for {model}, {dataset}, {max_train_instance}')
            instance_path = Path(os.path.join(results_path, model, dataset, max_train_instance))
            with open(instance_path /"requests_results.pkl", 'rb') as f:
                    requests_results = pickle.load(f)
            shot_metrics = ShotMetrics(requests_results,dataset,max_train_instance)
            final_metrics = shot_metrics.evaluate()
            with open(instance_path /"final_metrics.pkl", 'wb') as f:
                    pickle.dump(final_metrics,f)



    
