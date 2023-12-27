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
results_path = os.path.join(working_path, "inference_result-_subject")
models = os.listdir(results_path)
for model in models:
    datasets = os.listdir(os.path.join(results_path, model))
    results_per_train_instances = {n:[] for n in range(6)}
    for dataset in datasets:
        instances_id = []
        instances_metrics = []
        max_train_instances = os.listdir(os.path.join(results_path, model, dataset))
        for max_train_instance in tqdm(max_train_instances, desc="Computing per instance result"):
            logging.info(f'Computing results for {model}, {dataset}, {max_train_instance}')
            instance_path = Path(os.path.join(results_path, model, dataset, max_train_instance))
            with open(instance_path /"scenario_results.pkl", 'rb') as f:
                    scenario_results = pickle.load(f)
            results_per_train_instances[max_train_instance].append(scenario_results)
    overlaps = {n:[] for n in range(6)}
    for n in range(6):
        subject_overlap: SubjectOverlap() = SubjectOverlap(results_per_train_instances[n])
        overlaps[n]=subject_overlap.compute_overlap()    
    with open(Path(results_path,model,"overlaps.pkl"),"wb") as f:
        pickle.dump(overlaps,f)


    
