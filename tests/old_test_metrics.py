import torch
import dadapy
import numpy
import sklearn
from MCQA_Benchmark.metrics.metrics import Geometry, RunGeometry
import os
import pickle
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
working_path = "/orfeo/scratch/dssc/zenocosini" 
results_path = os.path.join(working_path, "inference_result")
models = os.listdir(results_path)
for model in models:
    if "13b" in model:
        continue
    datasets = os.listdir(os.path.join(results_path, model))
    for dataset in datasets:
        instances_id = []
        instances_metrics = []
        max_train_instances = os.listdir(os.path.join(results_path, model, dataset))
        for max_train_instance in tqdm(max_train_instances, desc="Computing per instance result"):
            logging.info(f'Computing results for {model}, {dataset}, {max_train_instance}')
            if "hidden_states.pkl" in os.listdir(os.path.join(results_path, model, dataset, max_train_instance)):
                with open(os.path.join(results_path, model, dataset, max_train_instance,"intrinsic_dim.pkl"), 'rb') as f:
                    intrinsic_dim= pickle.load(f)
                with open(os.path.join(results_path, model, dataset, max_train_instance,"agg_metrics.pkl"), 'rb') as f:
                    agg_metrics= pickle.load(f)
                id_path = os.path.join(results_path, model, dataset, max_train_instance, "intrinsic_dim.pkl")
                metrics_path = os.path.join(results_path, model, dataset, max_train_instance, "agg_metrics.pkl")
                instances_id.append(intrinsic_dim)
                instances_metrics.append(agg_metrics)
        # Save difference accross run metrics
        geometry = Geometry(instances_id, instances_metrics)
        id_acc = geometry.id_acc
        with open(os.path.join(results_path, model, dataset, "id_acc.pkl"), 'wb') as f:
            pickle.dump(id_acc,f)
    

    
