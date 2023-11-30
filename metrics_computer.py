import torch
import dadapy
import numpy
from hidden_states_geometry.geometry import Geometry, RunGeometry
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
        if "opt" in model and "commonsense" in dataset:
            continue

        instances_id = []
        instances_metrics = []
        max_train_instances = os.listdir(os.path.join(results_path, model, dataset))
        for max_train_instance in tqdm(max_train_instances, desc="Computing per instance result"):
            logging.info(f'Computing results for {model}, {dataset}, {max_train_instance}')
            if "hidden_states.pkl" in os.listdir(os.path.join(results_path, model, dataset, max_train_instance)):
                with open(os.path.join(results_path, model, dataset, max_train_instance,"hidden_states.pkl"), 'rb') as f:
                    raw_hidden_states = pickle.load(f)
                with open(os.path.join(results_path, model, dataset, max_train_instance,"metrics.pkl"), 'rb') as f:
                    raw_metrics = pickle.load(f)
                hidden_geometry=RunGeometry(raw_hidden_states,raw_metrics)
                id_path = os.path.join(results_path, model, dataset, max_train_instance, "intrinsic_dim.pkl")
                metrics_path = os.path.join(results_path, model, dataset, max_train_instance, "agg_metrics.pkl")
                instances_id.append(hidden_geometry.instances_id)
                instances_metrics.append(hidden_geometry.metrics)
                #save intrinsic dimension
                with open(id_path, 'wb') as f:  
                    pickle.dump(hidden_geometry.instances_id,f)
                #save metrics
                with open(metrics_path, 'wb') as f:  
                    pickle.dump(hidden_geometry.metrics,f)
        # Save difference accross run metrics
        #geometry = Geometry(instances_id, instances_metrics)
        #id_acc = geometry.id_acc
        #with open(os.path.join(results_path, model, dataset, "id_acc.pkl"), 'wb') as f:
        #    pickle.dump(id_acc,f)
    

    
