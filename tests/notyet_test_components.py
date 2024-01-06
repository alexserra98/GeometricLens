import sys
import os
#append to Pyhton path the path to the parent directory
sys.path.append("/home/alexserra98/helm-suite")

from inference_id.metrics.metrics import *

import pickle
import logging
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

working_path = "/home/alexserra98/helm-suite/inference_id"
results_path = os.path.join(working_path, "tests/assets/components/overlap_short_result")
models = os.listdir(results_path)
for model in models:
    datasets = os.listdir(os.path.join(results_path, model))
    results_per_train_instances = {n:[] for n in range(1)}
    for dataset in datasets:
        instances_id = []
        instances_metrics = []
        max_train_instances = os.listdir(os.path.join(results_path, model, dataset))
        logging.info(f'Collecting results for {model}, {dataset}')
        for max_train_instance in max_train_instances:
            instance_path = Path(os.path.join(results_path, model, dataset, max_train_instance))
            with open(instance_path /"scenario_results.pkl", 'rb') as f:
                    scenario_results = pickle.load(f)
            results_per_train_instances[int(max_train_instance)].append(scenario_results)
    overlaps = {n:[] for n in range(6)}
    for n in tqdm(range(1), desc="Computing overlap..."):
        subject_overlap: SubjectOverlap = SubjectOverlap(results_per_train_instances[n])
        overlaps[n]=subject_overlap.compute_overlap()    

    logging.info(f'Saving results...')
    with open(Path(results_path,model,"overlaps.pkl"),"wb") as f:
        pickle.dump(overlaps,f)


    
