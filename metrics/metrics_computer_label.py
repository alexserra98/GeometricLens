from inference_id.metrics.metrics import *
import os
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
import sys

label = sys.argv[1]
data_directory = sys.argv[2]

class OverlapClasses(Enum):
    letter = LetterOverlap
    subject = SubjectOverlap
assert label in list(OverlapClasses.__members__), "Label not supported"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
working_path = "/orfeo/scratch/dssc/zenocosini" 
results_path = os.path.join(working_path, data_directory)
models = os.listdir(results_path)
for model in models:
    datasets = os.listdir(os.path.join(results_path, model))
    results_per_train_instances = {n:[] for n in range(6)}
    datasets.sort()
    for dataset in datasets:
        instances_id = []
        instances_metrics = []
        max_train_instances = os.listdir(os.path.join(results_path, model, dataset))
        logging.info(f'Collecting results for {model}, {dataset}')
        for max_train_instance in max_train_instances:
            instance_path = Path(os.path.join(results_path, model, dataset, max_train_instance))
            with open(instance_path /"scenario_results.pkl", 'rb') as f:
                    scenario_results = pickle.load(f)
            #import pdb; pdb.set_trace()
            results_per_train_instances[int(max_train_instance)].append(scenario_results)
    overlaps = {n:[] for n in range(6)}
    for n in tqdm(range(6), desc="Computing overlap..."):
        overlap_instance = OverlapClasses[label].value(results_per_train_instances[n])
        overlaps[n] = overlap_instance.compute_overlap()

    logging.info(f'Saving results...')
    file_name = f"{label}_overlaps.pkl"
    with open(Path(results_path, model, file_name), "wb") as f:
        pickle.dump(overlaps, f)




    
