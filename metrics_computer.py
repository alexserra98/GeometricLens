from hidden_states_geometry.geometry import Geometry, RunGeometry
import os
import pickle
working_path = os.getcwd()
results_path = os.path.join(working_path, "results")
models = os.listdir(results_path)
for model in models:
    datasets = os.listdir(os.path.join(results_path, model))
    for dataset in datasets:
        instances_id = []
        instances_metrics = []
        max_train_instances = os.listdir(os.path.join(results_path, model, dataset))
        for max_train_instance in max_train_instances:
            runs = os.listdir(os.path.join(results_path, model, dataset, max_train_instance))
            for run in runs:
                if "hidden_states.pkl" in os.listdir(os.path.join(results_path, model, dataset, max_train_instance, run)):
                    with open(os.path.join(results_path, model, dataset, max_train_instance, run, "hidden_states.pkl"), 'rb') as f:
                        raw_hidden_states = pickle.load(f)
                    with open(os.path.join(results_path, model, dataset, max_train_instance, run, "metrics.pkl"), 'rb') as f:
                        raw_metrics = pickle.load(f)
                    hidden_geometry=RunGeometry(raw_hidden_states,raw_metrics)
                    id_path = os.path.join(results_path, model, dataset, max_train_instance, run, "intrinsic_dim.pkl")
                    metrics_path = os.path.join(results_path, model, dataset, max_train_instance, run, "metrics.pkl")
                    instances_id.append(hidden_geometry.instances_id)
                    instances_metrics.append(hidden_geometry.metrics)
                    #save intrinsic dimension
                    with open(id_path, 'wb') as f:  
                        pickle.dump(hidden_geometry.instances_id,f)
                    #save metrics
                    with open(metrics_path, 'wb') as f:  
                        pickle.dump(hidden_geometry.metrics,f)
        # Save difference accross run metrics
        geometry = Geometry(instances_id, instances_metrics)
        id_acc = geometry.id_acc
        with open(os.path.join(results_path, model, dataset, "id_acc.pkl"), 'wb') as f:
            pickle.dump(id_acc,f)
    

    
