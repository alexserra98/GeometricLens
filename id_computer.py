from hidden_states_geometry.geometry import Geometry, RunGeometry
import os
import pickle
working_path = os.getcwd()
results_path = os.path.join(working_path, "results")
for folder_name, subfolders, filenames in os.walk(results_path):
    dataset = folder_name.split("/")[-1]
    print("Calculating ID " + dataset )
    if not subfolders:
        assert "hidden_states.pkl" in filenames, "There is no hidden_states.pkl file in the folder " + folder_name
        assert "metrics.pkl" in filenames, "There is no metrics.pkl file in the folder " + folder_name
        #load hidden states
        with open(os.path.join(folder_name, "hidden_states.pkl"), 'rb') as f:
            raw_hidden_states = pickle.load(f)
        with open(os.path.join(folder_name, "metrics.pkl"), 'rb') as f:
            raw_metrics = pickle.load(f)
        hidden_geometry=RunGeometry(raw_hidden_states,raw_metrics)
        id_path = os.path.join(folder_name, "intrinsic_dim.pkl")
        metrics_path = os.path.join(folder_name, "metrics.pkl")
        #save intrinsic dimension
        with open(id_path, 'wb') as f:  
            pickle.dump(hidden_geometry.instances_id,f)
        #save metrics
        with open(metrics_path, 'wb') as f:  
            pickle.dump(hidden_geometry.metrics,f)
    
