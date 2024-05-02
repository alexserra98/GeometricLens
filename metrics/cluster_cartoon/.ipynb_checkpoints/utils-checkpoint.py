import sys
sys.path.append("/u/dssc/zenocosini/helm_suite/MCQA_Benchmark")
from metrics.utils import *
from metrics.query import DataFrameQuery
from common.utils import *

#from sklearn.feature_selection import mutual_info_regression MISSIN?
from dadapy.data import Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import KernelPCA
from sklearn.metrics import pairwise_distances
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from pathlib import Path
from collections import Counter

_PATH = Path("/orfeo/scratch/dssc/zenocosini/mmlu_result/")

class Pipeline():
    def __init__(self):
        self.df = None
    
    def preprocess(self, query):
        hidden_states,_,hidden_states_stat = self.retrieve(query)
        filtered_hidden_states, repeated_indices = filter_indentical_rows(hidden_states)
        return filtered_hidden_states, hidden_states_stat, repeated_indices
        

            
    def compute(self):
        z = 0.05
        models = ["meta-llama/Llama-2-7b-hf",
                  "meta-llama/Llama-2-13b-hf",
                  "meta-llama/Llama-2-70b-hf",
                  "meta-llama/Llama-3-8b-hf",
                  "meta-llama/Llama-3-70b-hf",]
        rows = []
        for model in models:
            for shot in [0,5]:
                dict_query = {"method":"last",
                              "model_name": model,
                              "train_instances": shot}
                hidden_states, hidden_states_stat, repeated_indices = self.preprocess(dict_query)
                for layer in [3,7,20,33]:
                    print(f'Iteration with model:{model}, shot:{shot}, layer:{layer}')
                    # compute clustering
                    data = Data(hidden_states[:,layer,:])
                    data.compute_distances(maxk=1000)
                    clusters_assignement = data.compute_clustering_ADP(Z=z)
                    
                    # filter small cluster
                    cluster_to_remove = np.where(np.bincount(clusters_assignement)<20)
                    bad_indices = np.where(np.isin(clusters_assignement, cluster_to_remove[0]))[0]
                    clusters_assignement = np.delete(clusters_assignement, bad_indices)
                    label_per_row = np.delete(label_per_row, bad_indices)
                    
                    
                    # compute centroids
                    distance_matrix = self.compute_centroids(data, cluster_to_remove)
                    
                    # compute attributes
                    cluster_counts, most_represented = self.attributes_df(label_per_row, 
                                                                          clusters_assignement, 
                                                                          hidden_states_stat, 
                                                                          repeated_indices)
                    
                    row = [model, shot, layer, most_represented, distance_matrix, cluster_counts]
                    rows.append(row)
        
        df = pd.DataFrame(rows, columns = ["model", 
                                           "shot", 
                                           "layer", 
                                           "most_represented", 
                                           "distance_matrix", 
                                           "cluster_counts"])
        self.df = df
                    
    def plot(self):
        for index, row in self.df.iterrows():
            model = row["model"]
            shot = row["shot"]
            layer = row["layer"]
            most_represented = row["most_represented"]
            distance_matrix = row["distance_matrix"]
            cluster_counts = row["cluster_counts"]    
            clusters, counts = zip(*sorted(cluster_counts.items()))
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
            positions = mds.fit_transform(distance_matrix)
            fig, ax = plt.subplots()
            norm = Normalize(vmin=0, vmax=1)
            cmap = plt.cm.coolwarm  
            scalar_map = ScalarMappable(norm=norm, cmap=cmap)
            cluster_colors = scalar_map.to_rgba(most_represented["percentage"])
            for pos, count, cluster, color in zip(positions, counts, clusters,cluster_colors ):
                ax.scatter(pos[0], pos[1], s= count, color=color, label=color, edgecolors='black', linewidth=1)
            ax.set_title('Cluster Visualization')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            plt.colorbar(scalar_map, ax=ax, label='Percentage of STEM subjects')
            plt.title(f"Model: {model}, Shot: {shot}, Layer: {layer}") 
            plt.show()
            plt.savefig(_PATH / f"model_{model}_shot_{shot}_layer_{layer}.png")
        
    def retrieve(self, query):
        tsm = TensorStorageManager()
        query = DataFrameQuery(query)
        hidden_states,_, hidden_states_stat = tsm.retrieve_tensor(query, "npy")
        return hidden_states, hidden_states_stat
        
    def attributes_df(self, 
                      label_per_row, 
                      clusters_assignement, 
                      hidden_states_stat, 
                      repeated_indices):
        cluster_sub = np.concatenate([np.expand_dims(label_per_row,axis =0),np.expand_dims(clusters_assignement,axis =0)], axis = 0)
        cluster_sub = cluster_sub.T
        
        sub_binary = self.stem(hidden_states_stat, 
                               repeated_indices,
                               cluster_sub)
        
        df = pd.DataFrame({"type": sub_binary, "cluster": cluster_sub[:,1]})
        df["type"] = df["type"].astype(int)
        most_represented = df.groupby('cluster')['type'].agg(lambda x: x.mean()).reset_index()
        most_represented.rename(columns={"type":"percentage"}, inplace=True)
        
        cluster_counts = Counter(cluster_sub[:, 1])
        
        return cluster_counts, most_represented
    
    def stem(self, 
             hidden_states_stat, 
             repeated_indices,
             cluster_sub):
        subjects = hidden_states_stat["datasets"]
        subjects = np.delete(subjects, repeated_indices)
        subjects_list = np.unique(subjects)
        stem_subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry',
                 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics',
                 'computer_security', 'conceptual_physics', 'electrical_engineering', 'high_school_biology',
                 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics',
                 'high_school_physics', 'high_school_statistics', 'machine_learning', 'medical_genetics',
                 'professional_medicine', 'virology', 'elementary_mathematics']

        # Determine STEM and non-STEM
        is_stem = np.isin(subjects_list, stem_subjects)
        stem = subjects_list[is_stem]
        not_stem = subjects_list[~is_stem]
        
        map_sub_ind = {class_name: n for n,class_name in enumerate(set(subjects))}
        map_ind_sub = {n: class_name for n,class_name in enumerate(set(subjects))}
        sub_binary = list(map(lambda r: r in stem,list(map(lambda r: map_ind_sub[r],cluster_sub[:,0]))))
        return sub_binary
    
    def compute_centroids(self, data, cluster_to_remove):
        # compute centroids
        indices_per_cluster = data.cluster_indices
        centroids = list(map(lambda r: np.mean(data.X[r],axis=0), indices_per_cluster))
        centroids = list(map(lambda r: np.mean(data.X[r],axis=0), indices_per_cluster))
        centroids = np.stack(centroids)
        centroids = np.delete(centroids,cluster_to_remove, axis=0)

        distance_matrix = pairwise_distances(centroids, metric='l2')
        
        return distance_matrix
            
    def run(self):
        self.compute()
        self.plot()
        
    
def find_identical_rows(matrix):
    # View the rows as a structured array to handle them as tuples
    dtype = [('row', matrix.dtype, matrix.shape[1])]
    structured_array = matrix.view(dtype)
    
    # Find unique rows and their indices
    _, inverse_indices = np.unique(structured_array, return_inverse=True)
    
    # Find where each row appears for the first time and their counts
    unique_rows, counts = np.unique(inverse_indices, return_counts=True)
    
    # Filter out unique rows, keeping only duplicates
    repeated_indices = [np.where(inverse_indices == i)[0] for i in unique_rows[counts > 1]]
    
    return repeated_indices

def filter_indentical_rows(hidden_states):
    repeated_indices = find_identical_rows(hidden_states)
    filtered_hidden_states = np.delete(hidden_states, np.concatenate(repeated_indices))
    return filtered_hidden_states, repeated_indices