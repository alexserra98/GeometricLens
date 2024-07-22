import sys
sys.path.append("/u/dssc/zenocosini/helm_suite/MCQA_Benchmark")

from metrics.query import DataFrameQuery
from common.utils import *
from metrics.utils import *

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
#!export OMP_NUM_THREADS=1
_PATH = Path("/orfeo/scratch/dssc/zenocosini/mmlu_result/")
_PATH_RESULT = Path("/orfeo/cephfs/home/dssc/zenocosini/helm_suite/MCQA_Benchmark/metrics/cluster_cartoon")

class Pipeline():
    def __init__(self):
        self.df = None
    
    def compute(self):	
        z = 1.68
        models = ["lama-2-7b",
                  "lama-2-7b-chat",
                  "lama-2-13b-chat",
                  "lama-2-70b",
                  "lama-2-70b-chat",
                  "lama-3-8b",
                  "lama-3-70b",
                  "mistral-1-7b",
                  "mistral-1-7b-chat",
                 ]
        models = ["llama-3-8b",
                  "llama-3-8b-chat",
                  "llama-3-8b-ft"
                 ]
        models = ["llama-3-70b",
                  "llama-3-70b-chat",
                  "llama-2-70b",
                  "llama-2-70b-chat",
                 ]
        rows = []
        for model in models:
            for shot in [0,5]:
                if "70" in model and shot == 5 and "chat" not in model:
                    shot = 4
                elif "70" in model and "chat" in model and shot == 5:
                    continue
                dict_query = {"method":"last",
                              "model_name": model,
                              "train_instances": shot}
                hidden_states, logits, hidden_states_stat, repeated_indices = self.preprocess(dict_query)
                for layer in [69, 73]:
                    print(f'Iteration with model:{model}, shot:{shot}, layer:{layer}')
                    if layer == 33:
                        base_repr = logits
                    else:
                        base_repr = hidden_states[:,layer,:]
                    
                    clusters_assignement,data = self.compute_cluster_assignment(base_repr,z)

                    # filter small cluster
                    cluster_to_remove = np.where(np.bincount(clusters_assignement)<30)
                    bad_indices = np.where(np.isin(clusters_assignement, cluster_to_remove[0]))[0]
                    clusters_assignement = np.delete(clusters_assignement, bad_indices)
                    filtered_hidden_states_stat = hidden_states_stat.drop(bad_indices)
                    
                    # compute centroids
                    distance_matrix = self.compute_centroids(data, cluster_to_remove)
                    
                    # compute attributes
                    cluster_counts, most_represented = self.attributes_df(clusters_assignement, 
                                                                          filtered_hidden_states_stat)
                    
                    row = [model, 
                           shot, 
                           layer, 
                           most_represented, 
                           distance_matrix, 
                           cluster_counts]
                    rows.append(row)
                checkpoint_path = Path(_PATH_RESULT / f"df_result/checkpoint.pkl")
                
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(rows, f)
            
        df = pd.DataFrame(rows, columns = ["model", 
                                           "shot", 
                                           "layer", 
                                           "most_represented", 
                                           "distance_matrix", 
                                           "cluster_counts"])
        self.df = df
        self.df.to_pickle(_PATH_RESULT / f"df_result/df_llama_3_8b_letter.pkl")
    def preprocess(self, query):
        hidden_states, logits, hidden_states_stat = self.retrieve(query)
        
        filtered_hidden_states, filtered_logits, filter_hidden_states_stat, repeated_indices = filter_indentical_rows(hidden_states, logits, hidden_states_stat)
        return filtered_hidden_states, \
               filtered_logits, \
               filter_hidden_states_stat, \
               repeated_indices
    def retrieve(self, query):
        tsm = TensorStorageManager(instances_per_sub = 200)
        query = DataFrameQuery(query)
        hidden_states,logits, hidden_states_stat = tsm.retrieve_tensor(query, "npy")
        return hidden_states,logits, hidden_states_stat                
    
    def compute_cluster_assignment(self, base_repr, z):
        data = Data(coordinates=base_repr, maxk=100)
        ids, _, _ = data.return_id_scaling_gride(range_max=100)
        data.set_id(ids[3])
        data.compute_density_kNN(k=16)
        clusters_assignment = data.compute_clustering_ADP(Z=z, halo=False)
        return clusters_assignment, data
    
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
        # self.plot()
        
    def attributes_df(self,  
                      clusters_assignement, 
                      hidden_states_stat):
        # subjects = hidden_states_stat["dataset"]
        # map_sub_ind = {class_name: n for n,class_name in enumerate(set(subjects))}
        # map_ind_sub = {n: class_name for n,class_name in enumerate(set(subjects))}
        
        # label_per_row = np.array([map_sub_ind[e] for e in subjects])
        # cluster_sub = np.concatenate([np.expand_dims(label_per_row,axis =0),np.expand_dims(clusters_assignement,axis =0)], axis = 0)
        # cluster_sub = cluster_sub.T

        # # Binary division of subjects
        # stem,not_stem = self.stem(hidden_states_stat)
        # sub_binary = list(map(lambda r: r in stem,list(map(lambda r: map_ind_sub[r],cluster_sub[:,0]))))
        # df = pd.DataFrame({"type": sub_binary, "cluster": cluster_sub[:,1]})
        # df["type"] = df["type"].astype(int)

        # Macrocategories division for subject
        # natural_science, formal_science, humanities, social_science = self.macro_category(hidden_states_stat)
        
        # def label_cluster(r):
        #     if r in natural_science:
        #         return "natural_science"
        #     elif r in formal_science:
        #         return "formal_science"
        #     elif r in humanities:
        #         return "humanities"
        #     elif r in social_science:
        #         return "social_science"
        # sub_category = list(map(lambda r: label_cluster(r),list(map(lambda r: map_ind_sub[r],cluster_sub[:,0]))))
        # df = pd.DataFrame({"type": sub_category, "cluster": cluster_sub[:,1]})

        # Division by letters
        predictions = hidden_states_stat["only_ref_pred"]
        df = pd.DataFrame({"type": predictions, "cluster": clusters_assignement})
        def compute_fraction_type(series):
            type_counts = series.value_counts(normalize=True)
            return type_counts.to_dict()
                
        # Macrocategories division of subject
        # most_represented = df.groupby('cluster')['type'].agg(compute_fraction_type).reset_index() 

        # Binary division of subject
        # most_represented = df.groupby('cluster')['type'].agg(lambda x: x.mean()).reset_index()

        # Division by letters
        most_represented = df.groupby('cluster')['type'].agg(compute_fraction_type).reset_index() 
        most_represented.rename(columns={"type":"percentage"}, inplace=True)
        cluster_counts = Counter(clusters_assignement)
        
        return cluster_counts, most_represented
    
    def stem(self, 
             hidden_states_stat):
        subjects = hidden_states_stat["dataset"]
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
        
        return stem, not_stem
    
    def macro_category(self,
                       hidden_states_stat):
        
        subjects = hidden_states_stat["dataset"]
        subjects_list = np.unique(subjects)
        sub_natural_science =["anatomy", "astronomy", "clinical_knowledge","college_biology",
                          "college_chemistry", "college_physics", "conceptual_physics", "high_school_biology",
                          "high_school_chemistry", "high_school_physics", "professional_medicine",
                          "college_medicine", "human_aging", "medical_genetics", "nutrition","virology"]
        sub_formal_science = ["abstract_algebra", "college_computer_science", "college_mathematics",
                          "computer_security", "electrical_engineering", "elementary_mathematics",
                          "high_school_computer_science", "high_school_mathematics", "high_school_statistics", "machine_learning"]
        sub_humanities  = ["high_school_european_history", "formal_logic", "high_school_us_history", "high_school_world_history",
            	       "international_law", "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios","philosophy",
            	       "prehistory", "professional_law", "world_religions"]
        sub_social_science = ["business_ethics", "high_school_government_and_politics", "high_school_macroeconomics",
            		      "high_school_geography", "high_school_microeconomics", "econometrics", "high_school_psychology",
                          "human_sexuality", "professional_psychology", "public_relations", "security_studies",
                          "sociology", "us_foreign_policy", "management", "marketing", "professional_accounting", "global_facts", "miscellaneous"]

        # Determine macrocategory
        is_natural_science = np.isin(subjects_list, sub_natural_science)
        is_formal_science = np.isin(subjects_list, sub_formal_science)
        is_humanities = np.isin(subjects_list, sub_humanities)
        is_social_science = np.isin(subjects_list, sub_social_science)
        
        natural_science = subjects_list[is_natural_science]
        formal_science = subjects_list[is_formal_science]
        humanities = subjects_list[is_humanities]
        social_science = subjects_list[is_social_science]
        # print(f"{natural_science}, {formal_science}, {humanities}, {social_science}")
        return natural_science, formal_science, humanities, social_science

    def answered_letter(self,
                       hidden_states_stat):
        letters = hidden_states_stat["predictions"]
        letters_list = np.unique(letter)
        # Determine macrocategory
        is_a = np.isin(letters_list, ["A"])
        is_b = np.isin(letters_list, ["B"])
        is_c = np.isin(letters_list, ["C"])
        is_d = np.isin(letters_list, ["D"])
        
        a = subjects_list[is_natural_science]
        b = subjects_list[is_formal_science]
        c = subjects_list[is_humanities]
        d = subjects_list[is_social_science]
        # print(f"{natural_science}, {formal_science}, {humanities}, {social_science}")
        return a,b,c,d

        
    
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

def filter_indentical_rows(hidden_states,logits, hidden_states_stat):
    # we pick only 7 because the repeated indices are supposed to be equal at each layer
    repeated_indices = find_identical_rows(hidden_states[:,7,:])
    indices_to_exclude = [index[0] for index in repeated_indices]
    filtered_hidden_states = np.delete(hidden_states, indices_to_exclude, axis=0)
    filtered_logits = np.delete(logits, indices_to_exclude, axis=0)
    filtered_hidden_states_stat = hidden_states_stat.drop(indices_to_exclude)
    filtered_hidden_states_stat.reset_index(inplace=True)
    return filtered_hidden_states, filtered_logits, filtered_hidden_states_stat, repeated_indices