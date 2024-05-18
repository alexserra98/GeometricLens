from collections import Counter
from dadapy._cython import cython_overlap as c_ov
import numpy as np
import warnings
import torch
import pickle
from pairwise_distances import compute_distances
import sys
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from dadapy import data

from scipy.stats import entropy
from collections import Counter


arr = np.array([1, 1, 3, 4])

Counter(arr).most_common()[0][1]


def return_data_overlap(indices_base, indices_other, k=30, subjects=None):

    assert indices_base.shape[0] == indices_other.shape[0]
    ndata = indices_base.shape[0]

    overlaps_full = c_ov._compute_data_overlap(
        ndata, k, indices_base.astype(int), indices_other.astype(int)
    )

    overlaps = np.mean(overlaps_full)
    if subjects is not None:
        overlaps = {}
        for subject in np.unique(subjects):
            mask = subject == subjects
            overlaps[subject] = np.mean(overlaps_full[mask])

    return overlaps


def _label_imbalance_helper(labels, k, class_fraction):
    if k is not None:
        max_k = k
        k_per_sample = np.array([k for _ in range(len(labels))])

    k_per_class = {}
    class_count = Counter(labels)
    # potentially overwrites k_per_sample
    if class_fraction is not None:
        for label, count in class_count.items():
            class_k = int(count * class_fraction)
            k_per_class[label] = class_k
            if class_k == 0:
                k_per_class[label] = 1
                warnings.warn(
                    f" max_k < 1 for label {label}. max_k set to 1.\
                    Consider increasing class_fraction.",
                    stacklevel=2,
                )
        max_k = max([k for k in k_per_class.values()])
        k_per_sample = np.array([k_per_class[label] for label in labels])

    class_weights = {label: 1 / count for label, count in class_count.items()}
    sample_weights = np.array([class_weights[label] for label in labels])

    return k_per_sample, sample_weights, max_k


def return_label_overlap(
    dist_indices,
    labels,
    k=None,
    avg=True,
    class_fraction=None,
    weighted=True,
):
    k_per_sample, sample_weights, max_k = _label_imbalance_helper(
        labels, k, class_fraction
    )
    # print(k_per_sample, sample_weights, max_k)
    assert len(labels) == dist_indices.shape[0]

    neighbor_index = dist_indices[:, 1 : max_k + 1]

    ground_truth_labels = np.repeat(np.array([labels]).T, repeats=max_k, axis=1)

    overlaps = np.equal(np.array(labels)[neighbor_index], ground_truth_labels)

    if class_fraction is not None:
        nearest_neighbor_rank = np.arange(max_k)[np.newaxis, :]
        # should this overlap entry be discarded?
        mask = nearest_neighbor_rank >= k_per_sample[:, np.newaxis]
        # mask out the entries to be discarded
        overlaps[mask] = False

    overlaps = overlaps.sum(axis=1) / k_per_sample
    if avg and weighted:
        overlaps = np.average(overlaps, weights=sample_weights)
    elif avg:
        overlaps = np.mean(overlaps)

    return overlaps


def analyze(
    base_path,
    layer,
    dataset_mask,
    clusters,
    intrinsic_dim,
    overlaps,
    spec,
):

    base_repr = torch.load(f"{base_path}/l{layer}_target.pt")
    base_repr = base_repr.to(torch.float64).numpy()

    with open(f"{base_path}/statistics_target.pkl", "rb") as f:
        stats = pickle.load(f)
    subjects = stats["subjects"]
    subjects_to_int = {sub: i for i, sub in enumerate(np.unique(subjects))}
    subj_label = np.array([subjects_to_int[sub] for sub in subjects])

    letters = stats["answers"]
    letters_to_int = {letter: i for i, letter in enumerate(np.unique(letters))}
    letter_label = np.array([letters_to_int[sub] for sub in letters])

    # balance the test set if asked
    is_balanced = ""
    if dataset_mask is not None:
        base_repr = base_repr[dataset_mask]
        subj_label = subj_label[dataset_mask]
        letter_label = letter_label[dataset_mask]

    # remove identical points
    base_unique, base_idx, base_inverse = np.unique(
        base_repr, axis=0, return_index=True, return_inverse=True
    )
    indices = np.sort(base_idx)
    base_repr = base_repr[indices]
    subj_label = subj_label[indices]
    letter_label = letter_label[indices]

    # ***********************************************************************

    maxk = 1000
    assert indices.shape[0] > maxk, (indices.shape[0], maxk)
    distances_base, dist_index_base, mus, _ = compute_distances(
        X=base_repr,
        n_neighbors=maxk + 1,
        n_jobs=1,
        working_memory=2048,
        range_scaling=2048,
        argsort=False,
    )

    for halo in [False]:
        is_halo = ""
        if halo:
            is_halo = "-halo"
        for z in [0, 0.5, 1, 1.6, 2.1]:

            d = data.Data(distances=(distances_base, dist_index_base))
            ids, _, _ = d.return_id_scaling_gride(range_max=100)
            d.set_id(ids[3])
            intrinsic_dim[f"ids-{spec}"].append(ids)
            d.compute_density_kNN(k=16)
            assignment = d.compute_clustering_ADP(Z=z, halo=halo)

            mask = np.ones(len(assignment), dtype=bool)
            if halo:
                mask = assignment != -1

            clusters[f"nclus-{spec}-z{z}{is_halo}"].append(d.N_clusters)
            population = []
            for clus in d.cluster_indices:
                population.append(len(clus))
            clusters[f"population-{spec}-z{z}{is_halo}"].append(population)

            clusters[f"subjects-ami-{spec}-z{z}{is_halo}"].append(
                adjusted_mutual_info_score(assignment[mask], subj_label[mask])
            )

            clusters[f"subjects-ari-{spec}-z{z}{is_halo}"].append(
                adjusted_rand_score(assignment[mask], subj_label[mask])
            )

            entropies = []
            most_commons = []
            for clus in d.cluster_indices:
                assert halo == False
                composition = subj_label[np.array(clus)]
                most_commons.append(Counter(composition).most_common()[0][1])
                h = entropy(composition)
                entropies.append(h)

            clusters[f"subj-entropies-{spec}-z{z}{is_halo}"].append(entropies)
            clusters[f"subj-most_common-{spec}-z{z}{is_halo}"].append(most_commons)

            # *********************************************************************

            clusters[f"letters-ami-{spec}-z{z}{is_halo}"].append(
                adjusted_mutual_info_score(assignment[mask], letter_label[mask])
            )

            clusters[f"letters-ari-{spec}-z{z}{is_halo}"].append(
                adjusted_rand_score(assignment[mask], letter_label[mask])
            )

            entropies = []
            most_commons = []
            for clus in d.cluster_indices:
                assert halo == False
                composition = letter_label[np.array(clus)]
                most_commons.append(Counter(composition).most_common()[0][1])
                h = entropy(composition)
                entropies.append(h)

            clusters[f"letters-entropies-{spec}-z{z}{is_halo}"].append(entropies)
            clusters[f"letters-most_common-{spec}-z{z}{is_halo}"].append(most_commons)
            # ************************************************************************************

    print(clusters[f"letters-ari-{spec}-z{z}{is_halo}"])
    sys.stdout.flush()

    for class_fraction in [0.3, 0.5]:
        overlaps[f"subjects-{spec}-{class_fraction}"].append(
            return_label_overlap(
                dist_indices=dist_index_base,
                labels=list(subj_label),
                class_fraction=class_fraction,
            )
        )
    for class_fraction in [0.03, 0.1, 0.3]:
        overlaps[f"letters-{spec}-{class_fraction}"].append(
            return_label_overlap(
                dist_indices=dist_index_base,
                labels=list(letter_label),
                class_fraction=class_fraction,
            )
        )
    return clusters, intrinsic_dim, overlaps
