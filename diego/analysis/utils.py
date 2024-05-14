from collections import Counter
from dadapy._cython import cython_overlap as c_ov
import numpy as np
import warnings


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
