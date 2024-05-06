from collections import defaultdict
import torch
from dadapy._cython import cython_overlap as c_ov
from pairwise_distances import compute_distances
import sys
import numpy as np
import pickle

import os
import argparse
from collections import Counter
from dadapy import data
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--finetuned_mode",
        type=str,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./results",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
    )
    parser.add_argument("--num_shots", type=int, default=None)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    return args


args = parse_args()
os.makedirs(args.results_path, exist_ok=True)

base_dir = "/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens/repo/results"


print(f"processing model: {args.model}")
print(f"processing epochs: {args.epochs}")
print(f"processing daset: {args.eval_dataset}")
print(f"num_shots: {args.num_shots}")
sys.stdout.flush()

if args.eval_dataset == "test":
    mask_dir = (
        "/u/area/ddoimo/ddoimo/finetuning_llm/open-instruct/open_instruct/my_utils"
    )
    dataset_mask = np.load(f"{mask_dir}/test_mask_100.npy")


# ****************************************************************************************


cpt = ""
if args.ckpt is not None:
    ckpts = [args.ckpt]
    cpt = "_cpt{args.ckpt}"
else:
    ckpts = np.arange(args.epochs + 1)

overlaps = defaultdict(list)
clusters = defaultdict(list)
intrinsic_dim = defaultdict(list)

for epoch in ckpts[::-1]:
    # layer 0 is all overlapped
    for layer in range(1, 34):
        print(f"processing {args.num_shots} epoch {epoch} layer {layer}")
        sys.stdout.flush()

        # ************************************

        if args.num_shots is not None:
            base_path = f"{base_dir}/mmlu/{args.model}/{args.num_shots}shot"
            name = f"base_{args.num_shots}"

        else:
            base_path = f"{base_dir}/finetuned_{args.finetuned_mode}/evaluated_{args.eval_dataset}/{args.model}/{args.epochs}epochs/epoch_{epoch}"
            name = f"finetuned_{args.finetuned_mode}_eval_{args.eval_dataset}"

        base_repr = torch.load(f"{base_path}/l{layer}_target.pt")
        base_repr = base_repr.to(torch.float64).numpy()

        with open(f"{base_path}/statistics_target.pkl", "rb") as f:
            stats = pickle.load(f)
        subjects = np.array(stats["subjects"])

        # balance the test set if asked
        is_balanced = ""
        if dataset_mask is not None:
            is_balanced = "_balanced"
            base_repr = base_repr[dataset_mask]
            subjects = subjects[dataset_mask]
            # check that all the subjects have the same frequency
            frequences = Counter(subjects).values()
            assert len(np.unique(list(frequences))) == 1
            # check that the frequency is 100
            assert np.unique(list(frequences))[0] == 100, (
                np.unique(list(frequences))[0],
                frequences,
            )

        subjetcs = np.repeat(np.arange(len(subjects)), 100)
        # remove identical points
        base_unique, base_idx, base_inverse = np.unique(
            base_repr, axis=0, return_index=True, return_inverse=True
        )
        indices = np.sort(base_idx)

        base_repr = base_repr[indices]
        subjects = subjects[indices]

        # ***********************************************************************

        maxk = 300
        assert indices.shape[0] > maxk, (indices.shape[0], maxk)
        distances_base, dist_index_base, mus, _ = compute_distances(
            X=base_repr,
            n_neighbors=maxk + 1,
            n_jobs=1,
            working_memory=2048,
            range_scaling=maxk + 2,
            argsort=False,
        )

        for halo in [True, False]:
            is_halo = ""
            if halo:
                is_halo = "-halo"
            for z in [0.5, 1, 1.6, 2.1]:

                d = data.Data(distances=(distances_base, dist_index_base))
                ids, _, _ = d.return_id_scaling_gride(range_max=100)
                d.set_id(ids[3])
                intrinsic_dim[f"ids-ep{epoch}"].append(ids)
                d.compute_density_kNN(k=16)
                assignment = d.compute_clustering_ADP(Z=z, halo=halo)

                mask = np.ones(len(assignment), dtype=bool)
                if halo:
                    mask = assignment != -1

                clusters[f"ami-ep{epoch}-z{z}{is_halo}"].append(
                    adjusted_mutual_info_score(assignment[mask], subjects[mask])
                )

                clusters[f"ari-ep{epoch}-z{z}{is_halo}"].append(
                    adjusted_rand_score(assignment[mask], subjects[mask])
                )

        for k in [30, 100]:
            overlaps[f"ep{epoch}_k{k}"].append(
                return_label_overlap(
                    dist_indices=dist_index_base,
                    labels=subjects,
                    k=k,
                )
            )

    with open(
        f"{args.results_path}/overlap_subject_{args.model}_{name}_epoch{args.epochs}.pkl",
        "wb",
    ) as f:
        pickle.dump(overlaps, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"{args.results_path}/cluster_subjetcs_{args.model}_{name}_epoch{args.epochs}.pkl",
        "wb",
    ) as f:
        pickle.dump(clusters, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"{args.results_path}/ids_{args.model}_{name}_epoch{args.epochs}.pkl",
        "wb",
    ) as f:
        pickle.dump(intrinsic_dim, f, protocol=pickle.HIGHEST_PROTOCOL)
