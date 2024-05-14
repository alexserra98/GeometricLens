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
    parser.add_argument("--finetuned_mode", type=str, default=None)
    parser.add_argument(
        "--mask_dir",
        type=str,
    )
    parser.add_argument(
        "--model_name",
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
    parser.add_argument("--samples_subject", type=int, default=200)
    parser.add_argument("--num_shots", type=int, default=None)
    parser.add_argument("--ckpt", type=int, default=None)
    args = parser.parse_args()
    return args


# **************************************************************************


args = parse_args()
os.makedirs(args.results_path, exist_ok=True)

base_dir = "/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens/repo/results"


print(f"processing model: {args.model_name}")
print(f"processing epochs: {args.epochs}")
print(f"processing daset: {args.eval_dataset}")
print(f"num_shots: {args.num_shots}")
sys.stdout.flush()

if args.eval_dataset == "test":
    mask_dir = args.mask_dir
    if args.samples_subject == 100:
        dataset_mask = np.load(f"{mask_dir}/test_mask_100.npy")
    if args.samples_subject == 200:
        dataset_mask = np.load(f"{mask_dir}/test_mask_200.npy")
    else:
        assert False, "wrong samples subject"


# *****************************************************************************


if args.finetuned_mode is not None:
    cpt = ""
    if args.ckpt is not None:
        ckpts = [args.ckpt]
        cpt = "_cpt{args.ckpt}"
    else:
        ckpts = np.arange(args.epochs + 1)
else:
    ckpts = [0]


if args.model_name == "llama3-8b":
    nlayers = 34
elif args.model_name == "llama2-13b":
    nlayers = 42
else:
    assert False, "wrong model name"


overlaps = defaultdict(list)
clusters = defaultdict(list)
intrinsic_dim = defaultdict(list)

for epoch in ckpts[::-1]:
    # layer 0 is all overlapped
    for layer in range(1, nlayers):
        if args.finetuned_mode is not None:
            print(f"processing {args.num_shots} epoch {epoch} layer {layer}")
            sys.stdout.flush()
        else:
            print(f"processing {args.num_shots} layer {layer}")
            sys.stdout.flush()
        # ************************************
        if args.finetune_mode is None:
            assert args.num_shots is not None
            # if args.question_sampled:
            #     base_path = f"{base_dir}/evaluated_test/questions_sampled/{args.model_name}/{args.num_shots}shot"
            #     name = f"base_question_sampled_{args.num_shots}"
            # else:
            base_path = f"{base_dir}/mmlu/{args.model_name}/{args.num_shots}shot"
            name = f"base_{args.num_shots}"

        else:
            base_path = f"{base_dir}/finetuned_{args.finetuned_mode}/evaluated_{args.eval_dataset}/{args.model_name}/{args.epochs}epochs/epoch_{epoch}"
            name = f"finetuned_{args.finetuned_mode}_eval_{args.eval_dataset}_epoch{args.epochs}"

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
            is_balanced = "_balanced"
            base_repr = base_repr[dataset_mask]
            subj_label = subj_label[dataset_mask]
            letter_label = letter_label[dataset_mask]
            # check that all the subjects have the same frequency
            # frequences = Counter(subjects).values()
            # assert len(np.unique(list(frequences))) == 1
            # # check that the frequency is 100
            # assert np.unique(list(frequences))[0] == 100, (
            #     np.unique(list(frequences))[0],
            #     frequences,
            # )

        # subjetcs = np.repeat(np.arange(len(subjects)), 100)
        # remove identical points
        base_unique, base_idx, base_inverse = np.unique(
            base_repr, axis=0, return_index=True, return_inverse=True
        )
        indices = np.sort(base_idx)
        base_repr = base_repr[indices]
        subj_label = subj_label[indices]
        letter_label = letter_label[indices]
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
                # intrinsic_dim[f"ids-ep{epoch}"].append(ids)
                d.compute_density_kNN(k=16)
                assignment = d.compute_clustering_ADP(Z=z, halo=halo)

                mask = np.ones(len(assignment), dtype=bool)
                if halo:
                    mask = assignment != -1

                clusters[f"subjects-ami-ep{epoch}-z{z}{is_halo}"].append(
                    adjusted_mutual_info_score(assignment[mask], subj_label[mask])
                )

                clusters[f"subjects-ari-ep{epoch}-z{z}{is_halo}"].append(
                    adjusted_rand_score(assignment[mask], subj_label[mask])
                )

                clusters[f"letters-ami-ep{epoch}-z{z}{is_halo}"].append(
                    adjusted_mutual_info_score(assignment[mask], letter_label[mask])
                )

                clusters[f"letters-ari-ep{epoch}-z{z}{is_halo}"].append(
                    adjusted_rand_score(assignment[mask], letter_label[mask])
                )

        for class_fraction in [0.3, 0.5]:
            overlaps[f"subjects-ep{epoch}_{class_fraction}"].append(
                return_label_overlap(
                    dist_indices=dist_index_base,
                    labels=subj_label,
                    class_fraction=class_fraction,
                )
            )

            overlaps[f"letters-ep{epoch}_{class_fraction}"].append(
                return_label_overlap(
                    dist_indices=dist_index_base,
                    labels=letter_label,
                    class_fraction=class_fraction,
                )
            )

    with open(
        f"{args.results_path}/overlap_subject_{args.model_name}_{name}.pkl",
        "wb",
    ) as f:
        pickle.dump(overlaps, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"{args.results_path}/cluster_subjetcs_{args.model_name}_{name}.pkl",
        "wb",
    ) as f:
        pickle.dump(clusters, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"{args.results_path}/ids_{args.model_name}_{name}.pkl",
        "wb",
    ) as f:
        pickle.dump(intrinsic_dim, f, protocol=pickle.HIGHEST_PROTOCOL)
