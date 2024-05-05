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
        type=str,
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
    )
    parser.add_argument(
        "--num_shots",
        type=int,
    )
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    return args


args = parse_args()
os.makedirs(args.results_path, exist_ok=True)

base_dir = "/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens/repo/results"


print(f"processing model: {args.model}")
print(f"processing epochs: {args.epochs}")
print(f"processing epoch_ckpt: {args.ckpt_epoch}")
print(f"processing daset: {args.eval_dataset}")
print(f"num_shots: {args.num_shots}")
sys.stdout.flush()

if args.eval_dataset == "test":
    mask_dir = (
        "/u/area/ddoimo/ddoimo/finetuning_llm/open-instruct/open_instruct/my_utils"
    )
    dataset_mask = np.load(f"{mask_dir}/test_mask_100.npy")


# ****************************************************************************************


if args.ckpt is not None:
    ckpts = [args.ckpt]
else:
    ckpts = np.arange(args.epochs + 1)

ov_repr = defaultdict(list)
cluster_comparison = defaultdict(list)

for epoch in ckpts:
    # layer 0 is all overlapped
    for layer in range(1, 34):
        print(f"processing {args.num_shots} epoch {epoch} layer {layer}")
        sys.stdout.flush()

        # ************************************

        pretrained_path = f"{base_dir}/mmlu/{args.model}/{args.num_shots}"
        base_repr = torch.load(f"{pretrained_path}/l{layer}_target.pt")
        base_repr = base_repr.to(torch.float64).numpy()

        finetuned_path = f"{base_dir}/finetuned_{args.finetuned_mode}/evaluated_{args.eval_dataset}/{args.model}/{args.epochs}epochs/epoch_{epoch}"
        finetuned_repr = torch.load(f"{finetuned_path}/l{layer}_target.pt")
        finetuned_repr = finetuned_repr.to(torch.float64).numpy()

        with open(f"{finetuned_path}/statistics_target.pkl", "rb") as f:
            stats = pickle.load(f)
        subjects = np.array(stats["subjects"])

        # balance the test set if asked
        is_balanced = ""
        if dataset_mask is not None:
            is_balanced = "_balanced"
            base_repr = base_repr[dataset_mask]
            finetuned_repr = finetuned_repr[dataset_mask]
            subjects = subjects[dataset_mask]
            # check that all the subjects have the same frequency
            frequences = Counter(subjects).values()
            assert len(np.unique(list(frequences))) == 1
            # check that the frequency is 100
            assert np.unique(list(frequences))[0] == 100, (
                np.unique(list(frequences))[0],
                frequences,
            )

            # remove identical points
            base_unique, base_idx, base_inverse = np.unique(
                base_repr, axis=0, return_index=True, return_inverse=True
            )
            finetuned_unique, finetuned_idx, finetuned_inverse = np.unique(
                finetuned_repr, axis=0, return_index=True, return_inverse=True
            )
            indices = np.intersect1d(base_idx, finetuned_idx)
            indices = np.sort(indices)

            base_repr = base_repr[indices]
            finetuned_repr = finetuned_repr[indices]
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

            distances_finetuned, dist_index_finetuned, mus, _ = compute_distances(
                X=finetuned_repr,
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
                    d.compute_density_kNN(k=16)
                    assignment_base = d.compute_clustering_ADP(Z=z, halo=halo)

                    d = data.Data(distances=(distances_base, dist_index_base))
                    ids, _, _ = d.return_id_scaling_gride(range_max=100)
                    d.set_id(ids[3])
                    d.compute_density_kNN(k=16)
                    assignment_finetuned = d.compute_clustering_ADP(Z=z, halo=halo)

                    mask = np.ones(len(assignment_finetuned), dtype=bool)
                    if halo:
                        is_core1 = assignment_base != -1
                        is_core2 = assignment_finetuned != -1
                        mask = np.logical_and(is_core1, is_core2)

                    cluster_comparison[f"ami-ep{epoch}-z{z}{is_halo}"].append(
                        adjusted_mutual_info_score(
                            assignment_base[mask], assignment_finetuned[mask]
                        )
                    )

                    cluster_comparison[f"ari-ep{epoch}-z{z}{is_halo}"].append(
                        adjusted_rand_score(
                            assignment_base[mask], assignment_finetuned[mask]
                        )
                    )

            for k in [30, 100, 300]:
                ov_repr[f"ep{epoch}_k{k}"].append(
                    return_data_overlap(
                        indices_base=dist_index_base,
                        indices_other=dist_index_finetuned,
                        k=k,
                        subjects=None,
                    )
                )

            ov_repr_tmp = return_data_overlap(
                indices_base=dist_index_base,
                indices_other=dist_index_finetuned,
                k=30,
                subjects=subjects,
            )

            for subject in np.unique(subjects):
                ov_repr[f"ep{epoch}_{subject}_k30"].append(ov_repr_tmp[subject])

with open(
    f"{args.results_path}/overlaps_{args.model}_finetuned_{args.finetuned_mode}_eval_{args.eval_dataset}{is_balanced}_epoch{args.epochs}_{args.num_shots}.pkl",
    "wb",
) as f:
    pickle.dump(ov_repr, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(
    f"{args.results_path}/cluster_comparison_{args.model}_finetuned_{args.finetuned_mode}_eval_{args.eval_dataset}{is_balanced}_epoch{args.epochs}_{args.num_shots}.pkl",
    "wb",
) as f:
    pickle.dump(cluster_comparison, f, protocol=pickle.HIGHEST_PROTOCOL)
