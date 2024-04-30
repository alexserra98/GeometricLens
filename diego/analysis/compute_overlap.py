from collections import defaultdict
import torch
from dadapy._cython import cython_overlap as c_ov
from pairwise_distances import compute_distances
import sys
import numpy as np
import pickle

import os
import argparse


def return_data_overlap(indices_base, indices_other, k=30, subjects=None):

    assert indices_base.shape[0] == indices_other.shape[0]
    ndata = indices_base.shape[0]

    overlaps_full = c_ov._compute_data_overlap(
        ndata, k, indices_base.astype(int), indices_other.astype(int)
    )

    overlaps = overlaps_full
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
    args = parser.parse_args()
    return args


args = parse_args()
os.makedirs(args.results_path, exist_ok=True)

base_dir = "/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens/repo/results"
for epoch in [0, 4]:
    print(f"processing epoch {epoch}")
    sys.stdout.flush()

    for mode in ["0shot", "5shot"]:
        print(f"processing {mode}")
        sys.stdout.flush()

        assert args.finetuned_mode in ["dev", "dev+val"]
        eval_options = ["test", "validation"]
        if args.finetuned_mode == "dev+val":
            eval_options = ["test"]

        for evaluation_mode in eval_options:
            overlaps = defaultdict(list)
            overlaps_subjects = defaultdict(list)

            for layer in range(34):
                # ************************************
                pretrained_path = f"{base_dir}/mmlu/{args.model}/{mode}"
                base = torch.load(f"{pretrained_path}/l{layer}_target.pt")

                base_ = base.to(torch.float64).numpy()
                base_unique, base_idx, base_inverse = np.unique(
                    base_, axis=0, return_index=True, return_inverse=True
                )
                # **********************************

                finetuned_path = f"{base_dir}/finetuned_{args.finetuned_mode}/evaluated_{evaluation_mode}/{args.model}/{epoch}epochs/epoch_{epoch}"
                finetuned = torch.load(f"{finetuned_path}/l{layer}_target.pt")

                finetuned_ = finetuned.to(torch.float64).numpy()
                finetuned_unique, finetuned_idx, finetuned_inverse = np.unique(
                    finetuned_, axis=0, return_index=True, return_inverse=True
                )

                indices = np.intersect1d(base_idx, finetuned_idx)

                maxk = 100
                distances_base, dist_index_base, mus, _ = compute_distances(
                    X=base_[indices],
                    n_neighbors=maxk + 1,
                    n_jobs=1,
                    working_memory=2048,
                    range_scaling=maxk + 2,
                    argsort=False,
                )

                distances_finetuned, dist_index_finetuned, mus, _ = compute_distances(
                    X=finetuned_[indices],
                    n_neighbors=maxk + 1,
                    n_jobs=1,
                    working_memory=2048,
                    range_scaling=maxk + 2,
                    argsort=False,
                )

                for k in [10, 100]:
                    overlaps[f"{k}"].append(
                        return_data_overlap(
                            indices_base=dist_index_base,
                            indices_other=dist_index_finetuned,
                            k=k,
                            subjects=None,
                        )
                    )

                with open(f"{finetuned_path}/statistics_target.pkl", "rb") as f:
                    stats = pickle.load(f)
                subjects = np.array(stats["subjects"])[indices]

                tmp = return_data_overlap(
                    indices_base=dist_index_base,
                    indices_other=dist_index_finetuned,
                    k=30,
                    subjects=subjects,
                )

                for subject in np.unique(subjects):
                    overlaps_subjects[subject].append(tmp[subject])

            with open(
                f"{args.results_path}/overlaps_{args.model}_finetuned_{args.finetuned_mode}_eval_{evaluation_mode}_epoch{epoch}_{mode}.pkl",
                "wb",
            ) as f:
                pickle.dump(overlaps, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(
                f"{args.results_path}/overlaps_{args.model}_finetuned_{args.finetuned_mode}_eval_{evaluation_mode}_epoch{epoch}_{mode}_subjects_k30.pkl",
                "wb",
            ) as f:
                pickle.dump(overlaps_subjects, f, protocol=pickle.HIGHEST_PROTOCOL)
