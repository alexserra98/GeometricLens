from collections import defaultdict
import torch
from dadapy._cython import cython_overlap as c_ov
from pairwise_distances import compute_distances
import sys
import numpy as np
import pickle


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


pretrained_path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/results/mmlu/llama-3-8b"
mode = "5shot"
base = torch.load(f"{pretrained_path}/{mode}/l32_target.pt")
base_ = base.to(torch.float64).numpy()
base_unique, base_idx, base_inverse = np.unique(
    base_, axis=0, return_index=True, return_inverse=True
)


finetuned_path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/results/finetuned/llama-3-8b/epoch_2"
finetuned = torch.load(f"{finetuned_path}/l32_target.pt")
finetuned_ = finetuned.to(torch.float64).numpy()
finetuned_unique, finetuned_idx, finetuned_inverse = np.unique(
    finetuned_, axis=0, return_index=True, return_inverse=True
)

indices = np.intersect1d(base_idx, finetuned_idx)

with open(f"{finetuned_path}/statistics_target.pkl", "rb") as f:
    stats = pickle.load(f)
subjects = np.array(stats["subjects"])[indices]


maxk = 50
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
subjects.shape
dist_index_finetuned.shape
overlaps = return_data_overlap(
    indices_base=dist_index_base,
    indices_other=dist_index_finetuned,
    k=30,
    subjects=subjects,
)

np.mean(list(overlaps.values()))


return_data_overlap(
    indices_base=dist_index_base,
    indices_other=dist_index_finetuned,
    k=30,
    subjects=subjects,
)


# numbers between 0 and n_unique
base_inverse.shape

dist_index[base_inverse]

base_inverse.shape


distances_base
dist_index_base

base_unique[base_inverse]


@torch.no_grad()
def compute_overlap(
    accelerator,
    model,
    val_loader,
    tokenizer,
    target_layers,
    embdims,
    dtypes,
    base_indices,
    subjects,
    results_dir,
    filename,
    ckpt_dir,
):
    target_layer_names = list(target_layers.values())
    name_to_idx = {val: key for key, val in target_layers.items()}

    model.eval()

    accelerator.print("representations extracted")
    sys.stdout.flush()

    act_dict = extr_act.hidden_states

    overlaps = defaultdict(dict)
    for i, (name, act) in enumerate(act_dict.items()):
        torch.save(act, f"{results_dir}/{name}{filename}.pt")

    print(f"{results_dir}/{name}{filename}.pt\n")

    # dirpath_actual = (
    #     "/u/area/ddoimo/ddoimo/finetuning_llm/open-instruct/results/llama-2-7b"
    # )
    # actual = torch.load(
    #     f"{dirpath_actual}/base_model.model.model.layers.0.input_layernorm_outepoch2.pt"
    # )

    # print(dirpath_actual)

    # dirpath_original = "/u/area/ddoimo/ddoimo/open/geometric_lens/repo/results/validation/llama-2-7b/0shot"
    # print("{dirpath original}")
    # expected = torch.load(f"{dirpath_original}/l0_hook_output_target.pt")

    # torch.testing.assert_close(actual, expected)

    # print("actual and original match\n")

    for shots in base_indices.keys():
        # for norm in base_indices[shots].keys():
        # ov_tmp = defaultdict(dict)
        ov_tmp = defaultdict()
        for i, (name, act) in enumerate(act_dict.items()):
            act = act.to(torch.float64).numpy()

            if name_to_idx[name] < 1:
                continue
            else:
                _, dist_index, _, _ = compute_distances(
                    X=act,
                    n_neighbors=40 + 1,
                    n_jobs=1,
                    working_memory=2048,
                    range_scaling=40 + 1,
                    argsort=False,
                )

                ov_tmp[name] = return_data_overlap(
                    indices_base=dist_index,
                    indices_other=base_indices[shots][name_to_idx[name]],
                    subjects=subjects,
                    k=30,
                )

        overlaps[shots] = ov_tmp

    model.train()
    return overlaps
