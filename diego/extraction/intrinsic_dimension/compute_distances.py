import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from intrinsic_dimension.pairwise_distances import compute_distances
from intrinsic_dimension.extract_activations import extract_activations
from transformers import PreTrainedModel, LlamaTokenizer
import sys
from accelerate import Accelerator
import pickle
import math
from dadapy import data
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    adjusted_mutual_info_score,
)


def get_embdims(model, dataloader, target_layers):
    embdims = defaultdict(lambda: None)
    dtypes = defaultdict(lambda: None)

    def get_hook(name, embdims):
        def hook_fn(module, input, output):
            try:
                embdims[name] = output.shape[-1]
                dtypes[name] = output.dtype
            except:
                embdims[name] = output[0].shape[-1]
                dtypes[name] = output[0].dtype

        return hook_fn

    handles = {}
    for name, module in model.named_modules():
        if name in target_layers:
            handles[name] = module.register_forward_hook(get_hook(name, embdims))

    batch = next(iter(dataloader))
    sys.stdout.flush()
    _ = model(batch["input_ids"].to("cuda"))

    for name, module in model.named_modules():
        if name in target_layers:
            handles[name].remove()

    assert len(embdims) == len(target_layers)
    return embdims, dtypes


def compute_accuracy(predictions, answers, subjects=None):

    # ground_truths is an array of letters, without trailing spaces
    # predictions is an array of tokens

    # we remove spaces in from of the letters
    accuracy = {}
    tot_ans = len(predictions)
    num_correct = 0
    for pred, ans in zip(predictions, answers):
        if pred == ans:
            num_correct += 1
    accuracy["micro"] = num_correct / tot_ans

    if subjects is not None:
        acc_subj = {}
        for subject in np.unique(subjects):
            mask = subject == subjects
            pred_tmp = predictions[mask]
            ans_tmp = answers[mask]

            tot_ans = len(ans_tmp)
            num_correct = 0
            for pred, ans in zip(pred_tmp, ans_tmp):
                if pred == ans:
                    num_correct += 1
            acc_tmp = num_correct / tot_ans

            acc_subj[subject] = acc_tmp

    accuracy["subjects"] = acc_subj
    accuracy["macro"] = np.mean(list(acc_subj.values()))

    return accuracy


@torch.inference_mode()
def compute_id(
    accelerator: Accelerator,
    model: PreTrainedModel,
    dataloader: DataLoader,
    tokenizer,
    target_layers,
    maxk=50,
    dirpath=".",
    filename="",
    use_last_token=False,
    remove_duplicates=True,
    save_distances=True,
    save_repr=False,
    print_every=100,
    prompt_search=False,
    time_stamp=None,
    few_shot_indices=None,
    few_shot_seed=None,
    acc_macro=None,
    measure_ari=False,
    mask_dir=None,
):
    model = model.eval()
    if accelerator.is_main_process:
        dirpath = str(dirpath).lower()
        os.makedirs(dirpath, exist_ok=True)

    if filename != "":
        filename = "_" + filename

    if use_last_token:
        filename = f"{filename}_target"
    else:
        filename = f"{filename}_mean"

    target_layer_names = list(target_layers.values())
    target_layer_labels = list(target_layers.keys())
    accelerator.print("layer_to_extract: ", target_layer_labels)
    embdims, dtypes = get_embdims(model, dataloader, target_layer_names)

    start = time.time()
    extr_act = extract_activations(
        accelerator,
        model,
        dataloader,
        target_layer_names,
        embdims,
        dtypes,
        use_last_token=use_last_token,
        print_every=print_every,
        prompt_search=prompt_search,
        time_stamp=time_stamp,
    )
    extr_act.extract(dataloader, tokenizer)
    accelerator.print(f"num_tokens: {extr_act.hidden_size/10**3}k")
    accelerator.print((time.time() - start) / 3600, "hours")

    if accelerator.is_main_process:
        # dictionary containing the representation
        if not prompt_search:
            act_dict = extr_act.hidden_states
            if save_repr:
                for i, (layer, act) in enumerate(act_dict.items()):
                    torch.save(act, f"{dirpath}/l{target_layer_labels[i]}{filename}.pt")
                    # torch.save(act, f"{dirpath}/{layer}{filename}.pt")

        predictions = extr_act.predictions  # tokens
        constrained_predictions = extr_act.constrained_predictions  # tokens
        processed_labels = extr_act.targets  # tokens

        answers = dataloader.dataset["answers"]  # letters
        ground_truths = dataloader.dataset["labels"]  # tokens
        subjects = dataloader.dataset["subjects"]

        # check
        # assert torch.all(ground_truths == processed_labels), (
        #     processed_labels,
        #     ground_truths,
        # )

        answers = np.array([ans.strip() for ans in answers])
        predictions = np.array([tokenizer.decode(pred).strip() for pred in predictions])
        constrained_predictions = np.array(
            [tokenizer.decode(pred).strip() for pred in constrained_predictions]
        )

        acc_pred = compute_accuracy(
            predictions,
            answers[: len(predictions)],
            np.array(subjects[: len(predictions)]),
        )
        if acc_macro is not None:
            is_close = math.isclose(acc_pred["macro"], acc_macro, rel_tol=1e-4)
            if not is_close:
                print("measured: ", acc_pred["macro"], "\nexpected: ", acc_macro)

        acc_constrained = compute_accuracy(
            constrained_predictions,
            answers[: len(constrained_predictions)],
            np.array(subjects[: len(predictions)]),
        )
        accelerator.print("exact_match constrained:", acc_constrained["macro"])

        accelerator.print("exact_match macro:", acc_pred["macro"])
        accelerator.print("exact_match micro:", acc_pred["micro"])
        for subject, acc in acc_pred["subjects"].items():
            accelerator.print(f"{subject}: {acc:.3f}\n")

        if prompt_search:
            examples = [42, 1042, 2042, 3042, 4042, 5042]
            with open(f"prompt_search_{time_stamp}.txt", "a") as f:
                f.write(f"accuracy: {acc_pred['macro']}\n")
                f.write(f"accuracy_constrained: {acc_constrained['macro']}\n\n")
                f.write(f"accuracy_sybj: {acc_pred['subjects']}\n\n")
                for ind in examples:
                    f.write(f"example {ind}\n")
                    f.write(f"{dataloader.dataset[ind]['prompt']} {answers[ind]}\n")
                    f.write(f"prediction: {predictions[ind]}\n\n")

        if not prompt_search:
            if few_shot_indices is not None:

                assert few_shot_seed is not None
                statistics = {
                    "accuracy": acc_pred,
                    "few_shot_indices": few_shot_indices,
                }
                with open(
                    f"{dirpath}/statistics{filename}_seed{few_shot_seed}.pkl", "wb"
                ) as f:
                    pickle.dump(statistics, f)

            else:
                metrics = None
                if measure_ari:
                    dataset_mask = None
                    if mask_dir is not None:
                        dataset_mask = np.load(f"{mask_dir}/test_mask_200.npy")
                    metrics = compute_ari(
                        act_dict=act_dict,
                        subjects=dataloader.dataset["subjetcs"],
                        dataset_mask=dataset_mask,
                    )

                statistics = {
                    "subjects": dataloader.dataset["subjects"],
                    "answers": answers,
                    "predictions": predictions,
                    "contrained_predictions": constrained_predictions,
                    "accuracy": acc_pred,
                    "constrained_accuracy": acc_constrained,
                    "few_shot_indices": few_shot_indices,
                    "metrics": metrics,
                }

                with open(f"{dirpath}/statistics{filename}.pkl", "wb") as f:
                    pickle.dump(statistics, f)

            if save_distances:
                for i, (layer, act) in enumerate(act_dict.items()):
                    act = act.to(torch.float64).numpy()

                    save_backward_indices = False
                    if remove_duplicates:
                        act, idx, inverse = np.unique(
                            act, axis=0, return_index=True, return_inverse=True
                        )
                        accelerator.print(len(idx), len(inverse))
                        if len(idx) == len(inverse):
                            # if no overlapping data has been found return the original ordred array
                            assert len(np.unique(inverse)) == len(inverse)
                            act = act[inverse]
                        else:
                            save_backward_indices = True
                        accelerator.print(f"unique_samples = {len(idx)}")
                        accelerator.print(f"num_duplicates = {len(inverse)-len(idx)}")

                    n_samples = act.shape[0]
                    if n_samples == 1:
                        accelerator.print(
                            f"{layer} has only one sample:distance matrices not computed"
                        )
                    else:
                        range_scaling = min(1050, n_samples - 1)
                        maxk = min(maxk, n_samples - 1)

                        start = time.time()
                        distances, dist_index, mus, _ = compute_distances(
                            X=act,
                            n_neighbors=maxk + 1,
                            n_jobs=1,
                            working_memory=2048,
                            range_scaling=range_scaling,
                            argsort=False,
                        )
                        accelerator.print((time.time() - start) / 60, "min")

                        np.save(
                            f"{dirpath}/l{target_layer_labels[i]}{filename}_dist",
                            distances,
                        )
                        np.save(
                            f"{dirpath}/l{target_layer_labels[i]}{filename}_index",
                            dist_index,
                        )
                        if save_backward_indices:
                            np.save(
                                f"{dirpath}/l{target_layer_labels[i]}{filename}_inverse",
                                inverse,
                            )
                        np.save(
                            f"{dirpath}/l{target_layer_labels[i]}{filename}_mus", mus
                        )


def compute_ari(act_dict, subjects, dataset_mask=None):
    metrics = defaultdict(list)

    for i, (layer, act) in enumerate(act_dict.items()):
        base_repr = act.to(torch.float64).numpy()

        subjects_to_int = {sub: i for i, sub in enumerate(np.unique(subjects))}
        subj_label = np.array([subjects_to_int[sub] for sub in subjects])

        # balance the test set if asked
        if dataset_mask is not None:
            base_repr = base_repr[dataset_mask]
            subj_label = subj_label[dataset_mask]

        # remove identical points
        _, base_idx, _ = np.unique(
            base_repr, axis=0, return_index=True, return_inverse=True
        )
        indices = np.sort(base_idx)
        base_repr = base_repr[indices]
        subj_label = subj_label[indices]

        # **************************************************************************

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

        d = data.Data(distances=(distances_base, dist_index_base))
        ids, _, _ = d.return_id_scaling_gride(range_max=100)
        d.set_id(ids[3])
        d.compute_density_kNN(k=16)
        assignment = d.compute_clustering_ADP(Z=1.6, halo=False)

        metrics["ari"].append(adjusted_rand_score(assignment, subj_label))
        metrics["ami"].append(adjusted_mutual_info_score(assignment, subj_label))
        metrics["completeness"].append(completeness_score(assignment, subj_label))
        metrics["homogeneity"].append(homogeneity_score(assignment, subj_label))

    return metrics
