from datasets import load_dataset
import torch
from functools import partial
from datasets.utils.logging import disable_progress_bar
import numpy as np
from dataset_utils.utils import MMLU_Dataset

disable_progress_bar()


def encode_func(example, tokenizer, max_seq_length, text_field):
    example_text = example[text_field]
    tokenized_example = tokenizer(
        example_text.strip(),
        add_special_tokens=False,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    # in the output they will be converted to lists but at least we can apply flatten
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def get_text_dataset(
    filepath=None,
    tokenizer=None,
    max_seq_length=2048,
    num_processes=1,
    text_field="text",
    n_caption_per_image=5,
    file_format="csv",
    split="train",
):
    raw_dataset = load_dataset(
        file_format,
        data_files=filepath,
        split=split,
    )
    if n_caption_per_image != 5:
        # this reduces the number of samples
        assert n_caption_per_image < 5, "number of caption should be smaller than 5"
        caption_ids = np.arange(5)[:n_caption_per_image]
        raw_dataset = raw_dataset.filter(lambda x: x["caption_id"] in caption_ids)

    encode_function = partial(
        encode_func,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        text_field=text_field,
    )

    tokenized_dataset = raw_dataset.map(
        encode_function,
        batched=False,
        num_proc=num_processes,
        load_from_cache_file=False,
        remove_columns=[
            name
            for name in raw_dataset.column_names
            if name not in ["input_ids", "labels", "attention_mask"]
        ],
    )
    # the output is always list of lists
    tokenized_dataset.set_format(type="pt")

    return tokenized_dataset
