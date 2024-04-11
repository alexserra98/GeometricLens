from datasets import load_dataset
import torch
from functools import partial
from datasets.utils.logging import disable_progress_bar
import numpy as np

# from dataset_utils.utils import MMLU_Dataset
import sys


disable_progress_bar()


def filter_out_long_sequences(tokenized_dataset, max_seq_len):

    tot_examples = tokenized_dataset.num_rows
    tokenized_datasets = tokenized_dataset.filter(
        lambda example: len(example["input_ids"]) < max_seq_len
    )
    tot_filtered_examples = tokenized_datasets.num_rows

    if tot_filtered_examples < tot_examples:
        diff = tot_examples - tot_filtered_examples
        print(
            f"you filter out {diff} examples, {diff/tot_examples*100: .2f}% of the total"
        )
        sys.stdout.flush()
    return tokenized_dataset


# prompt builder
class MMLU_Dataset:
    # num_few_shots = # shots
    # model_name number_istences to remove
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        accelerator,
        num_few_shots=0,
        subject=None,
        num_processes=1,
        num_samples=None,
    ):

        self.dataset = "mmlu"
        self.subject = subject
        if subject is not None:
            self.dataset = f"mmlu:{self.subject}"
        # we add space because the prompt format ends with ":" without a space.
        # comparing the answers in the token space requires this construction.
        self.answers = np.array([" A", " B", " C", " D"])
        self.num_few_shots = num_few_shots
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_processes = num_processes
        self.num_samples = num_samples
        self.accelerator = accelerator

    def construct_question(self, question, choices, answer, include_answer=False):
        # added strip
        prompt = f"{question.strip()}\n"
        for i, choice in enumerate(choices):
            # added strip
            prompt += f"{self.answers[i]}. {choice.strip()}\n"
        # added space to final answers
        prompt += "Answer:"
        if include_answer:
            prompt += f" {self.answers[answer]}\n\n"
        return prompt

    # prompt contruction.buils to operate on list of inputs.
    def construct_prompt(self, batch, tokenizer, dev_set, max_seq_len, num_few_shots):
        prompts = []

        questions = batch["question"]  # list of strings
        subjects = batch["subject"]  # list of strings
        choices = batch["choices"]  # list of list of strings
        answer_indices = np.array(batch["answer"])  # array of integers

        # build a dict of subsets of the dev set with the subject of the batch
        if num_few_shots > 0:
            local_dev_set = {}
            for subject in set(subjects):
                local_dev_set[subject] = dev_set.filter(
                    lambda dev_example: dev_example["subject"] == subject,
                )

        for i, question in enumerate(questions):
            prompt = f"The following are multiple choice questions (with answers) about {subjects[i]}.\n\n"
            current_subject = subjects[i]
            for j in range(num_few_shots):
                shot = local_dev_set[current_subject][j]
                prompt += self.construct_question(
                    shot["question"],
                    shot["choices"],
                    shot["answer"],
                    include_answer=True,
                )
            question = self.construct_question(
                questions[i], choices[i], answer_indices[i]
            )
            prompt += question
            prompts.append(prompt)

        # tokenization part
        tokenized_examples = [
            tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=False,
                add_special_tokens=True,
            ).input_ids.flatten()
            for prompt in prompts
        ]

        # targets are tokenized with space included
        tokenized_labels = [
            tokenizer(
                self.answers[index], return_tensors="pt", add_special_tokens=False
            ).input_ids.flatten()
            for index in answer_indices
        ]

        attention_mask = [
            torch.ones_like(input_ids) for input_ids in tokenized_examples
        ]

        return {
            "prompt": prompts,
            "answers": [self.answers[index] for index in answer_indices],
            "subjects": subjects,
            "input_ids": tokenized_examples,
            "labels": tokenized_labels,
            "attention_mask": attention_mask,
        }

    def construct_dataset(self):
        """
        Construct the request instances for the scenario
        """
        # removed trust remote code
        self.accelerator.print("loading dataset")
        split = "test"
        if self.num_samples is not None:
            split = f"test[:{self.num_samples}]"
        if self.subject is not None:
            dataset = load_dataset("cais/mmlu", self.subject, split=split)
        else:
            dataset = load_dataset("cais/mmlu", "all", split=split)

        few_shot_dataset = None
        if self.num_few_shots > 0 and self.num_few_shots <= 5:
            few_shot_dataset = load_dataset("cais/mmlu", "all", split="dev")
        elif self.num_few_shots > 5:
            few_shot_dataset = load_dataset("cais/mmlu", "all", split="dev+val")

        encode_function = partial(
            self.construct_prompt,
            tokenizer=self.tokenizer,
            dev_set=few_shot_dataset,
            max_seq_len=self.max_seq_len,
            num_few_shots=self.num_few_shots,
        )
        self.accelerator.print("tokenization started")
        sys.stdout.flush()
        tokenized_dataset = dataset.map(
            encode_function,
            batched=True,
            batch_size=self.num_processes,
            num_proc=self.num_processes,
            load_from_cache_file=False,
        )

        self.accelerator.print("tokenization finished")
        sys.stdout.flush()

        # remove examples loger than max seq len maybe not necessary at all
        # list of list is made list of tensors
        tokenized_dataset.set_format(type="pt")
        tokenized_dataset = filter_out_long_sequences(
            tokenized_dataset, self.max_seq_len
        )

        return tokenized_dataset
