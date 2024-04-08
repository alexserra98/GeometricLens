import os
import time
import torch
import numpy as np
from collections import defaultdict
import math
import torch.distributed as dist
import psutil

rng = np.random.default_rng(42)
# ***************************************************


class extract_activations:
    def __init__(
        self,
        model,
        dataloader,
        target_layers,
        embdim,
        dtypes,
        nsamples,
        use_last_token=False,
        print_every=100,
    ):
        self.model = model
        # embedding size
        self.embdim = embdim
        # number of samples to collect (e.g. 10k)
        self.nsamples = nsamples
        # whether to compute the id on the last token /class_token
        self.use_last_token = use_last_token
        self.print_every = print_every

        self.micro_batch_size = dataloader.batch_size
        self.nbatches = len(dataloader)
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.global_batch_size = self.world_size * self.micro_batch_size
        self.hidden_size = 0

        if self.rank == 0:
            print(
                "before hidden states RAM Used (GB):",
                psutil.virtual_memory()[3] / 10**9,
            )

        self.init_hidden_states(target_layers, dtypes=dtypes)
        self.init_hooks(target_layers)

    def init_hidden_states(self, target_layers, dtypes):
        # dict containing the representations extracted in a sigle forward pass
        self.hidden_states_tmp = defaultdict(lambda: None)

        # dict storing the all the representations
        self.hidden_states = {}
        for name in target_layers:
            self.hidden_states[name] = torch.zeros(
                (self.nsamples, self.embdim[name]), dtype=dtypes[name]
            )

    def init_hooks(self, target_layers):
        for name, module in self.model.named_modules():
            if name in target_layers:
                module.register_forward_hook(
                    self._get_hook(name, self.hidden_states_tmp)
                )

    def _get_hook(self, name, hidden_states):
        if self.world_size > 1:

            def hook_fn(module, input, output):
                hidden_states[name] = output

        else:

            def hook_fn(module, input, output):
                hidden_states[name] = output.cpu()

        return hook_fn

    def _gather_and_update_fsdp(self, mask, is_last_batch):
        # batch size ==  1 we handle just this setup for world size > 1
        assert mask.shape[0] == 1

        # all gather the sequence lengths from all ranks
        seq_len = torch.sum(mask, dim=1)  # 1x 1 tensor
        seq_len_list = [torch.zeros_like(seq_len) for _ in range(self.world_size)]
        dist.all_gather(seq_len_list, seq_len)
        max_size = max(seq_len_list).item()
        size_diff = max_size - seq_len.item()
        num_current_tokens = sum(seq_len_list).item()

        for _, (name, hidden_state) in enumerate(self.hidden_states_tmp.items()):
            _, _, embdim = hidden_state.shape

            # pad the activations in all ranks to be of the shape 1 x max_seq_len x embd
            if size_diff > 0:
                padding = torch.zeros(
                    (1, size_diff, embdim), device="cuda", dtype=hidden_state.dtype
                )
                hidden_state = torch.cat((hidden_state, padding), dim=1)

            if self.rank == 0:
                # gather the activations to rank 0
                states_list = [
                    torch.zeros(
                        (1, max_size, embdim), device="cuda", dtype=hidden_state.dtype
                    )
                    for _ in range(self.world_size)
                ]
                dist.gather(hidden_state, states_list, dst=0)

                # move to cpu, remove padding, and update hidden states
                num_current_tokens = self._update_hidden_state_fsdp(
                    states_list, seq_len_list, name, num_current_tokens, is_last_batch
                )
            else:
                dist.gather(hidden_state, dst=0)

            dist.barrier()
            del hidden_state

        if self.rank == 0:
            self.hidden_size += num_current_tokens

    def _update_hidden_state_fsdp(
        self, states_list, seq_len_list, name, num_current_tokens, is_last_batch
    ):
        if self.use_last_token:
            act_tmp = torch.cat(
                [
                    state.squeeze()[seq_len.item() - 1 : seq_len.item()]
                    for state, seq_len in zip(states_list, seq_len_list)
                ],
                dim=0,
            ).cpu()
            assert act_tmp.shape[0] == self.world_size
            num_current_tokens = act_tmp.shape[0]
        else:
            act_tmp = torch.cat(
                [
                    state.squeeze()[: seq_len.item()].mean(dim=0, keepdims=True)
                    for state, seq_len in zip(states_list, seq_len_list)
                ],
                dim=0,
            ).cpu()
            assert act_tmp.shape[0] == self.world_size
            num_current_tokens = act_tmp.shape[0]

        if is_last_batch:
            self.hidden_states[name][self.hidden_size :] = act_tmp
        else:
            self.hidden_states[name][
                self.hidden_size : self.hidden_size + num_current_tokens
            ] = act_tmp

        return num_current_tokens

    def _update_hidden_state(self, mask, is_last_batch):
        seq_len = torch.sum(mask, dim=1)
        mask = mask.unsqueeze(-1)
        for _, (name, activations) in enumerate(self.hidden_states_tmp.items()):
            if self.use_last_token:
                batch_size = seq_len.shape[0]
                act_tmp = activations[
                    torch.arange(batch_size), torch.tensor(seq_len) - 1
                ]
            else:
                denom = torch.sum(mask, dim=1)  # batch x 1
                # act_tmp -> batch x seq_len x embed
                # mask -> batch x seq_len x 1
                # denom -> batch x 1
                act_tmp = torch.sum(activations * mask, dim=1) / denom

            num_current_tokens = act_tmp.shape[0]
            if is_last_batch:
                self.hidden_states[name][self.hidden_size :] = act_tmp
            else:
                self.hidden_states[name][
                    self.hidden_size : self.hidden_size + num_current_tokens
                ] = act_tmp
        self.hidden_size += num_current_tokens

    @torch.no_grad()
    def extract(self, dataloader):
        self.t_load = 0
        self.t_forw = 0
        self.t_extr = 0
        self.gather_time = 0
        start = time.time()
        is_last_batch = False
        for i, data in enumerate(dataloader):
            if (i + 1) == self.nbatches:
                is_last_batch = True

            mask = data["attention_mask"] != 0
            mask = mask.to("cuda")
            batch = data["input_ids"].to("cuda")
            _ = self.model(batch)

            if self.world_size > 1:
                self._gather_and_update_fsdp(mask, is_last_batch)
            else:
                self._update_hidden_state(mask.cpu(), is_last_batch)

            if self.rank == 0:

                if (i + 1) % (self.print_every // self.global_batch_size) == 0:
                    torch.cuda.synchronize()
                    end = time.time()
                    self.fabric.print(
                        f"{(i+1)*self.global_batch_size/1000}k data, \
                        batch {i+1}/{self.nbatches}, \
                        tot_time: {(end-start)/60: .3f}min, "
                    )
