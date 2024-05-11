#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from transformers import (
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    BertConfig,
    BertModel,
    BertTokenizer
)

import math
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Generator, Optional
from torch import nn
import lightning as L
import numpy as np
import torch
from einops import rearrange, reduce, repeat
from gluonts.model import Input, InputSpec
from gluonts.torch import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    ExpandDimArray,
    TestSplitSampler,
    Transformation,
)
from gluonts.transform.split import TFTInstanceSplitter
from jaxtyping import Bool, Float, Int
from torch.distributions import Distribution

from uni2ts.common.torch_util import safe_div
from uni2ts.loss.packed import PackedNLLLoss as _PackedNLLLoss

from uni2ts.model.moirai.module import MoiraiModule
from uni2ts.model.llm_moirai.finetune import get_data_description, calculate_lags


class SampleNLLLoss(_PackedNLLLoss):
    def reduce_loss(
        self,
        loss: Float[torch.Tensor, "batch seq_len #dim"],
        prediction_mask: Optional[Bool[torch.Tensor, "batch seq_len"]],
        observed_mask: Optional[Bool[torch.Tensor, "batch seq_len #dim"]],
        sample_id: Optional[Int[torch.Tensor, "batch seq_len"]],
        variate_id: Optional[Int[torch.Tensor, "batch seq_len"]],
    ) -> Float[torch.Tensor, "batch"]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        mask = prediction_mask.unsqueeze(-1) * observed_mask
        tobs = reduce(
            id_mask
            * reduce(
                mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loss = safe_div(loss, tobs)
        return (loss * mask).sum(dim=(-1, -2))


class LlmMoiraiForecast(L.LightningModule):
    """
    Moirai for forecasting. Load the ckpt from fine-tuning or pre-training.
    """

    def __init__(
        self,
        module_kwargs: dict[str, Any],
        llm_kwargs: dict[str, Any],
        task_kwargs: dict[str, Any],
        data: str,
        prediction_length: int,
        target_dim: int,                     # Get from meta data of test dataset
        feat_dynamic_real_dim: int,
        past_feat_dynamic_real_dim: int,
        context_length: int,
        patch_size: int | str = "auto",
        num_samples: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()  # PL: save all the hyperparameters passed to init into self.hparams
        self.module = MoiraiModule(**module_kwargs)
        self.d_llm = llm_kwargs['d_llm']
        self.llm_layers = llm_kwargs['llm_layers']
        self.llm_model = self._set_llm_model(llm_kwargs['llm_model'])  # LLM is frozen
        self.llm_tokenizer = self._set_llm_tokenizer(llm_kwargs['llm_model'])
        if isinstance(self.hparams.patch_size, int):
            self.projector = nn.Linear(self.d_llm, patch_size)
        else:
            self.projector = nn.ModuleList([nn.Linear(self.d_llm, ps) for ps in self.module.patch_sizes])
        self.per_sample_loss_func = SampleNLLLoss()
        self.data_description = get_data_description(data)

    def _set_llm_model(self, llm_model):
        """
        Adapt from https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py
        """

        if llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = self.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    attn_implementation="eager",
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    attn_implementation="eager",
                    # load_in_4bit=True
                )

        elif llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = self.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

        elif llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = self.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
        else:
            raise Exception('LLM model is not defined')

        # Freeze LLM's parameters
        for param in llm_model.parameters():
            param.requires_grad = False

        return llm_model

    def _set_llm_tokenizer(self, llm_model):
        """
        Adapt from https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py
        """

        if llm_model == 'LLAMA':
            try:
                tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )

        elif llm_model == 'GPT2':
            try:
                tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif llm_model == 'BERT':
            try:
                tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': pad_token})
            tokenizer.pad_token = pad_token

        return tokenizer

    def create_predictor(
        self,
        batch_size: int,
        device: str = "auto",
    ) -> PyTorchPredictor:

        ts_fields = []
        if self.hparams.feat_dynamic_real_dim > 0:
            ts_fields.append("feat_dynamic_real")
            ts_fields.append("observed_feat_dynamic_real")

        past_ts_fields = []
        if self.hparams.past_feat_dynamic_real_dim > 0:
            past_ts_fields.append("past_feat_dynamic_real")
            past_ts_fields.append("past_observed_feat_dynamic_real")

        # Instance splitter used by the Temporal Fusion Transformer model.
        # Returns known dynamic features as a single tensor of shape […, context_length + prediction_length, …] without splitting it into past & future parts
        instance_splitter = TFTInstanceSplitter(
            instance_sampler=TestSplitSampler(),  # defines how the windows are cut/sampled
            past_length=self.past_length,  # context_len + prediction_len if ps is 'auto'
            future_length=self.hparams.prediction_length,
            observed_value_field="observed_target",
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )

        return PyTorchPredictor(
            input_names=self.prediction_input_names,
            prediction_net=self,
            batch_size=batch_size,
            prediction_length=self.hparams.prediction_length,
            input_transform=self.get_default_transform() + instance_splitter,
            device=device,
        )

    def describe_inputs(self, batch_size: int = 1) -> InputSpec:
        data = {
            "past_target": Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.target_dim,
                ),
                dtype=torch.float,
            ),
            "past_observed_target": Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.target_dim,
                ),
                dtype=torch.bool,
            ),
            "past_is_pad": Input(
                shape=(batch_size, self.past_length),
                dtype=torch.bool,
            ),
        }
        if self.hparams.feat_dynamic_real_dim > 0:
            data["feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length + self.hparams.prediction_length,
                    self.hparams.feat_dynamic_real_dim,
                ),
                dtype=torch.float,
            )
            data["observed_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length + self.hparams.prediction_length,
                    self.hparams.feat_dynamic_real_dim,
                ),
                dtype=torch.bool,
            )
        if self.hparams.past_feat_dynamic_real_dim > 0:
            data["past_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.past_feat_dynamic_real_dim,
                ),
                dtype=torch.float,
            )
            data["past_observed_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.past_feat_dynamic_real_dim,
                ),
                dtype=torch.bool,
            )
        return InputSpec(data=data, zeros_fn=torch.zeros)

    @property
    def prediction_input_names(self) -> list[str]:
        return list(self.describe_inputs())

    @property
    def training_input_names(self):
        return self.prediction_input_names + ["future_target", "future_observed_values"]

    @property
    def past_length(self) -> int:
        return (
            self.hparams.context_length + self.hparams.prediction_length
            if self.hparams.patch_size == "auto"
            else self.hparams.context_length
        )

    def context_token_length(self, patch_size: int) -> int:
        """
        Number of patches/tokens in context range
        """
        return math.ceil(self.hparams.context_length / patch_size)

    def prediction_token_length(self, patch_size) -> int:
        """
        Number of patches/tokens in prediction range
        """
        return math.ceil(self.hparams.prediction_length / patch_size)

    @property
    def max_patch_size(self) -> int:
        return max(self.module.patch_sizes)

    def forward(
        self,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        past_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
        past_observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
        num_samples: Optional[int] = None,
    ) -> Float[torch.Tensor, "batch sample future_time *tgt"]:

        # ToDo: For now we don't use patch_size == auto
        if self.hparams.patch_size == "auto":
            val_loss = []
            preds = []
            for patch_size in self.module.patch_sizes:
                val_loss.append(
                    self._val_loss(
                        patch_size=patch_size,
                        target=past_target[..., : self.past_length, :],  # context + predict_len
                        observed_target=past_observed_target[..., : self.past_length, :],
                        is_pad=past_is_pad[..., : self.past_length],
                        feat_dynamic_real=(
                            feat_dynamic_real[..., : self.past_length, :]
                            if feat_dynamic_real is not None
                            else None
                        ),
                        observed_feat_dynamic_real=(
                            observed_feat_dynamic_real[..., : self.past_length, :]
                            if observed_feat_dynamic_real is not None
                            else None
                        ),
                        # Only use context range for these two.
                        past_feat_dynamic_real=(
                            past_feat_dynamic_real[..., : self.hparams.context_length, :]
                            if past_feat_dynamic_real is not None
                            else None
                        ),
                        past_observed_feat_dynamic_real=(
                            past_observed_feat_dynamic_real[..., : self.hparams.context_length, :]
                            if past_observed_feat_dynamic_real is not None
                            else None
                        ),
                    )
                )
                distr = self._get_distr(
                    patch_size,
                    past_target[..., -self.hparams.context_length :, :],
                    past_observed_target[..., -self.hparams.context_length :, :],
                    past_is_pad[..., -self.hparams.context_length :],
                    (
                        feat_dynamic_real[..., -self.past_length :, :]
                        if feat_dynamic_real is not None
                        else None
                    ),
                    (
                        observed_feat_dynamic_real[..., -self.past_length :, :]
                        if observed_feat_dynamic_real is not None
                        else None
                    ),
                    (
                        past_feat_dynamic_real[..., -self.hparams.context_length :, :]
                        if past_feat_dynamic_real is not None
                        else None
                    ),
                    (
                        past_observed_feat_dynamic_real[
                            ..., -self.hparams.context_length :, :
                        ]
                        if past_observed_feat_dynamic_real is not None
                        else None
                    ),
                )
                preds.append(
                    self._format_preds(
                        patch_size,
                        distr.sample(
                            torch.Size((num_samples or self.hparams.num_samples,))
                        ),
                        past_target.shape[-1],
                    )
                )
            val_loss = torch.stack(val_loss)  # (patch_sizes, bs)
            preds = torch.stack(preds)
            idx = val_loss.argmin(dim=0)  # bs; for each sample, use the patch_size with the lowest val loss
            return preds[idx, torch.arange(len(idx), device=idx.device)]
        else:
            distr = self._get_distr(
                self.hparams.patch_size,
                past_target,
                past_observed_target,
                past_is_pad,
                feat_dynamic_real,
                observed_feat_dynamic_real,
                past_feat_dynamic_real,
                past_observed_feat_dynamic_real,
            )

            # preds: (sample batch combine_seq patch)
            preds = distr.sample(torch.Size((num_samples or self.hparams.num_samples,)))
            return self._format_preds(
                self.hparams.patch_size, preds, past_target.shape[-1]
            )

    def _val_loss(  # ToDo: We don't use patch_size == auto for now.
        self,
        patch_size: int,
        target: Float[torch.Tensor, "batch time tgt"],
        observed_target: Bool[torch.Tensor, "batch time tgt"],
        is_pad: Bool[torch.Tensor, "batch time"],
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        past_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
        past_observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
    ) -> Float[torch.Tensor, "batch"]:
        # convert format
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            patch_size,
            # Slice the context if self.hparams.patch_size is 'auto'
            past_target=target[..., : self.hparams.context_length, :],
            past_observed_target=observed_target[..., : self.hparams.context_length, :],
            past_is_pad=is_pad[..., : self.hparams.context_length],

            # future is included in target if self.hparams.patch_size is 'auto', else None.
            future_target=target[..., self.hparams.context_length :, :],
            future_observed_target=observed_target[..., self.hparams.context_length :, :],
            future_is_pad=is_pad[..., self.hparams.context_length :],

            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )

        # get predictions
        distr = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            torch.ones_like(time_id, dtype=torch.long) * patch_size,
        )
        val_loss = self.per_sample_loss_func(
            pred=distr,
            target=target,
            prediction_mask=prediction_mask,
            observed_mask=observed_mask,
            sample_id=sample_id,
            variate_id=variate_id,
        )
        return val_loss

    def _get_distr(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        past_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
        past_observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
    ) -> Distribution:
        # convert format
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )

        # For each TS in the batch, generate a prompt
        prompt = self._get_sample_prompt(target, observed_mask, prediction_mask)

        #  Get LLM reprs of prompt.
        prompt = self.llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        prompt = prompt.to(self.llm_model.device)
        prompt_reprs = self.llm_model(input_ids=prompt.input_ids,
                                      attention_mask=prompt.attention_mask).last_hidden_state  # (bs, num_prompt_patches, d_llm)

        # ToDo: Add a Q-former to reduce prompt length to a fixed length!
        #  Samples from different batches have different prompt length.

        # Use projector to map r_p to e_p. (bs, num_prompt_patches, patch_size)
        if isinstance(self.hparams.patch_size, int):
            prompt_prefix = self.projector(prompt_reprs)
        else:
            # Todo: Use the specific patch_size for each sample...
            prompt_prefix = self.projector[patch_size](prompt_reprs)

        batch_size = prompt_prefix.size(0)
        num_prompt_tokens = prompt_prefix.size(1)

        # Prepend prompt, modify the masks and ids.

        #  Pad last dim of prompt_prefix to max_patch_size.
        padded_prompt_prefix = torch.zeros((prompt_prefix.size(0), prompt_prefix.size(1), max(self.module.patch_sizes)),
                                           dtype=prompt_prefix.dtype,
                                           device=target.device)
        padded_prompt_prefix[:, :, :prompt_prefix.size(2)] = prompt_prefix  # (bs, num_prompt_patches, max_patch_size)

        #  First patch of each sample starts with non-observed values due to padding.
        #  Can we directly prepend prompt to target? Then that patch will be [True, True, False, ..., True, ...]
        #  Time Series is cut into separated segments by the mask... Not consecutive.
        #  Can! Not necessary to handle it. It is common for patching.
        prompt_observed_mask = torch.zeros((prompt_prefix.size(0), prompt_prefix.size(1), max(self.module.patch_sizes)),
                                           dtype=observed_mask.dtype,
                                           device=observed_mask.device)
        prompt_observed_mask[:, :, :prompt_prefix.size(2)] = True

        target = torch.cat([padded_prompt_prefix, target], dim=1)
        observed_mask = torch.cat([prompt_observed_mask, observed_mask], dim=1)

        # Each item is an individual sample and none of patches is completely padded, so all sample_ids are ones.
        sample_id = torch.cat(
            [torch.ones((batch_size, num_prompt_tokens), dtype=sample_id.dtype, device=sample_id.device), sample_id],
            dim=1
        )

        # Todo: For uni-channel are as below. How to deal with flatten multi-channel? prompt as a new variate?
        # Treat prompt patches as TS patches, so we need to add original time_id by num_prompt_patches.
        # Then concat with [0,..., num_prompt_patches].
        # No Sequence packing, so no worry about the ending patches are padded and with time id of zeros.
        time_id = torch.cat(
            [torch.arange(0, num_prompt_tokens, dtype=time_id.dtype, device=time_id.device).repeat(time_id.size(0), 1),
             time_id + num_prompt_tokens],
            dim=1
        )

        # ToDo: For uni-channel, duplicate. For flatten multi-channel, create a new variate.
        #  Cannot be the same as the ones in exsisting variates. Need to in the max_dim range.
        variate_id = repeat(
            variate_id[:, 0],
            'batch -> batch seq_len',
            seq_len=variate_id.shape[1] + num_prompt_tokens
        )

        prediction_mask = torch.cat(
            [torch.zeros((batch_size, num_prompt_tokens), dtype=prediction_mask.dtype, device=prediction_mask.device),
             prediction_mask],
            dim=1
        )

        # get predictions
        distr = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            torch.ones_like(time_id, dtype=torch.long) * patch_size,
        )
        return distr

    @torch.no_grad()
    def _get_sample_prompt(self, target, observed_mask, prediction_mask):
        prompt = []
        for b in range(target.size(0)):
            # Dataset Description
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.data_description};"
            )

            # Task description if task params are known.
            if self.hparams.prediction_length and self.hparams.context_length:
                prompt_ += f"Task description: forecast the next {str(self.hparams.prediction_length)} steps given the previous {str(self.hparams.context_length)} steps information; "

            # Todo: Compute the statistics for each sample.
            # In Moirai, each sample is a MTS? --> Compute channel-wise statistics.
            # It has been flattened. Need to use variate_id to compute.
            # Do we need to consider pad/missing values when computing statistics?

            # if self.hparams.prompt_statistics:

            # Mask indicating the observed tokens in context range for a sample b.
            mask = observed_mask[b] & ~prediction_mask[b].unsqueeze(-1).expand_as(observed_mask[b])
            valid_target = target[b][mask]  # A 1D tensor with observed context time steps
            min_value = torch.min(valid_target).item()
            max_value = torch.max(valid_target).item()
            median = torch.median(valid_target).item()
            lags = calculate_lags(valid_target)
            trend = valid_target.diff().sum()

            min_value_str = str(round(min_value, 4))
            max_value_str = str(round(max_value, 4))
            median_value_str = str(round(median, 4))
            lags_values_str = str(lags.tolist())
            prompt_ += (f"Input statistics: "
                        f"min value {min_value_str}, "
                        f"max value {max_value_str}, "
                        f"median value {median_value_str}, "
                        f"the trend of input is {'upward' if trend > 0 else 'downward'}, "
                        f"top 5 lags are : {lags_values_str};")

            prompt_ += "<|<end_prompt>|>"
            prompt.append(prompt_)

        return prompt


    @staticmethod
    def _patched_seq_pad(
        patch_size: int,
        x: torch.Tensor,
        dim: int,
        left: bool = True,
        value: Optional[float] = None,
    ) -> torch.Tensor:
        """
        pad x along dimension `dim` so that the size of `dim` is a multiple of patch_size.
        """
        if dim >= 0:
            dim = -x.ndim + dim
        pad_length = -x.size(dim) % patch_size
        if left:
            pad = (pad_length, 0)
        else:
            pad = (0, pad_length)
        pad = (0, 0) * (abs(dim) - 1) + pad
        return torch.nn.functional.pad(x, pad, value=value)

    def _generate_time_id(
        self,
        patch_size: int,
        past_observed_target: Bool[torch.Tensor, "batch past_seq tgt"],
    ) -> tuple[
        Int[torch.Tensor, "batch past_token"], Int[torch.Tensor, "batch future_token"]
    ]:

        # (bs, num_patches). Patches from unobserved range are False, others are True.
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),  # Pad along the time step dimension
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(past_seq_id.cumsum(dim=-1) - 1, min=0)  # Cumulate  ToDo: starts from 0. Cannot distinguish padded and the 1st ts patch. And how is the time_id in Train?
        batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
        future_seq_id = (
            repeat(
                torch.arange(
                    self.prediction_token_length(patch_size),
                    device=past_observed_target.device,
                ),
                f"prediction -> {batch_shape} prediction",
            )
            + past_seq_id.max(dim=-1, keepdim=True).values
            + 1
        )

        end = 1
        return past_seq_id, future_seq_id

    def _convert(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        future_target: Optional[Float[torch.Tensor, "batch future_time tgt"]] = None,           # Not used if ps is auto
        future_observed_target: Optional[Bool[torch.Tensor, "batch future_time tgt"]] = None,   # Not used if ps is auto
        future_is_pad: Optional[Bool[torch.Tensor, "batch future_time"]] = None,                # Not used if ps is auto
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        past_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
        past_observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
    ) -> tuple[
        Float[torch.Tensor, "batch combine_seq patch"],  # target
        Bool[torch.Tensor, "batch combine_seq patch"],  # observed_mask
        Int[torch.Tensor, "batch combine_seq"],  # sample_id
        Int[torch.Tensor, "batch combine_seq"],  # time_id
        Int[torch.Tensor, "batch combine_seq"],  # variate_id
        Bool[torch.Tensor, "batch combine_seq"],  # prediction_mask
    ]:

        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target = []
        observed_mask = []
        sample_id = []
        time_id = []
        variate_id = []
        prediction_mask = []
        dim_count = 0

        # P represents the number of patches for each variate.
        # If self.hparams.patch_size is not 'auto', set future_target as zeros
        if future_target is None:
            future_target = torch.zeros(
                batch_shape
                + (
                    self.hparams.prediction_length,
                    past_target.shape[-1],
                ),
                dtype=past_target.dtype,
                device=device,
            )

        # Patching and flatten MTS to one sequence --> (bs, P x num_tgt, patch_size)
        # target will be a list of 2 tensors:
        # [(bs, P_past x num_tgt, patch_size), (bs, P_future x num_tgt, patch_size)]

        # ToDo: Note when add prefix to past_target, add before _patched_seq_pad
        target.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(patch_size, past_target, -2, left=True),  # Past is padded on the left
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),  # Pad the last dim to max_patch_size.
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_target, -2, left=False),  # Future is padded on the right
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )

        # If self.hparams.patch_size is not 'auto', set future_observed_target as ones
        if future_observed_target is None:
            future_observed_target = torch.ones(
                batch_shape
                + (
                    self.hparams.prediction_length,
                    past_observed_target.shape[-1],
                ),
                dtype=torch.bool,
                device=device,
            )

        # observed_mask. Same as the above.
        # will be a list of 2 boolean tensors:
        # [(bs, P_past x num_tgt, patch_size), (bs, P_future x num_tgt, patch_size)]
        observed_mask.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_target, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_observed_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )

        # If self.hparams.patch_size is not 'auto', set future_is_pad as zeros
        if future_is_pad is None:
            future_is_pad = torch.zeros(
                batch_shape + (self.hparams.prediction_length,),
                dtype=torch.long,
                device=device,
            )

        # sample_id: If the patch in the flatten sequence is from padding.
        # will be a list of 2 0/1 tensors: [(bs, tgt x P_past), (bs, tgt x P_future)]
        # 1: not all tokens are from padding, 0: all tokens in a patch are from padding.
        # For eval, no sequence packing. Each item is a sample. So only 0 and 1 in sample_id.
        sample_id.extend(
            [    # past_is_pad is in shape of "batch past_time".
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1  # Pad with 1
                            )
                            == 0
                        ).int(),  # Turn to 0 / 1; if padded is 0.
                        "... (seq patch) -> ... seq",  # (bs, P)
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",  #  (bs, tgt x P)
                    dim=past_target.shape[-1],
                ),
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
            ]
        )

        # 'past_seq_id': (bs, P_past) time id for patches in context range.
        # 'future_seq_id': (bs, P_future) time id for patches in prediction range.
        # If all the tokens in a patch (of all variates) are from padding, then the id of that patch is 0.
        # Only may appear on past's left
        # Starts from 0.  [0, 0, ..., 0, 1, 2, ...]
        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, past_observed_target
        )

        # Time id for patches in flatten sequence.
        # A list with 2*num_tgt tensors: [past_seq_id * num_tgt, future_seq_id * num_tgt]
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )

        # Variate id for patches in flatten sequence.
        # Each id is from 0 to num_tgt-1.  Repeat in a way that [0,1,2] --> [0, 0, 1, 1, 2, 2]
        # A list with 2 tensors: [(bs, P_past x num_tgt), (bs, P_future x num_tgt)]
        variate_id.extend(
            [
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                ),
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    future=self.prediction_token_length(patch_size),
                ),
            ]
        )

        dim_count += past_target.shape[-1]  # Update dim_count

        # Prediction mask. Not in the shape of flatten sequence.
        # A list of 2 Boolean tensors: [(bs, P_past * num_tgt), (bs, P_future * num_tgt)]
        prediction_mask.extend(
            [
                torch.zeros(
                    batch_shape
                    + (self.context_token_length(patch_size) * past_target.shape[-1],),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.ones(
                    batch_shape
                    + (
                        self.prediction_token_length(patch_size)
                        * past_target.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
            ]
        )

        if feat_dynamic_real is not None:
            if observed_feat_dynamic_real is None:
                raise ValueError(
                    "observed_feat_dynamic_real must be provided if feat_dynamic_real is provided"
                )

            target.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., : self.hparams.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., self.hparams.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                ]
            )
            observed_mask.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., : self.hparams.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., self.hparams.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                ]
            )
            sample_id.extend(
                [
                    repeat(
                        reduce(
                            (
                                self._patched_seq_pad(
                                    patch_size, past_is_pad, -1, left=True
                                )
                                == 0
                            ).int(),
                            "... (seq patch) -> ... seq",
                            "max",
                            patch=patch_size,
                        ),
                        "... seq -> ... (dim seq)",
                        dim=feat_dynamic_real.shape[-1],
                    ),
                    torch.ones(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                ]
            )
            time_id.extend(
                [past_seq_id] * feat_dynamic_real.shape[-1]
                + [future_seq_id] * feat_dynamic_real.shape[-1]
            )
            variate_id.extend(
                [
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                        past=self.context_token_length(patch_size),
                    ),
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                        future=self.prediction_token_length(patch_size),
                    ),
                ]
            )
            dim_count += feat_dynamic_real.shape[-1]
            prediction_mask.extend(
                [
                    torch.zeros(
                        batch_shape
                        + (
                            self.context_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.zeros(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                ]
            )

        if past_feat_dynamic_real is not None:
            if past_observed_feat_dynamic_real is None:
                raise ValueError(
                    "past_observed_feat_dynamic_real must be provided if past_feat_dynamic_real is provided"
                )
            target.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                )
            )
            observed_mask.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                )
            )
            sample_id.append(
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_feat_dynamic_real.shape[-1],
                )
            )
            time_id.extend([past_seq_id] * past_feat_dynamic_real.shape[-1])

            variate_id.append(
                repeat(
                    torch.arange(past_feat_dynamic_real.shape[-1], device=device)
                    + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                )
            )
            dim_count += past_feat_dynamic_real.shape[-1]
            prediction_mask.append(
                torch.zeros(
                    batch_shape
                    + (
                        self.context_token_length(patch_size)
                        * past_feat_dynamic_real.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                )
            )

        # Concatenate past and future along the patch axis.
        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        prediction_mask = torch.cat(prediction_mask, dim=-1)
        return (
            target,           # (bs, P_past + P_future, max_patch_size)
            observed_mask,    # (bs, P_past + P_future, max_patch_size), Boolean
            sample_id,        # (bs, P_past + P_future), 0/1
            time_id,          # (bs, P_past + P_future)
            variate_id,       # (bs, P_past + P_future)
            prediction_mask,  # (bs, P_past + P_future), Boolean
        )

    def _format_preds(
        self,
        patch_size: int,
        preds: Float[torch.Tensor, "sample batch combine_seq patch"],
        target_dim: int,
    ) -> Float[torch.Tensor, "batch sample future_time *tgt"]:
        start = target_dim * self.context_token_length(patch_size)
        end = start + target_dim * self.prediction_token_length(patch_size)
        preds = preds[..., start:end, :patch_size]
        preds = rearrange(
            preds,
            "sample ... (dim seq) patch -> ... sample (seq patch) dim",  # dim x seq = end - start
            dim=target_dim,
        )[..., : self.hparams.prediction_length, :]
        return preds.squeeze(-1)  # (batch, sample,)?

    def get_default_transform(self) -> Transformation:
        """
        Add GluonTS transformations.
        """
        transform = AsNumpyArray(
            field="target",
            expected_ndim=1 if self.hparams.target_dim == 1 else 2,
            dtype=np.float32,
        )

        if self.hparams.target_dim == 1:
            transform += ExpandDimArray(field="target", axis=0)

        transform += AddObservedValuesIndicator(
            target_field="target",
            output_field="observed_target",
            dtype=bool,
        )

        if self.hparams.feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(
                field="feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            )
            transform += AddObservedValuesIndicator(
                target_field="feat_dynamic_real",
                output_field="observed_feat_dynamic_real",
                dtype=bool,
            )

        if self.hparams.past_feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(
                field="past_feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            )
            transform += AddObservedValuesIndicator(
                target_field="past_feat_dynamic_real",
                output_field="past_observed_feat_dynamic_real",
                dtype=bool,
            )
        return transform
