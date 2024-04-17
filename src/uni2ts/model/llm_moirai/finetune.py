

import transformers
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

from uni2ts.model.moirai import MoiraiModule
from collections.abc import Callable, Sequence
from typing import Any, Optional
import lightning as L
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import nn
from uni2ts.loss.packed import PackedDistributionLoss, PackedNLLLoss
from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)
from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear
from uni2ts.optim import SchedulerType, get_scheduler
from uni2ts.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    DefaultPatchSizeConstraints,
    DummyValueImputation,
    EvalCrop,
    EvalMaskedPrediction,
    EvalPad,
    ExtendMask,
    FixedPatchSizeConstraints,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SampleDimension,
    SelectFields,
    SequencifyField,
    Transformation,
)


def get_data_description(data: str):
    if 'ETT' in data:
        file = 'ETT'
    else:
        file = data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content


"""
Finetune this model can be:
    1) Finetune in the same way as MoiraiFinetune: No specific patch_size/context_length/prediction_length.
       Random sample sequence from the TS, mask a random ratio of subsequence as prediction range.
    2) Finetune with the given patch_size/context_length/prediction_length. Like MoiraiForecast.
"""


class LlmMoiraiFinetune(L.LightningModule):
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    )
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
    }

    def __init__(
        self,
        module_kwargs: dict[str, Any],  # Already provided in checkpoints of Moirai classes
        llm_kwargs: dict[str, Any],
        task_kwargs: dict[str, Any],  # If not provided, follow MoiraiFinetune's training strategy
        data: str,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        moirai_opt_mode='freeze'
    ):
        """
        Name moirai as module to enable to load ckpt from pretrained moirai.
        """

        super().__init__()
        self.save_hyperparameters()
        #  Pretrained weights are loaded in main function through load_from_checkpoint.
        self.module = MoiraiModule(**module_kwargs)

        # Set params related to the forecasting task.
        self.patch_size = task_kwargs['patch_size']
        self.prediction_length = task_kwargs['prediction_length']
        self.context_length = task_kwargs['context_length']

        # Load dataset description based on 'data'
        self.data_description = get_data_description(data)

    def init_after_loading_moirai(self):

        # Todo: Device?
        # Set the pretrianed LLM and tokenizer.
        self.d_llm = self.hparams.llm_kwargs['d_llm']
        self.llm_layers = self.hparams.llm_kwargs['llm_layers']
        self.llm_model = self._set_llm_model(self.hparams.llm_kwargs['llm_model'])  # LLM is frozen
        self.llm_tokenizer = self._set_llm_tokenizer(self.hparams.llm_kwargs['llm_model'])

        # Todo: multiple patch size in Moirai. To project r_txt to ts patch, we need multiple Linear layers.
        if isinstance(self.patch_size, int):
            self.projector = nn.Linear(self.d_llm, self.patch_size)
        else:
            self.projector = nn.ModuleList([nn.Linear(self.d_llm, patch_size) for patch_size in self.module.patch_sizes])

    def forward(
            self,
            target: Float[torch.Tensor, "*batch seq_len max_patch"],
            observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
            sample_id: Int[torch.Tensor, "*batch seq_len"],
            time_id: Int[torch.Tensor, "*batch seq_len"],
            variate_id: Int[torch.Tensor, "*batch seq_len"],
            prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
            patch_size: Int[torch.Tensor, "*batch seq_len"], # Can different patches in a sample have differnet patch_sizes?
            num_samples: Optional[int] = None,
    ) -> Float[torch.Tensor, "*batch sample seq_len max_patch"]:

        """
        Inputs have been patchfied and flattened.
        Add prompt prefix to the sequence.
        Add a new field of prompt? --> Need to modify moirai module
        """

        # For each TS in the batch, generate a prompt
        prompt = []
        for b in range(target.size(0)):  # Todo: Is it the correct way to get batch_size?
            prompt_ = self._get_sample_prompt()
            prompt.append(prompt_)

        #  Get LLM reprs of prompt. Tensor: [bs, prompt_len, d_llm]
        prompt = self.llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        prompt_reprs = self.llm_model(input_ids=prompt.input_ids,
                                      attention_mask=prompt.attention_mask).last_hidden_state
        # prompt_reprs = prompt_reprs[:, :, :self.d_ff]
        # prompt_reprs = torch.reshape(prompt_reprs, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # prompt_reprs = prompt_reprs.permute(0, 1, 3, 2).contiguous()

        # Use projector to map r_p to e_p. Tensor:[bs, prompt_len, patch_size/max_patch]?
        if isinstance(self.patch_size, int):
            prompt_prefix = self.projector(prompt_reprs)
        else:
            # Todo: Use the specific patch_size for each sample...
            prompt_prefix = self.projector[patch_size](prompt_reprs)

        # Todo: Prepend prompt, modify the masks and ids.
        prompt_target = torch.cat([prompt_prefix, target], dim=1)

        distr = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            patch_size,
        )
        preds = distr.sample(torch.Size((num_samples or self.hparams.num_samples,)))
        return rearrange(preds, "n b ... -> b n ...")

    def loss(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],  # Can different patches in a sample have differnet patch_sizes?
    ) -> Float[torch.Tensor, ""]:

        # For each TS in the batch, generate a prompt
        prompt = []
        for b in range(target.size(0)):  # Todo: Is it the correct way to get batch_size?
            prompt_ = self._get_sample_prompt()
            prompt.append(prompt_)

        #  Get LLM reprs of prompt. Tensor: [bs, prompt_len, d_llm]
        prompt = self.llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        prompt = prompt.to(self.llm_model.device)  # Is it okay if using multi GPU?
        prompt_reprs = self.llm_model(input_ids=prompt.input_ids,
                                      attention_mask=prompt.attention_mask).last_hidden_state
        # prompt_reprs = prompt_reprs[:, :, :self.d_ff]
        # prompt_reprs = torch.reshape(prompt_reprs, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # prompt_reprs = prompt_reprs.permute(0, 1, 3, 2).contiguous()

        # Use projector to map r_p to e_p. Tensor:[bs, prompt_len, patch_size/max_patch]?
        if isinstance(self.patch_size, int):
            prompt_prefix = self.projector(prompt_reprs)
        else:
            # Todo: Use the specific patch_size for each sample...
            # batch_size, seq_len, _ = prompt_reprs.shape
            # outputs = torch.zeros_like(x)
            #
            # # Iterate over each element in the batch and sequence
            # for b in range(batch_size):
            #     for s in range(seq_len):
            #         # Find the index of the projector based on the patch size
            #         projector_index = self.patch_sizes.index(patch_size[b, s].item())
            #         # Select the appropriate FC layer and apply it
            #         outputs[b, s] = self.projector[projector_index](x[b, s])

            prompt_prefix = self.projector[patch_size](prompt_reprs)

        # Todo:
        # Patch size doesn't get involved in training...
        # target's last dim is max_patch_size, 128 for example.
        # How to map r_p to e_p? How to concat?

        # Todo: Prepend prompt, modify the masks and ids.
        prompt_target = torch.cat([prompt_prefix, target], dim=1)

        distr = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            patch_size,
        )
        loss = self.hparams.loss_func(
            pred=distr,
            target=target,
            prediction_mask=prediction_mask,
            observed_mask=observed_mask,
            sample_id=sample_id,
            variate_id=variate_id,
        )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """"
        Same as MoiraiFinetune
        """

        self.llm_model.eval()
        # todo: To freeze dropout in LLM and Moirai (if frozen)
        # for module in self.module.modules():
        #     if isinstance(module, nn.Dropout):
        #         module.eval()

        loss = self.loss(**batch)
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            self.hparams.loss_func.__class__.__name__,
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """"
        Same as MoiraiFinetune
        """
        loss = self.loss(**batch)
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            "val_loss",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss

    def configure_optimizers(self) -> dict:
        """
        Set the optimizer:
            - Freeze params in LLM.
            - Always update params in Projector
            - Handle params in Moirai based on  moirai_opt_mode
        Follow the same optimizer setup as MoiraiFinetune.
        """
        # Freeze Moirai. Only update projectors
        if self.hparams.moirai_opt_mode == 'freeze':
            for param in self.module.parameters():
                param.requires_grad = False

        # Freeze params except for LN in Moirai.
        elif self.hparams.moirai_opt_mode == 'LN':
            for mn, m in self.module.named_modules():
                for pn, p in m.named_parameters():
                    if isinstance(m, nn.LayerNorm) or not p.requires_grad:
                        continue
                    else:
                        p.requires_grad = False

        # Finetune all the params of morai.
        elif self.hparams.moirai_opt_mode == 'full':
            pass
        else:
            raise NotImplementedError

        decay = set()
        no_decay = set()

        # Decay
        whitelist_params = (
            LearnedProjection,
            MultiInSizeLinear,
            MultiOutSizeLinear,
            nn.Linear,
        )

        # No decay
        blacklist_params = (
            BinaryAttentionBias,
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):  # All bias no decay
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):  # Weights in white decay
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):  # Weights in black no decay
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }

    def _set_llm_model(self, llm_model):
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
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
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

    # def _load_pretrianed_moirai_moudle(self):
    #     return None

    def _get_sample_prompt(self):
        # Dataset Description
        prompt = (
            f"<|start_prompt|>Dataset description: {self.data_description};"
        )

        # Task description if task params are known.
        if self.prediction_length and self.context_length:
            prompt += f"Task description: forecast the next {str(self.prediction_length)} steps given the previous {str(self.context_length)} steps information; "

        # Todo: Compute the statistics for each sample.
        # In Moirai, each sample is a MTS? --> Compute channel-wise statistics.
        # It has been flattened. Need to use variate_id to compute.
        # Do we need to consider pad/missing values when computing statistics?
        # if self.hparams.prompt_statistics:
        #     for channel in data_variates_names:
        #         min_values_str = str(min_values[b].tolist()[0])
        #         max_values_str = str(max_values[b].tolist()[0])
        #         median_values_str = str(medians[b].tolist()[0])
        #         lags_values_str = str(lags[b].tolist())
        #         prompt_ += f"Input statistics of {channel}: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}, the trend of input is {'upward' if trends > 0 else 'downward'}, top 5 lags are : {lags_values_str};"

        prompt += "<|<end_prompt>|>"
        return prompt

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def create_train_transform(self) -> Transformation:
        # Out of the PL Trainer pipeline, a custom step.
        # Called in cli/finetune.py to process the training dataset.
        # Todo: Can we specify context_length and prediction_length in Finetuning?
        return (
            SampleDimension(
                max_dim=self.hparams.max_dim,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            # add a new field of "patch_size" to data dict, randomly choose from range based on frequency
            + GetPatchSize(
                min_time_patches=self.hparams.min_patches,
                target_field="target",
                patch_sizes=self.module.patch_sizes,
                patch_size_constraints=DefaultPatchSizeConstraints(),
                offset=True,
            )
            # Crop fields in a data_entry in the temporal dimension based on a patch_size.
            + PatchCrop(
                min_time_patches=self.hparams.min_patches,
                max_patches=self.module.max_seq_len,
                will_flatten=True,
                offset=True,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + PackFields(
                output_field="target",
                fields=("target",),
            )
            + PackFields(
                output_field="past_feat_dynamic_real",
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
            )
            # Add a new field 'observed_mask'. Observed or missing: nan are False.
            + AddObservedMask(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                observed_mask_field="observed_mask",
                collection_type=dict,
            )
            # Impute the nan values.
            + ImputeTimeSeries(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                imputation_method=DummyValueImputation(value=0.0),
            )
            # Patching the series in fields into patches based on patch_size
            + Patchify(
                max_patch_size=max(self.module.patch_sizes),
                fields=("target", "observed_mask"),
                optional_fields=("past_feat_dynamic_real",),
            )
            # Add a new field "variate_id".
            + AddVariateIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                variate_id_field="variate_id",
                expected_ndim=3,
                max_dim=self.hparams.max_dim,
                randomize=True,
                collection_type=dict,
            )
            # Add a new field "time_id".
            + AddTimeIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                time_id_field="time_id",
                expected_ndim=3,
                collection_type=dict,
            )
            # Add a new field "prediction_mask"
            + MaskedPrediction(
                min_mask_ratio=self.hparams.min_mask_ratio,
                max_mask_ratio=self.hparams.max_mask_ratio,
                target_field="target",
                truncate_fields=("variate_id", "time_id", "observed_mask"),
                optional_truncate_fields=("past_feat_dynamic_real",),
                prediction_mask_field="prediction_mask",
                expected_ndim=3,
            )
            # Extend prediction_mask
            + ExtendMask(
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                mask_field="prediction_mask",
                expected_ndim=3,
            )
            + FlatPackCollection(
                field="variate_id",
                feat=False,
            )
            + FlatPackCollection(
                field="time_id",
                feat=False,
            )
            + FlatPackCollection(
                field="prediction_mask",
                feat=False,
            )
            + FlatPackCollection(
                field="observed_mask",
                feat=True,
            )
            + FlatPackFields(
                output_field="target",
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                feat=True,
            )
            + SequencifyField(field="patch_size", target_field="target")
            + SelectFields(fields=list(self.seq_fields))
        )

    def create_val_transform(
        self,
        offset: int,
        distance: int,
        prediction_length: int,
        context_length: int,
        patch_size: int,
    ) -> Transformation:
        return (
            GetPatchSize(
                min_time_patches=2,
                target_field="target",
                patch_sizes=self.module.patch_sizes,
                patch_size_constraints=FixedPatchSizeConstraints(patch_size),
                offset=True,
            )
            + EvalCrop(
                offset,
                distance,
                prediction_length,
                context_length,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + PackFields(
                output_field="target",
                fields=("target",),
            )
            + PackFields(
                output_field="past_feat_dynamic_real",
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
            )
            + EvalPad(
                prediction_pad=-prediction_length % patch_size,
                context_pad=-context_length % patch_size,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddObservedMask(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                observed_mask_field="observed_mask",
                collection_type=dict,
            )
            + ImputeTimeSeries(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                imputation_method=DummyValueImputation(value=0.0),
            )
            + Patchify(
                max_patch_size=max(self.module.patch_sizes),
                fields=("target", "observed_mask"),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddVariateIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                variate_id_field="variate_id",
                expected_ndim=3,
                max_dim=self.hparams.max_dim,
                randomize=True,
                collection_type=dict,
            )
            + AddTimeIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                time_id_field="time_id",
                expected_ndim=3,
                collection_type=dict,
            )
            + EvalMaskedPrediction(
                mask_length=-prediction_length % patch_size,
                target_field="target",
                truncate_fields=("variate_id", "time_id", "observed_mask"),
                optional_truncate_fields=("past_feat_dynamic_real",),
                prediction_mask_field="prediction_mask",
                expected_ndim=3,
            )
            + ExtendMask(
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                mask_field="prediction_mask",
                expected_ndim=3,
            )
            + FlatPackCollection(
                field="variate_id",
                feat=False,
            )
            + FlatPackCollection(
                field="time_id",
                feat=False,
            )
            + FlatPackCollection(
                field="prediction_mask",
                feat=False,
            )
            + FlatPackCollection(
                field="observed_mask",
                feat=True,
            )
            + FlatPackFields(
                output_field="target",
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                feat=True,
            )
            + SequencifyField(field="patch_size", target_field="target")
            + SelectFields(fields=list(self.seq_fields))
        )
