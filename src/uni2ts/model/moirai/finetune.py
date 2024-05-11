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

from collections.abc import Callable, Sequence
from typing import Any, Optional

import lightning as L
import lightning.pytorch as pl
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

from .module import MoiraiModule


class MoiraiFinetune(L.LightningModule):
    """
    Moirai for Fine-tuning
    """

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
        module_kwargs: dict[str, Any],
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
    ):
        """
        MoiraiFinetune doesn't specify context_length/prediction_length/patch_size.
        These configs are sampled from predefined ranges during training.
        And they are specified when creating val datasets during validation.
        """

        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters()
        self.module = MoiraiModule(**module_kwargs)

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],  # max_seq_len, max_patch_size
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],  # 0/1. If a patch is padded (dim 1).
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
        num_samples: Optional[int] = None,
    ) -> Float[torch.Tensor, "*batch sample seq_len max_patch"]:
        """
        See MoiraiModule.forward() for explaination of these params
        """
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
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, ""]:
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

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # `**` unpacks a dict's items: each key-value pair is passed as a kwarg to the function
        loss = self.loss(**batch)

        # Why compute batch_size like this?
        # To remove the samples with all padding patches from a batch.
        # I.e. A sample with all sample_ids are 0.
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

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
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

        # # Set all params to Non-trainable
        # for pn, p in self.named_parameters():
        #     if p.requires_grad:
        #         p.requires_grad = False
        # # Finetune Norm layers
        # for mn, m in self.named_modules():
        #     if isinstance(m, RMSNorm):
        #         for pn, p in m.named_parameters():
        #             p.requires_grad = True
        #
        #     if isinstance(m, MultiInSizeLinear):
        #         for pn, p in m.named_parameters():
        #             p.requires_grad = True
        #
        #     if isinstance(m, MultiOutSizeLinear):
        #         for pn, p in m.named_parameters():
        #             p.requires_grad = True

    ##################################################
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
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

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

    def create_train_transform(self) -> Transformation:
        """
        Transformation per sample for train dataset.
        Called in cli/finetune.py to process the training dataset.
        By default, each data_entry is the entire record of a channel.
        """

        return (
            # # ToDo: What is the aim of this?  Remove if ds is mode of 'wide_multivariate'
            SampleDimension(
                max_dim=self.hparams.max_dim,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            ) +


            # add a new field of "patch_size" to data dict.
            # randomly choose from range based on frequency
            # ToDo: each sample has a different patch_size?
            GetPatchSize(
                min_time_patches=self.hparams.min_patches,
                target_field="target",
                patch_sizes=self.module.patch_sizes,  # (64,)  self.module.patch_sizes
                patch_size_constraints=DefaultPatchSizeConstraints(),
                offset=True,
            )

            # Crop fields in a data_entry in the temporal dimension based on a patch_size.
            # Sequences in fields will be cropped into a size of random multiple of patch sizes.
            # Crop size (num_patches) is randomly sampled from [min_time_patches, max_time_patches].
            # Start point of cropping is randomly selected. So each sample has a different num_patches.
            # ToDo: Design a crop transformation to crop a given fixed context length and prediction length
            + PatchCrop(
                min_time_patches=self.hparams.min_patches,  # minimum number of patches for time dimension
                max_patches=self.module.max_seq_len,  # max number of patches for time * dim dimension (if flatten)
                will_flatten=True,  # Sequences will be flatten subsequently.
                offset=True,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )

            # Todo: Why need Pack?
            + PackFields(
                output_field="target",
                fields=("target",),
            )
            + PackFields(
                output_field="past_feat_dynamic_real",
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
            )
            # Add a new field 'observed_mask'. Nan are False.
            + AddObservedMask(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                observed_mask_field="observed_mask",
                collection_type=dict,
            )
            # Impute nan values.
            + ImputeTimeSeries(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                imputation_method=DummyValueImputation(value=0.0),
            )
            # Patchify TS record into patches.
            # No matter used patch_size, pad all the patches to max_patch_size.
            # Shape of patchified fields: (1, n_patch, max_patch_size)
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
                randomize=True,  # Why randomize?
                collection_type=dict,
            )
            # Add Time_id for each patch. These ids are in order.
            + AddTimeIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                time_id_field="time_id",
                expected_ndim=3,
                collection_type=dict,
            )

            # Add a new field "prediction_mask". Random mask patches in the end for prediction.
            # Mask ratio is uniformly sampled from [min_mask_ratio, max_mask_ratio]
            # For truncate_fields, truncate the part corresponding to prediction patches.
            # ToDo: Different samples have different prediction masks?
            # ToDo: Use EvalMaskedPrediction for a specific prediction length?
            + MaskedPrediction(
                min_mask_ratio=self.hparams.min_mask_ratio,
                max_mask_ratio=self.hparams.max_mask_ratio,
                target_field="target",
                truncate_fields=("variate_id", "time_id", "observed_mask"),
                optional_truncate_fields=("past_feat_dynamic_real",),
                prediction_mask_field="prediction_mask",
                expected_ndim=3,
            )
            # Extend prediction_mask for "past_feat_dynamic_real" (If it exists)
            # set another prediction mask with all False for it in field "prediction_mask".
            # So there will be 2 items in field "prediction_mask".
            + ExtendMask(
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                mask_field="prediction_mask",
                expected_ndim=3,
            )

            # Turn item in field into nparray. Flat along time dimension then pack.
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
        offset: int,                # Offset to val split. After offset is used.
        distance: int,              # distance bt prediction windows, equal to prediction_length
        prediction_length: int,
        context_length: int,
        patch_size: int,
    ) -> Transformation:

        """
        Transformation per sample for val dataset.
        These args are from the config yaml of finetune's val dataset.
        By default, each data_entry is the entire record of a channel.
        Be called when preparing a batch of data (__get_item__)
        """
        return (
            # Initial 'target' is [(L, ), ..., (L, )], as _pa_column_to_numpy of HuggingFaceDatasetIndexer.
            # Only 1 series in 'target' if build data in 'wide'. Have multi series if build in 'wide_multivariate'
            # Add a new field of "patch_size" to data dict
            # randomly choose from range based on frequency
            GetPatchSize(
                min_time_patches=2,
                target_field="target",
                patch_sizes=self.module.patch_sizes,   # (64,)  self.module.patch_sizes
                patch_size_constraints=FixedPatchSizeConstraints(patch_size),
                offset=True,
            )

            # For each sample, crop the [prediction_length context_length] region of sequence
            # The region is computed based on its 'window' (window id)
            + EvalCrop(
                offset,
                distance,
                prediction_length,
                context_length,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )

            # Turn variables in fields as nparray. Then pack the fields.
            # Make no difference for wide data. Then 'target' is nparray:  (1, L)
            # For wide_mts data, pack the series into MTS. Then 'target' is a nparray: (C, L)
            + PackFields(
                output_field="target",
                fields=("target",),
            )
            + PackFields(
                output_field="past_feat_dynamic_real",
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
            )

            # Pad along the time dimension so that can get a multiple of patches
            # Padded values are Nan.
            + EvalPad(
                prediction_pad=-prediction_length % patch_size,  # -: compute how much to pad
                context_pad=-context_length % patch_size,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )

            # Add observed_mask. Nan is False. Same shape of target.
            + AddObservedMask(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                observed_mask_field="observed_mask",
                collection_type=dict,
            )

            # Impute Nan
            + ImputeTimeSeries(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                imputation_method=DummyValueImputation(value=0.0),
            )

            # Patchify TS record into patches.
            # No matter used patch_size, pad all the patches to max_patch_size!
            # Shape of patchified fields: (1, n_patch, max_patch_size)
            + Patchify(
                max_patch_size=max(self.module.patch_sizes),
                fields=("target", "observed_mask"),
                optional_fields=("past_feat_dynamic_real",),
            )

            # ToDo: What does it mean? Why set randomize=True?
            #  It seems that add a random number to each data_entry. No matter the Variate.
            + AddVariateIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                variate_id_field="variate_id",
                expected_ndim=3,
                max_dim=self.hparams.max_dim,
                randomize=True,
                collection_type=dict,
            )

            # Add Time_id for each patch. These ids are in order.
            + AddTimeIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                time_id_field="time_id",
                expected_ndim=3,
                collection_type=dict,
            )

            # Todo: mask_length?
            #  Now it is the length of padding in prediction.
            + EvalMaskedPrediction(
                # mask_length=-prediction_length % patch_size,
                mask_length=prediction_length // patch_size,  # num of patches for prediction?
                target_field="target",
                truncate_fields=("variate_id", "time_id", "observed_mask"),
                optional_truncate_fields=("past_feat_dynamic_real",),
                prediction_mask_field="prediction_mask",
                expected_ndim=3,
            )
            # Extend prediction_mask for "past_feat_dynamic_real" (If it exists)
            # set another prediction mask with all False for it in field "prediction_mask".
            # So there will be 2 items in field "prediction_mask".
            + ExtendMask(
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                mask_field="prediction_mask",
                expected_ndim=3,
            )
            # Turn item in field into nparray. Flat multiple variates into a sequence.
            # If feat=False, field should be    then output shape is (num_patch, )
            # If feat=True, field should be     then output shape is (num_patch, feat). feat represents max_patch_size here.
            # * is the number of patches
            + FlatPackCollection(
                field="variate_id",
                feat=False,
            )
            + FlatPackCollection(
                field="time_id",
                feat=False,
            )
            # If "past_feat_dynamic_real" exist, * will be 2* num_patches?
            + FlatPackCollection(
                field="prediction_mask",
                feat=False,
            )
            + FlatPackCollection(
                field="observed_mask",
                feat=True,
            )
            # Seems pack target and past_feat_dynamic_real, and set it as target.
            # If "past_feat_dynamic_real" exist, * will be 2* num_patches?
            + FlatPackFields(
                output_field="target",
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                feat=True,
            )
            # Transform patch_size from int to a sequence (num_patches, patch_size).
            # For each sample, patch_size is the same.
            + SequencifyField(field="patch_size", target_field="target")
            # Only use self.seq_fields for a sample (dict).
            # Discard fields: item_id, freq, start, which are created by building datasets.
            + SelectFields(fields=list(self.seq_fields))
        )
    # field 'sample_id' is added after getting a batch of data.
    # It indicates the padding patches which are padded to the max_patch_num in a batch.
    # Therefore, it is not sample-wise transformation, Not in the transformation chain.

class MoiraiLinearProbe(MoiraiFinetune): ...
