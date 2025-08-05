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

import math
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution

from uni2ts.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedNLLLoss,
    PackedPointLoss,
)
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
    AddSampleIndex,
    AddTimeIndex,
    AddVariateIndex,
    DummyValueImputation,
    EvalCrop,
    EvalMaskedPrediction,
    EvalPad,
    ExtendMask,
    FinetunePatchCrop,
    FixedPatchSizeConstraints,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    PackFields,
    Patchify,
    SelectFields,
    SequencifyField,
    Transformation,
)

from .module import MoiraiModule


class SetTrainableParamsMixin:
    """
    A basic version of Mixin class for setting trainable parameters. Haven't applied it in the current model.
    """

    @staticmethod
    def set_trainable_params(
        finetune_pattern: str,
        named_parameters: Generator[tuple[str, torch.nn.Parameter], None, None],
    ):
        if finetune_pattern == "full":
            pass
        elif finetune_pattern == "freeze_ffn":
            for pn, p in named_parameters:
                if "ffn" in pn:
                    p.requires_grad = False
        elif finetune_pattern == "head_only":
            for pn, p in named_parameters:
                if "param_proj" not in pn:
                    p.requires_grad = False
        else:
            raise ValueError("Unsupported finetune pattern {}".format(finetune_pattern))


class MoiraiFinetune(L.LightningModule):
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
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[MoiraiModule] = None,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        context_length: Optional[int | list[int]] = None,
        prediction_length: Optional[int | list[int]] = None,
        patch_size: Optional[int] = None,
        finetune_pattern: str | list[str] = "full",
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"

        if num_training_steps is not None:
            assert (
                num_warmup_steps <= num_training_steps
            ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = MoiraiModule(**module_kwargs) if module is None else module

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.finetune_pattern = finetune_pattern

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        distr = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size,
        )
        return distr

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        distr = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )
        loss = self.hparams.loss_func(
            pred=distr,
            **{
                field: batch[field]
                for field in [
                    "target",
                    "prediction_mask",
                    "observed_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"train/{self.hparams.loss_func.__class__.__name__}",
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
        distr = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )
        val_loss = self.hparams.loss_func(
            pred=distr,
            **{
                field: batch[field]
                for field in [
                    "target",
                    "prediction_mask",
                    "observed_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"val/{self.hparams.loss_func.__class__.__name__}",
            val_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )

        if self.hparams.val_metric is not None:
            val_metrics = (
                self.hparams.val_metric
                if isinstance(self.hparams.val_metric, list)
                else [self.hparams.val_metric]
            )
            for metric_func in val_metrics:
                if isinstance(metric_func, PackedPointLoss):
                    pred = distr.sample(torch.Size((self.hparams.num_samples,)))
                    pred = torch.median(pred, dim=0).values
                elif isinstance(metric_func, PackedDistributionLoss):
                    pred = distr
                else:
                    raise ValueError(f"Unsupported loss function: {metric_func}")

                metric = metric_func(
                    pred=pred,
                    **{
                        field: batch[field]
                        for field in [
                            "target",
                            "prediction_mask",
                            "observed_mask",
                            "sample_id",
                            "variate_id",
                        ]
                    },
                )

                self.log(
                    f"val/{metric_func.__class__.__name__}",
                    metric,
                    on_step=self.hparams.log_on_step,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    rank_zero_only=True,
                )

        return val_loss

    def configure_optimizers(self) -> dict:
        decay = set()
        no_decay = set()

        if self.finetune_pattern == "full":
            pass
        elif self.finetune_pattern == "freeze_ffn":
            for pn, p in self.named_parameters():
                if "ffn" in pn:
                    p.requires_grad = False
        elif self.finetune_pattern == "head_only":
            for pn, p in self.named_parameters():
                if "param_proj" not in pn:
                    p.requires_grad = False
        else:
            raise ValueError(
                "Unsupported finetune pattern {}".format(self.finetune_pattern)
            )

        whitelist_params = (
            LearnedProjection,
            MultiInSizeLinear,
            MultiOutSizeLinear,
            nn.Linear,
        )
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
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
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
            SchedulerType.CONSTANT,  # Use constant lr scheduler
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

    @property
    def train_transform_map(
        self,
    ) -> dict[str | type, Callable[..., Transformation]]:
        def default_train_transform(
            distance: int,
            prediction_length: int,
            context_length: int,
            patch_size: int,
        ):
            return (
                GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_sizes=self.module.patch_sizes,
                    patch_size_constraints=FixedPatchSizeConstraints(patch_size),
                    offset=True,
                )
                + FinetunePatchCrop(
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
                    randomize=False,  # Disable random variate id
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + AddSampleIndex(  # Since sequence packing is not used, need to add sample id in transformation
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    sample_id_field="sample_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + EvalMaskedPrediction(
                    mask_length=math.ceil(prediction_length / patch_size),
                    target_field="target",
                    truncate_fields=(
                        "variate_id",
                        "time_id",
                        "observed_mask",
                        "sample_id",
                    ),
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
                    field="sample_id",
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

        return defaultdict(lambda: default_train_transform)

    @property
    def val_transform_map(
        self,
    ) -> dict[str | type, Callable[..., Transformation]]:
        def default_val_transform(
            offset: int,
            distance: int,
            prediction_length: int,
            context_length: int,
            patch_size: int,
        ):
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
                    randomize=False,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + AddSampleIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    sample_id_field="sample_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + EvalMaskedPrediction(
                    mask_length=math.ceil(prediction_length / patch_size),
                    target_field="target",
                    truncate_fields=(
                        "variate_id",
                        "time_id",
                        "observed_mask",
                        "sample_id",
                    ),
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
                    field="sample_id",
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

        return defaultdict(lambda: default_val_transform)


class MoiraiLinearProbe(MoiraiFinetune): ...
