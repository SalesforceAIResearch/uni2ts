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
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Generator, List, Optional

import lightning as L
import numpy
import numpy as np
import torch
from einops import rearrange, reduce, repeat
from gluonts.model import Input, InputSpec
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.torch import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    CausalMeanValueImputation,
    ExpandDimArray,
    TestSplitSampler,
    Transformation,
)
from gluonts.transform.split import TFTInstanceSplitter
from jaxtyping import Bool, Float, Int

from uni2ts.transform.imputation import CausalMeanImputation

from .module import Moirai2Module


class Moirai2Forecast(L.LightningModule):
    def __init__(
        self,
        prediction_length: int,
        target_dim: int,
        feat_dynamic_real_dim: int,
        past_feat_dynamic_real_dim: int,
        context_length: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[Moirai2Module] = None,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        if module_kwargs and "attn_dropout_p" in module_kwargs:
            module_kwargs["attn_dropout_p"] = 0
        if module_kwargs and "dropout_p" in module_kwargs:
            module_kwargs["dropout_p"] = 0

        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = Moirai2Module(**module_kwargs) if module is None else module
        self.module.eval()

    @contextmanager
    def hparams_context(
        self,
        prediction_length: Optional[int] = None,
        target_dim: Optional[int] = None,
        feat_dynamic_real_dim: Optional[int] = None,
        past_feat_dynamic_real_dim: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> Generator["Moirai2Forecast", None, None]:
        kwargs = {
            "prediction_length": prediction_length,
            "target_dim": target_dim,
            "feat_dynamic_real_dim": feat_dynamic_real_dim,
            "past_feat_dynamic_real_dim": past_feat_dynamic_real_dim,
            "context_length": context_length,
        }
        old_hparams = deepcopy(self.hparams)
        for kw, arg in kwargs.items():
            if arg is not None:
                self.hparams[kw] = arg

        yield self

        for kw in kwargs:
            self.hparams[kw] = old_hparams[kw]

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
        instance_splitter = TFTInstanceSplitter(
            instance_sampler=TestSplitSampler(),
            past_length=self.past_length,
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
            forecast_generator=QuantileForecastGenerator(self.module.quantile_levels),
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
        return self.hparams.context_length

    def context_token_length(self, patch_size: int) -> int:
        return math.ceil(self.hparams.context_length / patch_size)

    def prediction_token_length(self, patch_size) -> int:
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
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
    ) -> Float[torch.Tensor, "batch num_quantiles future_time *tgt"]:
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            self.module.patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )

        per_var_context_token = self.context_token_length(self.module.patch_size)
        total_context_token = self.hparams.target_dim * per_var_context_token
        per_var_predict_token = self.prediction_token_length(self.module.patch_size)
        total_predict_token = self.hparams.target_dim * per_var_predict_token

        # In the case of 3 variables, per_var_context_token=3, per_var_predict_token=2.
        # The variate_id is: 0 0 0 1 1 1 2 2 2 [p0] [p0] [p1] [p1] [p2] [p2]
        # pred_index is initialized to be the indices of 2, 5, 8
        # assign_index is initialized to be the indices of 9, 11, 13
        pred_index = torch.arange(
            start=per_var_context_token - 1,
            end=total_context_token,
            step=per_var_context_token,
        )
        assign_index = torch.arange(
            start=total_context_token,
            end=total_context_token + total_predict_token,
            step=per_var_predict_token,
        )
        quantile_prediction = repeat(
            target,
            "... patch_size -> ... num_quantiles patch_size",
            num_quantiles=len(self.module.quantile_levels),
            patch_size=self.module.patch_size,
        ).clone()

        preds = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            training_mode=False,
        )

        def structure_multi_predict(
            per_var_predict_token,
            pred_index,
            assign_index,
            preds,
        ):
            preds = rearrange(
                preds,
                "... (predict_token num_quantiles patch_size) -> ... predict_token num_quantiles patch_size",
                predict_token=self.module.num_predict_token,
                num_quantiles=self.module.num_quantiles,
                patch_size=self.module.patch_size,
            )
            preds = rearrange(
                preds[..., pred_index, :per_var_predict_token, :, :],
                "... pred_index predict_token num_quantiles patch_size -> ... (pred_index predict_token) num_quantiles patch_size",
            )
            adjusted_assign_index = torch.cat(
                [
                    torch.arange(start=idx, end=idx + per_var_predict_token)
                    for idx in assign_index
                ]
            )
            return preds, adjusted_assign_index

        if per_var_predict_token <= self.module.num_predict_token:
            preds, adjusted_assign_index = structure_multi_predict(
                per_var_predict_token,
                pred_index,
                assign_index,
                preds,
            )
            quantile_prediction[..., adjusted_assign_index, :, :] = preds
            return self._format_preds(
                self.module.num_quantiles,
                self.module.patch_size,
                quantile_prediction,
                self.hparams.target_dim,
            )
        else:
            # expand batch_size to batch_size, num_quantiles for recursive quantile forecast
            expand_target = repeat(
                target,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_prediction_mask = repeat(
                prediction_mask,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_observed_mask = repeat(
                observed_mask,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_sample_id = repeat(
                sample_id,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_time_id = repeat(
                time_id,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_variate_id = repeat(
                variate_id,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.module.quantile_levels),
                batch_size=target.shape[0],
            ).clone()

            preds, adjusted_assign_index = structure_multi_predict(
                self.module.num_predict_token,
                pred_index,
                assign_index,
                preds,
            )
            quantile_prediction[..., adjusted_assign_index, :, :] = preds

            # choose current quantiles values as point forecast for next-step prediction
            expand_target[..., adjusted_assign_index, :] = rearrange(
                preds,
                "... predict_token num_quantiles patch_size -> ... num_quantiles predict_token patch_size",
                num_quantiles=self.module.num_quantiles,
                patch_size=self.module.patch_size,
                predict_token=self.module.num_predict_token,
            )
            expand_prediction_mask[..., adjusted_assign_index] = False

            remain_step = per_var_predict_token - self.module.num_predict_token
            while remain_step > 0:
                preds = self.module(
                    expand_target,
                    expand_observed_mask,
                    expand_sample_id,
                    expand_time_id,
                    expand_variate_id,
                    expand_prediction_mask,
                    training_mode=False,
                )

                pred_index = assign_index + self.module.num_predict_token - 1
                assign_index = pred_index + 1
                preds, adjusted_assign_index = structure_multi_predict(
                    (
                        self.module.num_predict_token
                        if remain_step - self.module.num_predict_token > 0
                        else remain_step
                    ),
                    pred_index,
                    assign_index,
                    preds,
                )
                quantile_prediction_next_step = rearrange(
                    preds,
                    "... num_quantiles_prev pred_index num_quantiles patch_size -> ... pred_index (num_quantiles_prev num_quantiles) patch_size",
                    num_quantiles=self.module.num_quantiles,
                    patch_size=self.module.patch_size,
                )
                # compute quantile forecast based on quantile forecast from quantile samples
                quantile_prediction_next_step = torch.quantile(
                    quantile_prediction_next_step,
                    torch.tensor(
                        self.module.quantile_levels,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    dim=-2,
                )
                quantile_prediction[..., adjusted_assign_index, :, :] = rearrange(
                    quantile_prediction_next_step,
                    "num_quantiles ... patch_size -> ... num_quantiles patch_size",
                )

                # choose current quantiles values as point forecast for next-step prediction
                expand_target[..., adjusted_assign_index, :] = rearrange(
                    quantile_prediction_next_step,
                    "num_quantiles batch_size predict_token patch_size -> batch_size num_quantiles predict_token patch_size",
                    num_quantiles=self.module.num_quantiles,
                    patch_size=self.module.patch_size,
                    predict_token=len(adjusted_assign_index),
                )
                expand_prediction_mask[..., adjusted_assign_index] = False

                remain_step -= self.module.num_predict_token

            return self._format_preds(
                self.module.num_quantiles,
                self.module.patch_size,
                quantile_prediction,
                self.hparams.target_dim,
            )

    def predict(
        self,
        past_target: List[Float[np.ndarray, "past_time tgt"]],
        feat_dynamic_real: Optional[
            List[Float[np.ndarray, "batch past_time tgt"]]
        ] = None,
        past_feat_dynamic_real: Optional[
            List[Float[np.ndarray, "batch past_time tgt"]]
        ] = None,
    ) -> Float[numpy.ndarray, "batch num_quantiles future_time *tgt"]:

        # only support univariate forecast now
        # implementation refer to https://github.com/awslabs/gluonts/blob/v0.15.x/src/gluonts/transform/split.py#L523
        data_entry = {
            "past_target": past_target,
            "feat_dynamic_real": feat_dynamic_real,
            "past_feat_dynamic_real": past_feat_dynamic_real,
        }

        data_entry["past_observed_target"] = [~np.isnan(x) for x in past_target]
        if feat_dynamic_real:
            data_entry["observed_feat_dynamic_real"] = [
                ~np.isnan(x) for x in feat_dynamic_real
            ]
        else:
            data_entry["observed_feat_dynamic_real"] = None

        if past_feat_dynamic_real:
            data_entry["past_observed_feat_dynamic_real"] = [
                ~np.isnan(x) for x in past_feat_dynamic_real
            ]
        else:
            data_entry["past_observed_feat_dynamic_real"] = None

        # check and use the same imputation strategy as pretraining
        impute = CausalMeanImputation()

        def process_sample(sample):
            arr = np.asarray(sample)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            if np.issubdtype(arr.dtype, np.number) and np.isnan(arr).any():
                arr = impute(arr)
            return arr

        for key, value in data_entry.items():
            if value is not None:
                data_entry[key] = [process_sample(sample) for sample in value]

        data_entry["past_is_pad"] = np.zeros(
            (len(data_entry["past_target"]), self.hparams.context_length), dtype=bool
        )

        # pad or slice context data
        for key in data_entry:
            if data_entry[key] is not None and isinstance(data_entry[key], list):
                for idx in range(len(data_entry[key])):
                    if data_entry[key][idx].shape[0] > self.hparams.context_length:
                        data_entry[key][idx] = data_entry[key][idx][
                            -self.hparams.context_length :, :
                        ]
                    else:
                        pad_length = (
                            self.hparams.context_length - data_entry[key][idx].shape[0]
                        )
                        pad_block = np.full(
                            (pad_length, 1),
                            data_entry[key][idx][0],
                            dtype=data_entry[key][idx].dtype,
                        )  # alternative: padding with 0
                        data_entry[key][idx] = np.concatenate(
                            [pad_block, data_entry[key][idx]], axis=0
                        )
                        if key == "past_target":
                            data_entry["past_is_pad"][idx, :pad_length] = True

        for k in ["past_target", "feat_dynamic_real", "past_feat_dynamic_real"]:
            if data_entry[k] is not None:
                data_entry[k] = torch.tensor(
                    np.array(data_entry[k]), device=self.device, dtype=torch.float32
                )

        for k in [
            "past_observed_target",
            "observed_feat_dynamic_real",
            "past_observed_feat_dynamic_real",
            "past_is_pad",
        ]:
            if data_entry[k] is not None:
                data_entry[k] = torch.tensor(
                    np.array(data_entry[k]), device=self.device, dtype=torch.bool
                )

        with torch.no_grad():
            predictions = self(**data_entry).detach().cpu().numpy()
        return predictions

    @staticmethod
    def _patched_seq_pad(
        patch_size: int,
        x: torch.Tensor,
        dim: int,
        left: bool = True,
        value: Optional[float] = None,
    ) -> torch.Tensor:
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
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(
            past_seq_id.cummax(dim=-1).values.cumsum(dim=-1) - 1, min=0
        )
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
        return past_seq_id, future_seq_id

    def _convert(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        future_target: Optional[Float[torch.Tensor, "batch future_time tgt"]] = None,
        future_observed_target: Optional[
            Bool[torch.Tensor, "batch future_time tgt"]
        ] = None,
        future_is_pad: Optional[Bool[torch.Tensor, "batch future_time"]] = None,
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
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

        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, past_observed_target
        )

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
        target.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(patch_size, past_target, -2, left=True),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
            ]
        )
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
                    (0, 0),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_observed_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
            ]
        )
        if future_is_pad is None:
            future_is_pad = torch.zeros(
                batch_shape + (self.hparams.prediction_length,),
                dtype=torch.long,
                device=device,
            )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
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
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )
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
        dim_count += past_target.shape[-1]
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
                        (0, 0),
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
                        (0, 0),
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
                        (0, 0),
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
                        (0, 0),
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
                    (0, 0),
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
                    (0, 0),
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

        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        prediction_mask = torch.cat(prediction_mask, dim=-1)
        return (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        )

    # Note: have not tested on multivariate data
    def _format_preds(
        self,
        num_quantiles: int,
        patch_size: int,
        preds: Float[torch.Tensor, "batch combine_seq patch"],
        target_dim: int,
    ) -> Float[torch.Tensor, "batch num_quantiles future_time *tgt"]:
        start = target_dim * self.context_token_length(patch_size)
        end = start + target_dim * self.prediction_token_length(patch_size)
        preds = preds[..., start:end, :num_quantiles, :patch_size]
        preds = rearrange(
            preds,
            "... (dim seq) num_quantiles patch -> ... num_quantiles (seq patch) dim",
            dim=target_dim,
        )[..., : self.hparams.prediction_length, :]
        return preds.squeeze(-1)

    def get_default_transform(self) -> Transformation:
        transform = AsNumpyArray(
            field="target",
            expected_ndim=1 if self.hparams.target_dim == 1 else 2,
            dtype=np.float32,
        )
        if self.hparams.target_dim == 1:
            transform += AddObservedValuesIndicator(
                target_field="target",
                output_field="observed_target",
                imputation_method=CausalMeanValueImputation(),
                dtype=bool,
            )
            transform += ExpandDimArray(field="target", axis=0)
            transform += ExpandDimArray(field="observed_target", axis=0)
        else:
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
