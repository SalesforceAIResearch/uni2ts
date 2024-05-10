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

from typing import Optional

import numpy as np
import pytest
import torch

from uni2ts.distribution import StudentTOutput
from uni2ts.loss.packed import PackedDistributionLoss, PackedNLLLoss
from uni2ts.model.moirai import MoiraiForecast, MoiraiPretrain


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 16],
)
@pytest.mark.parametrize("d_model, num_layers", [(64, 1), (128, 2)])
@pytest.mark.parametrize(
    "max_seq_len, max_dim",
    [(128, 128), (256, 256)],
)
@pytest.mark.parametrize(
    "patch_sizes",
    [(8, 16, 32), (8, 16, 32, 64)],
)
@pytest.mark.parametrize(
    "scaling",
    [True, False],
)
def test_moirai_pretrain(
    batch_size: int,
    d_model: int,
    num_layers: int,
    max_seq_len: int,
    max_dim: int,
    patch_sizes: tuple[int, ...],
    scaling: bool,
    loss_func: PackedDistributionLoss = PackedNLLLoss(),
):
    model = MoiraiPretrain(
        module_kwargs=dict(
            distr_output=StudentTOutput(),
            d_model=d_model,
            num_layers=num_layers,
            patch_sizes=patch_sizes,
            max_seq_len=max_seq_len,
            attn_dropout_p=0.0,
            dropout_p=0.0,
            scaling=scaling,
        ),
        min_mask_ratio=0.1,
        max_mask_ratio=0.2,
        max_dim=max_dim,
        min_patches=1,
        loss_func=loss_func,
        num_training_steps=10,
        num_warmup_steps=1,
    )

    target = torch.randn(batch_size, max_seq_len, max(patch_sizes))
    observed_mask = torch.ones(
        batch_size, max_seq_len, max(patch_sizes), dtype=torch.bool
    )
    sample_id = torch.ones(batch_size, max_seq_len, dtype=torch.int64)
    time_id = (
        torch.arange(max_seq_len, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
    )
    variate_id = torch.randint(max_dim, (batch_size, max_seq_len), dtype=torch.long)
    prediction_mask = torch.cat(
        [
            torch.zeros(batch_size, max_seq_len // 2, dtype=torch.bool),
            torch.ones(batch_size, max_seq_len // 2, dtype=torch.bool),
        ],
        dim=-1,
    )
    patch_size = torch.ones(batch_size, max_seq_len, dtype=torch.long) * 16

    distr = model(
        target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
        patch_size,
    )
    prediction = distr.sample((model.hparams.num_samples,))
    loss = model.hparams.loss_func(
        pred=distr,
        target=target,
        prediction_mask=prediction_mask,
        observed_mask=observed_mask,
        sample_id=sample_id,
        variate_id=variate_id,
    )

    assert prediction.shape == (
        model.hparams.num_samples,
        batch_size,
        max_seq_len,
        max(patch_sizes),
    )
    assert not prediction.isnan().any()
    assert loss.shape == ()
    assert not loss.isnan()


@pytest.mark.parametrize(
    "target_dim",
    [1, 2],
)
@pytest.mark.parametrize(
    "past_feat_dynamic_real_dim",
    [None, 1, 2],
)
def test_moirai_transform_map(
    create_data_entry,
    target_dim: int,
    past_feat_dynamic_real_dim: Optional[int],
    length: int = 1000,
    max_seq_len: int = 256,
    max_dim: int = 256,
    patch_sizes: tuple[int, ...] = (8, 16, 32, 64),
    d_model: int = 64,
    num_layers: int = 2,
    scaling: bool = True,
    loss_func: PackedDistributionLoss = PackedNLLLoss(),
):
    torch.manual_seed(0)
    np.random.seed(0)
    model = MoiraiPretrain(
        module_kwargs=dict(
            distr_output=StudentTOutput(),
            d_model=d_model,
            num_layers=num_layers,
            patch_sizes=patch_sizes,
            max_seq_len=max_seq_len,
            attn_dropout_p=0.0,
            dropout_p=0.0,
            scaling=scaling,
        ),
        max_dim=max_dim,
        min_mask_ratio=0.1,
        max_mask_ratio=0.2,
        min_patches=1,
        loss_func=loss_func,
        num_training_steps=10,
        num_warmup_steps=1,
    )
    transform_map = model.train_transform_map
    transform = transform_map["default"]()

    data_entry = create_data_entry(
        length=length,
        freq="H",
        target_dim=target_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
    )
    transformed_data_entry = transform(data_entry)

    # check field
    for field in MoiraiPretrain.seq_fields:
        assert (
            field in transformed_data_entry
        ), f"transformed_data_entry should have field {field}"

    # check length
    seq_len = len(data_entry["target"])
    for k, v in transformed_data_entry.items():
        assert len(v) == seq_len, (
            f"transformed_data_entry[{k}] has length {len(v)} "
            f"but should have length {seq_len}"
        )


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("context_length", [101, 152, 203])
@pytest.mark.parametrize("prediction_length", [20])
@pytest.mark.parametrize(
    "target_dim, feat_dynamic_real_dim, past_feat_dynamic_real_dim",
    [
        (1, 0, 0),
        (2, 0, 0),
        (1, 1, 0),
        (2, 2, 0),
        (1, 1, 1),
        (2, 2, 2),
    ],
)
@pytest.mark.parametrize("patch_size", [2, 4, 8])
def test_moirai_forecast(
    batch_size: int,
    context_length: int,
    prediction_length: int,
    target_dim: int,
    feat_dynamic_real_dim: int,
    past_feat_dynamic_real_dim: int,
    patch_size: int,
    num_samples: int = 2,
):
    model = MoiraiForecast(
        module_kwargs=dict(
            distr_output=StudentTOutput(),
            d_model=128,
            num_layers=2,
            patch_sizes=(2, 4, 8),
            max_seq_len=128,
            attn_dropout_p=0.0,
            dropout_p=0.0,
            scaling=True,
        ),
        context_length=context_length,
        prediction_length=prediction_length,
        target_dim=target_dim,
        feat_dynamic_real_dim=feat_dynamic_real_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        patch_size=patch_size,
        num_samples=num_samples,
    )

    inputs = model.describe_inputs(batch_size).zeros()
    inputs["past_observed_target"] = torch.ones_like(inputs["past_observed_target"])
    prediction = model(**inputs)
    if target_dim == 1:
        assert prediction.shape == (batch_size, num_samples, prediction_length)
    else:
        assert prediction.shape == (
            batch_size,
            num_samples,
            prediction_length,
            target_dim,
        )


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("context_length", [101, 152, 203])
@pytest.mark.parametrize("prediction_length", [20])
@pytest.mark.parametrize(
    "target_dim, feat_dynamic_real_dim, past_feat_dynamic_real_dim",
    [
        (1, 0, 0),
        (2, 0, 0),
        (1, 1, 0),
        (2, 2, 0),
        (1, 1, 1),
        (2, 2, 2),
    ],
)
def test_moirai_forecast_auto(
    batch_size: int,
    context_length: int,
    prediction_length: int,
    target_dim: int,
    feat_dynamic_real_dim: int,
    past_feat_dynamic_real_dim: int,
    num_samples: int = 2,
):
    model = MoiraiForecast(
        module_kwargs=dict(
            distr_output=StudentTOutput(),
            d_model=128,
            num_layers=2,
            patch_sizes=(2, 4, 8),
            max_seq_len=128,
            attn_dropout_p=0.0,
            dropout_p=0.0,
            scaling=True,
        ),
        context_length=context_length,
        prediction_length=prediction_length,
        target_dim=target_dim,
        feat_dynamic_real_dim=feat_dynamic_real_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        patch_size="auto",
        num_samples=num_samples,
    )

    inputs = model.describe_inputs(batch_size).zeros()
    inputs["past_observed_target"] = torch.ones_like(inputs["past_observed_target"])
    prediction = model(**inputs)
    if target_dim == 1:
        assert prediction.shape == (batch_size, num_samples, prediction_length)
    else:
        assert prediction.shape == (
            batch_size,
            num_samples,
            prediction_length,
            target_dim,
        )


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("context_length", [101, 152, 203])
@pytest.mark.parametrize("prediction_length", [20])
@pytest.mark.parametrize(
    "target_dim, feat_dynamic_real_dim, past_feat_dynamic_real_dim",
    [
        (1, 0, 0),
        (2, 0, 0),
        (1, 1, 0),
        (2, 2, 0),
        (1, 1, 1),
        (2, 2, 2),
    ],
)
@pytest.mark.parametrize("patch_size", [2, 4, 8])
def test_moirai_forecast_hparams_context(
    batch_size: int,
    context_length: int,
    prediction_length: int,
    target_dim: int,
    feat_dynamic_real_dim: int,
    past_feat_dynamic_real_dim: int,
    patch_size: int,
    num_samples: int = 2,
):
    model = MoiraiForecast(
        module_kwargs=dict(
            distr_output=StudentTOutput(),
            d_model=128,
            num_layers=2,
            patch_sizes=(2, 4, 8),
            max_seq_len=128,
            attn_dropout_p=0.0,
            dropout_p=0.0,
            scaling=True,
        ),
        context_length=999,
        prediction_length=999,
        target_dim=999,
        feat_dynamic_real_dim=999,
        past_feat_dynamic_real_dim=999,
        patch_size=999,
        num_samples=999,
    )

    with model.hparams_context(
        prediction_length=prediction_length,
        target_dim=target_dim,
        feat_dynamic_real_dim=feat_dynamic_real_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=num_samples,
    ) as model:
        inputs = model.describe_inputs(batch_size).zeros()
        inputs["past_observed_target"] = torch.ones_like(inputs["past_observed_target"])
        prediction = model(**inputs)
        if target_dim == 1:
            assert prediction.shape == (batch_size, num_samples, prediction_length)
        else:
            assert prediction.shape == (
                batch_size,
                num_samples,
                prediction_length,
                target_dim,
            )

    assert model.hparams.prediction_length == 999
    assert model.hparams.target_dim == 999
    assert model.hparams.feat_dynamic_real_dim == 999
    assert model.hparams.past_feat_dynamic_real_dim == 999
    assert model.hparams.context_length == 999
    assert model.hparams.patch_size == 999
    assert model.hparams.num_samples == 999
