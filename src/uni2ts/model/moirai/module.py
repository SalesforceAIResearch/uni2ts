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

from functools import partial

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution

from uni2ts.common.torch_util import mask_fill, packed_attention_mask
from uni2ts.distribution import DistributionOutput
from uni2ts.module.norm import RMSNorm
from uni2ts.module.packed_scaler import PackedNOPScaler, PackedStdScaler
from uni2ts.module.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from uni2ts.module.transformer import TransformerEncoder
from uni2ts.module.ts_embed import MultiInSizeLinear


class MoiraiModule(nn.Module):
    """Contains components of Moirai to ensure implementation is identical across models"""

    def __init__(
        self,
        distr_output: DistributionOutput,
        d_model: int,
        num_layers: int,
        patch_sizes: tuple[int, ...],
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_sizes = patch_sizes
        self.max_seq_len = max_seq_len
        self.scaling = scaling

        # how does this mask_encoding get learned?
        self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.in_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),  # Variate only
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,  # Shared
            d_ff=None,
        )
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_param_proj(d_model, patch_sizes)

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],        # (bs, P_past + P_future, patch_size)
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],  # (bs, P_past + P_future, patch_size)
        sample_id: Int[torch.Tensor, "*batch seq_len"],                 # (bs, P_past + P_future), 0/1
        time_id: Int[torch.Tensor, "*batch seq_len"],                   # (bs, P_past + P_future)
        variate_id: Int[torch.Tensor, "*batch seq_len"],                # (bs, P_past + P_future)
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],          # (bs, P_past + P_future)
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:

        # loc, scale are (bs, P_past + P_future, 1). The location and scale for that patch.
        # Patches have the same loc/scale if they are from the same variate.
        # If patches from padding, their loc/scale are zeros.
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),  # Observed and not in prediction range
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale

        # Project TS patches into representation, use the corresponding projection layer based on patch_size
        # (bs, P, d_model)
        reprs = self.in_proj(scaled_target, patch_size)

        # ToDo: Integrate LLM's aligned embeddings here

        # Replace prediction patches with mask encoding
        masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)
        reprs = self.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )
        distr_param = self.param_proj(reprs, patch_size)
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
        return distr
