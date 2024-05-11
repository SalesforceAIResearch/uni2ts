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
            var_attn_bias_layer=partial(BinaryAttentionBias),  # Binary attn bias for Variate id
            time_qk_proj_layer=partial(                        # RoPE for Time index
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,                          # Different layers use the same time RoPE QK
            d_ff=None,
        )
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_param_proj(d_model, patch_sizes)

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],        # (bs, num_patch, max_patch_size)
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],  # (bs, num_patch, max_patch_size)
        sample_id: Int[torch.Tensor, "*batch seq_len"],                 # (bs, num_patch), 0/1 in eval
        time_id: Int[torch.Tensor, "*batch seq_len"],                   # (bs, num_patch)
        variate_id: Int[torch.Tensor, "*batch seq_len"],                # (bs, num_patch)
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],          # (bs, num_patch)
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:

        """
        num_patches:
            - 'max_length' of model due to sequence packing in training;
            -  num of patches in context+prediction range in eval.

        :param target: Flatten patchified target (including future), padded to num_patch (dim 1) and max_patch_size (dim 2).
        :param observed_mask: If a time step is observed. (dim 2)
        :param sample_id: Sample_id in a packed bin. Starst from 1. If a patch is padded, its id is 0.
        :param time_id: Time id for each patch. For each sample, its time_id starts with 0 and increases in order.
        :param variate_id: Variate id for each patch. For each sample, randomly select this id in train; in order in eval.
        :param prediction_mask: Distinguish context and prediction patches
        :param patch_size: Patch size for each sample. Different samples may have different patch size.

        In MoiraiFinetune: Except sample_id, the others are from transformation.
        In MoiraiForecast: All are from _convert.
        """

        # loc, scale are (bs, num_patch, 1). The location and scale for that patch.
        # Patches have the same loc/scale if they are from the same variate and the same sample.
        # In another word, loc/scale are computed over a variate of a sample, and applied to the patches in it.
        # If patches from padding, their loc/scale are zeros and ones, respectively.
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),  # Observed and not in prediction range
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale

        # Project TS patches into representation, use the corresponding weight and bias based on patch_size.
        # (bs, num_patch, d_model)
        reprs = self.in_proj(scaled_target, patch_size)

        # ToDo: Integrate LLM's aligned embeddings here

        # Replace prediction patches with mask encoding
        masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)
        reprs = self.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),  # (bs, num_patch, num_patch). If patches are from the same sample.
            time_id=time_id,
            var_id=variate_id,
        )
        distr_param = self.param_proj(reprs, patch_size)
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
        return distr
