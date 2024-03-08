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

from typing import Callable, Optional

import torch
from jaxtyping import Float, PyTree
from torch.distributions import Pareto
from torch.nn import functional as F

from ._base import DistributionOutput


class ParetoOutput(DistributionOutput):
    distr_cls = Pareto
    args_dim = dict(scale=1, alpha=1)

    @property
    def domain_map(
        self,
    ) -> PyTree[
        Callable[[Float[torch.Tensor, "*batch 1"]], Float[torch.Tensor, "*batch"]], "T"
    ]:
        return dict(scale=self._scale, alpha=self._alpha)

    def _scale(
        self, scale: Float[torch.Tensor, "*batch 1"]
    ) -> Float[torch.Tensor, "*batch"]:
        epsilon = torch.finfo(scale.dtype).eps
        return F.softplus(scale).clamp_min(epsilon).squeeze(-1)

    def _alpha(
        self, alpha: Float[torch.Tensor, "*batch 1"]
    ) -> Float[torch.Tensor, "*batch"]:
        epsilon = torch.finfo(alpha.dtype).eps
        return (2.0 + F.softplus(alpha).clamp_min(epsilon)).squeeze(-1)


class ParetoFixedAlphaOutput(DistributionOutput):
    distr_cls = Pareto
    args_dim = dict(scale=1)

    def __init__(self, alpha: float = 3.0):
        assert alpha > 0.0
        self.alpha = alpha

    @property
    def domain_map(
        self,
    ) -> PyTree[
        Callable[[Float[torch.Tensor, "*batch 1"]], Float[torch.Tensor, "*batch"]], "T"
    ]:
        return dict(scale=self._scale)

    def _scale(
        self, scale: Float[torch.Tensor, "*batch 1"]
    ) -> Float[torch.Tensor, "*batch"]:
        epsilon = torch.finfo(scale.dtype).eps
        return F.softplus(scale).clamp_min(epsilon).squeeze(-1)

    def _distribution(
        self,
        distr_params: PyTree[Float[torch.Tensor, "*batch 1"], "T"],
        validate_args: Optional[bool] = None,
    ) -> Pareto:
        scale = distr_params["scale"]
        distr_params["alpha"] = torch.as_tensor(
            self.alpha, dtype=scale.dtype, device=scale.device
        )
        return self.distr_cls(**distr_params, validate_args=validate_args)
