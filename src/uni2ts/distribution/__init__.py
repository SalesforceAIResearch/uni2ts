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

from ._base import DistributionOutput, DistrParamProj
from .laplace import LaplaceFixedScaleOutput, LaplaceOutput
from .log_normal import LogNormalOutput
from .mixture import MixtureOutput
from .negative_binomial import NegativeBinomialOutput
from .normal import NormalFixedScaleOutput, NormalOutput
from .pareto import ParetoFixedAlphaOutput, ParetoOutput
from .student_t import StudentTOutput

DISTRIBUTION_OUTPUTS = [
    "LaplaceFixedScaleOutput",
    "LaplaceOutput",
    "LogNormalOutput",
    "MixtureOutput",
    "NegativeBinomialOutput",
    "NormalFixedScaleOutput",
    "NormalOutput",
    "ParetoFixedAlphaOutput",
    "StudentTOutput",
]

__all__ = ["DistrParamProj", "DistributionOutput"] + DISTRIBUTION_OUTPUTS
