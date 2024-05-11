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

from ._base import Chain, Identity, Transformation
from .crop import EvalCrop, PatchCrop, SpecifiedPatchCrop
from .feature import AddObservedMask, AddTimeIndex, AddVariateIndex, AddSampleIndex
from .field import LambdaSetFieldIfNotPresent, RemoveFields, SelectFields, SetValue
from .imputation import DummyValueImputation, ImputeTimeSeries, LastValueImputation
from .pad import EvalPad, Pad, PadFreq, PadOutRangeTokens
from .patch import (
    DefaultPatchSizeConstraints,
    FixedPatchSizeConstraints,
    GetPatchSize,
    Patchify,
    PatchSizeConstraints,
)
from .resample import SampleDimension
from .reshape import (
    FlatPackCollection,
    FlatPackFields,
    PackCollection,
    PackFields,
    SequencifyField,
    Transpose,
)
from .task import EvalMaskedPrediction, ExtendMask, MaskedPrediction

__all__ = [
    "AddObservedMask",
    "AddTimeIndex",
    "AddVariateIndex",
    "AddSampleIndex",
    "Chain",
    "DefaultPatchSizeConstraints",
    "DummyValueImputation",
    "EvalCrop",
    "EvalMaskedPrediction",
    "EvalPad",
    "ExtendMask",
    "FixedPatchSizeConstraints",
    "FlatPackCollection",
    "FlatPackFields",
    "GetPatchSize",
    "Identity",
    "ImputeTimeSeries",
    "LambdaSetFieldIfNotPresent",
    "LastValueImputation",
    "MaskedPrediction",
    "PackCollection",
    "PackFields",
    "Pad",
    "PadFreq",
    "PadOutRangeTokens",
    "PatchCrop",
    "PatchSizeConstraints",
    "Patchify",
    "RemoveFields",
    "SampleDimension",
    "SelectFields",
    "SpecifiedPatchCrop",
    "SequencifyField",
    "SetValue",
    "Transformation",
    "Transpose",
]
