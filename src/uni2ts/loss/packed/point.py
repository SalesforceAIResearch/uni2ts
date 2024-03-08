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

from .normalized import PackedNMAELoss, PackedNMSELoss, PackedNRMSELoss, PointNormType


class PackedMAELoss(PackedNMAELoss):
    def __init__(self):
        super().__init__(normalize=PointNormType.NONE)


class PackedMSELoss(PackedNMSELoss):
    def __init__(self):
        super().__init__(normalize=PointNormType.NONE)


class PackedRMSELoss(PackedNRMSELoss):
    def __init__(self):
        super().__init__(normalize=PointNormType.NONE)
