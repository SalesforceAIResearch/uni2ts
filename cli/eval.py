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

import hydra
import pandas as pd
import torch
from gluonts.time_feature import get_seasonality
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.eval_util.evaluation import evaluate_model


@hydra.main(version_base="1.3", config_path="conf/eval", config_name="default")
def main(cfg: DictConfig):
    # Set display options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.options.display.float_format = "{:.3f}".format

    test_data, metadata = call(cfg.data)
    batch_size = cfg.batch_size
    while True:
        model = call(cfg.model, _partial_=True, _convert_="all")(
            prediction_length=metadata.prediction_length,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
        )
        metrics = instantiate(cfg.metrics, _convert_="all")
        try:
            predictor = model.create_predictor(batch_size, cfg.device)
            res = evaluate_model(
                predictor,
                test_data=test_data,
                metrics=metrics,
                batch_size=cfg.batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=get_seasonality(metadata.freq),
            )
            print(res)
            output_dir = HydraConfig.get().runtime.output_dir
            writer = SummaryWriter(log_dir=output_dir)
            for name, metric in res.to_dict("records")[0].items():
                writer.add_scalar(f"{metadata.split}_metrics/{name}", metric)
            writer.close()
            break
        except torch.cuda.OutOfMemoryError:
            print(
                f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size//2}"
            )
            batch_size //= 2
            if batch_size < cfg.min_batch_size:
                print(
                    f"batch_size {batch_size} smaller than "
                    f"min_batch_size {cfg.min_batch_size}, ending evaluation"
                )
                break


if __name__ == "__main__":
    main()
