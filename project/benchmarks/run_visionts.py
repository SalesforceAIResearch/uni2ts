import argparse
import os
import warnings
from functools import partial

warnings.filterwarnings("ignore")

import einops
import numpy as np
import pandas as pd
import torch
from gluonts.ev.metrics import MAE, MASE, MSE, ND, NRMSE, SMAPE
from gluonts.itertools import batcher

# from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm
from visionts import VisionTS

from uni2ts.eval_util.data import get_gluonts_test_dataset, get_lsf_test_dataset
from uni2ts.eval_util.evaluation import evaluate_forecasts


def imputation_nan(array):
    """
    Impute missing value using Naive forecasting.
    """
    not_nan_mask = ~np.isnan(array)
    if not_nan_mask.all():
        return array
    if not not_nan_mask.any():
        return np.zeros_like(array)

    array_imputed = np.copy(array)
    for i in range(len(array)):
        if not not_nan_mask[i]:
            array_imputed[i] = array_imputed[i - 1]
    return array_imputed


POSSIBLE_SEASONALITIES = {
    "S": [3600],  # 1 hour
    "T": [1440, 10080],  # 1 day or 1 week
    "H": [24, 168],  # 1 day or 1 week
    "D": [7, 30, 365],  # 1 week, 1 month or 1 year
    "W": [52, 4],  # 1 year or 1 month
    "M": [12, 6, 3],  # 3 months, 6 months or 1 year
    "B": [5],
    "Q": [4, 2],  # 6 months or 1 year
}


def norm_freq_str(freq_str: str) -> str:
    base_freq = freq_str.split("-")[0]
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]
    return base_freq


def get_seasonality_list(freq: str) -> int:
    offset = pd.tseries.frequencies.to_offset(freq)
    base_seasonality_list = POSSIBLE_SEASONALITIES.get(norm_freq_str(offset.name), [])
    seasonality_list = []
    for base_seasonality in base_seasonality_list:
        seasonality, remainder = divmod(base_seasonality, offset.n)
        if not remainder:
            seasonality_list.append(seasonality)
    seasonality_list.append(1)
    return seasonality_list


def forcast_batch(batch, device, context_len, align_const, model: VisionTS):
    # input batch: list of [L, ].
    # return: [B, H]
    context = [
        torch.tensor(imputation_nan(entry[-context_len:])).view((1, -1, 1)).to(device)
        for entry in batch
    ]
    try:
        context_list = [torch.concatenate(context, dim=0)]
    except RuntimeError:
        # Context lengths are not the same. Should process one by one.
        context_list = context
    cur_forecast_samples = []
    for cur_context in context_list:
        real_context_length = cur_context.shape[1]
        if real_context_length != model.context_len:
            model.update_config(
                real_context_length,
                model.pred_len,
                model.periodicity,
                model.norm_const,
                align_const,
            )
        forecasts = model.forward(cur_context, fp64=True)
        forecasts = einops.rearrange(forecasts, "b t 1 -> b t").detach().cpu().numpy()
        cur_forecast_samples.append(forecasts)
    return np.concatenate(cur_forecast_samples, axis=0)


def convert_context_len(context_len, no_periodicity_context_len, periodicity):
    if periodicity == 1:
        context_len = no_periodicity_context_len
    # Round context length to the integer multiples of the period
    context_len = int(round(context_len / periodicity)) * periodicity
    return context_len


def evaluate(
    model,
    dataset,
    save_path,
    context_len,
    no_periodicity_context_len,
    device="cuda:0",
    checkpoint_dir="./ckpt",
    mae_arch="mae_base",
    batch_size=512,
    test_setting="monash",
    prediction_length=96,
    periodicity="autotune",
    norm_const=0.4,
    align_const=0.4,
):
    print("-" * 5, f"Evaluating {dataset} on {test_setting} setting", "-" * 5)
    if test_setting == "monash" or test_setting == "pf":
        get_dataset = get_gluonts_test_dataset  # for monash and pf, the prediction length can be inferred.
    elif test_setting == "lsf":
        get_dataset = partial(get_lsf_test_dataset, prediction_length=prediction_length)
    else:
        raise NotImplementedError(
            f"Cannot find the test setting {test_setting}. Please select from monash, pf, lsf."
        )
    test_data, metadata = get_dataset(dataset)
    if test_setting != "lsf":
        # for monash and pf, the prediction length can be inferred.
        prediction_length = metadata.prediction_length
    print(
        f"{test_setting} setting - prediction length for {dataset}: {prediction_length}"
    )

    while True:
        try:
            if model == "visionts":
                # Load VisionTS
                model = VisionTS(mae_arch, ckpt_dir=checkpoint_dir).to(device)
                print(
                    f"Model - VisionTS loaded with batch_size: {batch_size}, dataset = {dataset}, pred_len = {prediction_length}"
                )
            else:
                raise NotImplementedError(f"Model {model} not implemented")

            if periodicity == "autotune":
                # Validate seasonality list
                seasonality_list = get_seasonality_list(metadata.freq)
                best_valid_mae = float("inf")
                best_valid_p = 1
                for periodicity in seasonality_list:
                    cur_context_len = convert_context_len(
                        context_len, no_periodicity_context_len, periodicity
                    )
                    model.update_config(
                        cur_context_len,
                        prediction_length,
                        periodicity,
                        norm_const,
                        align_const,
                    )
                    cur_mae_list = []
                    for batch in tqdm(
                        list(batcher(test_data.input, batch_size=batch_size)),
                        desc=f"Validate periodicity: {periodicity}",
                    ):
                        input_batch = [x["target"][:-prediction_length] for x in batch]
                        label_batch = np.stack(
                            [x["target"][-prediction_length:] for x in batch]
                        )
                        cur_forecast_samples = forcast_batch(
                            input_batch, device, cur_context_len, align_const, model
                        )
                        assert cur_forecast_samples.shape == label_batch.shape
                        cur_mae_list.append(np.abs(label_batch - cur_forecast_samples))
                    cur_valid_mae = np.mean(np.concatenate(cur_mae_list, axis=0))
                    if cur_valid_mae < best_valid_mae:
                        best_valid_p = periodicity
                        best_valid_mae = cur_valid_mae
                        print(
                            f"autotune: P = {periodicity} | valid mae = {cur_valid_mae}, accept!"
                        )
                    else:
                        print(
                            f"autotune: P = {periodicity} | valid mae = {cur_valid_mae}, reject!"
                        )
                periodicity = best_valid_p
            elif periodicity == "freq":
                periodicity = get_seasonality_list(metadata.freq)[0]
            else:
                periodicity = int(periodicity)

            # Generate forecast samples
            forecast_samples = []
            cur_context_len = convert_context_len(
                context_len, no_periodicity_context_len, periodicity
            )
            print(f"Use periodicity = {periodicity}, context len = {cur_context_len}")
            model.update_config(
                cur_context_len, prediction_length, periodicity, norm_const, align_const
            )
            for batch in tqdm(
                list(batcher(test_data.input, batch_size=batch_size)),
                desc="Forecasting",
            ):
                batch = [x["target"] for x in batch]
                cur_forecast_samples = forcast_batch(
                    batch, device, cur_context_len, align_const, model
                )
                forecast_samples.append(cur_forecast_samples)
            forecast_samples = np.concatenate(forecast_samples, axis=0)
            break
        except torch.cuda.OutOfMemoryError:
            print(
                f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size//2}"
            )
            batch_size //= 2

    # Convert forecast samples into gluonts SampleForecast objects
    sample_forecasts = []
    for item, ts in zip(forecast_samples, test_data.input):
        forecast_start_date = ts["start"] + len(ts["target"])
        sample_forecasts.append(
            SampleForecast(
                samples=np.reshape(item, (1, -1)), start_date=forecast_start_date
            )
        )

    # Evaluate
    metrics_df = evaluate_forecasts(
        sample_forecasts,
        test_data=test_data,
        metrics=[
            MSE(),
            MAE(),
            SMAPE(),
            MASE(),
            ND(),
            NRMSE(),
        ],
    )
    metrics_df.index = [dataset]
    print(metrics_df)
    metrics_df.to_csv(save_path)
    print(f"Results saved to {save_path}")
    print("-" * 5, f"Evaluation of {dataset} complete", "-" * 5)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a model and dataset, then make predictions."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to use"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--periodicity",
        type=str,
        required=True,
        help=(
            "Time series periodicity length. Can be the following param: "
            + "(1) 'autotune': find the best periodicity on the validation set based on frequency "
            + "(2) 'freq': use the pre-defined periodicity based on frequency "
            + "(3) An integer: use the given periodicity."
        ),
    )
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Directory to save the results"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./ckpt/",
        help="Path to load the model. Auto download if not exists.",
    )
    parser.add_argument("--context_len", type=int, default=1000, help="Context length.")
    parser.add_argument(
        "--no_periodicity_context_len",
        type=int,
        default=1000,
        help="Context length for data with periodicity = 1.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for generating samples"
    )
    parser.add_argument("--run_name", type=str, default="test", help="Name of the run")
    parser.add_argument(
        "--test_setting",
        type=str,
        default="monash",
        choices=["monash", "lsf", "pf"],
        help="Name of the test setting",
    )
    parser.add_argument(
        "--pred_length", type=int, default=96, help="Prediction length for LSF dataset"
    )
    parser.add_argument(
        "--norm_const", type=float, default=0.4, help="Hyperparameter (r) of VisionTS"
    )
    parser.add_argument(
        "--align_const", type=float, default=0.4, help="Hyperparameter (c) of VisionTS"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device. cuda or cpu"
    )
    parser.add_argument(
        "--vision_model_arch",
        type=str,
        default="mae_base",
        choices=["mae_base", "mae_large", "mae_huge"],
        help="Backbone of VisionTS",
    )

    args = parser.parse_args()

    output_dir = os.path.join(args.save_dir, args.run_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.test_setting == "lsf":
        save_dir = os.path.join(output_dir, f"{args.dataset}_{args.pred_length}.csv")
    else:
        save_dir = os.path.join(output_dir, f"{args.dataset}.csv")

    with torch.no_grad():
        evaluate(
            args.model,
            args.dataset,
            save_dir,
            context_len=args.context_len,
            no_periodicity_context_len=args.no_periodicity_context_len,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
            test_setting=args.test_setting,
            prediction_length=args.pred_length,
            periodicity=args.periodicity,
            norm_const=args.norm_const,
            align_const=args.align_const,
        )
