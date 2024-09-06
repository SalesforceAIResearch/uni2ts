import argparse
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import timesfm
import torch
from gluonts.itertools import batcher
from paxml import checkpoints
from tqdm.auto import tqdm

from uni2ts.eval_util.data import get_gluonts_test_dataset, get_lsf_test_dataset

QUANTILES = list(np.arange(1, 10) / 10.0)
EPS = 1e-7


def _mse(y_pred, y_true):
    """mse loss."""
    return np.square(y_pred - y_true)


def _mae(y_pred, y_true):
    """mae loss."""
    return np.abs(y_pred - y_true)


def _smape(y_pred, y_true):
    """_smape loss."""
    abs_diff = np.abs(y_pred - y_true)
    abs_val = (np.abs(y_true) + np.abs(y_pred)) / 2
    abs_val = np.where(abs_val > EPS, abs_val, 1.0)
    abs_diff = np.where(abs_val > EPS, abs_diff, 0.0)
    return abs_diff / abs_val


def evaluate(
    model,
    dataset,
    save_path,
    context_len=512,
    checkpoint_path=None,
    batch_size=512,
    test_setting="monash",
    pred_length=96,
):
    print("-" * 5, f"Evaluating {dataset} on {test_setting} setting", "-" * 5)
    if test_setting == "monash" or test_setting == "pf":
        get_dataset = get_gluonts_test_dataset  # for monash and pf, the prediction length can be inferred.
    elif test_setting == "lsf":
        get_dataset = partial(get_lsf_test_dataset, prediction_length=pred_length)
    else:
        raise NotImplementedError(
            f"Cannot find the test setting {test_setting}. Please select from monash, pf, lsf."
        )
    test_data, metadata = get_dataset(dataset)
    if test_setting == "lsf":
        prediction_length = pred_length
        # print(f"LSF setting - prediction length for {dataset}: {prediction_length}")
    else:  # for monash and pf, the prediction length can be inferred.
        prediction_length = metadata.prediction_length
    print(
        f"{test_setting} setting - prediction length for {dataset}: {prediction_length}"
    )

    while True:
        try:
            if model == "timesfm":
                # Load timesfm
                model = timesfm.TimesFm(
                    context_len=context_len,
                    horizon_len=prediction_length,
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=20,
                    model_dims=1280,
                    backend="gpu",
                    per_core_batch_size=batch_size,
                    quantiles=QUANTILES,
                )
                model.load_from_checkpoint(
                    # model_path,
                    checkpoint_path=checkpoint_path,
                    checkpoint_type=checkpoints.CheckpointType.FLAX,
                )
                print("Model - TimesFM loaded with batch_size:", batch_size)
            else:
                raise NotImplementedError(f"Model {model} not implemented")
            # Generate forecast samples
            forecast_samples = []
            start_time = time.time()
            for batch in tqdm(batcher(test_data.input, batch_size=batch_size)):
                # ipdb.set_trace()
                context = [torch.tensor(entry["target"]) for entry in batch]
                int_freq = timesfm.freq_map(
                    metadata.freq
                )  # to get the int frequency as used in tiemsfm
                lfreq = [int_freq] * len(context)
                _, forecasts = model.forecast(context, lfreq)
                median_forecasts = forecasts[
                    :, :, 5
                ]  # get the median forecast with size (bs, pred_len)
                forecast_samples.append(median_forecasts)
            end_time = time.time()
            break
        except torch.cuda.OutOfMemoryError:
            print(
                f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size//2}"
            )
            batch_size //= 2

    # ipdb.set_trace()
    input_batches = batcher(test_data.input, batch_size=batch_size)
    label_batches = batcher(test_data.label, batch_size=batch_size)

    smape_run_losses = []
    mse_run_losses = []
    mae_run_losses = []
    num_elements = 0
    abs_sum = 0
    for input_batch, label_batch, forecast_batch in tqdm(
        zip(input_batches, label_batches, forecast_samples)
    ):
        labels = np.array([torch.tensor(entry["target"]) for entry in label_batch])
        forecasts_batch = np.array(forecast_batch)
        assert (
            labels.shape == forecasts_batch.shape
        ), f"Labels shape {labels.shape} != Forecasts shape {forecasts_batch.shape}"
        mae_run_losses.append(_mae(forecasts_batch, labels).sum())
        mse_run_losses.append(_mse(forecasts_batch, labels).sum())
        smape_run_losses.append(_smape(forecasts_batch, labels).sum())
        num_elements += labels.shape[0] * labels.shape[1]
        abs_sum += np.abs(labels).sum()

    mse_val = np.sum(mse_run_losses) / num_elements
    result_dict = {
        "mse": mse_val,
        "smape": np.sum(smape_run_losses) / num_elements,
        "mae": np.sum(mae_run_losses) / num_elements,
        "wape": np.sum(mae_run_losses) / abs_sum,
        "nrmse": np.sqrt(mse_val) / (abs_sum / num_elements),
        "num_elements": num_elements,
        "abs_sum": abs_sum,
        "total_time_eval": end_time - start_time,
    }
    metrics_df = pd.DataFrame(result_dict, index=[dataset])
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
        "--model_path", type=str, default=None, help="Path to load the model"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Directory to save the results"
    )
    parser.add_argument("--context_len", type=int, default=512, help="Context length")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for generating samples"
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

    args = parser.parse_args()

    output_dir = os.path.join(args.save_dir, args.run_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.test_setting == "lsf":
        save_dir = os.path.join(output_dir, f"{args.dataset}_{args.pred_length}.csv")
    else:
        save_dir = os.path.join(output_dir, f"{args.dataset}.csv")
    evaluate(
        args.model,
        args.dataset,
        save_dir,
        context_len=args.context_len,
        checkpoint_path=args.model_path,
        batch_size=args.batch_size,
        test_setting=args.test_setting,
        pred_length=args.pred_length,
    )
