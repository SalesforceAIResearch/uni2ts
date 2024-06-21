import argparse
import os

import numpy as np
import torch
from chronos import ChronosPipeline
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.itertools import batcher

# from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm

from uni2ts.eval_util.data import get_gluonts_test_dataset
from uni2ts.eval_util.evaluation import evaluate_forecasts
from uni2ts.eval_util.metrics import MedianMSE


def evaluate(pipeline, dataset, save_path, num_samples=20, batch_size=512):
    print("-" * 5, f"Evaluating {dataset}", "-" * 5)
    test_data, metadata = get_gluonts_test_dataset(dataset)
    prediction_length = metadata.prediction_length

    while True:
        try:
            # Generate forecast samples
            forecast_samples = []
            for batch in tqdm(batcher(test_data.input, batch_size=batch_size)):
                context = [torch.tensor(entry["target"]) for entry in batch]
                forecast_samples.append(
                    pipeline.predict(
                        context,
                        prediction_length=prediction_length,
                        num_samples=num_samples,
                        limit_prediction_length=False,  # We disable the limit on prediction length.
                    ).numpy()
                )
            forecast_samples = np.concatenate(forecast_samples)
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
            SampleForecast(samples=item, start_date=forecast_start_date)
        )

    # Evaluate
    metrics_df = evaluate_forecasts(
        sample_forecasts,
        test_data=test_data,
        metrics=[
            MSE(),
            MAE(),
            MAPE(),
            SMAPE(),
            MSIS(),
            RMSE(),
            NRMSE(),
            ND(),
            MASE(),
            MedianMSE(),
            MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
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
        "--model_path", type=str, required=True, help="Path to load the model"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Directory to save the results"
    )
    parser.add_argument(
        "--num_samples", type=int, default=20, help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for generating samples"
    )
    parser.add_argument("--run_name", type=str, default="test", help="Name of the run")

    args = parser.parse_args()
    # Load Chronos
    pipeline = ChronosPipeline.from_pretrained(
        # "amazon/chronos-t5-small",
        args.model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    output_dir = os.path.join(args.save_dir, args.run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    evaluate(pipeline, args.dataset, os.path.join(output_dir, f"{args.dataset}.csv"))
