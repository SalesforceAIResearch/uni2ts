import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from agent import MoiraiAgentTimeSeriesForecast
from datasets import load_dataset
from dotenv import load_dotenv
from gift_eval import Dataset
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
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality
from tqdm import tqdm

logging.getLogger("gluonts.model.predictor").setLevel(logging.ERROR)
logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
DATASET_PROPERTIES_FILE = "assets/dataset_properties.json"
GIFT_EVAL_PARQUET_DIR = "Salesforce/GiftEvalParquet"


CANDIDATE_MODELS = ("chronos", "timesfm", "tirex")

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Define datasets and fallback model
short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
all_datasets = list(set(short_datasets.split() + med_long_datasets.split()))
dataset_properties_map = json.load(open(DATASET_PROPERTIES_FILE))


def extract_quantiles_prediction(df):
    """Extract quantiles predictions and convert them into glutonts compatible format
    The input df should have fields 'quantiles_0' to 'quantiles_8'
    """
    quantiles = []
    for i in range(9):
        quantiles.append(df[f"quantile_{i}"])

    stacked_lists = [np.stack(li, axis=0) for li in quantiles]
    combined = np.stack(stacked_lists, axis=1)
    quantile_forecasts = [
        QuantileForecast(
            forecast_arrays=x,
            start_date=pd.Period(
                df["future_start"].iloc[i], freq=df["frequency"].iloc[i]
            ),
            forecast_keys=[
                "0.1",
                "0.2",
                "0.3",
                "0.4",
                "0.5",
                "0.6",
                "0.7",
                "0.8",
                "0.9",
            ],
        )
        for i, x in enumerate(combined)
    ]
    return quantile_forecasts


def eval_dataset(dataset, ds_config, df):
    print(f"Processing {ds_config}")
    test_data = dataset.test_data
    L = test_data.prediction_length
    season_length = get_seasonality(dataset.freq)

    pred_cols = pd.json_normalize(df["final_pred"])
    df = df.drop(columns=["final_pred"]).join(pred_cols)
    quantile_forecasts = extract_quantiles_prediction(df)

    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]

    results = evaluate_forecasts(
        forecasts=quantile_forecasts,
        test_data=test_data,
        metrics=metrics,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )
    results.insert(loc=0, column="dataset", value=ds_config)
    return results


def get_prediction_df(dataset_config):
    # dataset_name = dataset_config.replace("/", "_") + ".parquet"
    # test_samples = load_dataset(
    #     "parquet",
    #     data_files={"x": os.path.join(GIFT_EVAL_PARQUET_DIR, dataset_name)},
    #     split="x",
    # )
    dataset_name = dataset_config.replace("/", "_")
    test_samples = load_dataset(GIFT_EVAL_PARQUET_DIR, dataset_name, split="train")
    num_samples = len(test_samples)
    results = []

    for sample_idx in tqdm(range(num_samples)):
        example_data = test_samples[sample_idx]

        time_series_data = {
            "history_seq": np.array(
                example_data["history_value"], dtype="float"
            ).tolist(),
            "history_window": [
                example_data["history_start"],
                example_data["history_end"],
            ],
            "frequency": example_data["frequency"],
            "future_window": [example_data["future_start"], example_data["future_end"]],
            "pred_length": len(example_data["future_value"]),
        }

        final_pred, response, is_diverse = moirai_agent(time_series_data)

        results.append(
            {
                "sample_idx": sample_idx,
                "response": response,
                "is_diverse": is_diverse,
                "final_pred": final_pred,
                "future_start": example_data["future_start"],
                "future_end": example_data["future_end"],
                "frequency": example_data["frequency"],
            }
        )

    df = pd.DataFrame(results)
    return df


def reformatted_metrics_for_leaderboard(row):
    reformatted = {}
    # .columns will give the keys, .iloc[0] will get the value for the first (only) row
    for key in row.columns:
        if key == "dataset":
            continue
        else:
            reformatted[f"eval_metrics/{key}"] = row.iloc[0][key]
    return reformatted


if __name__ == "__main__":
    moirai_agent = MoiraiAgentTimeSeriesForecast(
        model_names=CANDIDATE_MODELS, llm_repo_id="Salesforce/moirai-agent"
    )
    # get time series information

    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
    )
    parser.add_argument("--out_name", type=str, default="all_results.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = os.path.join(args.out_dir, args.out_name)

    # Preserve previous behavior: each run produces a fresh output CSV.
    if os.path.exists(out_name):
        os.remove(out_name)

    argv = []
    output_columns = None
    all_datasets = ["ett2/H"]  # DEBUGGING ONLY REMOVE THIS
    for ds_name in tqdm(sorted(all_datasets), desc="Processing datasets"):
        ds_key = ds_name.split("/")[0]
        terms = ["short", "medium", "long"]
        for term in terms:
            if (
                term == "medium" or term == "long"
            ) and ds_name not in med_long_datasets.split():
                continue

            if "/" in ds_name:
                ds_key = ds_name.split("/")[0]
                ds_freq = ds_name.split("/")[1]
                ds_key = ds_key.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
            else:
                ds_key = ds_name.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
                ds_freq = dataset_properties_map[ds_key]["frequency"]

            ds_config = f"{ds_key}/{ds_freq}/{term}"
            # Initialize the dataset
            to_univariate = (
                False
                if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
                else True
            )
            dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
            season_length = get_seasonality(dataset.freq)
            dataset_config = f"{ds_key}/{ds_freq}/{term}"
            df = get_prediction_df(dataset_config)
            out = eval_dataset(dataset, ds_config, df)

            domain = dataset_properties_map[ds_key]["domain"]
            num_variates = dataset_properties_map[ds_key]["num_variates"]
            result_metrics = reformatted_metrics_for_leaderboard(out)

            # Construct the values in the same order as the keys that will be used in the DataFrame
            row_values = {
                "dataset": ds_config,  # "dataset"
                "model": "MoiraiAgent",  # "model"
                **result_metrics,
                "domain": domain,  # "domain"
                "num_variates": num_variates,  # "num_variates"
            }
            if output_columns is None:
                output_columns = list(row_values.keys())

            # Append per-dataset results immediately.
            row_df = pd.DataFrame(
                [{col: row_values.get(col, np.nan) for col in output_columns}]
            )
            row_df.to_csv(
                out_name,
                mode="a",
                header=not os.path.exists(out_name),
                index=False,
            )
