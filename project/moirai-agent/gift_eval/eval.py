import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from tsf_models import (
    get_chronos_forecast_fn,
    get_moirai_forecast_fn,
    get_timesfm_forecast_fn,
    get_tirex_forecast_fn,
)

logging.getLogger("gluonts.model.predictor").setLevel(logging.ERROR)
logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
DATASET_PROPERTIES_FILE = "assets/dataset_properties.json"
GIFT_EVAL_PARQUET_DIR = "Salesforce/GiftEvalParquet"

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

AVAILABLE_MODELS = {
    "chronos": get_chronos_forecast_fn,
    "timesfm": get_timesfm_forecast_fn,
    "tirex": get_tirex_forecast_fn,
    "moirai": get_moirai_forecast_fn,
}
FORECAST_FEATURES = [
    "median",
    "quantile_0",
    "quantile_1",
    "quantile_2",
    "quantile_3",
    "quantile_4",
    "quantile_5",
    "quantile_6",
    "quantile_7",
    "quantile_8",
]
CANDIDATE_MODELS = ("chronos", "timesfm", "tirex")


# load candidate models
class TimeSeriesForecasters:
    def __init__(self, model_names=("chronos", "timesfm", "tirex")):
        self.model_names = model_names
        self.forecasters = {}
        for m_name in model_names:
            self.forecasters[m_name] = AVAILABLE_MODELS[m_name](device="cuda")
            print(f"==> {m_name} loaded successfully")

    def get_forecasts(self, history_values, pred_length):
        candidate_preds = {}
        for m_name in self.model_names:
            response = self.forecasters[m_name]([history_values], [pred_length])[0]
            for k, v in response.items():
                if k in FORECAST_FEATURES:
                    assert np.isnan(np.array(v)).sum() == 0
                    assert len(v) == pred_length
            candidate_preds[m_name] = response
        return candidate_preds

    def _get_mixture_pred(self, candidate_preds, model_names):
        feature_keys = list(candidate_preds[model_names[0]].keys())
        mixture_pred = {}
        all_quantiles = []
        for key in feature_keys:
            if key not in FORECAST_FEATURES:
                mixture_pred[key] = candidate_preds[model_names[0]][key]
            else:
                if key != "median":
                    k_quantiles = np.array(
                        [candidate_preds[m_name][key] for m_name in model_names]
                    )
                    all_quantiles.append(k_quantiles)
        all_quantiles = np.quantile(
            np.concatenate(all_quantiles, axis=0),
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            axis=0,
        )
        mixture_pred["median"] = all_quantiles[4].tolist()
        for i in range(9):
            mixture_pred[f"quantile_{i}"] = all_quantiles[i].tolist()
        return mixture_pred

    def get_best_pred(self, best_model, candidate_preds, model_names):
        mixture_pred = self._get_mixture_pred(candidate_preds, model_names)
        if best_model != "mixture":
            best_median = candidate_preds[best_model]["median"]
            offset = np.array(best_median) - np.array(mixture_pred["median"])
            for key in FORECAST_FEATURES:
                mixture_pred[key] = (
                    np.array(mixture_pred[key]) + np.array(offset)
                ).tolist()
        return mixture_pred


class TimeSeriesProcessor:
    def __init__(self, max_future_length, max_history_future_ratio):
        self.max_future_length = max_future_length
        self.max_history_future_ratio = max_history_future_ratio

    def _get_norm_factor(self, history_values):
        """
        Get the norm factor of the history values and future values.
        """
        history_values = pd.Series(history_values, dtype="float").to_numpy()
        valid_mask = ~np.isnan(history_values)
        if valid_mask.sum() == 0:
            return 0, 1
        valid_values = history_values[valid_mask]
        mean = np.mean(valid_values)
        std = np.clip(np.std(valid_values), a_min=1e-5, a_max=None)
        return mean, std

    def _normalize_values(self, values, mean, std):
        """
        Normalize the values using the mean and std.
        """
        values = pd.Series(values, dtype="float").to_numpy()
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() == 0:
            return values
        normalized_values = (values[valid_mask] - mean) / std
        values[valid_mask] = normalized_values
        return values.tolist()

    def _compute_mae(self, pred_values, target_values):
        pred_values = np.array(pred_values)
        target_values = np.array(target_values)
        valid_mask_future = ~np.isnan(target_values)
        if valid_mask_future.sum() == 0:
            return 0
        diff = pred_values[valid_mask_future] - target_values[valid_mask_future]
        mae = np.mean(np.abs(diff))
        if np.isnan(mae):
            return np.inf
        else:
            return mae

    def _get_best_model(self, metrics: dict):
        best_model = None
        best_mae = np.inf
        for m_name, m_mae in metrics.items():
            try:
                m_mae = float(m_mae)
            except:
                m_mae = np.nan
            if ~np.isnan(m_mae) and m_mae < best_mae:
                best_model, best_mae = m_name, m_mae
        return best_model

    def _downsample_time_series(
        self,
        history_values,
        history_timestamps,
        candidate_preds,
        model_names,
        future_timestamps,
        future_values=None,
        downsample_step=1,
    ):
        downspl_future_values = (
            future_values[0::downsample_step] if future_values is not None else None
        )
        downspl_future_timestamps = future_timestamps[0::downsample_step]
        downspl_candidate_preds = {}
        for m_name in model_names:
            downspl_candidate_preds[m_name] = candidate_preds[m_name][
                0::downsample_step
            ]

        downspl_history_values = history_values[-downsample_step::-downsample_step][
            ::-1
        ]
        downspl_history_timestamps = history_timestamps[
            -downsample_step::-downsample_step
        ][::-1]
        return (
            downspl_history_values,
            downspl_history_timestamps,
            downspl_candidate_preds,
            downspl_future_timestamps,
            downspl_future_values,
        )

    def _truncate_time_series(
        self,
        history_values,
        history_timestamps,
        candidate_preds,
        model_names,
        future_timestamps,
        max_ctx_length,
        max_future_length,
    ):
        ctx_length = min(len(history_values), max_ctx_length)
        truncated_history_values = history_values[-ctx_length:]
        truncated_history_timestamps = history_timestamps[-ctx_length:]

        pred_length = min(len(future_timestamps), max_future_length)
        truncated_future_timestamps = future_timestamps[:pred_length]
        truncated_candidate_preds = {}
        for m_name in model_names:
            truncated_candidate_preds[m_name] = candidate_preds[m_name][:pred_length]

        return (
            truncated_history_values,
            truncated_history_timestamps,
            truncated_candidate_preds,
            truncated_future_timestamps,
            ctx_length,
            pred_length,
        )

    def __call__(self, time_series_data, model_names):
        ###->>>>>>>> ###->>>>>>>> ###->>>>>>>> ###->>>>>>>>
        # load raw data and get the best model
        raw_history_values = time_series_data["history_seq"]
        raw_history_start, raw_history_end = time_series_data["history_window"]
        raw_history_frequency = time_series_data["frequency"]
        raw_history_timestamps = pd.date_range(
            start=raw_history_start, end=raw_history_end, freq=raw_history_frequency
        )
        # raw_future_values = time_series_data['future_seq']
        raw_future_start, raw_future_end = time_series_data["future_window"]
        raw_pred_timestamps = pd.date_range(
            start=raw_future_start, end=raw_future_end, freq=raw_history_frequency
        )
        raw_pred_length = time_series_data["pred_length"]

        # normalize values and get the best model
        mean, std = self._get_norm_factor(raw_history_values)
        norm_history_values = self._normalize_values(raw_history_values, mean, std)

        norm_candidate_preds = {}
        for m_name in model_names:
            _pred = time_series_data["candidate_preds"][m_name]["median"]
            norm_candidate_preds[m_name] = self._normalize_values(_pred, mean, std)

        norm_candidate_cvs = {}
        _cv_metrics = {}
        for m_name in model_names:
            _pred = time_series_data["candidate_crossval"][m_name]["median"]
            _cv_metrics[m_name] = self._compute_mae(
                _pred, raw_history_values[-raw_pred_length:]
            )  # crossval on the last part of the history values
            norm_candidate_cvs[m_name] = self._normalize_values(_pred, mean, std)
        cv_ranking = [
            m_name
            for m_name in sorted(_cv_metrics.keys(), key=lambda x: _cv_metrics[x])
        ]

        # downsample and truncate the normalized time series, form a global view
        downspl_step = max(1, raw_pred_length // self.max_future_length)
        (
            global_history_values,
            global_history_timestamps,
            global_candidate_preds,
            global_future_timestamps,
            _,
        ) = self._downsample_time_series(
            norm_history_values,
            raw_history_timestamps,
            norm_candidate_preds,
            model_names,
            raw_pred_timestamps,
            None,
            downspl_step,
        )
        max_ctx_length = int(self.max_history_future_ratio * self.max_future_length)
        (
            global_history_values,
            global_history_timestamps,
            global_candidate_preds,
            global_future_timestamps,
            _,
            _,
        ) = self._truncate_time_series(
            global_history_values,
            global_history_timestamps,
            global_candidate_preds,
            model_names,
            global_future_timestamps,
            max_ctx_length,
            self.max_future_length,
        )

        global_candidate_cvs = {}
        for m_name in model_names:
            global_candidate_cvs[m_name] = norm_candidate_cvs[m_name][
                -downspl_step::-downspl_step
            ][::-1]
            len_cv = len(global_candidate_cvs[m_name])
        global_cv_label = global_history_values[-len_cv:]

        # convert to string
        global_history_string = ",".join([f"{v:.3f}" for v in global_history_values])
        global_candidate_pred_strings = {}
        for m_name in model_names:
            global_candidate_pred_strings[m_name] = ",".join(
                [f"{v:.3f}" for v in global_candidate_preds[m_name]]
            )
        global_candidate_cvs_strings = {}
        for m_name in model_names:
            global_candidate_cvs_strings[m_name] = ",".join(
                [f"{v:.3f}" for v in global_candidate_cvs[m_name]]
            )

        global_cv_label_string = ",".join([f"{v:.3f}" for v in global_cv_label])

        # build sample info
        history_info = {
            "history_window": f'[{global_history_timestamps[0].strftime("%Y-%m-%d %H:%M:%S")}, {global_history_timestamps[-1].strftime("%Y-%m-%d %H:%M:%S")}]',
            "history_values": global_history_string,
        }
        pred_info = {
            "future_window": f'[{global_future_timestamps[0].strftime("%Y-%m-%d %H:%M:%S")}, {global_future_timestamps[-1].strftime("%Y-%m-%d %H:%M:%S")}]',
            "model_names": model_names,
            "candidate_preds": global_candidate_pred_strings,
        }
        cv_info = {
            "crossval_window": f'[{global_history_timestamps[-len_cv].strftime("%Y-%m-%d %H:%M:%S")}, {global_history_timestamps[-1].strftime("%Y-%m-%d %H:%M:%S")}]',
            "crossval_ground_truth": global_cv_label_string,
            "crossval_preds": global_candidate_cvs_strings,
            "crossval_error_ranking": " < ".join(cv_ranking),
        }
        instruction = (
            "You are given a sequence of history values and several future predictions by candidate models. "
            "Analyze the future predictions by the candidates and their cross-validation errors on the last part of the history values. "
            "Select the optimal future predictions. Enclose the name of the best model by \\boxed{ and }. "
        )

        query = (
            f"{json.dumps(history_info, indent=2)}"
            "\n\n\n"
            f"{json.dumps(pred_info, indent=2)}"
            "\n\n\n"
            f"{json.dumps(cv_info, indent=2)}"
            "\n\n\n"
            f"{instruction}"
        )
        return {"query": query}

    def parse_answer(self, answer, model_names):
        match = re.search(r"\\boxed{(.+)}", answer, re.DOTALL)
        if match:
            content = match.group(1)
            ans = content.strip("\n").strip()
        else:
            ans = None
        assert ans in model_names
        return ans

    def check_preds_diversity(
        self,
        all_preds,
        diversity_threshold=0.2,
        max_forecast_length=40,
        history_values=None,
    ):
        mean, std = self._get_norm_factor(history_values)
        all_preds = {
            m_name: self._normalize_values(all_preds[m_name]["median"], mean, std)
            for m_name in all_preds
        }

        if diversity_threshold == 0.0:
            return True
        try:
            preds = [p for _, p in all_preds.items()]
            preds = np.array(preds, dtype=float)
            if max_forecast_length is not None:
                downspl_step = preds.shape[1] // max_forecast_length
                preds = preds[:, ::downspl_step]
            cand_pred_diversity = np.std(preds, axis=0).mean()
            return cand_pred_diversity > diversity_threshold
        except:
            return True


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


class MoiraiAgentTimeSeriesForecast:
    def __init__(
        self,
        model_names=("chronos", "timesfm", "tirex"),
        llm_repo_id="Salesforce/moirai-agent",
    ):
        self.model_names = model_names
        print("==> Loading candidate forecasters...")
        self.time_series_forecasters = TimeSeriesForecasters(model_names=model_names)

        print("==> Loading time series processor and LLM...")
        self.preprocessor = TimeSeriesProcessor(
            max_future_length=40, max_history_future_ratio=10
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_repo_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_repo_id, dtype="auto", device_map="auto"
        )

    def __call__(self, time_series_data):
        candidate_preds = self.time_series_forecasters.get_forecasts(
            time_series_data["history_seq"], time_series_data["pred_length"]
        )
        candidate_crossval = self.time_series_forecasters.get_forecasts(
            time_series_data["history_seq"][: -time_series_data["pred_length"]],
            time_series_data["pred_length"],
        )
        time_series_data["candidate_preds"] = candidate_preds
        time_series_data["candidate_crossval"] = candidate_crossval

        is_diverse = self.preprocessor.check_preds_diversity(
            time_series_data["candidate_preds"],
            history_values=time_series_data["history_seq"],
        )
        if not is_diverse:
            response = "mixture"
        else:
            chat_prompt = self.preprocessor(time_series_data, self.model_names)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": chat_prompt["query"]},
            ]
            prompt = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            llm_inputs = self.llm_tokenizer([prompt], return_tensors="pt").to(
                self.llm_model.device
            )
            generated_ids = self.llm_model.generate(**llm_inputs)
            response = self.llm_tokenizer.batch_decode(
                [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        llm_inputs.input_ids, generated_ids
                    )
                ],
                skip_special_tokens=True,
            )[0]
            response = self.preprocessor.parse_answer(response, self.model_names)

        final_pred = self.time_series_forecasters.get_best_pred(
            response,
            time_series_data["candidate_preds"],
            self.model_names,
        )
        return final_pred, response, is_diverse


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
