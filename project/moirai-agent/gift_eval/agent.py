import json
import logging
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
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


if __name__ == "__main__":
    moirai_agent = MoiraiAgentTimeSeriesForecast()
    time_series_data = {
        "history_seq": [1, 2, 3, 4, 5, 6],
        "history_window": ["2026-01-01 00:00:00", "2026-01-01 06:00:00"],
        "frequency": "H",
        "future_window": ["2026-01-01 07:00:00", "2026-01-01 10:00:00"],
        "pred_length": 4,
    }
    pred, _, _ = moirai_agent(time_series_data)
    print(pred)
