# %%
import argparse
import ast
import json
import os
from datetime import datetime

import numpy as np
from scipy.stats import gmean, gstd
from src.ctx_forecast.utils import parse_values_from_string
from tqdm import tqdm


def stats_between_percentile(all_values, p_low, p_high):
    # p_low and p_high are between 0 and 100
    all_values_sorted = np.sort(all_values)
    percentile_low = np.percentile(all_values_sorted, p_low)
    percentile_high = np.percentile(all_values_sorted, p_high)
    all_values_between_percentile = all_values_sorted[
        np.where(
            (all_values_sorted >= percentile_low)
            & (all_values_sorted <= percentile_high)
        )
    ]
    mean_percentile = np.round(gmean(all_values_between_percentile), 5)
    std_percentile = np.round(gstd(all_values_between_percentile), 5)
    return mean_percentile, std_percentile


def calculate_metrics(
    historical: list, future: list, prediction: list, id=None, roi_args=[{}]
):
    # check prediction
    hist_seq = np.array(historical)
    future_seq = np.array(future)
    if len(prediction) == 0:
        pred_seq = np.ones_like(future_seq) * np.mean(hist_seq)
        length_correct = False
    else:
        if len(prediction) == len(future_seq):
            pred_seq = np.array(prediction)
            length_correct = True
        else:
            if len(prediction) < len(future_seq):
                pred_seq = np.array(
                    prediction + [prediction[-1]] * (len(future_seq) - len(prediction))
                )
                length_correct = False
            else:
                pred_seq = np.array(prediction[: len(future_seq)])
                length_correct = False

    metrics = {}
    for roi_arg in roi_args:
        roi = roi_arg["roi"]
        alpha = roi_arg["alpha"]
        beta = roi_arg["beta"]
        if len(roi) > 0:
            mask = np.zeros_like(future_seq)
            mask[np.array(roi).ravel()] = True
            scale = np.where(mask, alpha, beta)
            scale = scale / scale.sum() * len(future_seq)
        else:
            scale = np.ones_like(future_seq)

        scaled_diff = scale * (pred_seq - future_seq)

        valid_values = np.array(future)
        nmae = np.mean(np.abs(scaled_diff)) / np.clip(
            np.max(valid_values) - np.min(valid_values), a_min=1e-6, a_max=np.inf
        )
        nrmse = np.sqrt(np.mean(scaled_diff**2)) / np.clip(
            np.max(valid_values) - np.min(valid_values), a_min=1e-6, a_max=np.inf
        )
        metrics[f'NMAE_{roi_arg["alpha"]}_{roi_arg["beta"]}'] = nmae
        metrics[f'NRMSE_{roi_arg["alpha"]}_{roi_arg["beta"]}'] = nrmse
        metrics[f'Length_Correct_{roi_arg["alpha"]}_{roi_arg["beta"]}'] = length_correct

    return metrics


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    dirname = os.path.dirname(args.results_dir)
    # scan all json files in the results_dir
    gen_result_dict = {}
    if args.results_dir.endswith(".json"):
        with open(args.results_dir, "r") as f:
            gen_result_dict = json.load(f)
    else:
        json_files = [f for f in os.listdir(args.results_dir) if f.endswith(".json")]
        for json_file in json_files:
            with open(os.path.join(args.results_dir, json_file), "r") as f:
                gen_result_dict.update(json.load(f))

    # sort gen_result_dict by key, key is a str of int
    gen_result_dict = dict(sorted(gen_result_dict.items(), key=lambda x: int(x[0])))

    ###->>>>>>>> ###->>>>>>>> ###->>>>>>>> ###->>>>>>>>
    # calculate metrics
    for key in tqdm(gen_result_dict):
        item = gen_result_dict[key]
        future = item["future_values"]
        future = [float(f) for f in future.split(",")]

        historical = item["history_values"]
        historical = [float(h) for h in historical.split(",")]

        response = item["response"]
        prediction = parse_values_from_string(
            response, start_tag="\\boxed{", end_tag="}", entry_sep=","
        )

        if len(historical) == 0 or len(future) == 0:
            print(f"===== invalid data {key}: missing history or future =====")
            del gen_result_dict[key]
        else:
            try:
                roi = list(ast.literal_eval(item["roi"]))
            except:
                roi = []

            roi_args = [
                {
                    "roi": roi,
                    "alpha": 1,
                    "beta": 1,
                },
                {
                    "roi": roi,
                    "alpha": 3,
                    "beta": 1,
                },
            ]
            out = calculate_metrics(
                historical, future, prediction, id=key, roi_args=roi_args
            )
            gen_result_dict[key]["metrics"] = out

    with open(os.path.join(dirname, "metrics.json"), "w") as f:
        json.dump(gen_result_dict, f, indent=4)

    ###->>>>>>>> ###->>>>>>>> ###->>>>>>>> ###->>>>>>>>
    # summary of metrics
    summary = {}
    metric_keys = []
    for id in gen_result_dict:
        for key in gen_result_dict[id]["metrics"].keys():
            for m_key in ["NMAE", "NRMSE"]:
                if key.startswith(m_key):
                    metric_keys.append(key)
                    break
        break

    # metrics for all
    summary["all"] = {}
    for m in metric_keys:
        all_m = [gen_result_dict[id]["metrics"][m] for id in gen_result_dict]
        all_m = np.array(all_m)
        mean_m, std_m = np.round(gmean(all_m), 5), np.round(gstd(all_m), 5)
        mean_p_0_99, std_p_0_99 = stats_between_percentile(all_m, 0, 99)
        summary["all"].update(
            {
                m: {
                    "all": f"{mean_m} +/- {std_m}",
                    "percetile 0-99": f"{mean_p_0_99} +/- {std_p_0_99}",
                }
            }
        )

    # metrics for samples with roi
    summary["subset_roi"] = {}
    for m in metric_keys:
        all_m = []
        for id in gen_result_dict:
            if gen_result_dict[id]["roi"]:
                all_m.append(gen_result_dict[id]["metrics"][m])

        if len(all_m) > 0:
            all_m = np.array(all_m)
            mean_m, std_m = np.round(gmean(all_m), 5), np.round(gstd(all_m), 5)
            mean_p_0_99, std_p_0_99 = stats_between_percentile(all_m, 0, 99)
            summary["subset_roi"].update(
                {
                    m: {
                        "all": f"{mean_m} +/- {std_m}",
                        "percetile 0-99": f"{mean_p_0_99} +/- {std_p_0_99}",
                    }
                }
            )

    # metrics for samples without roi
    summary["subset_non-roi"] = {}
    for m in metric_keys:
        all_m = []
        for id in gen_result_dict:
            if not gen_result_dict[id]["roi"]:
                all_m.append(gen_result_dict[id]["metrics"][m])

        if len(all_m) > 0:
            all_m = np.array(all_m)
            mean_m, std_m = np.round(gmean(all_m), 5), np.round(gstd(all_m), 5)
            mean_p_0_99, std_p_0_99 = stats_between_percentile(all_m, 0, 99)
            summary["subset_non-roi"].update(
                {
                    m: {
                        "all": f"{mean_m} +/- {std_m}",
                        "percetile 0-99": f"{mean_p_0_99} +/- {std_p_0_99}",
                    }
                }
            )

    with open(os.path.join(dirname, "summary_roi.json"), "w") as f:
        json.dump(summary, f, indent=4)
