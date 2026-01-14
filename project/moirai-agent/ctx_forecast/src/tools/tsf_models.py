import random

import numpy as np
import torch


def set_random_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seeds()


def pad_to_left(
    series_list: list[list[int | float]], value=np.nan
) -> list[list[int | float]]:
    assert isinstance(series_list, list), "series_list must be a list"
    for d in series_list:
        assert isinstance(d, list), "d must be a list"
        for i in d:
            assert isinstance(i, (int, float)), "i must be a number"

    max_ctx_len = max(len(d) for d in series_list)
    output = [[value] * (max_ctx_len - len(d)) + d for d in series_list]
    return output


def get_chronos_forecast_fn(device="cuda"):
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        "s3://autogluon/chronos-2", device_map=device, dtype=torch.float32
    )
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def make_forecast_batch(
        data: list[list[int | float]], prediction_length: list[int]
    ):
        """
        Args:
            data: list[list[int|float]], a list of time-series sequences (each sequence is a list of numbers)
            prediction_length: list[int], a list of prediction lengths for each time-series sequence
        Returns:
            output: list[dict], a list of dictionaries, each containing the low, median, and high values of the forecast
                - low: list[float], the low values of the forecast
                - median: list[float], the median values of the forecast
                - high: list[float], the high values of the forecast
        """
        assert (
            len(data) == len(prediction_length) and len(data) > 0
        ), "data and prediction_length must have the same length and not empty"

        data = pad_to_left(data)
        max_pred_len = max(prediction_length)
        with torch.no_grad():
            data = [np.array(d) for d in data]
            quantile_forecast, _ = pipeline.predict_quantiles(
                inputs=data,
                prediction_length=max_pred_len,
                quantile_levels=quantile_levels,
            )

        output = []
        for i in range(len(data)):
            pred_len = prediction_length[i]
            quantiles = quantile_forecast[i][0, :pred_len, :].cpu().numpy()
            forecast_i = {
                "median": quantiles[:, 4].tolist(),
                "quantile_levels": [str(q) for q in quantile_levels],
            }
            for q_idx in range(len(quantile_levels)):
                forecast_i[f"quantile_{q_idx}"] = quantiles[:, q_idx].tolist()
            output.append(forecast_i)
        return output

    return make_forecast_batch


def get_timesfm_forecast_fn(device="cuda"):
    from timesfm import ForecastConfig, TimesFM_2p5_200M_torch

    model = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    quantile_levels = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]

    def make_forecast_batch(
        data: list[list[int | float]], prediction_length: list[int]
    ):
        assert len(data) == len(
            prediction_length
        ), "data and prediction_length must have the same length"
        max_pred_len = max(prediction_length)
        num_data = len(data)

        # prepare model
        max_context = max([len(d) for d in data])
        max_context = (
            (max_context + model.model.p - 1) // model.model.p
        ) * model.model.p
        max_context = min(15360, max_context)
        model.compile(
            ForecastConfig(
                max_context=max_context,
                max_horizon=max_pred_len,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

        with torch.no_grad():
            data = [np.array(d) for d in data]
            model.global_batch_size = num_data
            _, quantile_forecast = model.forecast(horizon=max_pred_len, inputs=data)

        output = []
        for i in range(num_data):
            pred_len = prediction_length[i]
            qunatiles = quantile_forecast[i, :pred_len, 1:]
            forecast_i = {
                "median": qunatiles[:, 4].tolist(),
                "quantile_levels": quantile_levels,
            }
            for q_idx in range(len(quantile_levels)):
                forecast_i[f"quantile_{q_idx}"] = qunatiles[:, q_idx].tolist()
            output.append(forecast_i)
        return output

    return make_forecast_batch


def get_tirex_forecast_fn(device="cuda"):
    from tirex import ForecastModel, load_model

    model: ForecastModel = load_model("NX-AI/TiRex", device=device)
    quantile_levels = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]

    def make_forecast_batch(
        data: list[list[int | float]], prediction_length: list[int]
    ):
        assert len(data) == len(
            prediction_length
        ), "data and prediction_length must have the same length"

        max_pred_len = max(prediction_length)
        with torch.no_grad():
            data = [np.array(d) for d in data]
            quantile_forecast, _ = model.forecast(
                context=data,
                prediction_length=max_pred_len,
                resample_strategy="frequency",
            )

        output = []
        for i in range(len(data)):
            pred_len = prediction_length[i]
            quantiles = quantile_forecast[i, :pred_len, :]

            forecast_i = {
                "median": quantiles[:, 4].tolist(),
                "quantile_levels": quantile_levels,
            }
            for q_idx in range(len(quantile_levels)):
                forecast_i[f"quantile_{q_idx}"] = quantiles[:, q_idx].tolist()
            output.append(forecast_i)
        return output

    return make_forecast_batch


def get_moirai_forecast_fn(device="cuda"):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    model_module = Moirai2Module.from_pretrained(f"Salesforce/moirai-2.0-R-small")
    quantile_levels = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]

    def make_forecast_batch(
        data: list[list[int | float]], prediction_length: list[int]
    ):
        assert len(data) == len(
            prediction_length
        ), "data and prediction_length must have the same length"
        max_pred_len = max(prediction_length)

        model = Moirai2Forecast(
            module=model_module,
            prediction_length=max_pred_len,
            context_length=4000,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(device)

        with torch.no_grad():
            data = [np.array(d) for d in data]
            forecast = model.predict(data)

        output = []
        for i in range(len(data)):
            pred_length = prediction_length[i]
            forecast_i = {
                "median": forecast[i, 4, :pred_length].tolist(),
                "quantile_levels": quantile_levels,
            }
            for q_idx in range(len(quantile_levels)):
                forecast_i[f"quantile_{q_idx}"] = forecast[
                    i, q_idx, :pred_length
                ].tolist()
            output.append(forecast_i)
        return output

    return make_forecast_batch


if __name__ == "__main__":
    chronos_forecast_fn = get_chronos_forecast_fn()
    timesfm_forecast_fn = get_timesfm_forecast_fn()
    tirex_forecast_fn = get_tirex_forecast_fn()
    moirai_forecast_fn = get_moirai_forecast_fn()

    data = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    prediction_length = [2, 2]
    print(chronos_forecast_fn(data, prediction_length))
    print(timesfm_forecast_fn(data, prediction_length))
    print(tirex_forecast_fn(data, prediction_length))
    print(moirai_forecast_fn(data, prediction_length))
