"""
End-to-end smoke test for uni2ts using the Moirai2 model.

Downloads the Moirai-2.0-R-small model weights from HuggingFace on first run
(~300 MB cached to ~/.cache/huggingface/).

Run with:
    python test/sample_test_run.py
"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

# Use non-interactive backend when no display is available (e.g. CI)
matplotlib.use("Agg")

PDT = 20    # prediction length
CTX = 200   # context length
BSZ = 32    # batch size
TEST = 100  # number of test time steps


def load_sample_data() -> pd.DataFrame:
    url = (
        "https://gist.githubusercontent.com/rsnirwan/"
        "c8c8654a98350fadd229b00167174ec4/raw/"
        "a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
    )
    print("Loading sample time series data...")
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    print(f"  Loaded DataFrame: {df.shape[0]} time steps, {df.shape[1]} series")
    return df


def build_test_data(df: pd.DataFrame):
    ds = PandasDataset(dict(df))
    _, test_template = split(ds, offset=-TEST)
    test_data = test_template.generate_instances(
        prediction_length=PDT,
        windows=TEST // PDT,
        distance=PDT,
    )
    return test_data


def load_model() -> Moirai2Forecast:
    print("Loading Moirai-2.0-R-small model (downloads on first run)...")
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
        prediction_length=PDT,
        context_length=CTX,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    print("  Model loaded successfully")
    return model


def run_forecast(model: Moirai2Forecast, test_data) -> tuple:
    print("Running forecasts...")
    predictor = model.create_predictor(batch_size=BSZ)
    forecasts = list(predictor.predict(test_data.input))

    inputs = list(test_data.input)
    labels = list(test_data.label)
    print(f"  Generated {len(forecasts)} forecast(s)")
    return inputs, labels, forecasts


def validate_forecasts(inputs, labels, forecasts) -> bool:
    print("Validating forecasts...")
    all_ok = True
    for i, (inp, label, forecast) in enumerate(zip(inputs, labels, forecasts)):
        # Use the median (p50) as the point estimate — QuantileForecast does not
        # store a separate "mean", so forecast.mean would warn and fall back here anyway.
        pred_mean = forecast.quantile("p50")

        if pred_mean.shape[-1] != PDT:
            print(f"  ERROR forecast {i}: expected length {PDT}, got {pred_mean.shape[-1]}")
            all_ok = False
            continue

        if np.isnan(pred_mean).any():
            print(f"  ERROR forecast {i}: prediction contains NaN values")
            all_ok = False
            continue

    if all_ok:
        print(f"  All {len(forecasts)} forecast(s) passed validation")
    return all_ok


def save_plot(inp, label, forecast, path: str = "test/sample_forecast.png"):
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_single(
        inp,
        label,
        forecast,
        context_length=CTX,
        ax=ax,
        name="Moirai2",
        show_label=True,
    )
    ax.set_title("Moirai-2.0-R-small — Sample Forecast")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  Plot saved to {path}")


def main():
    print("=" * 60)
    print("uni2ts end-to-end smoke test")
    print("=" * 60)

    df = load_sample_data()
    test_data = build_test_data(df)
    model = load_model()
    inputs, labels, forecasts = run_forecast(model, test_data)
    ok = validate_forecasts(inputs, labels, forecasts)

    print("Saving sample forecast plot...")
    save_plot(inputs[0], labels[0], forecasts[0])

    print("=" * 60)
    if ok:
        print("RESULT: All checks passed")
        sys.exit(0)
    else:
        print("RESULT: One or more checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
