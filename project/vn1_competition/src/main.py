from pathlib import Path

import numpy as np
import pandas as pd
import torch
from forecast import Forecast
from gluonts.dataset.pandas import PandasDataset


def load_context(file_path1, file_path2):
    df_sales_0 = pd.read_csv(file_path1)
    df_sales_1 = pd.read_csv(file_path2)
    df_context = pd.concat([df_sales_0, df_sales_1.iloc[:, 3:]], axis=1)
    df_context["item_id"] = (
        df_context["Client"].astype(str)
        + "-"
        + df_context["Warehouse"].astype(str)
        + "-"
        + df_context["Product"].astype(str)
    )
    df_context.drop(columns=["Client", "Warehouse", "Product"], inplace=True)
    cols = ["item_id"] + [col for col in df_context.columns if col != "item_id"]
    df_context = df_context[cols]
    df_context = df_context.T
    df_context.columns = df_context.iloc[0]
    df_context.drop(df_context.index[0], inplace=True)
    df_context.index = pd.to_datetime(df_context.index)
    future_dates = pd.date_range(start="2024-01-08", periods=13, freq="W-MON")
    df_future = pd.DataFrame(index=future_dates, columns=df_context.columns)
    return df_context, df_future


def predict(df_context, df_future, checkpoint_path):
    dataset = PandasDataset(dict(df_context))
    context_length = 65
    prediction_length = 13
    model = Forecast.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size="auto",
        num_samples=500,
        target_dim=1,
        feat_dynamic_real_dim=dataset.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=dataset.num_past_feat_dynamic_real,
    )
    predictor = model.create_predictor(batch_size=128)
    forecasts = predictor.predict(dataset)
    for forecast in forecasts:
        item_id = forecast.item_id
        samples = forecast.samples
        prediction = np.round(np.median(samples, axis=0), decimals=4)
        df_future[item_id] = pd.Series(prediction, index=df_future.index)
    df = df_future.copy()
    df = df.reset_index()
    df.rename(columns={"index": "date"}, inplace=True)
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.reset_index()
    df.rename(columns={"index": "item_id"}, inplace=True)
    df[["Client", "Warehouse", "Product"]] = df["item_id"].str.split("-", expand=True)
    df.drop(columns=["item_id"], inplace=True)
    cols = ["Client", "Warehouse", "Product"] + [
        col for col in df.columns if col not in ["Client", "Warehouse", "Product"]
    ]
    df = df[cols]
    date_cols = [
        col for col in df.columns if col not in ["Client", "Warehouse", "Product"]
    ]
    df.rename(
        columns={col: col.strftime("%Y-%m-%d") for col in date_cols}, inplace=True
    )
    return df


def vn1_competition_evaluation(df):
    assert all(col in df.columns for col in ["Client", "Warehouse", "Product"])
    df["Client"] = df["Client"].astype(np.int64)
    df["Warehouse"] = df["Warehouse"].astype(np.int64)
    df["Product"] = df["Product"].astype(np.int64)
    df = df.set_index(["Client", "Warehouse", "Product"])
    df.columns = pd.to_datetime(df.columns)
    assert ~df.isnull().any().any()

    # Load Objective
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    file_path = project_dir / "data/phase_2_sales.csv"
    objective = pd.read_csv(file_path).set_index(["Client", "Warehouse", "Product"])
    objective.columns = pd.to_datetime(objective.columns)
    assert (df.index == objective.index).all()
    assert (df.columns == objective.columns).all()
    # This is an important rule that we communicate to competitors.
    abs_err = np.nansum(abs(df - objective))
    err = np.nansum((df - objective))
    score = abs_err + abs(err)
    score /= objective.sum().sum()
    return score  # It's a percentage


def main():
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    file_path1 = project_dir / "data/phase_0_sales.csv"
    file_path2 = project_dir / "data/phase_1_sales.csv"
    checkpoint_path = (
        project_dir.parent.parent
        / "outputs/VN1/run1/checkpoints/epoch=18-step=1900.ckpt"
    )
    df_context, df_future = load_context(file_path1, file_path2)
    df = predict(df_context, df_future, checkpoint_path)
    score = vn1_competition_evaluation(df)
    print(score)


if __name__ == "__main__":
    main()
