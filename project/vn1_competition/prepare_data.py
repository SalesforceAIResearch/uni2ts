import os
from collections.abc import Generator
from typing import Any

import datasets
import pandas as pd
from datasets import Features, Sequence, Value


def train_example_gen_func() -> Generator[dict[str, Any], None, None]:
    for i, (product_id, df) in enumerate(train_df.items()):
        yield {
            "target": df.to_numpy(),
            "start": df.index[0],
            "freq": pd.infer_freq(df.index),
            "item_id": f"item_{i}",
        }


def val_example_gen_func() -> Generator[dict[str, Any], None, None]:
    for i, (product_id, df) in enumerate(val_df.items()):
        yield {
            "target": df.to_numpy(),
            "start": df.index[0],
            "freq": pd.infer_freq(df.index),
            "item_id": f"item_{i}",
        }


def get_data(file_path1, file_path2):
    df_sales_0 = pd.read_csv(file_path1)
    df_sales_1 = pd.read_csv(file_path2)
    df_sales = pd.concat([df_sales_0, df_sales_1.iloc[:, 3:]], axis=1)
    df_sales["item_id"] = (
        df_sales["Client"].astype(str)
        + "-"
        + df_sales["Warehouse"].astype(str)
        + "-"
        + df_sales["Product"].astype(str)
    )
    df_sales.drop(columns=["Client", "Warehouse", "Product"], inplace=True)
    cols = ["item_id"] + [col for col in df_sales.columns if col != "item_id"]
    df_sales = df_sales[cols]
    df_sales = df_sales.T
    df_sales.columns = df_sales.iloc[0]
    df_sales.drop(df_sales.index[0], inplace=True)
    df_sales.index = pd.to_datetime(df_sales.index)
    return df_sales


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path1 = os.path.join(current_dir, "data/phase_0_sales.csv")
file_path2 = os.path.join(current_dir, "data/phase_1_sales.csv")

df_sales = get_data(file_path1, file_path2)
df_sales = df_sales.iloc[70:, :]
df_sales.index.name = "timestamp"
zero_ratios = (df_sales == 0).mean()
cols_to_drop = zero_ratios[zero_ratios > 0.5].index
df_sales = df_sales.drop(columns=cols_to_drop)

train_df = df_sales.iloc[:-16, :]
val_df = df_sales.iloc[:, :]

features = Features(
    dict(
        target=Sequence(Value("float32")),
        start=Value("timestamp[s]"),
        freq=Value("string"),
        item_id=Value("string"),
    )
)

train_dataset = datasets.Dataset.from_generator(
    train_example_gen_func, features=features
)
val_dataset = datasets.Dataset.from_generator(val_example_gen_func, features=features)
train_dataset_path = os.path.join(current_dir, "train_dataset")
val_dataset_path = os.path.join(current_dir, "val_dataset")
train_dataset.save_to_disk(train_dataset_path)
val_dataset.save_to_disk(val_dataset_path)
