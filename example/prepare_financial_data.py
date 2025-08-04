import polars as pl
import pandas as pd
from pathlib import Path
from collections.abc import Generator
from typing import Any
import datasets
from datasets import Features, Sequence, Value

def load_and_prepare_data():
    """
    Loads 1-hour BTC-USD data for 2015, prepares it for uni2ts,
    and saves it as a Hugging Face dataset.
    """
    # Define the path to the data
    data_path = "/home/dev/data/ohlcv/asset_class=crypto/freq=1h/symbol=BTC/year=2015/month=*/part.parquet"

    # Load the data using polars
    scan = pl.scan_parquet(
        data_path,
        hive_partitioning=True,
    )

    # Select and rename columns
    df_pl = scan.select(
        pl.col("ts"),
        pl.col("open"),
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        pl.col("volume"),
    ).sort("ts").collect()

    # Convert to pandas for compatibility with the rest of the script
    df_pd = df_pl.to_pandas()
    df_pd.set_index('ts', inplace=True)

    # Define the generator function for the dataset
    def multivar_example_gen_func() -> Generator[dict[str, Any], None, None]:
        yield {
            "target": df_pd[['open', 'high', 'low', 'close', 'volume']].to_numpy().T,  # array of shape (var, time)
            "start": df_pd.index[0],
            "freq": pd.infer_freq(df_pd.index),
            "item_id": "BTC",
        }

    # Define the features of the dataset
    features = Features(
        dict(
            target=Sequence(
                Sequence(Value("float32")), length=len(df_pd.columns)
            ),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            item_id=Value("string"),
        )
    )

    # Create the Hugging Face dataset
    hf_dataset = datasets.Dataset.from_generator(
        multivar_example_gen_func, features=features
    )

    # Save the dataset to disk
    dataset_path = "financial_dataset_btc_2015"
    hf_dataset.save_to_disk(dataset_path)
    print(f"Dataset saved to {dataset_path}")

if __name__ == "__main__":
    load_and_prepare_data()
