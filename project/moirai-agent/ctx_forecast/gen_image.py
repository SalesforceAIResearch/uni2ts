import argparse
import glob
import os

import pandas as pd
from src.utils.image import plot_historical_and_save
from tqdm import tqdm


def main(in_file, out_file, img_root):
    df = pd.read_parquet(in_file)
    hist_values = df["history_values"].tolist()
    img_paths = []
    for i, v in enumerate(hist_values):
        v = [float(x) for x in v.split(",")]

        start = df["history_start"].iloc[i]

        freq = df["frequency"].iloc[i]
        timestamps = pd.date_range(start=start, periods=len(v), freq=freq)
        ts_list = timestamps.strftime("%Y-%m-%d %H:%M:%S").tolist()

        item = {"Date": ts_list, "Value": v}

        img_path = os.path.join(img_root, f"image_{i}.jpg")
        img_paths.append(img_path)
        plot_historical_and_save(item, img_path)
    df["image_path"] = img_paths

    df.to_parquet(out_file)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script to plot time series")
    parser.add_argument("--in_file", default="./gift_ctx.parquet", required=True)
    parser.add_argument("--out_file", default="./gift_ctx_image.parquet", required=True)
    parser.add_argument("--img_root", default="./images/", required=True)

    args = parser.parse_args()
    main(args.in_file, args.out_file, args.img_root)
