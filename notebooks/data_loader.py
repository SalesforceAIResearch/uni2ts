# Drop-in replacement for read_parquet_hive (Polars-friendly)
from pathlib import Path
from typing import Union, Optional, Any, List
import polars as pl

def read_parquet_hive(
    root_path: Union[str, Path],
    *,
    asset_class: str,
    symbol: str,
    freq: str,
    year: Optional[int] = None,
    month: Optional[int] = None,
    columns: Optional[List[str]] = None,
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Reads OHLCV data from a hive-partitioned Parquet lake with Polars.
    Ensures `ts` is timezone-aware in UTC and places it as the first column.
    """
    base_path = Path(root_path) / f"asset_class={asset_class}" / f"freq={freq}" / f"symbol={symbol}"

    # Build path pattern
    if year is not None and month is not None:
        search_path = base_path / f"year={year}" / f"month={month:02d}"
    elif year is not None:
        search_path = base_path / f"year={year}"
    else:
        search_path = base_path

    # Always include ts if columns filtering was requested
    read_cols = columns
    if read_cols is not None and "ts" not in read_cols:
        read_cols = ["ts"] + [c for c in read_cols]

    df = pl.read_parquet(
        str(search_path),
        hive_partitioning=True,
        columns=read_cols,
        **kwargs,
    )

    if "ts" in df.columns:
        ts_dtype = df.schema["ts"]
        dtype_name = ts_dtype.__class__.__name__  # "Utf8" or "Datetime", etc.

        if dtype_name == "Utf8":
            # Parse strings to timezone-aware UTC
            df = df.with_columns(
                pl.col("ts").str.to_datetime(time_zone="UTC", strict=False)
            )
        elif dtype_name == "Datetime":
            # Determine if tz is set on the dtype
            tz = getattr(ts_dtype, "time_zone", None)
            if tz is None:
                # Naive -> mark as UTC
                df = df.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
            elif tz != "UTC":
                # Convert to UTC if needed
                df = df.with_columns(pl.col("ts").dt.convert_time_zone("UTC"))
        else:
            # Unexpected type: make a best-effort parse to UTC
            df = df.with_columns(
                pl.col("ts").cast(pl.Utf8).str.to_datetime(time_zone="UTC", strict=False)
            )

        # Put ts first without Pandas ops
        if df.columns[0] != "ts":
            df = df.select(["ts"] + [c for c in df.columns if c != "ts"])

    # Require OHLC; treat volume as optional since some indices may lack it
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns in loaded data: {', '.join(sorted(missing))}")

    return df
