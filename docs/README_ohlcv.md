# OHLCV Parquet Lake

Columnar, hive-partitioned Parquet dataset for intraday and daily OHLCV across multiple asset classes. Optimized for local analytics with DuckDB and Polars. No server involved.

## Layout

Base folder: `/home/dev/data/ohlcv`

```

asset\_class=\<crypto|fx|index|etf|equity|futures>/
freq=<1min|15min|1h|4h|1d>/
symbol=\<TICKER\_OR\_PAIR>/
year=YYYY/
month=MM/
part.parquet

````

Partition folders are hive-style. Engines expose `asset_class`, `freq`, `symbol`, `year`, `month` as virtual columns.

## Schema

Each `part.parquet` contains:

| column       | type                     | notes                                  |
|--------------|--------------------------|----------------------------------------|
| ts           | TIMESTAMP WITH TIME ZONE | always stored in UTC                   |
| open         | DOUBLE                   |                                        |
| high         | DOUBLE                   |                                        |
| low          | DOUBLE                   |                                        |
| close        | DOUBLE                   |                                        |
| volume       | DOUBLE                   | may be null for some indices           |
| asset_class  | STRING                   | materialized for convenience           |
| symbol       | STRING                   | materialized for convenience           |

Compression: ZSTD with Parquet statistics enabled.

## Time zone rules

- All stored timestamps are UTC.
- Source files:
  - **Crypto** timestamps are UTC in vendor files.
  - **FX** and **Index** timestamps are Eastern Time in vendor files. During ingest they are localized to America/New_York, then converted to UTC.
  - **ETF** and **Equity** timestamps are Eastern Time in vendor files. Same conversion to UTC during ingest.

## Resampling

1 min to higher intervals uses:
- open = first
- high = max
- low = min
- close = last
- volume = sum

Windows are left-closed and labeled by the window start.

Anchors for window boundaries:
- Crypto: anchored to UTC for all intervals, including 1d.
- FX, Index, ETF, Equity: anchored to America/New_York for all intervals, including 1d.

This means daily bars for non-crypto represent calendar days in Eastern Time. Crypto daily bars represent calendar days in UTC.

## File sizing

Monthly partitions per symbol. For very early years with sparse activity some files are small. For heavy symbols and recent years, files are larger. If needed later, row group size can be tuned when writing.

## Ingestion outline

Ingestion is done with Polars:
- Parse and localize timestamps per rules above.
- Deduplicate on `(symbol, ts)`.
- Write monthly Parquet under hive partitions.
- Build 15min, 1h, 4h, 1d from 1min using the anchors above.

Helpers in the notebook:
- `read_crypto_1min_utc`, `read_fx_1min`, `read_index_1min`, `read_crypto_equity_etf_1min`
- `resample_ohlcv_anchored`
- `write_monthly_parquet`
- `ingest_folder`, `unzip_and_ingest_archives`

## Query examples

### DuckDB

Set session to UTC for display:

```sql
SET TimeZone='UTC';
````

Scan ETH 15 min for July 2025:

```sql
SELECT symbol, ts, open, high, low, close, volume
FROM read_parquet(
  '/home/dev/data/ohlcv/asset_class=crypto/freq=15min/symbol=ETH/year=2025/month=07/part.parquet',
  hive_partitioning=1
)
ORDER BY ts
LIMIT 20;
```

Count rows per symbol at 1 min:

```sql
SELECT symbol, COUNT(*) AS rows, MIN(ts) AS min_ts, MAX(ts) AS max_ts
FROM read_parquet('/home/dev/data/ohlcv/asset_class=equity/freq=1min/symbol=*/year=*/month=*/part.parquet', hive_partitioning=1)
GROUP BY symbol
ORDER BY symbol
LIMIT 50;
```

Daily bars for SPY in 2022:

```sql
SELECT ts, open, high, low, close, volume
FROM read_parquet('/home/dev/data/ohlcv/asset_class=etf/freq=1d/symbol=SPY/year=2022/month=*/part.parquet', hive_partitioning=1)
ORDER BY ts;
```

### Polars (lazy)

```python
import polars as pl

scan = pl.scan_parquet(
    "/home/dev/data/ohlcv/asset_class=fx/freq=1h/symbol=USDJPY/year=2024/month=*/part.parquet",
    hive_partitioning=True,
)

df = (
    scan
    .filter(pl.col("ts").is_between(pl.datetime(2024,1,1, time_unit="ms", time_zone="UTC"),
                                    pl.datetime(2024,4,1, time_unit="ms", time_zone="UTC")))
    .group_by([pl.col("ts").dt.truncate("1d").alias("d")])
    .agg(pl.col("close").last().alias("close_last"))
    .sort("d")
).collect()
```

## Conventions

* Symbols appear as given by the vendor. Crypto pairs like `ETH-BTC` are stored as a single `symbol` value.
* Only minutes with trades are present in source files. Gaps indicate zero traded volume.
* Volume units: shares for equities and ETFs, contracts for futures, base currency units for FX, native units for crypto.
* Index files may not have volume.

## Known variations and options

* Adjustment type for ETF and Equity files is implied by the source filename (for example `_adjsplit`). The current layout does not add an `adjustment` partition. If you need to store multiple adjustment types side by side, add a partition like `adjustment=<unadj|adjsplit|adjtotal>` and propagate it during write.
* If you prefer daily bars in UTC for all assets, change the resampling anchor to `calendar_tz='UTC'` when building `1d`.

## Tips for WSL

* Keep data on the Linux filesystem, for example under `/home/dev`. Reading from `/mnt/c` is slower.
* Renaming the lake root is safe. Paths are discovered at query time. If you keep absolute paths in your own catalogs, store them relative to `LAKE_ROOT`.

