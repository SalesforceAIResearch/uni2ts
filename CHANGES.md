# Commit: Python 3.13 + updated dependency compatibility

**Message:** `fix: Python 3.13 + updated dependency compatibility (pandas 2.2, PyTorch 2.11, gluonts 0.16, datasets 4.x)`

## Summary

Relaxed version pins in `pyproject.toml` to support Python 3.13 (`numpy`, `scipy`, `torch`, `gluonts`, `datasets`). Fixed resulting breakages across the codebase:

- **`src/uni2ts/transform/patch.py`**: Updated `DEFAULT_RANGES` keys to pandas 2.2+ frequency aliases (`H`→`h`, `T`→`min`, `M`→`ME`, etc.)
- **`src/uni2ts/data/builder/simple.py`**: Replaced `Dataset.from_generator` with `Dataset.from_list` to avoid Windows Arrow file-locking in datasets 4.x; added `.astype("float32").squeeze()` for stricter schema enforcement
- **`src/uni2ts/model/moirai2/forecast.py`**: Added `_QuantilePredictionNetWrapper` to adapt `Moirai2Forecast.forward()` output to the gluonts 0.16.x `QuantileForecastGenerator` interface (now expects `((tensor,), loc, scale)`)
- **`test/module/test_attention.py`**: Updated all-masked attention assertion for PyTorch 2.11 behaviour change
- **`test/data/builder/test_simple.py`**: Added explicit `del hf_dataset` before `shutil.rmtree` to release Arrow file handles on Windows
- **`test/transform/test_patch.py`**: Fixed frequency alias lookup to use `norm_freq_str`
- **`README.md`**: Fixed Getting Started code block (added missing `moirai2` imports, removed unused imports, replaced hardcoded lengths with variables)
- **`test/sample_test_run.py`**: Added end-to-end smoke test; uses `forecast.quantile("p50")` instead of `forecast.mean` to avoid gluonts `QuantileForecast` warning
