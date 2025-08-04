# Multivariate OHLCV Forecasting Strategy

## 1. Introduction and Objectives

### Purpose
This document outlines a comprehensive strategy for training a Moirai model on multivariate OHLCV (Open, High, Low, Close, Volume) data using the uni2ts framework. It provides detailed guidance on data preparation, model configuration, training, and evaluation.

### Goals
- Create a specialized forecasting model for financial time series data
- Leverage the multivariate nature of OHLCV data to improve forecast accuracy
- Establish a reproducible workflow for training and fine-tuning models on financial data
- Achieve competitive performance compared to domain-specific financial forecasting models

### Success Criteria
- **Primary Metrics**: CRPS (Continuous Ranked Probability Score) and MSIS (Mean Scaled Interval Score) for probabilistic evaluation
- **Secondary Metrics**: MSE (Mean Squared Error) and MAE (Mean Absolute Error) for point forecast evaluation
- **Practical Metrics**: Directional accuracy and profitability metrics for trading applications

## 2. Data Preparation Pipeline

### 2.1 Data Selection and Extraction

#### Asset Selection
- **Asset Class**: Start with crypto assets (e.g., BTC, ETH) due to their 24/7 trading and cleaner data patterns
- **Symbols**: Begin with major assets (BTC, ETH) and gradually expand to include more assets
- **Frequency**: Use 1h data as a balance between signal granularity and sequence length
- **Date Range**: Use at least 2 years of data, with the most recent 20% reserved for validation

#### Data Extraction with Polars
```python
import polars as pl

def extract_ohlcv_data(asset_class, symbol, freq, start_year, end_year, start_month=1, end_month=12):
    """
    Extract OHLCV data for a specific asset from the Parquet data lake.
    
    Parameters:
    -----------
    asset_class : str
        Asset class (crypto, fx, equity, etc.)
    symbol : str
        Symbol or ticker
    freq : str
        Frequency (1min, 15min, 1h, 4h, 1d)
    start_year, end_year : int
        Start and end years
    start_month, end_month : int
        Start and end months (default: full year)
        
    Returns:
    --------
    polars.DataFrame
        DataFrame containing the OHLCV data
    """
    # Construct the path pattern
    path_pattern = f"/home/dev/data/ohlcv/asset_class={asset_class}/freq={freq}/symbol={symbol}/year={{{start_year}..{end_year}}}/month={{{start_month}..{end_month}}}/part.parquet"
    
    # Scan the parquet files
    scan = pl.scan_parquet(
        path_pattern,
        hive_partitioning=True,
    )
    
    # Select and rename columns
    df = scan.select(
        pl.col("ts"),
        pl.col("open"),
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        pl.col("volume"),
    ).sort("ts").collect()
    
    return df
```

#### Timezone Handling
- **Crypto**: Data is stored in UTC, no conversion needed
- **FX, Equity, ETF, Index**: Data is stored in UTC but represents trading in Eastern Time
- **Handling**: Use the appropriate timezone information when creating datetime features

```python
def add_datetime_features(df, asset_class):
    """
    Add datetime features to the DataFrame based on the asset class.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing OHLCV data with a 'ts' column
    asset_class : str
        Asset class (crypto, fx, equity, etc.)
        
    Returns:
    --------
    polars.DataFrame
        DataFrame with additional datetime features
    """
    # Define the timezone based on asset class
    tz = "UTC" if asset_class == "crypto" else "America/New_York"
    
    # Add datetime features
    df = df.with_columns([
        pl.col("ts").dt.year().alias("year"),
        pl.col("ts").dt.month().alias("month"),
        pl.col("ts").dt.day().alias("day"),
        pl.col("ts").dt.hour().alias("hour"),
        pl.col("ts").dt.weekday().alias("weekday"),
        pl.col("ts").dt.is_weekend().alias("is_weekend"),
    ])
    
    return df
```

### 2.2 Data Preprocessing

#### Handling Missing Values
- **Detection**: Identify missing timestamps in the expected frequency grid
- **Imputation**: Use forward fill for missing values, with a maximum limit to avoid stale data
- **Gaps**: For large gaps (e.g., market closures), consider adding a binary indicator feature

```python
def handle_missing_values(df, freq, max_ffill_periods=3):
    """
    Handle missing values in OHLCV data.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing OHLCV data
    freq : str
        Frequency string (e.g., '1h', '1d')
    max_ffill_periods : int
        Maximum number of periods to forward fill
        
    Returns:
    --------
    polars.DataFrame
        DataFrame with missing values handled
    """
    # Convert frequency string to pandas frequency
    pd_freq = freq.replace('min', 'T')
    
    # Create a complete timestamp grid
    min_ts = df["ts"].min()
    max_ts = df["ts"].max()
    
    # Create a reference DataFrame with all expected timestamps
    date_range = pl.date_range(min_ts, max_ts, interval=pd_freq, closed="left")
    ref_df = pl.DataFrame({"ts": date_range})
    
    # Join with the original data
    filled_df = ref_df.join(df, on="ts", how="left")
    
    # Add a binary column indicating missing data
    filled_df = filled_df.with_column(
        pl.when(pl.col("open").is_null()).then(1).otherwise(0).alias("is_missing")
    )
    
    # Forward fill missing values up to max_ffill_periods
    for col in ["open", "high", "low", "close", "volume"]:
        filled_df = filled_df.with_column(
            pl.col(col).forward_fill(limit=max_ffill_periods)
        )
    
    # Drop any remaining nulls
    filled_df = filled_df.drop_nulls()
    
    return filled_df
```

#### Outlier Detection and Handling
- **Z-score Method**: Identify values beyond 3 standard deviations
- **Modified Z-score**: Use median absolute deviation for robustness
- **Handling**: Cap outliers at threshold values rather than removing them

```python
def handle_outliers(df, columns=["open", "high", "low", "close", "volume"], threshold=3.0, method="zscore"):
    """
    Detect and handle outliers in OHLCV data.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing OHLCV data
    columns : list
        List of columns to check for outliers
    threshold : float
        Threshold for outlier detection (default: 3.0)
    method : str
        Method for outlier detection ('zscore' or 'modified_zscore')
        
    Returns:
    --------
    polars.DataFrame
        DataFrame with outliers handled
    """
    result_df = df.clone()
    
    for col in columns:
        if method == "zscore":
            # Calculate z-scores
            mean = result_df[col].mean()
            std = result_df[col].std()
            z_scores = (result_df[col] - mean) / std
            
            # Cap outliers
            result_df = result_df.with_column(
                pl.when(z_scores > threshold).then(mean + threshold * std)
                .when(z_scores < -threshold).then(mean - threshold * std)
                .otherwise(pl.col(col))
                .alias(col)
            )
        elif method == "modified_zscore":
            # Calculate modified z-scores using median absolute deviation
            median = result_df[col].median()
            mad = result_df[col].sub(median).abs().median() * 1.4826  # Scaling factor for normal distribution
            
            # Cap outliers
            result_df = result_df.with_column(
                pl.when((pl.col(col) - median).abs() / mad > threshold).then(
                    pl.when(pl.col(col) > median).then(median + threshold * mad)
                    .otherwise(median - threshold * mad)
                )
                .otherwise(pl.col(col))
                .alias(col)
            )
    
    return result_df
```

#### Normalization
- **Per-Series Normalization**: Apply instance-wise normalization to each series
- **Methods**: StandardScaler (z-score) or MinMaxScaler (0-1 range)
- **Metadata Retention**: Store normalization parameters for later denormalization

```python
def normalize_ohlcv(df, method="standard", columns=["open", "high", "low", "close", "volume"]):
    """
    Normalize OHLCV data and return normalization parameters.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing OHLCV data
    method : str
        Normalization method ('standard' or 'minmax')
    columns : list
        List of columns to normalize
        
    Returns:
    --------
    tuple
        (normalized_df, norm_params)
    """
    norm_df = df.clone()
    norm_params = {}
    
    for col in columns:
        if method == "standard":
            # Standard scaling (z-score)
            mean = df[col].mean()
            std = df[col].std()
            norm_df = norm_df.with_column(
                ((pl.col(col) - mean) / std).alias(col)
            )
            norm_params[col] = {"mean": mean, "std": std}
        elif method == "minmax":
            # Min-max scaling to [0, 1]
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            norm_df = norm_df.with_column(
                ((pl.col(col) - min_val) / range_val).alias(col)
            )
            norm_params[col] = {"min": min_val, "max": max_val}
    
    return norm_df, norm_params
```

### 2.3 Feature Engineering

#### Technical Indicators
- **Trend Indicators**: Moving Averages (SMA, EMA, WMA), MACD, Parabolic SAR
- **Momentum Indicators**: RSI, Stochastic Oscillator, CCI
- **Volatility Indicators**: Bollinger Bands, ATR, Standard Deviation
- **Volume Indicators**: OBV, Volume ROC, Accumulation/Distribution Line

```python
def add_technical_indicators(df):
    """
    Add technical indicators to OHLCV data.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing OHLCV data
        
    Returns:
    --------
    polars.DataFrame
        DataFrame with additional technical indicators
    """
    # Convert to pandas for easier calculation with TA-Lib or pandas-ta
    pdf = df.to_pandas()
    
    # Moving Averages
    pdf['sma_5'] = pdf['close'].rolling(window=5).mean()
    pdf['sma_10'] = pdf['close'].rolling(window=10).mean()
    pdf['sma_20'] = pdf['close'].rolling(window=20).mean()
    pdf['ema_5'] = pdf['close'].ewm(span=5, adjust=False).mean()
    pdf['ema_10'] = pdf['close'].ewm(span=10, adjust=False).mean()
    pdf['ema_20'] = pdf['close'].ewm(span=20, adjust=False).mean()
    
    # MACD
    pdf['macd'] = pdf['ema_12'] - pdf['ema_26']
    pdf['macd_signal'] = pdf['macd'].ewm(span=9, adjust=False).mean()
    pdf['macd_hist'] = pdf['macd'] - pdf['macd_signal']
    
    # RSI
    delta = pdf['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    pdf['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    pdf['bb_middle'] = pdf['close'].rolling(window=20).mean()
    pdf['bb_std'] = pdf['close'].rolling(window=20).std()
    pdf['bb_upper'] = pdf['bb_middle'] + 2 * pdf['bb_std']
    pdf['bb_lower'] = pdf['bb_middle'] - 2 * pdf['bb_std']
    
    # ATR
    high_low = pdf['high'] - pdf['low']
    high_close = (pdf['high'] - pdf['close'].shift()).abs()
    low_close = (pdf['low'] - pdf['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    pdf['atr_14'] = true_range.rolling(14).mean()
    
    # Convert back to polars
    result_df = pl.from_pandas(pdf)
    
    # Drop NaN values resulting from indicators calculation
    result_df = result_df.drop_nulls()
    
    return result_df
```

#### Cross-Asset Features
- **Market-wide Indicators**: Include features from market indices or sector ETFs
- **Correlation Features**: Rolling correlations with related assets
- **Relative Strength**: Performance relative to broader market or sector

```python
def add_cross_asset_features(main_df, reference_assets, window=20):
    """
    Add cross-asset features to the main DataFrame.
    
    Parameters:
    -----------
    main_df : polars.DataFrame
        DataFrame containing the main asset's OHLCV data
    reference_assets : dict
        Dictionary mapping asset names to their DataFrames
    window : int
        Window size for rolling calculations
        
    Returns:
    --------
    polars.DataFrame
        DataFrame with additional cross-asset features
    """
    # Convert to pandas for easier calculation
    pdf = main_df.to_pandas()
    pdf.set_index('ts', inplace=True)
    
    # Process each reference asset
    for asset_name, asset_df in reference_assets.items():
        # Convert reference asset to pandas and align index
        ref_pdf = asset_df.to_pandas()
        ref_pdf.set_index('ts', inplace=True)
        
        # Ensure both DataFrames have the same index
        common_index = pdf.index.intersection(ref_pdf.index)
        pdf = pdf.loc[common_index]
        ref_pdf = ref_pdf.loc[common_index]
        
        # Calculate relative performance
        pdf[f'rel_perf_{asset_name}'] = pdf['close'] / pdf['close'].shift(1) / (ref_pdf['close'] / ref_pdf['close'].shift(1))
        
        # Calculate rolling correlation
        pdf[f'corr_{asset_name}'] = pdf['close'].rolling(window=window).corr(ref_pdf['close'])
        
        # Calculate beta (market sensitivity)
        returns = pdf['close'].pct_change()
        ref_returns = ref_pdf['close'].pct_change()
        cov = returns.rolling(window=window).cov(ref_returns)
        var = ref_returns.rolling(window=window).var()
        pdf[f'beta_{asset_name}'] = cov / var
    
    # Convert back to polars
    pdf.reset_index(inplace=True)
    result_df = pl.from_pandas(pdf)
    
    # Drop NaN values
    result_df = result_df.drop_nulls()
    
    return result_df
```

#### Market Regime Indicators
- **Volatility Regimes**: VIX-like indicators for different asset classes
- **Trend Strength**: ADX or similar indicators to identify trending vs. ranging markets
- **Seasonality**: Day-of-week, hour-of-day, month-of-year effects

```python
def add_regime_indicators(df):
    """
    Add market regime indicators to OHLCV data.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing OHLCV data
        
    Returns:
    --------
    polars.DataFrame
        DataFrame with additional regime indicators
    """
    # Convert to pandas for calculations
    pdf = df.to_pandas()
    
    # Volatility indicator (similar to VIX calculation)
    returns = pdf['close'].pct_change()
    pdf['volatility_20'] = returns.rolling(window=20).std() * (252 ** 0.5)  # Annualized
    
    # Trend strength indicator (ADX-like)
    # Simplified calculation for demonstration
    up_move = pdf['high'].diff()
    down_move = pdf['low'].diff().multiply(-1)
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    tr = pd.concat([
        (pdf['high'] - pdf['low']).abs(),
        (pdf['high'] - pdf['close'].shift()).abs(),
        (pdf['low'] - pdf['close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    pdf['adx_14'] = dx.rolling(window=14).mean()
    
    # Market regime classification
    pdf['regime'] = 'neutral'
    pdf.loc[(pdf['adx_14'] > 25) & (plus_di > minus_di), 'regime'] = 'uptrend'
    pdf.loc[(pdf['adx_14'] > 25) & (plus_di < minus_di), 'regime'] = 'downtrend'
    pdf.loc[pdf['adx_14'] < 20, 'regime'] = 'ranging'
    
    # One-hot encode the regime
    regime_dummies = pd.get_dummies(pdf['regime'], prefix='regime')
    pdf = pd.concat([pdf, regime_dummies], axis=1)
    
    # Convert back to polars
    pdf.reset_index(inplace=True)
    result_df = pl.from_pandas(pdf)
    
    # Drop NaN values
    result_df = result_df.drop_nulls()
    
    return result_df
```

#### Feature Selection
- **Correlation Analysis**: Remove highly correlated features
- **Feature Importance**: Use tree-based models to rank feature importance
- **Domain Knowledge**: Prioritize features with known predictive power in financial markets

```python
def select_features(df, target_col='close', max_correlation=0.95, top_n=20):
    """
    Select the most relevant features using correlation analysis and feature importance.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing features
    target_col : str
        Target column for prediction
    max_correlation : float
        Maximum allowed correlation between features
    top_n : int
        Number of top features to select
        
    Returns:
    --------
    list
        List of selected feature names
    """
    # Convert to pandas for correlation analysis
    pdf = df.to_pandas()
    
    # Calculate correlation matrix
    corr_matrix = pdf.corr().abs()
    
    # Remove highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > max_correlation)]
    pdf_filtered = pdf.drop(columns=to_drop)
    
    # Calculate feature importance using a tree-based model
    from sklearn.ensemble import RandomForestRegressor
    
    # Prepare data
    X = pdf_filtered.drop(columns=[target_col, 'ts'])
    y = pdf_filtered[target_col]
    
    # Train a random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Select top N features
    selected_features = feature_importance.head(top_n)['feature'].tolist()
    
    return selected_features
```

### 2.4 Dataset Creation

#### Hugging Face Dataset Generator
- **Multivariate Format**: Structure data as (variates, time) for uni2ts compatibility
- **Metadata**: Include frequency, start timestamp, and item identifier
- **Features Schema**: Define the schema for the Hugging Face dataset

```python
from collections.abc import Generator
from typing import Any
import datasets
from datasets import Features, Sequence, Value
import numpy as np

def create_multivariate_ohlcv_dataset(df, feature_cols=None, target_cols=None, item_id="BTC"):
    """
    Create a Hugging Face dataset for multivariate OHLCV data.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing OHLCV data and features
    feature_cols : list
        List of feature column names to include as dynamic covariates
    target_cols : list
        List of target column names to forecast (default: ['close'])
    item_id : str
        Identifier for the time series
        
    Returns:
    --------
    datasets.Dataset
        Hugging Face dataset
    """
    # Default columns if not specified
    if target_cols is None:
        target_cols = ['close']
    
    if feature_cols is None:
        feature_cols = ['open', 'high', 'low', 'volume']
    
    # Convert to pandas for easier handling
    pdf = df.to_pandas()
    pdf.set_index('ts', inplace=True)
    
    # Extract target and feature arrays
    target_array = pdf[target_cols].to_numpy().T  # Shape: (variates, time)
    feature_array = pdf[feature_cols].to_numpy().T  # Shape: (variates, time)
    
    # Define the generator function
    def multivar_example_gen_func() -> Generator[dict[str, Any], None, None]:
        yield {
            "target": target_array,  # array of shape (var, time)
            "feat_dynamic_real": feature_array,  # array of shape (var, time)
            "start": pdf.index[0],
            "freq": pd.infer_freq(pdf.index),
            "item_id": item_id,
        }
    
    # Define the features schema
    features = Features(
        dict(
            target=Sequence(
                Sequence(Value("float32")), length=len(target_cols)
            ),
            feat_dynamic_real=Sequence(
                Sequence(Value("float32")), length=len(feature_cols)
            ),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            item_id=Value("string"),
        )
    )
    
    # Create the dataset
    hf_dataset = datasets.Dataset.from_generator(
        multivar_example_gen_func, features=features
    )
    
    return hf_dataset
```

#### Train/Validation/Test Splitting
- **Time-Based Split**: Use chronological splitting to avoid data leakage
- **Validation Size**: Use 20% of data for validation
- **Test Size**: Use 10% of data for final testing

```python
def create_train_val_test_datasets(df, val_ratio=0.2, test_ratio=0.1, feature_cols=None, target_cols=None, item_id="BTC"):
    """
    Create train, validation, and test datasets from OHLCV data.
    
    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing OHLCV data and features
    val_ratio : float
        Ratio of data to use for validation
    test_ratio : float
        Ratio of data to use for testing
    feature_cols : list
        List of feature column names
    target_cols : list
        List of target column names
    item_id : str
        Identifier for the time series
        
    Returns:
    --------
    tuple
        (train_dataset, val_dataset, test_dataset)
    """
    # Sort by timestamp
    df = df.sort("ts")
    
    # Calculate split indices
    n = len(df)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - val_size - test_size
    
    # Split the DataFrame
    train_df = df.slice(0, train_size)
    val_df = df.slice(train_size, val_size)
    test_df = df.slice(train_size + val_size, test_size)
    
    # Create datasets
    train_dataset = create_multivariate_ohlcv_dataset(train_df, feature_cols, target_cols, item_id)
    val_dataset = create_multivariate_ohlcv_dataset(val_df, feature_cols, target_cols, item_id)
    test_dataset = create_multivariate_ohlcv_dataset(test_df, feature_cols, target_cols, item_id)
    
    # Save datasets
    train_dataset.save_to_disk(f"datasets/{item_id}_train")
    val_dataset.save_to_disk(f"datasets/{item_id}_val")
    test_dataset.save_to_disk(f"datasets/{item_id}_test")
    
    return train_dataset, val_dataset, test_dataset
```

#### Complete Data Pipeline
- **End-to-End Process**: Combine all steps into a single pipeline
- **Configuration**: Make the pipeline configurable for different assets and timeframes
- **Caching**: Implement caching for intermediate results to speed up experimentation

```python
def ohlcv_data_pipeline(
    asset_class,
    symbol,
    freq,
    start_year,
    end_year,
    feature_engineering=True,
    cross_asset_features=False,
    reference_assets=None,
    normalization_method="standard",
    val_ratio=0.2,
    test_ratio=0.1
):
    """
    End-to-end data pipeline for OHLCV data.
    
    Parameters:
    -----------
    asset_class : str
        Asset class (crypto, fx, equity, etc.)
    symbol : str
        Symbol or ticker
    freq : str
        Frequency (1min, 15min, 1h, 4h, 1d)
    start_year, end_year : int
        Start and end years
    feature_engineering : bool
        Whether to add technical indicators
    cross_asset_features : bool
        Whether to add cross-asset features
    reference_assets : dict
        Dictionary of reference assets for cross-asset features
    normalization_method : str
        Method for normalization ('standard' or 'minmax')
    val_ratio, test_ratio : float
        Ratios for validation and test splits
        
    Returns:
    --------
    tuple
        (train_dataset, val_dataset, test_dataset, norm_params)
    """
    # 1. Extract data
    df = extract_ohlcv_data(asset_class, symbol, freq, start_year, end_year)
    
    # 2. Add datetime features
    df = add_datetime_features(df, asset_class)
    
    # 3. Handle missing values
    df = handle_missing_values(df, freq)
    
    # 4. Handle outliers
    df = handle_outliers(df)
    
    # 5. Feature engineering
    if feature_engineering:
        df = add_technical_indicators(df)
    
    # 6. Add cross-asset features
    if cross_asset_features and reference_assets is not None:
        df = add_cross_asset_features(df, reference_assets)
    
    # 7. Add market regime indicators
    df = add_regime_indicators(df)
    
    # 8. Select features
    feature_cols = select_features(df, target_col='close')
    target_cols = ['close']
    
    # 9. Normalize data
    df_norm, norm_params = normalize_ohlcv(df, method=normalization_method)
    
    # 10. Create datasets
    train_dataset, val_dataset, test_dataset = create_train_val_test_datasets(
        df_norm, val_ratio, test_ratio, feature_cols, target_cols, symbol
    )
    
    return train_dataset, val_dataset, test_dataset, norm_params
```

## 3. Model Architecture and Configuration

### 3.1 Moirai Model Overview

#### Architecture Components
- **Patchified Masked Encoder**: Splits input into non-overlapping patches
- **Any-variate Attention**: Handles arbitrary number of variates with permutation-equivariance
- **Mixture Distribution Output**: Provides probabilistic forecasts with flexible distribution types

#### Model Sizes
- **Small (14M parameters)**: For quick experimentation and resource-constrained environments
- **Base (91M parameters)**: Balanced performance and computational requirements
- **Large (311M parameters)**: Maximum accuracy for production use

#### Pre-trained Checkpoints
- **moirai_1.1_R_small**: `Salesforce/moirai-1.1-R-small`
- **moirai_1.1_R_base**: `Salesforce/moirai-1.1-R-base`
- **moirai_1.1_R_large**: `Salesforce/moirai-1.1-R-large`

### 3.2 Patch Size Selection

#### Frequency-to-Patch Mapping
- **Hourly Data (1h)**: Patch sizes 32 or 64
- **4-Hour Data (4h)**: Patch sizes 32 or 64
- **Daily Data (1d)**: Patch sizes 32 or 64
- **Minute Data (1min, 15min)**: Patch sizes 64 or 128

#### Automatic Selection
- Use `patch_size="auto"` to let uni2ts select the appropriate patch size based on frequency
- Manual override for specific requirements or experimentation

#### Impact on Performance
- Larger patches reduce sequence length and computational cost
- Smaller patches preserve temporal detail but increase computational requirements
- Ablation studies show that using the wrong patch size can significantly degrade performance

### 3.3 Context and Prediction Length

#### Context Length Selection
- **Minimum Context**: At least 5x the prediction length for stable forecasting
- **Maximum Context**: Limited by token budget (typically 512 tokens in pre-training)
- **Recommended Values**:
  - Hourly data (1h): 168-336 hours (1-2 weeks)
  - 4-Hour data (4h): 120-240 hours (20-40 days)
  - Daily data (1d): 60-120 days (2-4 months)

#### Prediction Length Selection
- **Short-term Forecasts**: 24-48 hours for hourly data, 7-14 days for daily data
- **Medium-term Forecasts**: 72-168 hours for hourly data, 30-60 days for daily data
- **Long-term Forecasts**: Consider using larger patch sizes for very long horizons

#### Token Budget Calculation
- **Formula**: `tokens ≈ (#variates) × ⌈(context + horizon) / patch_size⌉`
- **Example**: 
  - 5 variates (OHLCV)
  - 168 hours context
  - 24 hours prediction
  - Patch size 32
  - `tokens ≈ 5 × ⌈(168 + 24) / 32⌉ = 5 × 6 = 30 tokens`
- **Extended Example**:
  - 20 variates (OHLCV + 15 technical indicators)
  - 336 hours context
  - 48 hours prediction
  - Patch size 32
  - `tokens ≈ 20 × ⌈(336 + 48) / 32⌉ = 20 × 12 = 240 tokens`

### 3.4 Multivariate Configuration

#### Target vs. Covariates
- **Target Variable**: Typically 'close' price for financial forecasting
- **Dynamic Covariates**: Other OHLCV components and technical indicators
- **Past Dynamic Covariates**: Features that are only known in the context window

#### Handling Multiple Assets
- **Single Asset with Features**: Use 'close' as target, other OHLCV and indicators as covariates
- **Multiple Assets**: Include multiple assets as separate variates
- **Cross-Asset Relationships**: The any-variate attention mechanism will learn relationships between assets

#### Scaling Considerations
- **Per-Variate Scaling**: Each variate is scaled independently
- **Scale Retention**: Store scaling parameters for later denormalization
- **Normalization Method**: StandardScaler (z-score) is generally preferred for financial data

## 4. Training Configuration

### 4.1 Optimizer and Learning Rate

#### Optimizer Selection
- **AdamW**: Preferred optimizer with weight decay for regularization
- **Weight Decay**: 0.01 is a good starting point
- **Gradient Clipping**: Use gradient clipping to prevent exploding gradients

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    eps=1e-8
)
```

#### Learning Rate Schedule
- **Warmup**: Use 10% of total steps for warmup
- **Cosine Decay**: Cosine decay with restarts for the remaining steps
- **Initial Learning Rate**: 1e-4 is a good starting point

```python
from uni2ts.optim import get_scheduler, SchedulerType

scheduler = get_scheduler(
    name=SchedulerType.COSINE_WITH_RESTARTS,
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
    scheduler_specific_kwargs={"num_cycles": 3}
)
```

### 4.2 Loss Function

#### Primary Loss
- **Negative Log-Likelihood (NLL)**: For probabilistic forecasting
- **Distribution Type**: Student-t or mixture distribution for financial data

```python
from uni2ts.loss.packed import PackedNLLLoss
from uni2ts.distribution import StudentTOutput, MixtureOutput

# For Student-t distribution
distr_output = StudentTOutput()
loss_fn = PackedNLLLoss()

# For mixture distribution
distr_output = MixtureOutput([
    StudentTOutput(),
    NegativeBinomialOutput(),
    LogNormalOutput(),
    NormalOutput()
])
loss_fn = PackedNLLLoss()
```

#### Evaluation Metrics
- **CRPS**: For probabilistic forecast evaluation
- **MSE/MAE**: For point forecast evaluation
- **Directional Accuracy**: For trading applications

```python
from uni2ts.eval_util.metrics import MedianMSE
from gluonts.evaluation import make_evaluation_predictions, Evaluator

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
```

### 4.3 Training Parameters

#### Batch Size
- **Starting Point**: 32 for small model, 16 for base model, 8 for large model
- **Adjustment**: Increase or decrease based on available memory
- **Gradient Accumulation**: Use gradient accumulation for effectively larger batch sizes

#### Training Steps
- **Fine-tuning**: 10,000 - 50,000 steps depending on dataset size
- **Early Stopping**: Use validation loss with patience of 10 epochs
- **Checkpointing**: Save model every 1,000 steps

#### Data Loading
- **Packed Sequences**: Use packed sequences for efficient training
- **Shuffling**: Shuffle data at the epoch level
- **Prefetching**: Use prefetching to speed up data loading

## 5. Implementation Steps

### 5.1 Custom Dataset Builder

#### Financial Dataset Builder
- **Class Definition**: Create a custom `FinancialDatasetBuilder` class
- **Integration**: Integrate with the uni2ts data module
- **Configuration**: Make it configurable for different assets and timeframes

```python
# src/uni2ts/data/builder/financial.py
import datasets
from datasets.builder import DatasetBuilder
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

class FinancialDatasetBuilder(DatasetBuilder):
    """Builder for financial time series datasets."""
    
    def _info(self):
        return datasets.DatasetInfo(
            description="Financial time series dataset",
            features=datasets.Features({
                "target": datasets.Sequence(
                    datasets.Sequence(datasets.Value("float32")),
                    length=self.config.target_dim
                ),
                "feat_dynamic_real": datasets.Sequence(
                    datasets.Sequence(datasets.Value("float32")),
                    length=self.config.feat_dynamic_real_dim
                ),
                "start": datasets.Value("timestamp[s]"),
                "freq": datasets.Value("string"),
                "item_id": datasets.Value("string"),
            }),
            supervised_keys=None,
        )
    
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": self.config.data_dir / "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": self.config.data_dir / "val"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": self.config.data_dir / "test"},
            ),
        ]
    
    def _generate_examples(self, filepath):
        """Yields examples."""
        # Load the dataset
        dataset = datasets.load_from_disk(filepath)
        
        # Yield examples
        for i, example in enumerate(dataset):
            yield i, example
```

#### Integration with uni2ts
- **Update `__init__.py`**: Add the new builder to the module
- **Configuration**: Create configuration files for the dataset

```python
# src/uni2ts/data/builder/__init__.py
from .financial import FinancialDatasetBuilder

__all__ = [
    "FinancialDatasetBuilder",
    # ... other builders
]
```

### 5.2 Configuration Files

#### Data Configuration
- **Create `financial_btc.yaml`**: Define the configuration for BTC dataset
- **Parameters**: Specify dataset path, dimensions, and other parameters

```yaml
# cli/conf/finetune/data/financial_btc.yaml
_target_: uni2ts.data.builder.financial.FinancialDatasetBuilder
name: financial_btc
data_dir: ${oc.env:DATASET_DIR}/financial_btc
target_dim: 1
feat_dynamic_real_dim: 4
```

#### Model Configuration
- **Use Existing Model**: Use the pre-trained Moirai model
- **Parameters**: Specify model size, patch size, and other parameters

```yaml
# cli/conf/finetune/model/moirai_financial.yaml
defaults:
  - moirai_1.1_R_base
  
patch_size: 32
context_length: 168
prediction_length: 24
```

#### Training Configuration
- **Create `finetune_btc.yaml`**: Define the configuration for fine-tuning
- **Parameters**: Specify optimizer, learning rate, batch size, and other parameters

```yaml
# cli/conf/finetune/finetune_btc.yaml
defaults:
  - default
  - model: moirai_financial
  - data: financial_btc
  - _self_

run_name: btc_finetune
max_steps: 20000
val_check_interval: 1000
```

### 5.3 Training Execution

#### Command Line Interface
- **Use the CLI**: Use the uni2ts CLI for training
- **Parameters**: Specify configuration files and overrides

```bash
python -m cli.train \
  -cp conf/finetune \
  run_name=btc_finetune \
  model=moirai_financial \
  data=financial_btc \
  trainer.max_steps=20000 \
  trainer.val_check_interval=1000
```

#### Monitoring and Logging
- **TensorBoard**: Use TensorBoard for monitoring training progress
- **Checkpoints**: Save and load checkpoints for resuming training
- **Metrics**: Log metrics for evaluation

```bash
# Start TensorBoard
tensorboard --logdir=lightning_logs
```

### 5.4 Model Evaluation

#### Evaluation Script
- **Create Evaluation Script**: Create a script for evaluating the model
- **Metrics**: Calculate CRPS, MSE, MAE, and other metrics
- **Visualization**: Create visualizations of the forecasts

```python
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.eval_util.evaluation import evaluate_model
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import matplotlib.pyplot as plt

# Load the model
module = MoiraiModule.from_pretrained("path/to/checkpoint")
forecast_model = MoiraiForecast(
    prediction_length=24,
    target_dim=1,
    feat_dynamic_real_dim=4,
    context_length=168,
    module=module,
    patch_size="auto",
    num_samples=100,
)

# Create a predictor
predictor = forecast_model.create_predictor(batch_size=32)

# Load the test dataset
test_dataset = datasets.load_from_disk("datasets/BTC_test")

# Make predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_dataset,
    predictor=predictor,
    num_samples=100,
)

# Evaluate predictions
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
metrics = evaluator(ts_it, forecast_it)

# Print metrics
print(f"CRPS: {metrics['mean_wQuantileLoss']}")
print(f"MSE: {metrics['MSE']}")
print(f"MASE: {metrics['MASE']}")
```

#### Visualization
- **Plot Forecasts**: Create plots of the forecasts
- **Confidence Intervals**: Show confidence intervals for probabilistic forecasts
- **Comparison**: Compare with baseline models

```python
from uni2ts.eval_util.plot import plot_single

# Plot a single forecast
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plot_single(
    inp=test_data[0],
    label=test_data[0]["target"],
    forecast=forecasts[0],
    context_length=168,
    ax=ax
)
plt.savefig("forecast.png")
```

## 6. Iterative Refinement

### 6.1 Hyperparameter Tuning

#### Key Parameters
- **Patch Size**: Try different patch sizes (32, 64)
- **Context Length**: Experiment with different context lengths
- **Learning Rate**: Try different learning rates and schedules
- **Batch Size**: Experiment with different batch sizes

#### Tuning Strategy
- **Grid Search**: Systematically try different combinations
- **Random Search**: Randomly sample from parameter distributions
- **Bayesian Optimization**: Use Bayesian optimization for efficient search

```python
# Example hyperparameter grid
param_grid = {
    "patch_size": [32, 64],
    "context_length": [168, 336],
    "learning_rate": [1e-4, 5e-5, 1e-5],
    "batch_size": [16, 32, 64]
}
```

### 6.2 Feature Engineering

#### Feature Ablation
- **Remove Features**: Systematically remove features to identify important ones
- **Add Features**: Experiment with additional features
- **Feature Combinations**: Try different combinations of features

#### Advanced Features
- **Sentiment Analysis**: Include sentiment data from news or social media
- **Market Microstructure**: Include order book data if available
- **Macro Indicators**: Include macroeconomic indicators for longer-term forecasts

### 6.3 Model Ensemble

#### Ensemble Techniques
- **Model Averaging**: Average predictions from multiple models
- **Stacking**: Train a meta-model on the predictions of base models
- **Boosting**: Sequentially train models to correct errors of previous ones

#### Diversity Sources
- **Different Architectures**: Combine Moirai with other models (e.g., DeepAR, N-BEATS)
- **Different Hyperparameters**: Use models with different hyperparameters
- **Different Features**: Use models trained on different feature sets

```python
# Example ensemble code
def ensemble_forecasts(forecasts, weights=None):
    """
    Ensemble multiple forecasts.
    
    Parameters:
    -----------
    forecasts : list
        List of forecast objects
    weights : list
        List of weights for each forecast
        
    Returns:
    --------
    forecast
        Ensembled forecast
    """
    if weights is None:
        weights = [1.0 / len(forecasts)] * len(forecasts)
    
    # Combine samples
    samples = []
    for forecast, weight in zip(forecasts, weights):
        samples.append(forecast.samples * weight)
    
    # Create new forecast object
    ensembled_forecast = forecasts[0].copy()
    ensembled_forecast.samples = sum(samples)
    
    return ensembled_forecast
```

## 7. Appendix: Code Snippets and Examples

### 7.1 Complete Data Preparation Example

```python
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from collections.abc import Generator
from typing import Any
import datasets
from datasets import Features, Sequence, Value

# Extract data
df = extract_ohlcv_data("crypto", "BTC", "1h", 2020, 2022)

# Add datetime features
df = add_datetime_features(df, "crypto")

# Handle missing values
df = handle_missing_values(df, "1h")

# Handle outliers
df = handle_outliers(df)

# Add technical indicators
df = add_technical_indicators(df)

# Add market regime indicators
df = add_regime_indicators(df)

# Select features
feature_cols = select_features(df, target_col='close')
target_cols = ['close']

# Normalize data
df_norm, norm_params = normalize_ohlcv(df, method="standard")

# Create datasets
train_dataset, val_dataset, test_dataset = create_train_val_test_datasets(
    df_norm, 0.2, 0.1, feature_cols, target_cols, "BTC"
)

# Save normalization parameters
import json
with open("datasets/BTC_norm_params.json", "w") as f:
    json.dump(norm_params, f)
```

### 7.2 Complete Training Example

```python
import torch
import lightning as L
from uni2ts.model.moirai import MoiraiPretrain, MoiraiModule
from uni2ts.distribution import StudentTOutput
from uni2ts.optim import get_scheduler, SchedulerType

# Load the pre-trained model
module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base")

# Create a fine-tuning model
model = MoiraiPretrain(
    min_patches=4,
    min_mask_ratio=0.1,
    max_mask_ratio=0.2,
    max_dim=100,
    num_training_steps=20000,
    num_warmup_steps=2000,
    module=module,
    module_kwargs={
        "distr_output": StudentTOutput(),
        "d_model": 256,
        "num_layers": 4,
        "patch_sizes": (32, 64),
        "max_seq_len": 512,
        "attn_dropout_p": 0.1,
        "dropout_p": 0.1,
        "scaling": True,
    }
)

# Create a trainer
trainer = L.Trainer(
    max_steps=20000,
    val_check_interval=1000,
    logger=L.loggers.TensorBoardLogger("lightning_logs"),
    callbacks=[
        L.callbacks.ModelCheckpoint(
            dirpath="checkpoints",
            filename="moirai-{step}",
            save_top_k=3,
            monitor="val_loss"
        ),
        L.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        )
    ],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
)

# Train the model
trainer.fit(model, train_dataloader, val_dataloader)
```

### 7.3 Complete Evaluation Example

```python
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.eval_util.evaluation import evaluate_model
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import matplotlib.pyplot as plt
import pandas as pd
import json

# Load normalization parameters
with open("datasets/BTC_norm_params.json", "r") as f:
    norm_params = json.load(f)

# Load the fine-tuned model
module = MoiraiModule.from_pretrained("checkpoints/moirai-20000.ckpt")

# Create a forecast model
forecast_model = MoiraiForecast(
    prediction_length=24,
    target_dim=1,
    feat_dynamic_real_dim=len(feature_cols),
    context_length=168,
    module=module,
    patch_size="auto",
    num_samples=100,
)

# Create a predictor
predictor = forecast_model.create_predictor(batch_size=32)

# Load the test dataset
test_dataset = datasets.load_from_disk("datasets/BTC_test")

# Make predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_dataset,
    predictor=predictor,
    num_samples=100,
)

# Evaluate predictions
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
metrics = evaluator(ts_it, forecast_it)

# Print metrics
print(f"CRPS: {metrics['mean_wQuantileLoss']}")
print(f"MSE: {metrics['MSE']}")
print(f"MASE: {metrics['MASE']}")

# Denormalize forecasts
forecasts = list(forecast_it)
for i, forecast in enumerate(forecasts):
    # Denormalize samples
    mean = norm_params["close"]["mean"]
    std = norm_params["close"]["std"]
    forecast.samples = forecast.samples * std + mean

# Plot forecasts
for i in range(min(5, len(forecasts))):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_single(
        inp=test_data[i],
        label=test_data[i]["target"],
        forecast=forecasts[i],
        context_length=168,
        ax=ax
    )
    plt.savefig(f"forecast_{i}.png")
```

### 7.4 Troubleshooting Guide

#### Common Issues and Solutions

1. **Out of Memory Errors**
   - Reduce batch size
   - Use gradient accumulation
   - Use a smaller model
   - Use a larger patch size

2. **Slow Training**
   - Use a GPU if available
   - Use packed sequences
   - Use a larger batch size if memory allows
   - Use a smaller model

3. **Poor Performance**
   - Check data quality and preprocessing
   - Try different hyperparameters
   - Use a larger model
   - Add more features
   - Use ensemble methods

4. **Overfitting**
   - Use early stopping
   - Increase weight decay
   - Reduce model size
   - Use dropout
   - Use data augmentation

5. **Underfitting**
   - Use a larger model
   - Train for more steps
   - Reduce weight decay
   - Add more features
   - Use a smaller patch size
