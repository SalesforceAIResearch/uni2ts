#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import glob
import shutil
import tempfile
import re
import argparse
from pathlib import Path
from typing import Any, Callable, List, Optional, Dict

import polars as pl
import pandas as pd
import numpy as np
import datasets
from datasets import Features, Sequence, Value
from torch.utils.data import Dataset, ConcatDataset

from uni2ts.data.builder._base import DatasetBuilder
from uni2ts.data.dataset import MultiSampleTimeSeriesDataset, SampleTimeSeriesType
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Transformation, Identity
from uni2ts.common.env import env


class IterativeFinancialDatasetBuilder(DatasetBuilder):
    """
    A dataset builder that processes financial data iteratively from parquet files.
    This builder avoids loading the entire dataset into memory at once by processing
    assets in batches and using temporary storage.
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 10,  # Number of assets to process at once
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        temp_dir: Optional[str] = None,
        asset_class: str = "crypto",
        freq: str = "1h",
        years: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        max_ts: int = 128,
        combine_fields: tuple = ("target",),
    ):
        """
        Initialize the dataset builder.
        
        Args:
            data_path: Base path to the parquet files
            batch_size: Number of assets to process at once
            sample_time_series: Sampling strategy for time series
            temp_dir: Directory to store temporary processed data
            asset_class: Asset class to filter by (e.g., "crypto", "equity")
            freq: Frequency to filter by (e.g., "1h", "1d")
            years: List of years to include (e.g., ["2015", "2016"])
            symbols: List of asset symbols to include (e.g., ["BTC", "ETH"])
            max_ts: Maximum number of time series to combine
            combine_fields: Fields to combine when creating the dataset
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.sample_time_series = sample_time_series
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="uni2ts_")
        self.asset_class = asset_class
        self.freq = freq
        self.years = years
        self.symbols = symbols
        self.max_ts = max_ts
        self.combine_fields = combine_fields
        
    def build_dataset(self, asset_symbols: List[str] = None):
        """
        Process the parquet files and build datasets for the specified assets.
        
        Args:
            asset_symbols: List of asset symbols to process. If None, use self.symbols.
        """
        symbols_to_process = asset_symbols or self.symbols
        if symbols_to_process is None:
            # If no symbols specified, get all available symbols
            symbols_to_process = self._get_all_symbols()
            
        for symbol in symbols_to_process:
            asset_dir = self._get_asset_dir(symbol)
            if not asset_dir:
                print(f"Warning: No data found for symbol {symbol}")
                continue
                
            # Load and process the asset data
            try:
                hf_dataset = self._load_and_process_asset(asset_dir, symbol)
                
                # Save to temporary directory
                temp_path = os.path.join(self.temp_dir, f"{symbol}")
                hf_dataset.save_to_disk(temp_path)
                print(f"Processed and saved dataset for {symbol}")
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
    
    def load_dataset(self, transform_map: dict[Any, Callable[..., Transformation]]) -> Dataset:
        """
        Load datasets for all assets in the data path.
        
        Args:
            transform_map: Map of transformations to apply to the datasets
            
        Returns:
            A dataset containing all the processed assets
        """
        # Get all asset directories
        if self.symbols:
            asset_dirs = [self._get_asset_dir(symbol) for symbol in self.symbols]
            asset_dirs = [d for d in asset_dirs if d]  # Remove None values
        else:
            asset_dirs = self._get_asset_dirs()
        
        if not asset_dirs:
            raise ValueError("No asset directories found")
            
        # Process assets in batches
        datasets_list = []
        for i in range(0, len(asset_dirs), self.batch_size):
            batch_dirs = asset_dirs[i:i+self.batch_size]
            
            # Process this batch of assets
            batch_datasets = self._process_asset_batch(batch_dirs, transform_map)
            datasets_list.extend(batch_datasets)
            
            # Clean up temporary files for this batch
            self._cleanup_batch(batch_dirs)
        
        if not datasets_list:
            raise ValueError("No datasets were created")
            
        # Return a concatenated dataset
        return ConcatDataset(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
    
    def _get_all_symbols(self) -> List[str]:
        """Get all available symbols in the data path."""
        symbol_pattern = os.path.join(
            self.data_path, 
            f"asset_class={self.asset_class}", 
            f"freq={self.freq}", 
            "symbol=*"
        )
        symbol_dirs = glob.glob(symbol_pattern)
        return [self._extract_symbol(d) for d in symbol_dirs]
    
    def _get_asset_dir(self, symbol: str) -> Optional[str]:
        """Get the directory for a specific asset symbol."""
        asset_dir = os.path.join(
            self.data_path, 
            f"asset_class={self.asset_class}", 
            f"freq={self.freq}", 
            f"symbol={symbol}"
        )
        return asset_dir if os.path.exists(asset_dir) else None
    
    def _get_asset_dirs(self) -> List[str]:
        """Get all asset directories in the data path."""
        # Build the pattern based on the provided filters
        pattern = os.path.join(
            self.data_path, 
            f"asset_class={self.asset_class}", 
            f"freq={self.freq}", 
            "symbol=*"
        )
        
        # Get all matching directories
        asset_dirs = glob.glob(pattern)
        
        # Filter by symbols if provided
        if self.symbols:
            asset_dirs = [d for d in asset_dirs if self._extract_symbol(d) in self.symbols]
            
        return asset_dirs
    
    def _process_asset_batch(
        self, 
        batch_dirs: List[str], 
        transform_map: dict[Any, Callable[..., Transformation]]
    ) -> List[Dataset]:
        """
        Process a batch of assets and return datasets.
        
        Args:
            batch_dirs: List of asset directories to process
            transform_map: Map of transformations to apply to the datasets
            
        Returns:
            List of datasets for the processed assets
        """
        batch_datasets = []
        
        for asset_dir in batch_dirs:
            # Extract asset symbol from directory path
            symbol = self._extract_symbol(asset_dir)
            
            # Check if we already have processed data in the temp directory
            temp_path = os.path.join(self.temp_dir, f"{symbol}")
            if not os.path.exists(temp_path):
                # Load and process the asset data
                try:
                    hf_dataset = self._load_and_process_asset(asset_dir, symbol)
                    
                    # Save to temporary directory
                    hf_dataset.save_to_disk(temp_path)
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
                    continue
            
            try:
                # Get the appropriate transformation
                transform = self._get_transform(transform_map)
                
                # Create a dataset using the saved data
                dataset = MultiSampleTimeSeriesDataset(
                    HuggingFaceDatasetIndexer(datasets.load_from_disk(temp_path), uniform=False),
                    transform,
                    max_ts=self.max_ts,
                    combine_fields=self.combine_fields,
                    sample_time_series=self.sample_time_series,
                )
                
                batch_datasets.append(dataset)
            except Exception as e:
                print(f"Error creating dataset for {symbol}: {str(e)}")
                continue
        
        return batch_datasets
    
    def _extract_symbol(self, asset_dir: str) -> str:
        """Extract the asset symbol from the directory path."""
        # Use regex to extract the symbol from the directory path
        # Example: '/home/dev/data/ohlcv/asset_class=crypto/freq=1h/symbol=BTC' -> 'BTC'
        match = re.search(r'symbol=([^/]+)', asset_dir)
        if match:
            return match.group(1)
        else:
            # Fallback: get the last part of the path
            return os.path.basename(asset_dir).replace('symbol=', '')
    
    def _load_and_process_asset(self, asset_dir: str, symbol: str) -> datasets.Dataset:
        """
        Load and process data for a single asset.
        
        Args:
            asset_dir: Directory containing the asset data
            symbol: Symbol of the asset
            
        Returns:
            Processed Hugging Face dataset
        """
        # Build the pattern for parquet files
        if self.years:
            # If specific years are provided, only include those
            patterns = [f"{asset_dir}/year={year}/month=*/part.parquet" for year in self.years]
        else:
            # Otherwise, include all years
            patterns = [f"{asset_dir}/year=*/month=*/part.parquet"]
        
        # Check if any files match the patterns
        all_files = []
        for pattern in patterns:
            all_files.extend(glob.glob(pattern))
        
        if not all_files:
            raise ValueError(f"No parquet files found for {symbol} with patterns: {patterns}")
        
        # Load the data using polars
        scan = pl.scan_parquet(
            patterns,
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
        
        # Convert to pandas
        df_pd = df_pl.to_pandas()
        df_pd.set_index('ts', inplace=True)
        
        # Define the generator function for the dataset
        def multivar_example_gen_func():
            yield {
                "target": df_pd[['open', 'high', 'low', 'close', 'volume']].to_numpy().T,
                "start": df_pd.index[0],
                "freq": pd.infer_freq(df_pd.index) or "1h",  # Fallback to 1h if inference fails
                "item_id": symbol,
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
        return datasets.Dataset.from_generator(
            multivar_example_gen_func, features=features
        )
    
    def _get_transform(self, transform_map: dict[Any, Callable[..., Transformation]]) -> Transformation:
        """Get the appropriate transformation from the transform map."""
        if "default" in transform_map:
            return transform_map["default"]()
        elif MultiSampleTimeSeriesDataset in transform_map:
            return transform_map[MultiSampleTimeSeriesDataset]()
        else:
            return Identity()
    
    def _cleanup_batch(self, batch_dirs: List[str]):
        """Clean up temporary files for a batch of assets."""
        for asset_dir in batch_dirs:
            symbol = self._extract_symbol(asset_dir)
            temp_path = os.path.join(self.temp_dir, f"{symbol}")
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
    
    def cleanup(self):
        """Clean up all temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def save_dataset(self, dataset_name: str, storage_path: Optional[Path] = None):
        """
        Save the processed dataset to the custom data directory.
        
        Args:
            dataset_name: Name of the dataset to save
            storage_path: Path to store the dataset (defaults to CUSTOM_DATA_PATH)
        """
        if storage_path is None:
            storage_path = Path(env.CUSTOM_DATA_PATH)
        
        # Create the storage directory if it doesn't exist
        dataset_path = storage_path / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Process and save the dataset
        symbols_to_process = self.symbols
        if symbols_to_process is None:
            symbols_to_process = self._get_all_symbols()
            
        all_datasets = []
        
        for symbol in symbols_to_process:
            asset_dir = self._get_asset_dir(symbol)
            if not asset_dir:
                print(f"Warning: No data found for symbol {symbol}")
                continue
                
            try:
                # Load and process the asset data
                hf_dataset = self._load_and_process_asset(asset_dir, symbol)
                all_datasets.append(hf_dataset)
                print(f"Processed dataset for {symbol}")
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
        
        if not all_datasets:
            raise ValueError("No datasets were created")
        
        # Concatenate all datasets
        if len(all_datasets) > 1:
            # Combine all datasets into one
            combined_dataset = datasets.concatenate_datasets(all_datasets)
        else:
            combined_dataset = all_datasets[0]
        
        # Save the combined dataset
        combined_dataset.info.dataset_name = dataset_name
        combined_dataset.save_to_disk(dataset_path)
        print(f"Dataset saved to {dataset_path}")


def main():
    """Command-line interface for the IterativeFinancialDatasetBuilder."""
    parser = argparse.ArgumentParser(description="Build financial datasets from parquet files")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to create")
    parser.add_argument("--data_path", type=str, default="/home/dev/data/ohlcv", help="Path to parquet data")
    parser.add_argument("--asset_class", type=str, default="crypto", help="Asset class (crypto, equity, etc.)")
    parser.add_argument("--freq", type=str, default="1h", help="Frequency (1h, 1d, etc.)")
    parser.add_argument("--years", type=str, nargs="+", help="Years to include (e.g., 2015 2016 2017)")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to include (e.g., BTC ETH)")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of assets to process at once")
    parser.add_argument("--storage_path", type=str, help="Path to store the dataset (defaults to CUSTOM_DATA_PATH)")

    args = parser.parse_args()

    # Create the builder
    builder = IterativeFinancialDatasetBuilder(
        data_path=args.data_path,
        batch_size=args.batch_size,
        asset_class=args.asset_class,
        freq=args.freq,
        years=args.years,
        symbols=args.symbols,
    )

    # Save the dataset
    builder.save_dataset(args.dataset_name, Path(args.storage_path) if args.storage_path else None)


if __name__ == "__main__":
    main()
