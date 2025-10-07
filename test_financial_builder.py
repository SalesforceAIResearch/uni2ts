#!/usr/bin/env python
"""
Test script for the IterativeFinancialDatasetBuilder.
This script tests loading a small batch of assets to ensure the builder works correctly.
"""

import os
import tempfile
import shutil
from pathlib import Path

from uni2ts.data.builder import IterativeFinancialDatasetBuilder
from uni2ts.transform import Identity
from uni2ts.data.dataset import SampleTimeSeriesType

def test_builder():
    """Test the IterativeFinancialDatasetBuilder with a small batch of assets."""
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp(prefix="uni2ts_test_")
    
    try:
        # Create the builder with a small batch size and limited years
        builder = IterativeFinancialDatasetBuilder(
            data_path="/home/dev/data/ohlcv",
            batch_size=2,  # Small batch size for testing
            sample_time_series=SampleTimeSeriesType.PROPORTIONAL,
            temp_dir=temp_dir,
            asset_class="index",
            freq="1h",
            years=["2015"],  # Just one year for testing
            symbols=["SPX", "NDX"],  # Just two symbols for testing
            max_ts=128,
            combine_fields=("target",),
        )
        
        # Create a simple transform map
        transform_map = {"default": lambda: Identity()}
        
        # Try to load the dataset
        print("Loading dataset...")
        dataset = builder.load_dataset(transform_map)
        
        # Print some information about the dataset
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        
        # Try to access a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            if "target" in sample:
                # Convert to numpy array if it's a list
                import numpy as np
                target = sample['target']
                if isinstance(target, list):
                    target = np.array(target)
                print(f"Target shape: {target.shape}")
        
        # Clean up
        builder.cleanup()
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_builder()
