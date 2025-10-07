#!/usr/bin/env python
"""
Test script to verify the financial dataset can be loaded properly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), '.'))

from uni2ts.data.builder import SimpleDatasetBuilder
from uni2ts.transform import Identity

def test_dataset_loading():
    """Test that the financial dataset can be loaded properly."""
    print("Testing financial dataset loading...")
    
    # Create the builder
    builder = SimpleDatasetBuilder(
        dataset="financial_btc_2015_2020",
        weight=1000
    )
    
    # Create transform map
    transform_map = {"financial_btc_2015_2020": lambda: Identity()}
    
    # Load the dataset
    try:
        dataset = builder.load_dataset(transform_map)
        print(f"✓ Dataset loaded successfully with {len(dataset)} samples")
        
        # Check the first sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample keys: {list(sample.keys())}")
            
            # Check target structure
            if "target" in sample:
                target = sample['target']
                print(f"✓ Target type: {type(target)}")
                if hasattr(target, 'shape'):
                    print(f"✓ Target shape: {target.shape}")
                elif isinstance(target, list):
                    print(f"✓ Target length: {len(target)}")
                    if len(target) > 0:
                        print(f"✓ First variate length: {len(target[0])}")
            
            # Check metadata
            print(f"✓ Start timestamp: {sample.get('start', 'N/A')}")
            print(f"✓ Frequency: {sample.get('freq', 'N/A')}")
            print(f"✓ Item ID: {sample.get('item_id', 'N/A')}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        print("\n✓ Financial dataset test passed!")
    else:
        print("\n✗ Financial dataset test failed!")
        sys.exit(1)
