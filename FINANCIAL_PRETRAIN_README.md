# Financial Pre-training for MOIRAI

This directory contains scripts and configuration files for pre-training the MOIRAI model on financial time series data.

## Overview

The implementation allows for pre-training the MOIRAI model on large financial datasets stored in parquet format. It processes the data iteratively, one asset at a time, to avoid loading the entire dataset into memory at once.

## Files

- `src/uni2ts/data/builder/custom_financial_builder.py`: Custom dataset builder that processes financial data iteratively
- `cli/conf/pretrain/data/custom_financial.yaml`: Configuration file for the dataset
- `cli/conf/pretrain/model/moirai_financial.yaml`: Configuration file for the model
- `test_financial_builder.py`: Test script to verify the dataset builder works
- `run_financial_pretrain.sh`: Script to run the pre-training

## Usage

### Testing the Dataset Builder

Before running the full pre-training, you can test the dataset builder to ensure it works correctly:

```bash
./test_financial_builder.py
```

This will load a small batch of assets and verify that the dataset builder works as expected.

### Running Pre-training

To run the pre-training:

```bash
./run_financial_pretrain.sh
```

This will start the pre-training process using the configuration files in `cli/conf/pretrain/`.

### Customizing the Pre-training

You can customize the pre-training by modifying the configuration files:

- `cli/conf/pretrain/data/custom_financial.yaml`: Modify this file to change the dataset configuration (e.g., asset class, frequency, years, symbols)
- `cli/conf/pretrain/model/moirai_financial.yaml`: Modify this file to change the model configuration (e.g., model size, number of layers, patch sizes)

You can also modify the `run_financial_pretrain.sh` script to change the training parameters (e.g., number of epochs, batch size, learning rate).

## Monitoring Training

The training process logs metrics using TensorBoard. You can monitor the training by running:

```bash
tensorboard --logdir outputs/pretrain/moirai_financial/custom_financial/financial_pretrain/logs
```

This will start a TensorBoard server that you can access in your browser to view training metrics like loss over time.

## Checkpoints

The training process saves checkpoints in the `outputs/pretrain/moirai_financial/custom_financial/financial_pretrain/checkpoints` directory. You can use these checkpoints to resume training or for inference.

## Cleaning Up

The dataset builder creates temporary files in a directory specified by the `temp_dir` parameter. These files are automatically cleaned up after each batch is processed, but you can also manually clean them up by calling the `cleanup()` method on the dataset builder.
