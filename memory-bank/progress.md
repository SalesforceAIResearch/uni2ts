# Progress: Uni2TS Financial Model Fine-Tuning

## Current Status
We have a detailed plan for fine-tuning the Moirai model on the BTC dataset. The plan is based on a thorough review of the project's documentation and examples. We are ready to begin implementation.

## What Works
- The data loading and training pipeline is now understood.
- A pre-training job has been successfully run with an example dataset.
- The BTC dataset has been prepared and saved to disk.

## What's Left to Build
- A custom `DatasetBuilder` in `src/uni2ts/data/builder/financial.py`.
- An update to `src/uni2ts/data/builder/__init__.py` to include the new builder.
- A new data configuration file at `cli/conf/finetune/data/financial_btc_2015.yaml`.
- A new run configuration file at `cli/conf/finetune/finetune_btc.yaml`.
- The execution of the fine-tuning job.
- Documentation of the results in the memory bank.

## Known Issues
- None at this time.

## Evolution of Project Decisions
- The project has evolved from a general analysis of the repository to a concrete plan for fine-tuning a model on a specific dataset (2015 BTC-USD 1-hour data).
