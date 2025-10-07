#!/bin/bash
# Script to run pre-training on financial data using the MOIRAI model

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/home/dev/repos/uni2ts

# Create output directory if it doesn't exist
mkdir -p outputs/pretrain/moirai_financial/custom_financial

# Run the training script
echo "Starting pre-training on financial data..."
python -m cli.train -cp conf/pretrain \
    run_name=financial_pretrain \
    model=moirai_financial \
    data=custom_financial \
    trainer.max_epochs=100 \
    trainer.precision=16 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.gradient_clip_val=1.0 \
    train_dataloader.batch_size=32 \
    train_dataloader.num_batches_per_epoch=100

echo "Pre-training completed"
