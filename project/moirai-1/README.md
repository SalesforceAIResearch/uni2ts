# Moirai-1.0-R

Moirai is a Masked Encoder-based Universal Time Series Forecasting Transformer introduced in [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592).
This page contains information on reproducing the model training and evaluation results presented in the paper.

## Pre-trained Models

Pre-trained weights of the three model sizes can be found in the following table.

| # Model | # Parameters |
| :---: | :---: |
| [Moirai-1.0-R-Small](https://huggingface.co/Salesforce/moirai-1.0-R-small) | 14m |
| [Moirai-1.0-R-Base](https://huggingface.co/Salesforce/moirai-1.0-R-base) | 91m |
| [Moirai-1.0-R-Large](https://huggingface.co/Salesforce/moirai-1.0-R-large) | 311m |

## LOTSA Data v1

The Moirai-1.0-R series have been trained on the Large-scale Open Time Series Archive (LOTSA), also introduced in [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592). 
This dataset has been open sourced and can be found on the [Hugging Face Hub](https://huggingface.co/datasets/Salesforce/lotsa_data/).

## Pre-training
To reproduce the training runs, simply run the following commands.

**N.B.:** Run these commands in the base [uni2ts](../..) folder.

**N.B.:** You may need to adjust the gradient accumulation settings depending on your hardware.

```shell
python -m cli.train \ 
  -cp conf/pretrain \
  run_name=moirai_small \
  model=moirai_small \
  data=lotsa_v1_weighted \
  trainer.max_epochs=1_000 \
  train_dataloader.batch_size=256
```

```shell
python -m cli.train \ 
  -cp conf/pretrain \
  run_name=moirai_base \
  model=moirai_base \
  data=lotsa_v1_weighted \
  trainer.max_epochs=10_000 \
  train_dataloader.batch_size=256
```

```shell
python -m cli.train \ 
  -cp conf/pretrain \
  run_name=moirai_large \
  model=moirai_large \
  data=lotsa_v1_weighted \
  trainer.max_epochs=10_000 \
  train_dataloader.batch_size=256
```

## Evaluation

The following shell scripts can be used to reproduce the results on the Monash, Probabilistic Forecasting, and Long Sequence Forecasting settings as presented in the paper.

To run the probabilistic forecasting and long sequence forecasting, ensrue to add the following datasets to `LSF_PATH`, add `LSF_PATH` to the `.env` file.
* Datasets from Time-Series-Library: https://github.com/thuml/Time-Series-Library
* Walmart: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/overview
* Istanbul Traffic: https://www.kaggle.com/datasets/leonardo00/istanbul-traffic-index
* Turkey Power: https://www.kaggle.com/datasets/dharanikra/electrical-power-demand-in-turkey

**N.B.:** Run these commands in the base [uni2ts](../..) folder.

**N.B.:** Patch size and context lengths have been tuned based on the validation sets as described in the [paper](https://arxiv.org/abs/2402.02592), we directly fix them in the following scripts based on the validation set results.

```shell
# Monash TSF
./project/moirai-1/eval/monash_small.sh
./project/moirai-1/eval/monash_base.sh
./project/moirai-1/eval/monash_large.sh

# Probabilistic Forecasting
./project/moirai-1/eval/pf_small.sh
./project/moirai-1/eval/pf_base.sh
./project/moirai-1/eval/pf_large.sh

# Long Sequence Forecasting
./project/moirai-1/eval/lsf_small.sh
./project/moirai-1/eval/lsf_base.sh
./project/moirai-1/eval/lsf_large.sh
```
