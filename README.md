# Unified Training of Universal Time Series Transformers

Uni2TS is a PyTorch based library for research and applications related to Time Series Forecasting. It provides a unified framework for large-scale pre-training, fine-tuning, inference, and evaluation of Universal Time Series Transformers.

Related reading: [Moirai Paper](https://arxiv.org/abs/2402.02592), [Moirai Salesforce Blog](https://blog.salesforceairesearch.com/moirai/), [Moirai-MoE Paper](https://arxiv.org/abs/2410.10469), [Moirai-MoE Salesforce Blog](https://www.salesforce.com/blog/time-series-morai-moe/), [Moirai-MoE AI Horizon Forecast Blog](https://aihorizonforecast.substack.com/p/moirai-moe-upgrading-moirai-with), [Moirai-MoE Jiqizhixin Blog](https://mp.weixin.qq.com/s/LQvlgxx9vU965Yzy6RuBfQ).

## 🎉 What's New

* Nov 2024: The first general time series forecasting benchmark [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) is released. Moirai-Large achieves the best performance in the [Leaderboard](https://huggingface.co/spaces/Salesforce/GIFT-Eval)!

* Oct 2024: A new model Moirai-MoE! The preprint is available on [arXiv](https://arxiv.org/abs/2410.10469), along with the model weights of [Moirai-MoE-Small](https://huggingface.co/Salesforce/moirai-moe-1.0-R-small) and [Moirai-MoE-Base](https://huggingface.co/Salesforce/moirai-moe-1.0-R-base). Getting started with [inference code](https://github.com/SalesforceAIResearch/uni2ts/tree/main/project/moirai-moe-1) and [notebook examples](https://github.com/SalesforceAIResearch/uni2ts/tree/main/example)!

* Sep 2024: Released [Evaluation Code](https://github.com/SalesforceAIResearch/uni2ts/tree/main/project/benchmarks) of [TimesFM](https://arxiv.org/abs/2310.10688), [Chronos](https://arxiv.org/abs/2403.07815) and [VisionTS](https://arxiv.org/abs/2408.17253) on Monash, LSF and PF benchmarks.

* Jun 2024: Released Moirai-1.1-R model weights in [small](https://huggingface.co/Salesforce/moirai-1.1-R-small), [base](https://huggingface.co/Salesforce/moirai-1.1-R-base), and [large](https://huggingface.co/Salesforce/moirai-1.1-R-large).

* May 2024: The [Moirai Paper](https://arxiv.org/abs/2402.02592) has been accepted to ICML 2024 as an Oral presentation!

* Mar 2024: Release of Uni2TS library, along with [Moirai Paper](https://arxiv.org/abs/2402.02592), [Moirai-1.0-R Models](https://huggingface.co/collections/Salesforce/moirai-10-r-models-65c8d3a94c51428c300e0742), and [LOTSA Data](https://huggingface.co/datasets/Salesforce/lotsa_data/).

## ⚙️ Installation

1. Clone repository:
```shell
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts
```

2) Create virtual environment:
```shell
virtualenv venv
. venv/bin/activate
```

3) Build from source:
```shell
pip install -e '.[notebook]'
```

4) Create a `.env` file:
```shell
touch .env
```

We also support installation via PyPI.
```shell
pip install uni2ts
```

## 🏃 Getting Started

Let's see a simple example on how to use Uni2TS to make zero-shot forecasts from a pre-trained model. 
We first load our data using pandas, in the form of a wide DataFrame. 
Uni2TS relies on GluonTS for inference as it provides many convenience functions for time series forecasting, such as splitting a dataset into a train/test split and performing rolling evaluations, as demonstrated below.

```python
import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

MODEL = "moirai-moe"  # model name: choose from {'moirai', 'moirai-moe'}
SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 20  # prediction length: any positive integer
CTX = 200  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer

# Read data into pandas DataFrame
url = (
    "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
    "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
)
df = pd.read_csv(url, index_col=0, parse_dates=True)

# Convert into GluonTS dataset
ds = PandasDataset(dict(df))

# Split into train/test set
train, test_template = split(
    ds, offset=-TEST
)  # assign last TEST time steps as test set

# Construct rolling window evaluation
test_data = test_template.generate_instances(
    prediction_length=PDT,  # number of time steps for each prediction
    windows=TEST // PDT,  # number of windows in rolling window evaluation
    distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
)

# Prepare pre-trained model by downloading model weights from huggingface hub
if MODEL == "moirai":
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=PSZ,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
elif MODEL == "moirai-moe":
    model = MoiraiMoEForecast(
        module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=16,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )

predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)

inp = next(input_it)
label = next(label_it)
forecast = next(forecast_it)

plot_single(
    inp, 
    label, 
    forecast, 
    context_length=200,
    name="pred",
    show_label=True,
)
plt.show()
```

## 📔 Jupyter Notebook Examples
See the [example folder](example) for more examples on common tasks, e.g. visualizing forecasts, predicting from pandas DataFrame, etc.

## 💻 Command Line Interface
We provide several scripts which act as a [command line interface](cli) to easily run fine-tuning, evaluation, and even pre-training jobs. 
[Configurations](cli/conf) are managed with the [Hydra](https://hydra.cc/) framework.

### Fine-tuning
Firstly, let's see how to use Uni2TS to fine-tune a pre-trained model on your custom dataset. 
Uni2TS uses the [Hugging Face datasets library](https://github.com/huggingface/datasets) to handle data loading, and we first need to convert your dataset into the Uni2TS format. 
If your dataset is a simple pandas DataFrame, we can easily process your dataset with the following script.
We'll use the ETTh1 dataset from the popular [Long Sequence Forecasting benchmark](https://github.com/thuml/Time-Series-Library) for this example.
For more complex use cases, see [this notebook](example/prepare_data.ipynb) for more in-depth examples on how to use your custom dataset with Uni2TS.

1. To begin the process, add the path to the directory where you want to save the processed dataset into the ```.env``` file.
```shell
echo "CUSTOM_DATA_PATH=PATH_TO_SAVE" >> .env
```

2. Run the following script to process the dataset into the required format. For the ```dataset_type``` option, we support `wide`, `long` and `wide_multivariate`.
```shell
python -m uni2ts.data.builder.simple ETTh1 dataset/ETT-small/ETTh1.csv --dataset_type wide
```

However, we may want validation set during fine-tuning to perform hyperparameter tuning or early stopping.
To additionally split the dataset into a train and validation split we can use the mutually exclusive ```date_offset``` (datetime string) or ```offset``` (integer) options which determines the last time step of the train set.
The validation set will be saved as DATASET_NAME_eval.
```shell
python -m uni2ts.data.builder.simple ETTh1 dataset/ETT-small/ETTh1.csv --date_offset '2017-10-23 23:00:00'
```

3. Finally, we can simply run the fine-tuning script with the appropriate [training](cli/conf/finetune/data/etth1.yaml) and [validation](cli/conf/finetune/val_data/etth1.yaml) data configuration files.
```shell
python -m cli.train \
  -cp conf/finetune \
  run_name=example_run \ 
  model=moirai_1.0_R_small \ 
  data=etth1 \ 
  val_data=etth1  
```

### Evaluation

The evaluation script can be used to calculate evaluation metrics such as MSE, MASE, CRPS, and so on (see the [configuration file](cli/conf/eval/default.yaml)). 

Given a test split (see previous section on processing datasets), we can run the following command to evaluate it:
```shell
python -m cli.eval \ 
  run_name=example_eval_1 \
  model=moirai_1.0_R_small \
  model.patch_size=32 \ 
  model.context_length=1000 \
  data=etth1_test
```

Alternatively, we provide access to popular datasets, and can be toggled via the [data configurations](cli/conf/eval/data).
As an example, say we want to perform evaluation, again on the ETTh1 dataset from the popular [Long Sequence Forecasting benchmark](https://github.com/thuml/Time-Series-Library).
We first need to download the pre-processed datasets and put them in the correct directory, by setting up the TSLib repository and following the instructions.
Then, assign the dataset directory to the `LSF_PATH` environment variable:
```shell
echo "LSF_PATH=PATH_TO_TSLIB/dataset" >> .env
```

Thereafter, simply run the following script with the predefined [Hydra config file](cli/conf/eval/data/lsf_test.yaml):
```shell
python -m cli.eval \ 
  run_name=example_eval_2 \
  model=moirai_1.0_R_small \
  model.patch_size=32 \ 
  model.context_length=1000 \ 
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96 
```

### Pre-training
Now, let's see how you can pre-train your own model. 
We'll start with preparing the data for pre-training first, by downloading the [Large-scale Open Time Series Archive (LOTSA data)](https://huggingface.co/datasets/Salesforce/lotsa_data/).
Assuming you've already createed a `.env` file, run the following commands.
```shell
huggingface-cli download Salesforce/lotsa_data --repo-type=dataset --local-dir PATH_TO_SAVE
echo "LOTSA_V1_PATH=PATH_TO_SAVE" >> .env
```

Then, we can simply run the following script to start a pre-training job. 
See the [relevant](cli/train.py) [files](cli/conf/pretrain) on how to further customize the settings.
```shell
python -m cli.train \
  -cp conf/pretrain \
  run_name=first_run \
  model=moirai_small \
  data=lotsa_v1_unweighted
```

## 👀 Citation

If you're using this repository in your research or applications, please cite using the following BibTeX:

```markdown
@article{liu2024moiraimoe,
  title={Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts},
  author={Liu, Xu and Liu, Juncheng and Woo, Gerald and Aksu, Taha and Liang, Yuxuan and Zimmermann, Roger and Liu, Chenghao and Savarese, Silvio and Xiong, Caiming and Sahoo, Doyen},
  journal={arXiv preprint arXiv:2410.10469},
  year={2024}
}

@article{aksu2024gifteval,
  title={GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation},
  author={Aksu, Taha and Woo, Gerald and Liu, Juncheng and Liu, Xu and Liu, Chenghao and Savarese, Silvio and Xiong, Caiming and Sahoo, Doyen},
  journal={arXiv preprint arXiv:2410.10393},
  year={2024}
}

@inproceedings{woo2024moirai,
  title={Unified Training of Universal Time Series Forecasting Transformers},
  author={Woo, Gerald and Liu, Chenghao and Kumar, Akshat and Xiong, Caiming and Savarese, Silvio and Sahoo, Doyen},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```
