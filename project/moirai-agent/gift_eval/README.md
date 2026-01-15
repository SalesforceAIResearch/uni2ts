## Gift-Eval replication

Evaluate the MoiraiAgent on the Gift-Eval dataset and write metrics to a leaderboard compatible CSV file.

### Install

```sh
pip install -r requirements.txt
```

Note: Requires **Python >= 3.12**.

### Configure dataset path
Download Gifteval dataset [](https://huggingface.co/datasets/Salesforce/GiftEval) into a local path and set the path accordingly in `.env` file. 
Create `.env` file and set the following environment variable to the local path for GiftEval dataset:

```sh
GIFT_EVAL=/absolute/path/to/gift_eval
```

### Run

```sh
python eval.py --out_dir results --out_name all_results.csv
```


Note: This script is meant for demo replication purposes it is not fully optimized. Replicating results on the whole gifteval may take a long time depending on the resources used. Parallelizing tokenization jobs and integrating vllm for inference should speed it up significantly if needed.
