## GIFT-CTX Replication
This repo contains instructions to replicate the results of Moirai Agent on the GIFT-CTX benchmark.

### Set up environment
First, set up the environment by running:
```
pip install -r requirement.txt
```
And export your OpenAI key by:
```
export OPENAI_API_KEY="..."
```

### Prepare the dataset
- Download the `gift_ctx.parquet` data here: `https://huggingface.co/datasets/Salesforce/GIFT-CTX`
- Plot the historical data before runtime:
```
python gen_image.py --in_file gift_ctx.parquet --out_file gift_ctx_image.parquet --img_root img
```
Change the parameters to your project setting:
```
--in_file: path to the original GIFT-CTX dataset
--out_file: path to the new dataset with image path added
--img_root: path to the image folder
```
Note that Moirai Agent can work fine with just text input. To maximize its performance, we recommend to also provide the plot of historical data, which can be rendered beforehand to reduce runtime.
### Run Moirai Agent
To run Moirai Agent, first start the tools with:
```
bash run_tools.sh
```
For now, we provide agent with 2 tools: a python sandbox, and a timeseries foundation model.

To evaluate Moirai Agent on the GIFT-CTX benchmark with the default setting and replicate the reported results, run:
```
bash run.sh
```
Note that the results reported in the blog post were obtained with GPT-5.1 medium reasoning effort on Jan 07 2026. Please expect some minor differences with the official reported numbers because of the non-deterministic nature of LLMs. We provided our log files in `./results`.

To change the input path file, config file, output dir, input mode (text only, text with image), and parallelism, run with arguments:
```
bash run.sh [your_parquet_path] [your_config_path] [output_dir] [input_mode] [#jobs]
```

There are several configurations you can change in Moirai Agent such as LLM and its parameters, tools, and system prompt, which you can adjust in `src/ctx_forecast/config.py`. 


