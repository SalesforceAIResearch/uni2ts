#!/bin/bash

# build and launch python-sandbox docker

bash src/tools/python_sandbox/start_background.sh


# launch time series forecasting tools
CUDA_VISIBLE_DEVICES=7 python3 -m src.tools.tsf_services --model_name chronos_v2 &
# CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -m src.tools.tsf_services --model_name moirai &
