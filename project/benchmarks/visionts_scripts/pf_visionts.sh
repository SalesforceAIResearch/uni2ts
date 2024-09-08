#!/bin/bash

DATASET=visionts_scripts/pf_ds.txt

while read -r ds P L; do
  python run_visionts.py --dataset=${ds}  --run_name=visionts --save_dir=pf_results --test_setting=pf --model=visionts --periodicity=${P} --context_len=${L} --batch_size 512
done < ${DATASET}