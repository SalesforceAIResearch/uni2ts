#!/bin/bash

DATASET=visionts_scripts/lsf_ds.txt

while read -r ds P L; do
  for pl in 96 192 332 720;
  do
    python run_visionts.py --dataset=${ds}  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=${P} --context_len=${L} --pred_length=${pl} --batch_size 512
  done  
done < ${DATASET}