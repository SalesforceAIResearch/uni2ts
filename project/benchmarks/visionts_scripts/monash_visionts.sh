#!/bin/bash

DATASET=visionts_scripts/monash_ds.txt

while read -r ds P; do
  if [ "$P" -eq 1 ]; then
    context_len=300
  else
    context_len=1000
  fi
  python run_visionts.py --dataset=${ds}  --run_name=visionts --save_dir=monash_results --test_setting=monash --model=visionts --periodicity=${P} --context_len=${context_len} --batch_size 512
done < ${DATASET}