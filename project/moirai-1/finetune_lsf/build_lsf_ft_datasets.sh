#!/bin/bash
set -a
source .env
set +a

path_prefix=$LSF_PATH

for data in ETTh1 ETTh2; do
  python -m uni2ts.data.builder.simple \
    $data \
    "${path_prefix}/ETT-small/${data}.csv" \
    --dataset_type "wide_multivariate" \
    --offset 8640 \
    --normalize
done

for data in ETTm1 ETTm2; do
  python -m uni2ts.data.builder.simple \
    $data \
    "${path_prefix}/ETT-small/${data}.csv" \
    --dataset_type "wide" \
    --offset 34560 \
    --normalize
done