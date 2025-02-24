#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

cp=conf/eval
model=moirai_lightning_ckpt
data=ETTm2
cl=3000
ps=64
mode=S


cpp1=''
cpp2=''
cpp3=''
cpp4=''

index=1
for pl in 96 192 336 720; do
  case $index in
    1) cpp=$cpp1 ;;
    2) cpp=$cpp2 ;;
    3) cpp=$cpp3 ;;
    4) cpp=$cpp4 ;;
  esac

  exp_name=$(echo $cpp | cut -d'/' -f4)
  pretrained_model=$(echo $cpp | cut -d'/' -f5)
  ft_pattern=$(echo $cpp | cut -d'/' -f6)

  python -m cli.eval \
    -cp $cp \
    run_name="${exp_name}_${pretrained_model}_${ft_pattern}"  \
    model=$model \
    model.patch_size=$ps \
    model.context_length=$cl \
    model.checkpoint_path=$cpp \
    data=lsf_test \
    data.dataset_name=$data \
    data.mode=$mode \
    data.prediction_length=$pl

  index=$((index+1))
done
