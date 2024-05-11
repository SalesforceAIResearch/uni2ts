
export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

model=moirai_1.0_R_small

#python -m cli.finetune \
#  run_name=my_lsf_run \
#  model=$model \
#  data=etth1 \
#  val_data=etth1 & wait


#python -m cli.finetune \
#  run_name=my_lsf_run \
#  model=$model \
#  data=etth2 \
#  val_data=etth2 & wait


python -m cli.finetune \
  run_name=my_lsf_run \
  model=$model \
  data=ettm1 \
  val_data=ettm1 & wait


#python -m cli.finetune \
#  run_name=my_lsf_run \
#  model=$model \
#  data=ettm2 \
#  val_data=ettm2 & wait