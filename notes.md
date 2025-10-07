python -m cli.train \
  -cp conf/finetune \
  run_name=example_run \
  model=moirai_1.0_R_small \
  data=etth1 \
  val_data=etth1




python -m cli.train \
  -cp conf/finetune \
  run_name=optimized_run \
  model=moirai_1.0_R_small \
  data=etth1 \
  val_data=etth1 \
  tf32=false \
  trainer.precision=16-mixed \
  compile=default \
  trainer.accumulate_grad_batches=2 \
  train_dataloader.num_workers=4



tensorboard --logdir outputs/finetune/moirai_1.0_R_small/etth1/optimized_run/logs/



python -m cli.train \
  -cp conf/finetune \
  run_name=memory_efficient \
  model=moirai_1.0_R_small \
  data=etth1 \
  val_data=etth1 \
  tf32=false \
  trainer.precision=16-mixed \
  compile=default \
  train_dataloader.batch_size=64 \
  val_dataloader.batch_size=64 \
  train_dataloader.num_workers=4 \
  val_dataloader.num_workers=4 \
  val_data._args_.prediction_lengths=[96] \
  val_data._args_.context_lengths=[1000] \
  val_data._args_.patch_sizes=[32]



# After Agent

python -m cli.train \
  -cp conf/finetune \
  run_name=test_single_epoch \
  model=moirai_1.0_R_small \
  data=financial_btc_2015_2020 \
  tf32=false \
  trainer.precision=16-mixed \
  compile=default \
  trainer.accumulate_grad_batches=2 \
  train_dataloader.num_workers=4 \
  trainer.max_epochs=1


## Validation 

python -m cli.train \
  -cp conf/finetune \
  run_name=test_single_epoch \
  model=moirai_1.0_R_small \
  data=financial_btc_2015_2020 \
  val_data=financial_btc_2015_2020 \
  tf32=false \
  trainer.precision=16-mixed \
  compile=default \
  trainer.accumulate_grad_batches=2 \
  train_dataloader.num_workers=4 \
  trainer.max_epochs=1




python -m cli.train \
  -cp conf/finetune \
  run_name=test_single_epoch \
  model=moirai_1.0_R_small \
  data=financial_btc_2015_2020 \
  val_data=financial_btc_2015_2020 \
  tf32=false \
  trainer.precision=16-mixed \
  compile=default \
  trainer.accumulate_grad_batches=2 \
  train_dataloader.num_workers=4 \
  trainer.max_epochs=1
