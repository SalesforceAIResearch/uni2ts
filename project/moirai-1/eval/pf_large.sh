python -m cli.eval \
  run_name=pf_eval \
  model=moirai_1.0_R_large \
  model.patch_size=64 \
  model.context_length=1000 \
  data=gluonts_test \
  data.dataset_name=electricity

python -m cli.eval \
  run_name=pf_eval \
  model=moirai_1.0_R_large \
  model.patch_size=32 \
  model.context_length=5000 \
  data=gluonts_test \
  data.dataset_name=solar-energy

python -m cli.eval \
  run_name=pf_eval \
  model=moirai_1.0_R_large \
  model.patch_size=32 \
  model.context_length=5000 \
  data=gluonts_test \
  data.dataset_name=walmart

python -m cli.eval \
  run_name=pf_eval \
  model=moirai_1.0_R_large \
  model.patch_size=64 \
  model.context_length=5000 \
  data=gluonts_test \
  data.dataset_name=jena_weather

python -m cli.eval \
  run_name=pf_eval \
  model=moirai_1.0_R_large \
  model.patch_size=64 \
  model.context_length=5000 \
  data=gluonts_test \
  data.dataset_name=istanbul_traffic

python -m cli.eval \
  run_name=pf_eval \
  model=moirai_1.0_R_large \
  model.patch_size=64 \
  model.context_length=3000 \
  data=gluonts_test \
  data.dataset_name=turkey_power