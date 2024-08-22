model_size=tiny
model_path=amazon/chronos-t5-${model_size}
for ds in electricity solar-energy walmart jena_weather istanbul_traffic turkey_power
do
  python run_chronos.py --model_path=${model_path} --dataset=${ds}  --run_name=chronos-${model_size} --save_dir=pf_results_20 --test_setting=pf --num_samples=20
done