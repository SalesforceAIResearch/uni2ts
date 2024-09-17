for ds in electricity solar-energy walmart jena_weather istanbul_traffic turkey_power
do
  python run_visionts.py --dataset=${ds}  --run_name=visionts --save_dir=pf_results --test_setting=pf --model=visionts --periodicity=freq --context_len=2000 --batch_size 512
done
