for ds in ETTh1 ETTh2 ETTm1 ETTm2 electricity weather;
do
  for pl in 96 192 332 720;
  do
    python run_timesfm.py --dataset=${ds}  --run_name=timesfm --save_dir=lsf_results --test_setting=lsf --model=timesfm --pred_length=${pl} --batch_size=512
  done
done