for pl in 96 192 336 720; do
  python run_visionts.py --dataset=ETTh1  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=2880 --pred_length=${pl} --batch_size 512
done

for pl in 96 192 336 720; do
  python run_visionts.py --dataset=ETTh2  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=1728 --pred_length=${pl} --batch_size 512
done

for pl in 96 192 336 720; do
  python run_visionts.py --dataset=ETTm1  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=2304 --pred_length=${pl} --batch_size 512
done

for pl in 96 192 336 720; do
  python run_visionts.py --dataset=ETTm2  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=4032 --pred_length=${pl} --batch_size 512
done

for pl in 96 192 336 720; do
  python run_visionts.py --dataset=electricity  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=2880 --pred_length=${pl} --batch_size 512
done

for pl in 96 192 336 720; do
  python run_visionts.py --dataset=weather  --run_name=visionts --save_dir=lsf_results --test_setting=lsf --model=visionts --periodicity=freq --context_len=4032 --pred_length=${pl} --batch_size 512
done
