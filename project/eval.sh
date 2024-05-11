
# Reproduce; should be data.mode=M to align MTS setting in the paper?
export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

mode=S
run_name=my_lsf_eval
model=moirai_1.0_R_small

#for pl in 96 192 336 720; do
#  python -m cli.eval \
#    run_name=$run_name \
#    model=$model \
#    model.patch_size=64 \
#    model.context_length=5000 \
#    data=lsf_test \
#    data.dataset_name=ETTh1 \
#    data.mode=$mode \
#    data.prediction_length=$pl
#done

#for pl in 96 192 336 720; do
#  python -m cli.eval \
#    run_name=$run_name \
#    model=$model \
#    model.patch_size=64 \
#    model.context_length=3000 \
#    data=lsf_test \
#    data.dataset_name=ETTh2 \
#    data.mode=M \
#    data.prediction_length=$pl
#done


for pl in 96; do  # 192 336 720
  python -m cli.eval \
    run_name=$run_name \
    model=$model \
    model.patch_size=128 \
    model.context_length=4000 \
    data=lsf_test \
    data.dataset_name=ETTm1 \
    data.mode=S \
    data.prediction_length=$pl
done

#for pl in 96 192 336 720; do
#  python -m cli.eval \
#    run_name=$run_name \
#    model=$model \
#    model.patch_size=64 \
#    model.context_length=3000 \
#    data=lsf_test \
#    data.dataset_name=ETTm2 \
#    data.mode=S \
#    data.prediction_length=$pl
#done
#
#for pl in 96 192 336 720; do
#  python -m cli.eval \
#    run_name=$run_name \
#    model=$model \
#    model.patch_size=64 \
#    model.context_length=5000 \
#    data=lsf_test \
#    data.dataset_name=electricity \
#    data.mode=S \
#    data.prediction_length=$pl
#done
#
#for pl in 96 192 336 720; do
#  python -m cli.eval \
#    run_name=$run_name \
#    model=$model \
#    model.patch_size=128 \
#    model.context_length=2000 \
#    data=lsf_test \
#    data.dataset_name=weather \
#    data.mode=M \
#    data.prediction_length=$pl
#done


