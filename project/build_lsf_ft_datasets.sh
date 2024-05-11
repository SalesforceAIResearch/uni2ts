
ds_type="wide"  # "wide_multivariate"

for data in ETTh1 ETTh2; do
  python -m uni2ts.data.builder.simple \
    $data \
    "/home/eee/qzz/datasets/TSLib/long_term_forecast/ETT-small/${data}.csv" \
    --dataset_type $ds_type\
    --offset 8640 \
    --normalize
done


#for data in ETTm1 ETTm2; do
#  python -m uni2ts.data.builder.simple \
#    $data \
#    "/home/eee/qzz/datasets/TSLib/long_term_forecast/ETT-small/${data}.csv" \
#    --dataset_type $ds_type\
#    --offset 34560 \
#    --normalize
#done
#
#
#python -m uni2ts.data.builder.simple \
#  weather \
#  "/home/eee/qzz/datasets/TSLib/long_term_forecast/weather/weather.csv" \
#  --dataset_type $ds_type\
#  --offset 36887 \
#  --normalize
#
#
#python -m uni2ts.data.builder.simple \
#  electricity \
#  "/home/eee/qzz/datasets/TSLib/long_term_forecast/electricity/electricity.csv" \
#  --dataset_type $ds_type\
#  --offset 18412 \
#  --normalize




## Original builder without normalize
#for data in ETTh1 ETTh2; do
#  python -m uni2ts.data.builder.simple \
#    $data \
#    "/home/eee/qzz/datasets/TSLib/long_term_forecast/ETT-small/${data}.csv" \
#    --dataset_type "wide_multivariate"\
#    --offset 8640
#done

