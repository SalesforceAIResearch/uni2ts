for ds in m1_monthly monash_m3_monthly monash_m3_other m4_monthly m4_weekly m4_daily m4_hourly tourism_quarterly tourism_monthly cif_2016_6 cif_2016_12 australian_electricity_demand bitcoin pedestrian_counts vehicle_trips_without_missing kdd_cup_2018_without_missing weather nn5_daily_without_missing nn5_weekly car_parts_without_missing fred_md traffic_hourly traffic_weekly rideshare_without_missing hospital covid_deaths temperature_rain_without_missing sunspot_without_missing saugeenday us_births
do
  python run_visionts.py --dataset=${ds}  --run_name=visionts --save_dir=monash_results --test_setting=monash --model=visionts --periodicity=autotune --context_len=1000 --no_periodicity_context_len=300 --batch_size 512
done
