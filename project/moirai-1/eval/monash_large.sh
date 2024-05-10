for ds in us_births saugeenday sunspot_with_missing temperature_rain_with_missing covid_deaths hospital rideshare_with_missing traffic_weekly traffic_hourly fred_md car_parts_with_missing electricity_weekly electricity_hourly solar_weekly solar_10_minutes nn5_weekly nn5_daily_with_missing weather kdd_cup_2018_with_missing vehicle_trips_with_missing pedestrian_counts bitcoin_with_missing dominick australian_electricity_demand cif_2016_12 cif_2016_6 tourism_monthly tourism_quarterly m4_hourly m4_daily m4_weekly m4_monthly monash_m3_other monash_m3_monthly m1_monthly; do
  python -m cli.eval \
    run_name=monash_eval \
    model=moirai_1.0_R_large \
    model.patch_size=32 \
    model.context_length=1000 \
    data=monash \
    data.dataset_name=$ds
done;