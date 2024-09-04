# Benchmark
This directory contains the code and scripts for benchmarking. 


## Chronos
`run_chronos.py` is the code to run Chronos on a given dataset. 

`chronos_scripts` contains the scripts to run Chronos on different datasets.

### Examples
On Monash dataset:
```
sh chronos_scripts/monash_chronos_base.sh
```

On datasets for Probabilistic forecasting:
```
sh chronos_scripts/pf_chronos_base.sh
```


## TimesFM
`run_timesfm.py` is the code to run TimesFM on a given dataset.

`timesfm_scripts` contains the scripts to run TimesFM on different datasets.

### Examples
On Monash dataset:
```
sh monash_timesfm.sh 
```

On datasets for LSF: 
```
sh lsf_timesfm.sh
```