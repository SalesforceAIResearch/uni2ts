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


## VisionTS

- Paper: https://arxiv.org/abs/2408.17253
- Repo: https://github.com/Keytoyze/VisionTS

Before testing, you should first install VisionTS:

```bash
pip install visionts
```

`run_vision.py` is the code to run VisionTS on a given dataset.

`visionts_scripts` contains the scripts to run VisionTS on different datasets.

### Examples
On Monash dataset:
```bash
bash visionts_scripts/monash_visionts.sh 
```

On datasets for LSF: 
```bash
bash visionts_scripts/lsf_visionts.sh
```

On datasets for PF: 
```bash
bash visionts_scripts/pf_visionts.sh
```