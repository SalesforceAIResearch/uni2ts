# Finetuning of Moirai-1.0-R on LSF tasks

This page provides details on fine-tuning the Moirai-1.0-R series models. As examples, we provide the scripts to finetune `moirai-1.0-small` on ETTh and ETTm datasets, following the multivariate and univariate settings, respectively. We summarize the differences between finetuning and pretraining, as well as other key points, as follows:

## Dataset creation

Unlike pretraining, where samples of random lengths are randomly cropped from time series, we apply sliding windows to create offline datasets composed of fixed-length time series samples. The configurations of dataset creation can be found and revised in: `project/moirai-1/finetune_lsf/build_lsf_ft_datasets.sh`  and `src/uni2ts/data/builder/simple.py`.


* Dataset split follows the setup in [`src/uni2ts/eval_util/_lsf_dataset.py`](https://github.com/SalesforceAIResearch/uni2ts/blob/main/src/uni2ts/eval_util/_lsf_dataset.py). The same configurations are used in the corresponding files under `cli/conf/finetune/data` and `cli/conf/finetune/val_data`.
* The LSF setup requires normalizing the data using training statistics.
* We found that the number of training samples plays a vital role, which is determined by the distance of sliding windows. We use distance=1 by default. For large datasets, one can set distance to a larger value to reduce the computational cost per epoch.
* Dataset creation takes into account the choice between multivariate and univariate setups. Set `dataset_type` as `"wide_multivariate"` for multivariate and `dataset_type` as `"wide"` for univariate.


## Training 

* Patch size and context length are specified by users. We directly use the tuned values from the zero-shot evaluation setup.
* Set `data.mode` to `M` for multivariate and to `S` for univariate. Note that this needs to align with `dataset_type` in dataset creation.
* Sequence packing is not used during finetuning, as all samples have identical shapes. Thus, `sample_id` of each sample is added by transformation.
* We found that using a small learning rate (e.g. 5e-7) is suitable. 
* Unlike pertaining using `num_batches_per_epoch` in train_dataloader, here each epoch loops over all training samples.
* By default, the model updates all the parameters (finetune_pattern=full). One can set `finetune_pattern` to 'head_only' for linear probing. In this case, a larger learning rate is preferred (e.g. 1e-4 or 1e-3). We also allows to freeze FFN in the model by setting `finetune_pattern` to 'freeze_ffn'.
* A constant lr_optimizer is used for simplicity.

## Evaluation

* After finetuning, users need to add the relative checkpoint paths (starting with `.outputs/...`) to the corresponding eval shell scripts.
* The evaluation process remains identical to the original setup.

