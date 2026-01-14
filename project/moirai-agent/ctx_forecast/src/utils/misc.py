import logging
import os


def get_logger(save_path, log_file="log.txt"):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if os.path.exists(os.path.join(save_path, log_file)):
        os.remove(os.path.join(save_path, log_file))
    file_handler = logging.FileHandler(os.path.join(save_path, log_file))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


import numpy as np


def compute_nrmae(
    prediction: list[float], future_values: list[float], history_values: list[float]
):
    # get valid values
    history_values = np.array(history_values)
    valid_mask_history = ~np.isnan(history_values)
    future_values = np.array(future_values)
    valid_mask_future = ~np.isnan(future_values)

    if valid_mask_future.sum() == 0 or valid_mask_history.sum() == 0:
        return 0

    prediction = np.array(prediction)
    diff = prediction[valid_mask_future] - future_values[valid_mask_future]
    nrmae = np.mean(np.abs(diff)) / np.clip(
        np.mean(np.abs(history_values[valid_mask_history])), a_min=1, a_max=np.inf
    )
    if np.isnan(nrmae) or nrmae > 100:
        return 100
    else:
        return nrmae
