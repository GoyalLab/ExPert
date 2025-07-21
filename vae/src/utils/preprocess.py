import torch
import numpy as np
import logging
from anndata import AnnData
from src.utils.constants import REGISTRY_KEYS


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))

def scale_1d_array(
        x: np.ndarray, 
        zero_center: bool = True, 
        max_value: float | None = None, 
        abs: bool = True,
        log: bool = True,
        use_sigmoid: bool = True,
        check_scale: bool = True,
    ) -> np.ndarray:
    x = x.astype(float)
    # Check if x is already bounded [0,1]
    if check_scale and x.min() >= 0 and x.max() <= 1:
        return x
    if abs:
        x = np.abs(x)
    if log:
        x = np.log10(x)
    if zero_center:
        mean = np.mean(x)
        x -= mean
    std = np.std(x)
    if std != 0:
        x /= std
    else:
        x[:] = 0.0  # Avoid division by zero
    if max_value is not None:
        x = np.clip(x, -max_value, max_value)
    if use_sigmoid:
        x = sigmoid(x)
    return x
