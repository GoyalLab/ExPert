import yaml
import logging

import torch.nn as nn

from src.tune._statics import CONF_KEYS, NESTED_CONF_KEYS


# Recursive function to replace "nn.*" strings with actual torch.nn classes
def replace_nn_modules(d):
    if isinstance(d, dict):
        return {k: replace_nn_modules(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_nn_modules(v) for v in d]
    elif isinstance(d, str) and d.startswith("nn."):
        attr = d.split("nn.")[-1]
        return getattr(nn, attr)
    elif isinstance(d, str) and hasattr(nn, d):
        return getattr(nn, d)
    else:
        return d
    
def setup_config(config: dict) -> None:
    # Add schedule params to plan
    config[CONF_KEYS.PLAN][NESTED_CONF_KEYS.SCHEDULES_KEY] = config[CONF_KEYS.SCHEDULES]
    # Add plan to train
    config[CONF_KEYS.TRAIN][NESTED_CONF_KEYS.PLAN_KEY] = config[CONF_KEYS.PLAN]
    # Add encoder and decoder args to model
    config[CONF_KEYS.MODEL][NESTED_CONF_KEYS.ENCODER_KEY] = config[CONF_KEYS.ENCODER]
    config[CONF_KEYS.MODEL][NESTED_CONF_KEYS.DECODER_KEY] = config[CONF_KEYS.DECODER]
    # Add classifier args to model
    config[CONF_KEYS.MODEL][NESTED_CONF_KEYS.CLS_KEY] = config[CONF_KEYS.CLS]

def read_config(config_p: str, setup: bool = True) -> dict:
    """Read hyperparameter yaml file"""
    logging.info(f'Loading config file: {config_p}')
    with open(config_p, 'r') as f:
        config: dict = yaml.safe_load(f)
    # Check for config keys
    expected_keys = set(CONF_KEYS._asdict().values())
    assert expected_keys.issubset(config.keys()), f"Missing keys: {expected_keys - set(config.keys())}"
    # Convert nn modules to actual classes
    config = replace_nn_modules(config)
    if setup:
        # Set up nested structure
        setup_config(config=config)
    return config
