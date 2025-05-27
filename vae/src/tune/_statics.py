from typing import NamedTuple

class _HP_KEYS(NamedTuple):
    MODEL_PARAMS_KEY: str = 'model_params'
    DATA_PARAMS_KEY: str = 'data_params'
    TRAIN_PARAMS_KEY: str = 'train_params'
    UNKNOWN_CAT_KEY: str = 'unknown'

HP_KEYS = _HP_KEYS()