from typing import NamedTuple

class _HP_KEYS(NamedTuple):
    MODEL_PARAMS_KEY: str = 'model_params'
    DATA_PARAMS_KEY: str = 'data_params'
    TRAIN_PARAMS_KEY: str = 'train_params'
    UNKNOWN_CAT_KEY: str = 'unknown'

class _ConfKeys(NamedTuple):
    DATA: str = 'data'
    CLS: str = 'cls'
    ENCODER: str = 'encoder'
    DECODER: str = 'decoder'
    SCHEDULES: str = 'schedules'
    PLAN: str = 'plan'
    TRAIN: str = 'train'
    MODEL: str = 'model'

CONF_KEYS = _ConfKeys()

HP_KEYS = _HP_KEYS()