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

class _NestedConfKeys(NamedTuple):
    SCHEDULES_KEY: str = 'anneal_schedules'
    PLAN_KEY: str = 'plan_kwargs'
    ENCODER_KEY: str = 'extra_encoder_kwargs'
    DECODER_KEY: str = 'extra_decoder_kwargs'
    CLS_KEY: str = 'classifier_parameters'

CONF_KEYS = _ConfKeys()
NESTED_CONF_KEYS = _NestedConfKeys()
HP_KEYS = _HP_KEYS()
