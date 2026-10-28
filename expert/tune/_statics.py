# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

from typing import NamedTuple

class _HP_KEYS(NamedTuple):
    MODEL_PARAMS_KEY: str = 'model_params'
    DATA_PARAMS_KEY: str = 'data_params'
    TRAIN_PARAMS_KEY: str = 'train_params'
    UNKNOWN_CAT_KEY: str = 'unknown'

class _ConfKeys(NamedTuple):
    DATA: str = 'data'
    CLS: str = 'cls'
    ALIGNER: str = 'aligner'
    ENCODER: str = 'encoder'
    DECODER: str = 'decoder'
    SCHEDULES: str = 'schedules'
    PLAN: str = 'plan'
    TRAIN: str = 'train'
    MODEL: str = 'model'
    MODEL_SETUP: str = 'model_setup'

class _NestedConfKeys(NamedTuple):
    SCHEDULES_KEY: str = 'anneal_schedules'
    PLAN_KEY: str = 'plan_kwargs'
    ENCODER_KEY: str = 'extra_encoder_kwargs'
    DECODER_KEY: str = 'extra_decoder_kwargs'
    CLS_KEY: str = 'classifier_parameters'
    ALIGN_KEY : str = 'extra_aligner_kwargs'

CONF_KEYS = _ConfKeys()
NESTED_CONF_KEYS = _NestedConfKeys()
HP_KEYS = _HP_KEYS()
