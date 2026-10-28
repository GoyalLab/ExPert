# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

import logging
from rich.logging import RichHandler
import warnings

# Ignore depreciation warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Setup package logger
logging.basicConfig(
    level='INFO',
    format='%(filename)s | %(message)s',
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, show_time=True)]
)

log = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = ["ExPert"]


def __getattr__(name):
    # Lazy: importing ExPert eagerly drags in the full torch/scvi/lightning/scanpy
    # stack, which cold-imports for minutes off the network-mounted conda env. Defer
    # it so lightweight entrypoints that only touch expert.* subpackages (e.g. config
    # generation via expert.utils.io.sample_configs) don't pay that cost. PEP 562:
    # `from expert import ExPert` and `expert.ExPert` still work, importing on first use.
    if name == "ExPert":
        from expert.models._expert import ExPert
        return ExPert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

