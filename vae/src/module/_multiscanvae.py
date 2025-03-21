from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.distributions import Categorical, Normal
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F

from scvi import REGISTRY_KEYS
from scvi.data import _constants
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import Decoder, Encoder

from scvi.module._classifier import Classifier
from scvi.module._utils import broadcast_labels
from scvi.module._vae import VAE

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    from torch.distributions import Distribution

    from scvi.model.base import BaseModelClass