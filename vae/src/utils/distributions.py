from __future__ import annotations

from typing import TYPE_CHECKING

import warnings

from torch.distributions import Distribution, constraints
from torch.distributions import Normal as NormalTorch

import torch

from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from scvi import settings

def rescale_targets(x: torch.Tensor, scale: float = 4, min: float = -1.0, max: float = 1.0) -> torch.Tensor:
    """Rescale and clamp tensor values to specified range.
    
    Args:
        x: Input tensor
        scale: Scaling factor for standardized values
        min: Minimum allowed value 
        max: Maximum allowed value
    """
    return torch.clamp((x - x.mean()) * scale, min=min, max=max)

def log_mixture_normal(
    x: torch.Tensor,
    mu_1: torch.Tensor,
    mu_2: torch.Tensor,
    sigma_1: torch.Tensor,
    sigma_2: torch.Tensor,
    pi_logits: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a mixture normal model.

    pi_logits is the probability (logits) to be in the first component.

    Parameters
    ----------
    x
        Observed data
    mu_1
        Mean of the first normal component (shape: minibatch x features)
    mu_2
        Mean of the second normal component (shape: minibatch x features)
    sigma_1
        Standard deviation of the first normal component (shape: minibatch x features)
    sigma_2
        Standard deviation of the second normal component (shape: minibatch x features)
    pi_logits
        Probability of belonging to mixture component 1 (logits scale)
    eps
        Numerical stability constant
    log_fn
        log function
    """
    log = log_fn

    # Compute log probabilities for each normal component
    normal_1 = NormalTorch(mu_1, sigma_1)
    normal_2 = NormalTorch(mu_2, sigma_2 if sigma_2 is not None else sigma_1)

    log_prob_1 = normal_1.log_prob(x)
    log_prob_2 = normal_2.log_prob(x)

    # Compute the log mixture probabilities
    logsumexp = torch.logsumexp(torch.stack((log_prob_1, log_prob_2 - pi_logits)), dim=0)
    softplus_pi = torch.nn.functional.softplus(-pi_logits)

    log_mixture_normal_res = logsumexp - softplus_pi

    return log_mixture_normal_res



class NormalMixture(Distribution):
    """Normal mixture distribution.

    Parameters
    ----------
    mu1
        Mean of the component 1 distribution.
    mu2
        Mean of the component 2 distribution.
    sigma1
        Standard deviation for component 1.
    mixture_logits
        Logits scale probability of belonging to component 1.
    sigma2
        Standard deviation for component 2. If `None`, assumed to be equal to `sigma1`.
    validate_args
        Raise ValueError if arguments do not match constraints.
    """

    arg_constraints = {
        "mu1": constraints.real,
        "mu2": constraints.real,
        "sigma1": constraints.positive,
        "mixture_probs": constraints.half_open_interval(0.0, 1.0),
        "mixture_logits": constraints.real,
    }
    support = constraints.real

    def __init__(
        self,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        sigma1: torch.Tensor,
        mixture_logits: torch.Tensor,
        sigma2: torch.Tensor = None,
        validate_args: bool = False,
    ):
        (
            self.mu1,
            self.sigma1,
            self.mu2,
            self.mixture_logits,
        ) = broadcast_all(mu1, sigma1, mu2, mixture_logits)
        super().__init__(validate_args=validate_args)

        if sigma2 is not None:
            self.sigma2 = broadcast_all(mu1, sigma2)
        else:
            self.sigma2 = None

    @property
    def mean(self) -> torch.Tensor:
        pi = self.mixture_probs
        return pi * self.mu1 + (1 - pi) * self.mu2

    @lazy_property
    def mixture_probs(self) -> torch.Tensor:
        return logits_to_probs(self.mixture_logits, is_binary=True)

    @torch.inference_mode()
    def sample(
        self,
        sample_shape: torch.Size | tuple | None = None,
    ) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        pi = self.mixture_probs
        mixing_sample = torch.distributions.Bernoulli(pi).sample()
        mu = self.mu1 * mixing_sample + self.mu2 * (1 - mixing_sample)
        if self.sigma2 is None:
            sigma = self.sigma1
        else:
            sigma = self.sigma1 * mixing_sample + self.sigma2 * (1 - mixing_sample)
        normal_d = NormalTorch(mu, sigma)
        return normal_d.sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability."""
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn(
                "The value argument must be within the support of the distribution",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        return log_mixture_normal(
            value,
            self.mu1,
            self.mu2,
            self.sigma1,
            self.sigma2,
            self.mixture_logits,
        )

    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ", ".join(
            [
                f"{p}: "
                f"{self.__dict__[p] if self.__dict__[p].numel() == 1 else self.__dict__[p].size()}"
                for p in param_names
                if self.__dict__[p] is not None
            ]
        )
        return self.__class__.__name__ + "(" + args_string + ")"