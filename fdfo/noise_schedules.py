# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This source code is licensed under the NVIDIA Source Code License v1
# (Non-Commercial). The full license text is available in the LICENSE.txt
# file at the root of this repository.

"""
Utilities for customization of the sampling churn schedule. Main experiments
use the RawChurnScheduler, which just returns a specified gamma value.
Gradient weighting is included here but not in the training code for
simplicity.
"""

import torch
import math
from dataclasses import asdict
from fdfo.config import ChurnScheduleConfig, RawChurnScheduleConfig, PriorScheduleConfig, IntervalScheduleConfig

class Range:
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def logit(x):
    eps = 1e-7
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x / (1 - x))

def logit_normal(x, mu, sigma):
    eps = 1e-7
    x = torch.clamp(x, eps, 1.0 - eps)
    return 1 / (math.sqrt(2 * math.pi) * sigma * x * (1-x)) * torch.exp(-(logit(x) - mu)**2 / (2 * sigma**2))

def normalize(x):
    x = x / x.sum()
    return x

class Density:
    def pdf_discretized(self, ts, churn_scaling=False):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def sample_discretized(self, ts):
        pdf = self.pdf_discretized(ts)
        indices = torch.multinomial(pdf, 1)
        return ts[indices]

class LogitNormal(Density):
    def __init__(self, mu, sigma, weight=1.0):
        self.mu = mu
        self.sigma = sigma
        self.weight = weight

    def pdf(self, x):
        return logit_normal(x, self.mu, self.sigma)

    def pdf_discretized(self, ts, churn_scaling=False):
        ts = torch.as_tensor(ts, dtype=torch.float32)
        eps = 1e-6

        ts_clipped = torch.clamp(ts, eps, 1.0 - eps)
        pdf = logit_normal(ts_clipped, self.mu, self.sigma)
        pdf = torch.where(torch.isnan(pdf) | torch.isinf(pdf), torch.zeros_like(pdf), pdf)
        pdf = torch.where((ts == 0) | (ts == 1), torch.zeros_like(pdf), pdf)

        # Robust normalization
        pdf_sum = pdf.sum()
        if pdf_sum > eps:
            pdf = pdf / pdf_sum
        else:
            pdf = torch.ones_like(pdf) / len(pdf)

        pdf = pdf * self.weight
        if churn_scaling:
            pdf = torch.exp(pdf) - 1.0
        return pdf

    def sample(self):
        return sigmoid(torch.randn(()) * self.sigma + self.mu).item()


class Dirac(Density):
    def __init__(self, mu, weight=1.0):
        self.mu = mu
        self.weight = weight

    def pdf(self, x):
        return 0 * x # hack to keep the type and shape

    def pdf_discretized(self, ts, churn_scaling=False):
        ts = torch.as_tensor(ts, dtype=torch.float32)
        dists = torch.abs(ts - self.mu)
        min_idx = torch.argmin(dists)
        pdf = torch.zeros_like(ts)
        pdf[min_idx] = 1.0
        pdf = pdf * self.weight
        if churn_scaling:
            pdf = torch.exp(pdf) - 1.0
        return pdf

    def sample(self):
        return self.mu

class Scheduler:
    def __init__(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"

    def __str__(self):
        return self.__repr__()

    def sample_densities(self) -> tuple[Density, Density]:
        """Return two densities, one for the churn and one for the gradient application."""
        raise NotImplementedError

class IntervalScheduler(Scheduler):
    """Pick a random single time to churn around"""
    def __init__(self, mu: float = 1.0, sigma: float = 1.5, churn_weight: float = 1.0, churn_weight_sigma: float = 0.0, soft_sigma: float = 0.25):
        self.mu = mu
        self.sigma = sigma
        self.churn_weight = churn_weight
        self.churn_weight_sigma = churn_weight_sigma
        self.soft_sigma = soft_sigma

    def sample_densities(self):
        churn_mu = (torch.randn(()) * self.sigma + self.mu).item()
        churn_weight = torch.exp(torch.randn(()) * self.churn_weight_sigma + math.log(self.churn_weight)).item()
        return LogitNormal(churn_mu, self.soft_sigma, weight=churn_weight), LogitNormal(churn_mu, self.soft_sigma)

class PriorScheduler(Scheduler):
    """Churn only at the maximum noise level, and apply gradient broadly at all times.
    Note, we still need to choose how the applications are distributed."""
    def __init__(self, grad_mu: float = 0.2, grad_sigma: float = 1.2, churn_weight: float = 1.0, churn_weight_sigma: float = 0.0):
        self.grad_mu = grad_mu
        self.grad_sigma = grad_sigma
        self.churn_weight = churn_weight
        self.churn_weight_sigma = churn_weight_sigma

    def sample_densities(self):
        churn_weight = torch.exp(torch.randn(()) * self.churn_weight_sigma + math.log(self.churn_weight)).item()
        return Dirac(1.0, weight=churn_weight), LogitNormal(self.grad_mu, self.grad_sigma)

class Raw(Density):
    def __init__(self, weight=1.0):
        self.weight = weight

    def pdf(self, x):
        return self.weight

    def pdf_discretized(self, ts, churn_scaling=False):
        return torch.ones_like(ts) * self.weight

    def sample(self):
        raise NotImplementedError

class RawChurnScheduler(Scheduler):
    """Churn with a uniform distribution without any normalization/correction."""
    def __init__(self, churn_weight: float = 0.03, grad_mu: float = 0.5, grad_sigma: float = 1.0):
        self.churn_weight = churn_weight
        self.grad_mu = grad_mu
        self.grad_sigma = grad_sigma

    def sample_densities(self):
        return Raw(weight=self.churn_weight), LogitNormal(self.grad_mu, self.grad_sigma)

def build_churn_scheduler(config: ChurnScheduleConfig):
    schedule_map = {
        IntervalScheduleConfig: IntervalScheduler,
        PriorScheduleConfig: PriorScheduler,
        RawChurnScheduleConfig: RawChurnScheduler,
    }
    cls = schedule_map[type(config)]
    return cls(**asdict(config))
