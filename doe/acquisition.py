"""Acquisition functions for Bayesian optimization."""

import torch


def expected_improvement(mean, std, best_f, xi=0.01, maximize=True):
    """Closed-form Expected Improvement for a single (target) response.

    Parameters
    ----------
    mean, std : posterior mean / std of the target response at the candidates.
    best_f    : best target response observed so far.
    xi        : small margin nudging toward exploration.
    maximize  : True to maximize the response, False to minimize.

    Returns the EI of each candidate (higher = more promising to try next).
    """
    std = std.clamp_min(1e-12)
    normal = torch.distributions.Normal(0.0, 1.0)
    improvement = (mean - best_f - xi) if maximize else (best_f - mean - xi)
    z = improvement / std
    ei = improvement * normal.cdf(z) + std * torch.exp(normal.log_prob(z))
    ei = ei.clone()
    ei[std <= 1e-9] = 0.0  # no uncertainty => no expected improvement
    return ei
