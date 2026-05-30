"""Design-of-Experiment toolkit: multitask GP regression + Bayesian optimization.

Quick start
-----------
    from doe import MultitaskGP, BayesianOptimizer

    gp = MultitaskGP(train_x, train_y).fit()
    mean, lower, upper = gp.predict(test_x)

    opt = BayesianOptimizer(objective, num_factors=3, num_tasks=2)
    result = opt.run(num_init=6, num_iters=20)
"""

from .model import MultitaskGP
from .acquisition import expected_improvement
from .optimize import BayesianOptimizer
from . import plotting

__all__ = [
    "MultitaskGP",
    "expected_improvement",
    "BayesianOptimizer",
    "plotting",
]
