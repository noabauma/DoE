"""Demo: single-factor multitask GP regression using the ``doe`` module.

One input factor x in [0, 1] drives two correlated dummy responses. Shows the
basic fit/predict/plot workflow.
"""

import math

import torch

from doe import MultitaskGP
from doe import plotting


torch.manual_seed(0)

# Dummy data: 1 factor, 2 correlated responses (sine- and cosine-like).
train_x = torch.linspace(0, 1, 15)
noise = 0.05
train_y = torch.stack([
    torch.sin(train_x * 2 * math.pi) + noise * torch.randn(train_x.size()),
    torch.cos(train_x * 2 * math.pi) + noise * torch.randn(train_x.size()),
], dim=-1)

gp = MultitaskGP(train_x, train_y).fit(iters=60, verbose=True)

test_x = torch.linspace(0, 1, 100)
mean, lower, upper = gp.predict(test_x)

plotting.plot_1d(train_x, train_y, test_x, mean, lower, upper,
                 path="multitask_gp_example.png",
                 title="Multitask GP regression (dummy DoE data)")

print("\nLearned task covariance matrix:")
print(gp.task_covariance)
