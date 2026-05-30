"""Demo: multi-factor multitask GP regression using the ``doe`` module.

Three input factors drive two correlated responses. Fits the GP, scores it on a
held-out test set (parity plot), and reads off the ARD lengthscales -- which
reveal that the third factor barely matters.
"""

import torch

from doe import MultitaskGP
from doe import plotting


torch.manual_seed(0)

NUM_FACTORS = 3
NUM_TRAIN = 60
NUM_TEST = 40


def true_responses(x):
    """Hidden ground truth; f2 is deliberately almost irrelevant."""
    f0, f1, f2 = x[:, 0], x[:, 1], x[:, 2]
    base = torch.sin(2.5 * f0) + 0.5 * f1 ** 2
    y0 = base + 0.3 * f1
    y1 = 0.8 * base - 0.4 * f0 + 0.1 * f2
    return torch.stack([y0, y1], dim=-1)


train_x = torch.rand(NUM_TRAIN, NUM_FACTORS)
test_x = torch.rand(NUM_TEST, NUM_FACTORS)
train_y = true_responses(train_x) + 0.05 * torch.randn(NUM_TRAIN, 2)
test_y = true_responses(test_x)

gp = MultitaskGP(train_x, train_y).fit(iters=120, verbose=True)
mean, lower, upper = gp.predict(test_x)

rmse = plotting.plot_parity(
    test_y, mean, lower, upper,
    path="multitask_gp_multifactor.png",
    title=f"Multi-factor Multitask GP — {NUM_FACTORS} factors, 2 responses",
)

print("\nHeld-out RMSE per task:", [round(r.item(), 4) for r in rmse])
print("ARD lengthscales per factor (smaller = more influential):",
      [round(l.item(), 3) for l in gp.lengthscales])
print("Learned task covariance matrix:")
print(gp.task_covariance)
