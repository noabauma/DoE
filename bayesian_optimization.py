"""Demo: Bayesian optimization with a multitask GP using the ``doe`` module.

Goal: maximize response 1 over 3 input factors. Each experiment also measures a
correlated response 2, which the multitask GP exploits for free.
"""

import torch

from doe import BayesianOptimizer
from doe import plotting


torch.manual_seed(0)

NUM_FACTORS = 3
TRUE_OPT_X = torch.tensor([0.7, 0.3, 0.5])
NOISE = 0.02


def objective(x):
    """Measure both responses (with noise). Response 1 peaks at TRUE_OPT_X."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    f0, f1, f2 = x[:, 0], x[:, 1], x[:, 2]
    y0 = -((f0 - 0.7) ** 2 + (f1 - 0.3) ** 2 + 0.2 * (f2 - 0.5) ** 2)
    y1 = 0.8 * y0 + 0.2 * f0
    y = torch.stack([y0, y1], dim=-1)
    return y + NOISE * torch.randn_like(y)


opt = BayesianOptimizer(objective, num_factors=NUM_FACTORS, num_tasks=2,
                        target_task=0, maximize=True)
result = opt.run(num_init=6, num_iters=20)

true_max = objective(TRUE_OPT_X)[:, 0].item()  # ~0 (noisy peek at the peak)
print("\n--- Result ---")
print(f"True optimum near x={TRUE_OPT_X.numpy()} (response 1 ≈ 0)")
print(f"Found best at x={result['best_x'].numpy().round(3)} "
      f"with response 1 = {result['best_y']:.4f}")

plotting.plot_convergence(
    result["history"], true_optimum=0.0,
    path="bayesian_optimization.png",
    title="Bayesian optimization with a multitask GP",
)
