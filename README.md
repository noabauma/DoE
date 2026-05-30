# DoE
Design of Experiment software made by Engineers of Engineers

A small toolkit for Design of Experiment built on **GPyTorch multitask GP
regression**: model several correlated responses jointly across multiple input
factors, then let Bayesian optimization choose the next experiment to run.

## The `doe` module
| Module | What it provides |
|--------|------------------|
| `doe.MultitaskGP` | Multitask GP regression: ARD-RBF kernel over factors ⊗ learned task covariance. `fit`, `predict`, `posterior`, plus `lengthscales` and `task_covariance`. |
| `doe.BayesianOptimizer` | Closed-loop optimization of a target response using Expected Improvement. |
| `doe.expected_improvement` | The EI acquisition function (closed form). |
| `doe.plotting` | `plot_1d`, `plot_parity`, `plot_convergence`. |

```python
from doe import MultitaskGP, BayesianOptimizer

gp = MultitaskGP(train_x, train_y).fit()          # train_x: (n, factors), train_y: (n, tasks)
mean, lower, upper = gp.predict(test_x)

opt = BayesianOptimizer(objective, num_factors=3, num_tasks=2, target_task=0)
result = opt.run(num_init=6, num_iters=20)        # result["best_x"], result["history"], ...
```

## Demos
| Script | Shows |
|--------|-------|
| `multitask_gp_example.py` | Single factor, two correlated responses — basic fit/predict/plot. |
| `multitask_gp_multifactor.py` | Three factors — parity plot, ARD factor importance, task correlation. |
| `bayesian_optimization.py` | EI loop that proposes the next experiment and converges to the optimum. |

## Setup
```bash
python3 -m venv .venv
# bootstrap pip if your system venv lacks ensurepip:
#   curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install gpytorch matplotlib
```

## Run
```bash
.venv/bin/python multitask_gp_example.py
.venv/bin/python multitask_gp_multifactor.py
.venv/bin/python bayesian_optimization.py
```
