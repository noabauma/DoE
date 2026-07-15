# DoE
Design of Experiment software made by Engineers of Engineers

A small toolkit for Design of Experiment built on **GPyTorch multitask GP
regression**: model several correlated responses jointly across multiple input
factors, then let Bayesian optimization choose the next experiment to run.

## The `doe` module
| Module | What it provides |
|--------|------------------|
| `doe.MultitaskGP` | Multitask GP regression: ARD-RBF kernel over factors ⊗ learned task covariance. `fit`, `predict`, `posterior`, plus `lengthscales` and `task_covariance`. |
| `doe.BayesianOptimizer` | Closed-loop optimization of a target response using Expected Improvement (for callable objectives, e.g. simulations). |
| `doe.InteractiveOptimizer` | Human-in-the-loop DoE: proposes the next 1–3 concentration sets, you enter measured lab results, the GP adapts. Session save/load, best-so-far and predicted-optimum reports. |
| `doe.IngredientCosts` | Ingredient pricing: cost per experiment, total campaign spend, projected budget, and price of the yield (currency per response unit). |
| `doe.expected_improvement` | The EI acquisition function (closed form). |
| `doe.plotting` | `plot_1d`, `plot_parity`, `plot_convergence`. |

```python
from doe import MultitaskGP, BayesianOptimizer

gp = MultitaskGP(train_x, train_y).fit()          # train_x: (n, factors), train_y: (n, tasks)
mean, lower, upper = gp.predict(test_x)

opt = BayesianOptimizer(objective, num_factors=3, num_tasks=2, target_task=0)
result = opt.run(num_init=6, num_iters=20)        # result["best_x"], result["history"], ...
```

## Interactive DoE (human in the loop)

When the "objective" is a real lab experiment, use `InteractiveOptimizer`:
it proposes the next 1–3 concentration sets, you run them and type the
measured responses back in, and the model adapts before proposing again.
Ingredient prices let it estimate the cost of each proposed experiment, the
total campaign spend, and the price of the yield at the best settings.

### Web UI (recommended)

```bash
.venv/bin/python doe_server.py my_session.json        # http://localhost:8080
# from your laptop:  ssh -L 8080:localhost:8080 <user>@<lab machine>
```

The browser GUI covers the whole workflow: a setup wizard (ingredients with
concentration ranges and prices, responses, target), suggest 1–3 experiments,
enter measured results, and live interactive plots — optimization progress,
GP model slices with 95% bands, factor influence, response correlation, and
cost tracking with projections. Sessions auto-save after every result and the
history is exportable as CSV. The same session file works in the CLI below.

### Running the web UI as a systemd service

`doe.service` keeps the web UI running permanently (survives logouts and
reboots). It starts `doe_server.py my_session.json --host 0.0.0.0 --port 8080`
from this directory. Install and enable it once:

```bash
sudo cp doe.service /etc/systemd/system/doe.service
sudo systemctl daemon-reload
sudo systemctl enable --now doe.service   # start now and on every boot
```

Manage it with:

```bash
sudo systemctl start doe.service      # start
sudo systemctl stop doe.service       # stop
sudo systemctl restart doe.service    # restart (e.g. after editing the code)
systemctl status doe.service          # is it running?
journalctl -u doe.service -f          # follow the server logs
```

After changing `doe.service` itself, re-run the `cp` and `daemon-reload`
steps before restarting. Note that `--host 0.0.0.0` exposes the UI to the
whole network without authentication — change it to `127.0.0.1` in
`doe.service` if you only access it via `ssh -L` port forwarding.

### CLI

```bash
.venv/bin/python interactive_doe.py my_session.json   # resumes if it exists
```

Or from Python:

```python
from doe import InteractiveOptimizer, IngredientCosts

opt = InteractiveOptimizer(
    bounds=[[0, 5], [0, 2]], num_tasks=2, target_task=0,
    factor_names=["Li salt", "additive"],
    response_names=["capacity", "cycle life"],
    costs=IngredientCosts([1.2, 40.0], fixed_cost=15, currency="USD"),
    cost_aware=True,                    # prefer cheap experiments at equal EI
)
proposals = opt.suggest(3)              # next 1..3 concentration sets
opt.add_result(proposals[0], [155.2, 0.87])   # enter measured responses
opt.best()                              # best so far incl. cost per yield
opt.predicted_best()                    # model's predicted optimum
opt.spend(); opt.projected_spend(10)    # campaign cost so far / budgeted
opt.save("session.json")                # resume later with .load()
```

## Demos
| Script | Shows |
|--------|-------|
| `multitask_gp_example.py` | Single factor, two correlated responses — basic fit/predict/plot. |
| `multitask_gp_multifactor.py` | Three factors — parity plot, ARD factor importance, task correlation. |
| `bayesian_optimization.py` | EI loop that proposes the next experiment and converges to the optimum. |
| `interactive_doe.py` | Interactive terminal session: setup wizard, propose → measure → enter → adapt, with cost reports. |
| `doe_server.py` + `webui/` | Web GUI on `localhost:8080` for the whole interactive workflow, with live Plotly charts. |

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
