"""Interactive (human-in-the-loop) Bayesian DoE.

The optimizer proposes the next 1-3 concentration sets, the user runs those
experiments in the lab, types the measured responses back in, and the GP
refits so the next proposals adapt to the new results:

    opt = InteractiveOptimizer(bounds=[[0, 5], [0, 2]], num_tasks=2,
                               factor_names=["Li salt", "additive"],
                               response_names=["capacity", "cycle life"],
                               costs=IngredientCosts([1.2, 40.0], fixed_cost=15))
    proposals = opt.suggest(3)          # -> (3, num_factors) concentrations
    opt.add_result(proposals[0], [155.2, 0.87])
    proposals = opt.suggest(2)          # adapts to the new result
    opt.save("session.json")            # resume later with .load()
"""

import json
import os

import torch

from .model import MultitaskGP
from .acquisition import expected_improvement
from .cost import IngredientCosts

# Keys save()/load() own; everything else in a session file is user metadata
# (description, pending proposals, ...) and round-trips via ``meta``.
_SESSION_KEYS = frozenset({
    "bounds", "num_tasks", "target_task", "maximize", "cost_aware",
    "num_init", "factor_names", "response_names", "train_x", "train_y",
    "costs", "kernel", "poly_degree",
})


class InteractiveOptimizer:
    """Human-in-the-loop Bayesian optimization over ingredient concentrations.

    Until ``num_init`` results exist the proposals are a space-filling design
    (stratified random); after that a multitask GP is fit to all results and
    Expected Improvement on the target response picks the candidates. Batches
    of up to 3 are diversified by locally penalizing EI around already-picked
    candidates so the user gets distinct concentration sets to run in parallel.

    Parameters
    ----------
    bounds         : (num_factors, 2) low/high concentration per ingredient.
    num_tasks      : number of responses measured per experiment.
    target_task    : index of the response to optimize.
    maximize       : True to maximize the target response, False to minimize.
    costs          : optional IngredientCosts for pricing experiments/yield.
    cost_aware     : if True (and costs given), rank candidates by EI per
                     currency unit instead of plain EI -- cheaper experiments
                     that promise the same improvement are preferred.
    num_init       : results required before switching to GP-guided proposals.
    factor_names   : optional ingredient names for reports.
    response_names : optional response names for reports.
    kernel         : how the posterior is fit -- "rbf" (default, flexible GP)
                     or "poly" (posterior mean is an n-degree polynomial).
    poly_degree    : the n for kernel="poly" (1..10).
    """

    def __init__(self, bounds, num_tasks, target_task=0, maximize=True,
                 costs=None, cost_aware=False, num_init=4,
                 factor_names=None, response_names=None,
                 kernel="rbf", poly_degree=2):
        self.bounds = torch.as_tensor(bounds, dtype=torch.float32).reshape(-1, 2)
        self.num_factors = self.bounds.shape[0]
        self.num_tasks = num_tasks
        self.target_task = target_task
        self.maximize = maximize
        self.costs = costs
        self.cost_aware = cost_aware
        self.num_init = num_init
        self.factor_names = list(factor_names) if factor_names else [
            f"factor {i + 1}" for i in range(self.num_factors)
        ]
        self.response_names = list(response_names) if response_names else [
            f"response {i + 1}" for i in range(num_tasks)
        ]
        self.train_x = torch.empty(0, self.num_factors)
        self.train_y = torch.empty(0, num_tasks)
        self.gp = None
        self.meta = {}  # extra session-file keys (e.g. "description")
        self.poly_degree = 2
        self.set_kernel(kernel, poly_degree)

    # ------------------------------------------------------------------ data

    @property
    def num_results(self):
        return self.train_x.shape[0]

    def add_result(self, x, y):
        """Record one measured experiment: concentrations ``x``, responses ``y``."""
        x = torch.as_tensor(x, dtype=torch.float32).reshape(1, self.num_factors)
        y = torch.as_tensor(y, dtype=torch.float32).reshape(1, self.num_tasks)
        self.train_x = torch.cat([self.train_x, x])
        self.train_y = torch.cat([self.train_y, y])
        self.gp = None  # stale -- refit on next suggest()

    def add_results(self, x, y):
        """Record several measured experiments at once."""
        for xi, yi in zip(torch.as_tensor(x, dtype=torch.float32).reshape(-1, self.num_factors),
                          torch.as_tensor(y, dtype=torch.float32).reshape(-1, self.num_tasks)):
            self.add_result(xi, yi)

    def remove_result(self, index):
        """Delete one recorded experiment (e.g. a mistyped entry)."""
        if not 0 <= index < self.num_results:
            raise IndexError(f"no result #{index}")
        keep = torch.arange(self.num_results) != index
        self.train_x = self.train_x[keep]
        self.train_y = self.train_y[keep]
        self.gp = None

    def fitted_gp(self, fit_iters=100):
        """The multitask GP fit to all results (cached until data changes)."""
        if self.num_results < 2:
            raise ValueError("need at least 2 results to fit the GP")
        return self._fit(fit_iters)

    def set_kernel(self, kernel, poly_degree=None):
        """Choose how the posterior is fit.

        ``kernel="rbf"`` (the default) is the flexible GP; ``kernel="poly"``
        uses a polynomial kernel, whose posterior mean is exactly a
        degree-``poly_degree`` polynomial in the concentrations. All results
        are kept -- only the model refits.
        """
        if kernel not in ("rbf", "poly"):
            raise ValueError("kernel must be 'rbf' or 'poly'")
        if poly_degree is not None:
            poly_degree = int(poly_degree)
            if not 1 <= poly_degree <= 10:
                raise ValueError("poly_degree must be between 1 and 10")
            self.poly_degree = poly_degree
        self.kernel = kernel
        self.gp = None  # stale -- refit with the new kernel

    # ------------------------------------------------------------- proposals

    def _sample(self, n):
        """Stratified random samples within the bounds (space-filling-ish)."""
        strata = (torch.rand(n, self.num_factors)
                  + torch.stack([torch.randperm(n).float()
                                 for _ in range(self.num_factors)], dim=-1)) / n
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + (hi - lo) * strata

    def _fit(self, fit_iters):
        if self.gp is None:
            if self.kernel == "poly":
                # the polynomial kernel needs longer to find its output scale
                fit_iters = max(fit_iters, 300)
            self.gp = MultitaskGP(
                self.train_x, self.train_y,
                kernel=self.kernel, degree=self.poly_degree,
                x_bounds=self.bounds,
            ).fit(iters=fit_iters)
        return self.gp

    def suggest(self, n=1, candidate_pool=2000, fit_iters=100):
        """Propose the next ``n`` (1..3) concentration sets to run.

        Returns a tensor of shape (n, num_factors).
        """
        if not 1 <= n <= 3:
            raise ValueError("suggest between 1 and 3 experiments at a time")
        if self.num_results < self.num_init:
            return self._sample(n)

        gp = self._fit(fit_iters)
        cand = self._sample(candidate_pool)
        post = gp.posterior(cand)
        mean = post.mean[:, self.target_task]
        std = post.variance[:, self.target_task].clamp_min(1e-12).sqrt()
        col = self.train_y[:, self.target_task]
        best = (col.max() if self.maximize else col.min()).item()
        score = expected_improvement(mean, std, best, maximize=self.maximize)
        if self.cost_aware and self.costs is not None:
            score = score / self.costs.experiment_cost(cand).clamp_min(1e-9)

        # Greedy batch: after each pick, damp the score near it so the batch
        # covers distinct regions instead of proposing near-duplicates.
        span = (self.bounds[:, 1] - self.bounds[:, 0]).clamp_min(1e-9)
        picks = []
        score = score.clone()
        for _ in range(n):
            idx = score.argmax()
            picks.append(cand[idx])
            dist2 = (((cand - cand[idx]) / span) ** 2).sum(dim=-1)
            score = score * (1.0 - torch.exp(-dist2 / (2 * 0.1 ** 2)))
        return torch.stack(picks)

    # --------------------------------------------------------------- reports

    def best(self):
        """Best measured experiment so far (dict), or None before any result."""
        if self.num_results == 0:
            return None
        col = self.train_y[:, self.target_task]
        idx = (col.argmax() if self.maximize else col.argmin()).item()
        info = {
            "x": self.train_x[idx],
            "y": self.train_y[idx],
            "index": idx,
        }
        if self.costs is not None:
            info["cost"] = self.costs.experiment_cost(self.train_x[idx]).item()
            target = self.train_y[idx, self.target_task].item()
            if target > 0:
                info["cost_per_yield"] = info["cost"] / target
        return info

    def predicted_best(self, candidate_pool=5000, fit_iters=100):
        """Where the model currently believes the optimum is (dict).

        Searches the GP posterior mean over a random candidate pool -- an
        estimate of the best achievable settings, before actually running them.
        """
        if self.num_results < 2:
            return None
        gp = self._fit(fit_iters)
        cand = self._sample(candidate_pool)
        mean = gp.posterior(cand).mean
        col = mean[:, self.target_task]
        idx = (col.argmax() if self.maximize else col.argmin()).item()
        info = {"x": cand[idx], "y": mean[idx]}
        if self.costs is not None:
            info["cost"] = self.costs.experiment_cost(cand[idx]).item()
            target = mean[idx, self.target_task].item()
            if target > 0:
                info["cost_per_yield"] = info["cost"] / target
        return info

    def spend(self):
        """Total cost of all experiments run so far (0 if no costs given)."""
        if self.costs is None or self.num_results == 0:
            return 0.0
        return self.costs.batch_cost(self.train_x)

    def projected_spend(self, num_future):
        """Spend so far + estimate for ``num_future`` more experiments.

        Future runs are priced at the average concentration of the search
        space, which is the expected cost when the promising region is still
        unknown -- useful for budgeting the whole DoE campaign up front.
        """
        if self.costs is None:
            return 0.0
        mid = self.bounds.mean(dim=-1)
        return self.spend() + num_future * self.costs.experiment_cost(mid).item()

    # ----------------------------------------------------------- persistence

    def save(self, path):
        """Save the session (config + all results) to a JSON file.

        Extra keys in :attr:`meta` -- a session description, keys written by
        other tools -- are saved back too, so they survive round-trips. The
        file is replaced atomically.
        """
        state = {
            **self.meta,
            "bounds": self.bounds.tolist(),
            "num_tasks": self.num_tasks,
            "target_task": self.target_task,
            "maximize": self.maximize,
            "cost_aware": self.cost_aware,
            "num_init": self.num_init,
            "factor_names": self.factor_names,
            "response_names": self.response_names,
            "kernel": self.kernel,
            "poly_degree": self.poly_degree,
            "train_x": self.train_x.tolist(),
            "train_y": self.train_y.tolist(),
        }
        if self.costs is not None:
            state["costs"] = {
                "prices": self.costs.prices.tolist(),
                "fixed_cost": self.costs.fixed_cost,
                "names": self.costs.names,
                "currency": self.costs.currency,
            }
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        """Restore a session saved with :meth:`save`."""
        with open(path) as f:
            state = json.load(f)
        costs = None
        if "costs" in state:
            costs = IngredientCosts(
                state["costs"]["prices"],
                fixed_cost=state["costs"]["fixed_cost"],
                names=state["costs"]["names"],
                currency=state["costs"]["currency"],
            )
        opt = cls(
            bounds=state["bounds"],
            num_tasks=state["num_tasks"],
            target_task=state["target_task"],
            maximize=state["maximize"],
            costs=costs,
            cost_aware=state["cost_aware"],
            num_init=state["num_init"],
            factor_names=state["factor_names"],
            response_names=state["response_names"],
            kernel=state.get("kernel", "rbf"),          # older session files
            poly_degree=state.get("poly_degree", 2),    # predate the choice
        )
        if state["train_x"]:
            opt.train_x = torch.tensor(state["train_x"], dtype=torch.float32)
            opt.train_y = torch.tensor(state["train_y"], dtype=torch.float32)
        opt.meta = {k: v for k, v in state.items() if k not in _SESSION_KEYS}
        return opt
