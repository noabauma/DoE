"""Bayesian optimization loop built on the multitask GP."""

import torch

from .model import MultitaskGP
from .acquisition import expected_improvement


class BayesianOptimizer:
    """Bayesian optimization of one target response using a multitask GP.

    Each experiment measures ALL responses, and the multitask GP models them
    jointly -- so the correlated responses help even though only ``target_task``
    is optimized.

    Parameters
    ----------
    objective    : callable(x: (n, num_factors)) -> (n, num_tasks) responses.
                   Replace with a real lab/simulation measurement in practice.
    num_factors  : number of input factors.
    num_tasks    : number of responses measured per experiment.
    target_task  : index of the response to optimize.
    maximize     : True to maximize the target response, False to minimize.
    bounds       : (num_factors, 2) low/high per factor; defaults to unit cube.
    """

    def __init__(self, objective, num_factors, num_tasks, target_task=0,
                 maximize=True, bounds=None):
        self.objective = objective
        self.num_factors = num_factors
        self.num_tasks = num_tasks
        self.target_task = target_task
        self.maximize = maximize
        if bounds is None:
            bounds = torch.stack(
                [torch.zeros(num_factors), torch.ones(num_factors)], dim=-1
            )
        self.bounds = torch.as_tensor(bounds, dtype=torch.float32)

    def _sample(self, n):
        """Uniform random factor settings within the bounds."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + (hi - lo) * torch.rand(n, self.num_factors)

    def _best(self, y):
        col = y[:, self.target_task]
        return (col.max() if self.maximize else col.min()).item()

    def run(self, num_init=6, num_iters=20, candidate_pool=1000,
            fit_iters=100, verbose=True):
        """Run the optimization loop and return a result dict."""
        train_x = self._sample(num_init)
        train_y = self.objective(train_x)
        best = self._best(train_y)
        history = [best]
        if verbose:
            print(f"Init best (task {self.target_task}) after "
                  f"{num_init} runs: {best:.4f}")

        for it in range(num_iters):
            gp = MultitaskGP(train_x, train_y).fit(iters=fit_iters)

            cand = self._sample(candidate_pool)
            post = gp.posterior(cand)
            mean = post.mean[:, self.target_task]
            std = post.variance[:, self.target_task].clamp_min(1e-12).sqrt()
            ei = expected_improvement(mean, std, best, maximize=self.maximize)

            next_x = cand[ei.argmax()].unsqueeze(0)
            next_y = self.objective(next_x)
            train_x = torch.cat([train_x, next_x])
            train_y = torch.cat([train_y, next_y])
            best = self._best(train_y)
            history.append(best)
            if verbose:
                print(f"Iter {it + 1:2d}/{num_iters}: "
                      f"x={next_x.squeeze().numpy().round(3)} -> best = {best:.4f}")

        col = train_y[:, self.target_task]
        idx = (col.argmax() if self.maximize else col.argmin()).item()
        return {
            "train_x": train_x,
            "train_y": train_y,
            "best_x": train_x[idx],
            "best_y": best,
            "history": history,
        }
