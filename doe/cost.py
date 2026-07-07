"""Ingredient pricing and experiment cost estimation."""

import torch


class IngredientCosts:
    """Prices of the ingredients, used to cost experiments and the yield.

    An experiment mixes the ingredients at the given concentrations, so its
    material cost is the dot product of concentrations and per-unit prices,
    plus a fixed overhead per run (labour, machine time, consumables).

    Parameters
    ----------
    prices     : per-unit price of each ingredient, length ``num_factors`` --
                 the cost of one concentration unit of that ingredient.
    fixed_cost : fixed overhead added to every experiment.
    names      : optional ingredient names used in reports.
    currency   : label used in printed reports.
    """

    def __init__(self, prices, fixed_cost=0.0, names=None, currency="USD"):
        self.prices = torch.as_tensor(prices, dtype=torch.float32)
        self.fixed_cost = float(fixed_cost)
        self.names = list(names) if names else [
            f"ingredient {i + 1}" for i in range(len(self.prices))
        ]
        self.currency = currency

    def experiment_cost(self, x):
        """Cost of each experiment: fixed cost + concentrations . prices.

        ``x`` has shape (num_factors,) or (n, num_factors); returns (n,).
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x @ self.prices + self.fixed_cost

    def batch_cost(self, x):
        """Total cost of running all experiments in ``x``."""
        return self.experiment_cost(x).sum().item()

    def cost_per_yield(self, x, y):
        """Price of the yield: experiment cost divided by the response ``y``.

        ``y`` is the measured (or predicted) target response per experiment,
        e.g. grams of product -- the result is currency per response unit.
        """
        cost = self.experiment_cost(x)
        y = torch.as_tensor(y, dtype=torch.float32).reshape(cost.shape)
        return cost / y

    def describe(self, x):
        """Human-readable cost breakdown of a single experiment."""
        x = torch.as_tensor(x, dtype=torch.float32).squeeze()
        lines = []
        for name, conc, price in zip(self.names, x, self.prices):
            lines.append(f"  {name}: {conc.item():.4g} x "
                         f"{price.item():.4g} {self.currency} = "
                         f"{(conc * price).item():.4g} {self.currency}")
        if self.fixed_cost:
            lines.append(f"  fixed overhead: {self.fixed_cost:.4g} {self.currency}")
        total = self.experiment_cost(x).item()
        lines.append(f"  total: {total:.4g} {self.currency}")
        return "\n".join(lines)
