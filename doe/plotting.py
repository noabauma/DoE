"""Plotting helpers for the DoE multitask GP workflows."""

import torch
import matplotlib.pyplot as plt


def plot_1d(train_x, train_y, test_x, mean, lower, upper, path=None, title=None):
    """One subplot per task: observations, mean prediction, and 95% band.

    Only meaningful when there is a single input factor.
    """
    train_x = torch.as_tensor(train_x).squeeze().numpy()
    test_x = torch.as_tensor(test_x).squeeze().numpy()
    num_tasks = mean.shape[-1]

    fig, axes = plt.subplots(1, num_tasks, figsize=(5.5 * num_tasks, 4),
                             squeeze=False)
    for t, ax in enumerate(axes[0]):
        ax.plot(train_x, train_y[:, t].numpy(), "k*", label="observed")
        ax.plot(test_x, mean[:, t].numpy(), "b", label="mean prediction")
        ax.fill_between(test_x, lower[:, t].numpy(), upper[:, t].numpy(),
                        alpha=0.3, label="95% confidence")
        ax.set_title(f"Task {t + 1}")
        ax.set_xlabel("input factor x")
        ax.legend(loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_parity(test_y, mean, lower, upper, path=None, title=None):
    """One parity subplot per task (predicted vs. actual). Returns per-task RMSE."""
    num_tasks = mean.shape[-1]
    rmse = torch.sqrt(((mean - test_y) ** 2).mean(dim=0))

    fig, axes = plt.subplots(1, num_tasks, figsize=(5.5 * num_tasks, 5),
                             squeeze=False)
    for t, ax in enumerate(axes[0]):
        true_t = test_y[:, t].numpy()
        pred_t = mean[:, t].numpy()
        yerr = torch.stack([mean[:, t] - lower[:, t],
                            upper[:, t] - mean[:, t]]).numpy()
        ax.errorbar(true_t, pred_t, yerr=yerr, fmt="o", alpha=0.6,
                    ecolor="lightgray", label="prediction ± 2σ")
        lim = [min(true_t.min(), pred_t.min()), max(true_t.max(), pred_t.max())]
        ax.plot(lim, lim, "k--", label="perfect")
        ax.set_title(f"Task {t + 1}  (RMSE={rmse[t].item():.3f})")
        ax.set_xlabel("actual response")
        ax.set_ylabel("predicted response")
        ax.legend(loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return rmse


def plot_convergence(history, true_optimum=None, path=None, title=None):
    """Best-so-far target response vs. optimization iteration."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(history)), history, "o-", label="best found")
    if true_optimum is not None:
        ax.axhline(true_optimum, color="k", ls="--", label="true optimum")
    ax.set_xlabel("iteration (0 = initial design)")
    ax.set_ylabel("best target response so far")
    ax.set_title(title or "Bayesian optimization convergence")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, path)
    return fig


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=120)
        print(f"Saved plot to {path}")
