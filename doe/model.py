"""Multitask Gaussian Process regression model (GPyTorch)."""

import torch
import gpytorch


def _as_2d(x):
    """Coerce inputs to a (n, num_factors) float tensor."""
    x = torch.as_tensor(x, dtype=torch.float32)
    return x.unsqueeze(-1) if x.dim() == 1 else x


class _ExactMultitaskGP(gpytorch.models.ExactGP):
    """Internal ExactGP: ARD-RBF over factors ⊗ learned task covariance."""

    def __init__(self, train_x, train_y, likelihood, num_tasks, num_factors, rank):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=num_factors),
            num_tasks=num_tasks, rank=rank,
        )

    def forward(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class MultitaskGP:
    """Multitask GP regression over one or more input factors.

    Learns a separate ARD lengthscale per factor (so you can read off which
    factors matter) and a task-covariance matrix that couples the correlated
    responses, letting each experiment's responses inform one another.

    Parameters
    ----------
    train_x : array of shape (n,) or (n, num_factors)
    train_y : array of shape (n,) or (n, num_tasks)
    rank    : rank of the task covariance (1 is a good default).
    """

    def __init__(self, train_x, train_y, rank=1):
        self.train_x = _as_2d(train_x)
        train_y = torch.as_tensor(train_y, dtype=torch.float32)
        self.train_y = train_y.unsqueeze(-1) if train_y.dim() == 1 else train_y
        self.num_factors = self.train_x.shape[-1]
        self.num_tasks = self.train_y.shape[-1]
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        )
        self.model = _ExactMultitaskGP(
            self.train_x, self.train_y, self.likelihood,
            self.num_tasks, self.num_factors, rank,
        )

    def fit(self, iters=100, lr=0.1, verbose=False):
        """Train the GP hyperparameters by maximizing the marginal likelihood."""
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(iters):
            optimizer.zero_grad()
            loss = -mll(self.model(self.train_x), self.train_y)
            loss.backward()
            optimizer.step()
            if verbose and (i + 1) % max(1, iters // 5) == 0:
                print(f"  iter {i + 1:3d}/{iters} - loss {loss.item():.3f}")
        self.model.eval()
        self.likelihood.eval()
        return self

    def posterior(self, x):
        """Latent posterior (no observation noise). Use this for acquisition."""
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self.model(_as_2d(x))

    def predict(self, x):
        """Predictive mean and 95% bands incl. observation noise. For plots.

        Returns (mean, lower, upper), each of shape (n, num_tasks).
        """
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(_as_2d(x)))
            lower, upper = pred.confidence_region()
            return pred.mean, lower, upper

    @property
    def lengthscales(self):
        """ARD lengthscale per factor (smaller = more influential)."""
        return self.model.covar_module.data_covar_module.lengthscale.detach().squeeze()

    @property
    def task_covariance(self):
        """Learned covariance matrix between the responses (num_tasks²)."""
        return self.model.covar_module.task_covar_module.covar_matrix.to_dense().detach()
