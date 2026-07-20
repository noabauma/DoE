"""Multitask Gaussian Process regression model (GPyTorch)."""

import torch
import gpytorch


def _as_2d(x):
    """Coerce inputs to a (n, num_factors) float tensor."""
    x = torch.as_tensor(x, dtype=torch.float32)
    return x.unsqueeze(-1) if x.dim() == 1 else x


class _ExactMultitaskGP(gpytorch.models.ExactGP):
    """Internal ExactGP: data kernel (ARD-RBF default) ⊗ learned task covariance."""

    def __init__(self, train_x, train_y, likelihood, num_tasks, num_factors, rank,
                 data_kernel=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            data_kernel or gpytorch.kernels.RBFKernel(ard_num_dims=num_factors),
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
    train_x  : array of shape (n,) or (n, num_factors)
    train_y  : array of shape (n,) or (n, num_tasks)
    rank     : rank of the task covariance (1 is a good default).
    kernel   : "rbf" (default -- flexible GP) or "poly" -- a polynomial kernel,
               whose posterior mean is exactly a degree-``degree`` polynomial
               (all cross-terms between factors included). Note the prior
               shrinks the highest-order terms, so very high degrees (>= 5)
               behave like a smoothed version of a least-squares polynomial.
    degree   : polynomial degree, used only when ``kernel="poly"``.
    x_bounds : optional (num_factors, 2) low/high per factor. Only used by the
               polynomial kernel, which centers inputs to [-1, 1] so
               (x·x' + c)^degree stays well conditioned whatever the
               concentration units are; defaults to the training data's range.
    """

    def __init__(self, train_x, train_y, rank=1, kernel="rbf", degree=2,
                 x_bounds=None):
        self.train_x = _as_2d(train_x)
        train_y = torch.as_tensor(train_y, dtype=torch.float32)
        self.train_y = train_y.unsqueeze(-1) if train_y.dim() == 1 else train_y
        self.num_factors = self.train_x.shape[-1]
        self.num_tasks = self.train_y.shape[-1]
        self.kernel = kernel
        if kernel == "poly":
            b = (torch.as_tensor(x_bounds, dtype=torch.float32).reshape(-1, 2)
                 if x_bounds is not None
                 else torch.stack([self.train_x.min(0).values,
                                   self.train_x.max(0).values], dim=-1))
            self._x_lo = b[:, 0]
            self._x_span = (b[:, 1] - b[:, 0]).clamp_min(1e-9)
            # ScaleKernel: the polynomial kernel has no output scale of its
            # own and badly underfits data whose variance is far from 1
            data_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PolynomialKernel(power=int(degree)))
        elif kernel == "rbf":
            data_kernel = None
        else:
            raise ValueError("kernel must be 'rbf' or 'poly'")
        self._train_x_t = self._tx(self.train_x)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        )
        self.model = _ExactMultitaskGP(
            self._train_x_t, self.train_y, self.likelihood,
            self.num_tasks, self.num_factors, rank, data_kernel=data_kernel,
        )

    def _tx(self, x):
        """Kernel-space inputs (the polynomial kernel works on [-1, 1])."""
        if self.kernel == "poly":
            return (x - self._x_lo) / self._x_span * 2 - 1
        return x

    def fit(self, iters=100, lr=0.1, verbose=False):
        """Train the GP hyperparameters by maximizing the marginal likelihood."""
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(iters):
            optimizer.zero_grad()
            loss = -mll(self.model(self._train_x_t), self.train_y)
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
            return self.model(self._tx(_as_2d(x)))

    def predict(self, x):
        """Predictive mean and 95% bands incl. observation noise. For plots.

        Returns (mean, lower, upper), each of shape (n, num_tasks).
        """
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(self._tx(_as_2d(x))))
            lower, upper = pred.confidence_region()
            return pred.mean, lower, upper

    @property
    def lengthscales(self):
        """ARD lengthscale per factor (smaller = more influential).

        None for kernels without lengthscales (e.g. the polynomial kernel).
        """
        ls = self.model.covar_module.data_covar_module.lengthscale
        return None if ls is None else ls.detach().squeeze()

    @property
    def task_covariance(self):
        """Learned covariance matrix between the responses (num_tasks²)."""
        return self.model.covar_module.task_covar_module.covar_matrix.to_dense().detach()
