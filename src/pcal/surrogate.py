"""The surrogate: an exact GP, plus the exact leave-one-out downdate.

PREREGISTRATION.md Section 5 fixes the kernel as
ConstantKernel * Matern(nu=2.5, isotropic) + WhiteKernel, refit by marginal
likelihood at every step with n_restarts_optimizer=2. The isotropic length
scale is a deliberate pre-registered choice, not an oversight: anisotropic
optimization over the Magpie dimensions is this project's known compute
bottleneck, and PCA already supplies the dimensionality reduction.

Two standard deviations are distinguished throughout, because the two uses are
not the same question:

    observation sd  sqrt(var_f + noise)  - used wherever coverage of y(x) is
                                           evaluated, which is the estimand
    latent sd       sqrt(var_f)          - used inside acquisition functions

scikit-learn's predict(return_std=True) returns the observation sd when a
WhiteKernel is part of the kernel. The split is not pre-registered and is
recorded in preregistration/deviations.md.

Leave-one-out (Section 7.1) uses the closed form of Rasmussen and Williams
equation 5.12 rather than n explicit refits:

    mu_(-i)(x_i)    = y_i - alpha_i / [A^-1]_ii
    var_(-i)(x_i)   = 1 / [A^-1]_ii                with A = K + noise I

This is exact, not an approximation, at fixed hyperparameters - which is the
point: the refit must hold the kernel fixed so the comparison isolates the
removal of the point. tests/test_loo_refit.py checks it against an explicit
refit that physically excludes the row (validity test 6).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

_TINY = 1e-12


def build_kernel(cfg: dict):
    """The pre-registered kernel, built from configs/policies.yaml."""
    s = cfg["surrogate"]
    return (
        ConstantKernel(1.0, tuple(s["constant_bounds"]))
        * Matern(length_scale=float(s["length_scale_init"]),
                 length_scale_bounds=tuple(s["length_scale_bounds"]),
                 nu=float(s["nu"]))
        + WhiteKernel(noise_level=float(s["noise_level_init"]),
                      noise_level_bounds=tuple(s["noise_level_bounds"]))
    )


class GPSurrogate:
    def __init__(self, cfg: dict, random_state: int = 0):
        self.cfg = cfg
        self.s = cfg["surrogate"]
        self.random_state = int(random_state)
        self.gp: GaussianProcessRegressor | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.train_indices: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, indices=None) -> GPSurrogate:
        self.X_train = np.ascontiguousarray(np.asarray(X, dtype=float))
        self.y_train = np.asarray(y, dtype=float).ravel()
        self.train_indices = (None if indices is None
                              else np.asarray(indices, dtype=int).copy())
        self.gp = GaussianProcessRegressor(
            kernel=build_kernel(self.cfg),
            alpha=float(self.s["jitter"]),
            normalize_y=bool(self.s["normalize_y"]),
            n_restarts_optimizer=int(self.s["n_restarts_optimizer"]),
            random_state=self.random_state,
        ).fit(self.X_train, self.y_train)
        return self

    @property
    def noise_level(self) -> float:
        """The fitted WhiteKernel noise level."""
        kernel = self.gp.kernel_
        for name, value in kernel.get_params(deep=True).items():
            if name.endswith("noise_level") and isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def predict(self, X: np.ndarray, include_noise: bool = True):
        """Return (mean, sd). include_noise=False gives the latent sd."""
        mu, sd = self.gp.predict(np.asarray(X, dtype=float), return_std=True)
        sd = np.asarray(sd, dtype=float)
        if not include_noise:
            var = np.maximum(sd ** 2 - self.noise_level, _TINY)
            sd = np.sqrt(var)
        return np.asarray(mu, dtype=float).ravel(), sd

    def _inverse_diagonal(self) -> np.ndarray:
        """diag(A^-1) for A = K + noise I, from the fitted Cholesky factor."""
        L = self.gp.L_
        n = L.shape[0]
        Linv = solve_triangular(L, np.eye(n), lower=True)
        return np.sum(Linv ** 2, axis=0)

    def loo_predict(self):
        """Exact leave-one-out mean and observation sd at every training point.

        Each value is what the *current* surrogate would have predicted for that
        point had it not been in the training set - which is what Section 7.1
        needs to strip model drift out of a windowed residual.
        """
        if self.gp is None:
            raise RuntimeError("fit() before loo_predict()")
        alpha = np.asarray(self.gp.alpha_, dtype=float).ravel()
        dinv = self._inverse_diagonal()
        if np.any(dinv <= 0):
            raise np.linalg.LinAlgError("non-positive diagonal in A^-1")
        y = self.y_train
        if bool(self.s["normalize_y"]):
            y = (y - self.gp._y_train_mean) / self.gp._y_train_std
        mu = y - alpha / dinv
        var = 1.0 / dinv
        if bool(self.s["normalize_y"]):
            mu = mu * self.gp._y_train_std + self.gp._y_train_mean
            var = var * self.gp._y_train_std ** 2
        return mu, np.sqrt(np.maximum(var, _TINY))

    def refit_without(self, i: int):
        """Explicitly refit on every training row except i, hyperparameters held.

        Only tests/test_loo_refit.py uses this. It is the honest-but-slow
        counterpart of loo_predict(), and it is how validity test 6 confirms
        that the held-out point really is out.
        """
        if self.gp is None:
            raise RuntimeError("fit() before refit_without()")
        keep = np.ones(self.X_train.shape[0], dtype=bool)
        keep[int(i)] = False
        gp = GaussianProcessRegressor(
            kernel=self.gp.kernel_, optimizer=None,
            alpha=float(self.s["jitter"]),
            normalize_y=bool(self.s["normalize_y"]),
        ).fit(self.X_train[keep], self.y_train[keep])
        mu, sd = gp.predict(self.X_train[int(i)][None, :], return_std=True)
        return float(mu[0]), float(sd[0]), self.X_train[keep]

    def log_marginal_likelihood(self) -> float:
        return float(self.gp.log_marginal_likelihood_value_)

    def kernel_description(self) -> str:
        return str(self.gp.kernel_)
