"""Graded response model (and 2PL as its two-category case) fit by
marginal maximum likelihood with the Bock-Aitkin EM algorithm.

The trait is integrated out on a fixed grid of quadrature points with
N(0, 1) prior weights, which also sets the scale (mean 0, SD 1).

E-step: posterior weight of each grid point for each respondent, from
        which the expected number of respondents at each grid point
        choosing each category of each item follows.
M-step: for each item, maximize the expected complete-data
        log-likelihood over (a, d). Intercepts are kept ordered by
        writing d_k = d_{k-1} - exp(delta_k).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from ..simulate.items import ItemBank


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def quadrature(n_points: int = 61, bound: float = 6.0):
    theta = np.linspace(-bound, bound, n_points)
    w = np.exp(-0.5 * theta ** 2)
    return theta, w / w.sum()


def _cum(a, d, theta):
    """s[j, q, k] = P(X_j >= k | theta_q) for k = 0..K (with s_0 = 1, s_K = 0)."""
    J, K1 = d.shape
    z = a[:, None, None] * theta[None, :, None] + d[:, None, :]
    s = np.empty((J, len(theta), K1 + 2))
    s[..., 0] = 1.0
    s[..., 1:-1] = _sigmoid(z)
    s[..., -1] = 0.0
    return s


def category_probs(a, d, theta) -> np.ndarray:
    """P(X_j = k | theta_q), shape J x Q x K."""
    s = _cum(np.asarray(a, float), np.asarray(d, float), np.atleast_1d(theta))
    return np.clip(s[..., :-1] - s[..., 1:], 1e-300, 1.0)


def item_information(a, d, theta) -> np.ndarray:
    """Fisher information of each item at each theta, shape J x Q.

    I_j(theta) = a_j^2 * sum_k (w_k - w_{k+1})^2 / P_k, with
    w_k = s_k (1 - s_k). For the 2PL this reduces to a^2 P (1 - P).
    """
    a = np.asarray(a, float)
    s = _cum(a, np.asarray(d, float), np.atleast_1d(theta))
    w = s * (1 - s)
    P = np.clip(s[..., :-1] - s[..., 1:], 1e-300, 1.0)
    return a[:, None] ** 2 * ((w[..., :-1] - w[..., 1:]) ** 2 / P).sum(axis=2)


def _to_params(a, d):
    """(a, d) -> unconstrained vector [a, d1, log gaps...] per item."""
    J, K1 = d.shape
    out = np.empty((J, K1 + 1))
    out[:, 0] = a
    out[:, 1] = d[:, 0]
    if K1 > 1:
        out[:, 2:] = np.log(np.maximum(d[:, :-1] - d[:, 1:], 1e-6))
    return out


def _from_params(x):
    a = x[:, 0]
    d = x[:, 1:2] - np.concatenate(
        [np.zeros((len(a), 1)), np.cumsum(np.exp(x[:, 2:]), axis=1)], axis=1)
    return a, d


def _mstep_objective(flat, r, theta, J, K1):
    x = flat.reshape(J, K1 + 1)
    a, d = _from_params(x)
    s = _cum(a, d, theta)
    w = s * (1 - s)
    P = np.clip(s[..., :-1] - s[..., 1:], 1e-300, 1.0)
    ll = (r * np.log(P)).sum()
    ratio = r / P                                          # J x Q x K
    g_d = (w[..., 1:-1] * (ratio[..., 1:] - ratio[..., :-1])).sum(axis=1)
    g_a = (theta[None, :] * (ratio * (w[..., :-1] - w[..., 1:])).sum(axis=2)).sum(axis=1)
    gx = np.empty_like(x)
    gx[:, 0] = g_a
    gx[:, 1] = g_d.sum(axis=1)
    if K1 > 1:
        tail = np.cumsum(g_d[:, ::-1], axis=1)[:, ::-1]    # sum_{k>=m} g_d[k]
        gx[:, 2:] = -np.exp(x[:, 2:]) * tail[:, 1:]
    return -ll, -gx.ravel()


@dataclass
class IRTModel:
    a: np.ndarray
    d: np.ndarray
    loglik: float
    n_iter: int
    converged: bool
    theta_grid: np.ndarray
    prior: np.ndarray

    @property
    def bank(self) -> ItemBank:
        return ItemBank(self.a.copy(), self.d.copy())

    @property
    def b(self) -> np.ndarray:
        return -self.d / self.a[:, None]

    def info(self, theta) -> np.ndarray:
        return item_information(self.a, self.d, theta)

    def test_info(self, theta) -> np.ndarray:
        return self.info(theta).sum(axis=0)

    def eap(self, X):
        return eap_scores(self.a, self.d, X, self.theta_grid, self.prior)


def _onehot_loglik(X, logP):
    """log L[i, q] = sum_j log P_j(x_ij | theta_q); missing coded as -1."""
    n = X.shape[0]
    out = np.zeros((n, logP.shape[1]))
    for k in range(logP.shape[2]):
        out += (X == k).astype(float) @ logP[:, :, k]
    return out


def eap_scores(a, d, X, theta_grid=None, prior=None):
    """Expected a posteriori trait estimates and posterior SDs."""
    if theta_grid is None:
        theta_grid, prior = quadrature()
    X = np.asarray(X)
    logP = np.log(category_probs(a, d, theta_grid))
    lp = _onehot_loglik(X, logP) + np.log(prior)[None, :]
    lp -= lp.max(axis=1, keepdims=True)
    post = np.exp(lp)
    post /= post.sum(axis=1, keepdims=True)
    mean = post @ theta_grid
    sd = np.sqrt(np.maximum(post @ theta_grid ** 2 - mean ** 2, 0))
    return mean, sd


def fit_grm(X, n_categories: int | None = None, max_iter: int = 500,
            tol: float = 1e-4, n_quad: int = 61, verbose: bool = False) -> IRTModel:
    """Fit Samejima's graded response model by Bock-Aitkin EM.

    X : n x J integer matrix of responses 0..K-1, with -1 for missing.
    """
    X = np.asarray(X).astype(int)
    n, J = X.shape
    K = int(X.max()) + 1 if n_categories is None else n_categories
    K1 = K - 1
    theta, prior = quadrature(n_quad)
    # starting values from marginal cumulative proportions
    obs = X >= 0
    cum = np.stack([((X >= k) & obs).sum(0) / obs.sum(0) for k in range(1, K)], axis=1)
    cum = np.clip(cum, 0.01, 0.99)
    d0 = np.log(cum / (1 - cum)) * 1.2
    d0 = np.minimum.accumulate(d0 - np.arange(K1)[None, :] * 1e-3, axis=1)
    x = _to_params(np.ones(J), d0)
    onehots = [(X == k).astype(float) for k in range(K)]
    ll_old = -np.inf
    converged = False
    for it in range(1, max_iter + 1):
        a, d = _from_params(x)
        logP = np.log(category_probs(a, d, theta))
        lp = np.zeros((n, len(theta)))
        for k in range(K):
            lp += onehots[k] @ logP[:, :, k]
        lp += np.log(prior)[None, :]
        mx = lp.max(axis=1, keepdims=True)
        post = np.exp(lp - mx)
        norm = post.sum(axis=1, keepdims=True)
        ll = float((np.log(norm) + mx).sum())
        post /= norm
        r = np.stack([onehots[k].T @ post for k in range(K)], axis=2)   # J x Q x K
        res = optimize.minimize(_mstep_objective, x.ravel(), args=(r, theta, J, K1),
                                jac=True, method="L-BFGS-B",
                                options={"maxiter": 100})
        x = res.x.reshape(J, K1 + 1)
        if verbose and it % 10 == 0:
            print(f"iter {it:4d}  loglik {ll:.4f}")
        if abs(ll - ll_old) < tol:
            converged = True
            break
        ll_old = ll
    a, d = _from_params(x)
    return IRTModel(a, d, ll, it, converged, theta, prior)


def fit_2pl(X, **kw) -> IRTModel:
    """Two-parameter logistic model: the graded model with two categories."""
    return fit_grm(X, n_categories=2, **kw)
