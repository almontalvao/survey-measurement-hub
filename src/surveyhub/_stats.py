"""Small regression engines shared across modules, written from scratch.

* ``logistic_irls``: weighted logistic regression by iteratively reweighted
  least squares (Newton-Raphson on the log-likelihood).
* ``ols_cluster``: ordinary least squares with classical, HC1, or
  cluster-robust (CR1) standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogitResult:
    coef: np.ndarray
    se: np.ndarray
    loglik: float
    n_iter: int
    converged: bool

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(X, float) @ self.coef)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))   # numerically stable logistic


def logistic_irls(X, y, weights=None, max_iter: int = 100, tol: float = 1e-10,
                  ridge: float = 0.0) -> LogitResult:
    """Fit P(y=1|x) = logistic(x'beta) by Newton-Raphson.

    ``weights`` are frequency or survey weights (the score is weighted).
    ``ridge`` adds a small L2 penalty, useful under quasi-separation.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    beta = np.zeros(X.shape[1])
    converged = False
    for it in range(1, max_iter + 1):
        p = _sigmoid(X @ beta)
        grad = X.T @ (w * (y - p)) - ridge * beta
        H = (X * (w * p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    p = np.clip(_sigmoid(X @ beta), 1e-15, 1 - 1e-15)
    H = (X * (w * p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    ll = float(np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))
    return LogitResult(beta, se, ll, it, converged)


@dataclass
class OLSResult:
    coef: np.ndarray
    se: np.ndarray
    vcov: np.ndarray
    resid: np.ndarray


def ols_cluster(X, y, clusters=None, kind: str = "cluster") -> OLSResult:
    """OLS with ``kind`` in {"classical", "hc1", "cluster"} standard errors.

    The cluster-robust estimator is the CR1 sandwich with the usual
    small-sample factor G/(G-1) * (n-1)/(n-k).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    if kind == "classical":
        V = XtX_inv * (e @ e) / (n - k)
    elif kind == "hc1":
        meat = (X * e[:, None] ** 2).T @ X
        V = XtX_inv @ meat @ XtX_inv * n / (n - k)
    elif kind == "cluster":
        if clusters is None:
            raise ValueError("cluster-robust SEs need cluster ids")
        _, inv = np.unique(clusters, return_inverse=True)
        G = inv.max() + 1
        scores = np.zeros((G, k))
        np.add.at(scores, inv, X * e[:, None])
        meat = scores.T @ scores
        c = G / (G - 1) * (n - 1) / (n - k)
        V = c * XtX_inv @ meat @ XtX_inv
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return OLSResult(beta, np.sqrt(np.diag(V)), V, e)
