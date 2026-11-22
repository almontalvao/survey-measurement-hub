"""Exploratory factor analysis, from scratch.

* ``parallel_analysis``: Horn's method for the number of factors.
* ``efa``: iterated principal axis factoring, then varimax (orthogonal)
  or promax (oblique) rotation.
* ``congruence``: Tucker's congruence between estimated and true
  loadings, after matching factor order and sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


def _corr(X) -> np.ndarray:
    X = np.asarray(X, float)
    if X.shape[0] == X.shape[1] and np.allclose(X, X.T) and np.allclose(np.diag(X), 1):
        return X
    return np.corrcoef(X, rowvar=False)


@dataclass
class ParallelResult:
    n_factors: int
    observed: np.ndarray
    threshold: np.ndarray       # percentile of random-data eigenvalues


def parallel_analysis(X, n_sims: int = 200, percentile: float = 95,
                      seed=None) -> ParallelResult:
    """Retain the leading components whose eigenvalue beats the given
    percentile of eigenvalues from uncorrelated data of the same size."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    n, p = X.shape
    obs = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
    sims = np.empty((n_sims, p))
    for s in range(n_sims):
        Z = rng.standard_normal((n, p))
        sims[s] = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
    thr = np.percentile(sims, percentile, axis=0)
    above = obs > thr
    k = int(np.argmin(above)) if not above.all() else p
    return ParallelResult(k, obs, thr)


def varimax(L, gamma: float = 1.0, max_iter: int = 500, tol: float = 1e-10):
    """Kaiser-normalized varimax. Returns (rotated loadings, rotation)."""
    L = np.asarray(L, float)
    p, k = L.shape
    h = np.sqrt((L ** 2).sum(axis=1))
    A = L / h[:, None]
    R = np.eye(k)
    d_old = 0.0
    for _ in range(max_iter):
        B = A @ R
        u, s, vt = np.linalg.svd(
            A.T @ (B ** 3 - (gamma / p) * B @ np.diag((B ** 2).sum(axis=0))))
        R = u @ vt
        d = s.sum()
        if d_old and d / d_old < 1 + tol:
            break
        d_old = d
    return (A @ R) * h[:, None], R


def promax(L, power: int = 4):
    """Promax: varimax, then an oblique least-squares fit to a sharpened
    target. Returns (pattern loadings, total rotation T, factor corr Phi)."""
    V, R = varimax(L)
    Q = V * np.abs(V) ** (power - 1)
    U, *_ = np.linalg.lstsq(V, Q, rcond=None)
    d = np.diag(np.linalg.inv(U.T @ U))
    U = U @ np.diag(np.sqrt(d))
    pattern = V @ U
    T = R @ U
    Phi = np.linalg.inv(T.T @ T)
    return pattern, T, Phi


@dataclass
class EFAResult:
    loadings: np.ndarray        # rotated pattern loadings (p x k)
    communalities: np.ndarray
    Phi: np.ndarray             # factor correlations (identity if orthogonal)
    unrotated: np.ndarray
    n_iter: int

    @property
    def uniquenesses(self) -> np.ndarray:
        return 1 - self.communalities


def efa(X, n_factors: int, rotation: str = "promax", max_iter: int = 1000,
        tol: float = 1e-8) -> EFAResult:
    """Iterated principal axis factoring on the correlation matrix."""
    R = _corr(X)
    p = R.shape[0]
    h2 = 1 - 1 / np.diag(np.linalg.inv(R))          # squared multiple correlations
    for it in range(1, max_iter + 1):
        Rr = R.copy()
        np.fill_diagonal(Rr, h2)
        vals, vecs = np.linalg.eigh(Rr)
        order = np.argsort(vals)[::-1][:n_factors]
        L = vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))
        h2_new = (L ** 2).sum(axis=1)
        if np.max(np.abs(h2_new - h2)) < tol:
            h2 = h2_new
            break
        h2 = np.minimum(h2_new, 0.999)
    unrot = L
    if n_factors == 1 or rotation in (None, "none"):
        rot, Phi = unrot, np.eye(n_factors)
    elif rotation == "varimax":
        rot, _ = varimax(unrot)
        Phi = np.eye(n_factors)
    elif rotation == "promax":
        rot, _, Phi = promax(unrot)
    else:
        raise ValueError(f"unknown rotation {rotation!r}")
    # orient each factor so its largest loading is positive
    sign = np.sign(rot[np.abs(rot).argmax(axis=0), np.arange(n_factors)])
    rot = rot * sign
    Phi = Phi * np.outer(sign, sign)
    return EFAResult(rot, h2, Phi, unrot, it)


def congruence(L_est, L_true):
    """Tucker congruence per factor, after the best permutation and sign
    alignment of the estimated factors to the true ones.

    Returns (coefficients, aligned estimate)."""
    L_est = np.asarray(L_est, float)
    L_true = np.asarray(L_true, float)
    k = L_true.shape[1]
    best, best_perm = -np.inf, None
    for perm in permutations(range(k)):
        E = L_est[:, perm]
        c = (E * L_true).sum(0) / np.sqrt((E ** 2).sum(0) * (L_true ** 2).sum(0))
        score = np.abs(c).sum()
        if score > best:
            best, best_perm, best_c = score, perm, c
    aligned = L_est[:, best_perm] * np.sign(best_c)
    return np.abs(best_c), aligned
