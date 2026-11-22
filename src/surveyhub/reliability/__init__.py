"""Reliability coefficients for multi-item scales, from scratch.

Classical test theory writes each observed score as true score plus
error; reliability is the share of observed-score variance that is true
score. Cronbach's alpha equals that share only when every item measures
the trait equally well (tau-equivalence) and errors are uncorrelated.
McDonald's omega relaxes the first assumption by using CFA loadings, and
can absorb the second by modeling the correlated errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..factor.cfa import fit_cfa, CFAResult


def cronbach_alpha(X) -> float:
    X = np.asarray(X, float)
    k = X.shape[1]
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var))


def alpha_from_cov(S) -> float:
    S = np.asarray(S, float)
    k = S.shape[0]
    return float(k / (k - 1) * (1 - np.trace(S) / S.sum()))


def spearman_brown(r, factor):
    """Reliability of a test lengthened by ``factor`` (e.g. 2 = doubled)."""
    r = np.asarray(r, float)
    return factor * r / (1 + (factor - 1) * r)


def split_half(X, seed=None, n_splits: int = 1) -> float:
    """Spearman-Brown-corrected split-half reliability. With n_splits > 1,
    averages over random splits (odd/even split when n_splits == 1)."""
    X = np.asarray(X, float)
    k = X.shape[1]
    rng = np.random.default_rng(seed)
    vals = []
    for s in range(n_splits):
        idx = np.arange(k) if s == 0 else rng.permutation(k)
        a, b = idx[0::2], idx[1::2]
        r = np.corrcoef(X[:, a].sum(1), X[:, b].sum(1))[0, 1]
        vals.append(spearman_brown(r, 2))
    return float(np.mean(vals))


def item_total(X) -> np.ndarray:
    """Corrected item-total correlations (item vs sum of the other items)."""
    X = np.asarray(X, float)
    tot = X.sum(axis=1)
    return np.array([np.corrcoef(X[:, j], tot - X[:, j])[0, 1]
                     for j in range(X.shape[1])])


def alpha_if_deleted(X) -> np.ndarray:
    X = np.asarray(X, float)
    return np.array([cronbach_alpha(np.delete(X, j, axis=1))
                     for j in range(X.shape[1])])


def omega_from_fit(fit: CFAResult, group: int = 0) -> float:
    """Omega total for the unit-weighted sum score of a one-factor CFA.

    Modeled error covariances count as error, so they lower omega instead
    of inflating it the way they inflate alpha.
    """
    g = fit.groups[group]
    one = np.ones(g.Lambda.shape[0])
    common = one @ g.Lambda @ g.Phi @ g.Lambda.T @ one
    return float(common / (common + one @ g.Theta @ one))


@dataclass
class OmegaResult:
    omega: float
    fit: CFAResult


def omega(X, error_cov=()) -> OmegaResult:
    """Fit a one-factor CFA (optionally with correlated errors) and return
    omega total with the fitted model."""
    X = np.asarray(X, float)
    fit = fit_cfa(X, np.ones((X.shape[1], 1), bool), error_cov=error_cov)
    return OmegaResult(omega_from_fit(fit), fit)


def bootstrap_ci(stat, X, n_boot: int = 1000, level: float = 0.95, seed=None):
    """Percentile bootstrap CI for any statistic of the respondents x items
    matrix. Returns (estimate, lower, upper)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    n = X.shape[0]
    est = stat(X)
    boots = np.array([stat(X[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(est), float(lo), float(hi)


__all__ = ["cronbach_alpha", "alpha_from_cov", "spearman_brown", "split_half",
           "item_total", "alpha_if_deleted", "omega", "omega_from_fit",
           "OmegaResult", "bootstrap_ci"]
