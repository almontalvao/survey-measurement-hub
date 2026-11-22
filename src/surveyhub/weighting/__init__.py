"""Survey weighting and design-based inference, from scratch.

Adjustments (each returns a weight per respondent):
* ``poststratify``: match known population counts in each cell.
* ``rake``: iterative proportional fitting to several known margins at
  once (needs only the margins, not the full cross-classification),
  with optional weight trimming.
* ``pseudo_weights``: propensity weights for a nonprobability sample,
  from a logistic model that contrasts it with a reference sample.
* ``nonresponse_adjust``: inverse response-propensity adjustment for a
  probability sample where nonrespondents' covariates are known.

Diagnostics and variance:
* ``kish_deff`` / ``effective_n``: the price of unequal weights.
* ``linearized_se``: Taylor-linearization SE of a weighted (Hajek) mean,
  with optional strata and clusters.
* ``jackknife``: delete-a-group jackknife that re-runs the whole
  estimator, weighting step included, on every replicate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._stats import logistic_irls


# ------------------------------------------------------------- adjustments

def poststratify(cells, pop_totals, base_weights=None) -> np.ndarray:
    """``cells``: integer cell code per respondent; ``pop_totals``: known
    population count per cell code. Every cell must have a respondent."""
    cells = np.asarray(cells, int)
    pop_totals = np.asarray(pop_totals, float)
    w0 = np.ones(len(cells)) if base_weights is None else np.asarray(base_weights, float)
    samp = np.bincount(cells, weights=w0, minlength=len(pop_totals))
    if np.any((samp == 0) & (pop_totals > 0)):
        empty = np.flatnonzero((samp == 0) & (pop_totals > 0))
        raise ValueError(f"empty sample cells {empty.tolist()}: collapse cells first")
    return w0 * (pop_totals / np.where(samp > 0, samp, 1))[cells]


@dataclass
class RakeResult:
    weights: np.ndarray
    n_iter: int
    converged: bool
    max_margin_error: float       # largest relative gap to a target


def rake(margins: dict, targets: dict, base_weights=None, max_iter: int = 200,
         tol: float = 1e-10, trim: tuple | None = None,
         trim_rounds: int = 10) -> RakeResult:
    """Iterative proportional fitting.

    margins : {name: integer level code per respondent}
    targets : {name: population total per level}
    trim    : (low, high) bounds on weight / mean weight, e.g. (0.2, 5)
    """
    names = list(targets)
    codes = {v: np.asarray(margins[v], int) for v in names}
    tgt = {v: np.asarray(targets[v], float) for v in names}
    n = len(codes[names[0]])
    w = np.ones(n) if base_weights is None else np.asarray(base_weights, float).copy()
    total_it = 0

    def _ipf(w):
        nonlocal total_it
        for it in range(max_iter):
            total_it += 1
            for v in names:
                cur = np.bincount(codes[v], weights=w, minlength=len(tgt[v]))
                ratio = np.where(cur > 0, tgt[v] / np.where(cur > 0, cur, 1), 1.0)
                w = w * ratio[codes[v]]
            err = _margin_error(w)
            if err < tol:
                return w, True
        return w, False

    def _margin_error(w):
        e = 0.0
        for v in names:
            cur = np.bincount(codes[v], weights=w, minlength=len(tgt[v]))
            e = max(e, np.max(np.abs(cur - tgt[v]) / np.maximum(tgt[v], 1e-12)))
        return e

    w, ok = _ipf(w)
    if trim is not None:
        lo, hi = trim
        for _ in range(trim_rounds):
            m = w.mean()
            clipped = np.clip(w, lo * m, hi * m)
            if np.allclose(clipped, w):
                break
            w, ok = _ipf(clipped)
    return RakeResult(w, total_it, ok, _margin_error(w))


def dummies(codes: dict, intercept: bool = True) -> np.ndarray:
    """Main-effects design matrix (first level of each variable dropped)."""
    cols = [np.ones(len(next(iter(codes.values()))))] if intercept else []
    for v, x in codes.items():
        x = np.asarray(x, int)
        for lev in range(1, x.max() + 1):
            cols.append((x == lev).astype(float))
    return np.column_stack(cols)


def pseudo_weights(X_sample, X_reference, w_reference, ridge: float = 1e-6):
    """Propensity pseudo-weights for a nonprobability sample.

    Stack the sample (S = 1, weight 1) on a reference probability sample
    (S = 0, its survey weights, which sum to the population size) and fit
    P(S = 1 | x). The odds (1 - p) / p estimate how many population
    members each sample member stands for.
    """
    Xs = np.asarray(X_sample, float)
    Xr = np.asarray(X_reference, float)
    X = np.vstack([Xs, Xr])
    s = np.r_[np.ones(len(Xs)), np.zeros(len(Xr))]
    wt = np.r_[np.ones(len(Xs)), np.asarray(w_reference, float)]
    fit = logistic_irls(X, s, weights=wt, ridge=ridge)
    p = fit.predict(Xs)
    return (1 - p) / p


def nonresponse_adjust(X, responded, base_weights=None):
    """Inverse response-propensity weights for respondents of a
    probability sample. Returns weights for the responding rows only."""
    X = np.asarray(X, float)
    r = np.asarray(responded, bool)
    w0 = np.ones(len(r)) if base_weights is None else np.asarray(base_weights, float)
    fit = logistic_irls(X, r.astype(float), weights=w0)
    return w0[r] / fit.predict(X[r])


# -------------------------------------------------------- inference tools

def weighted_mean(y, w) -> float:
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    return float((w * y).sum() / w.sum())


def kish_deff(w) -> float:
    """Design effect from unequal weighting: n * sum(w^2) / sum(w)^2."""
    w = np.asarray(w, float)
    return float(len(w) * (w ** 2).sum() / w.sum() ** 2)


def effective_n(w) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def linearized_se(y, w, strata=None, clusters=None) -> float:
    """Taylor-linearization SE of the weighted mean sum(wy)/sum(w).

    The linearized variable is z_i = w_i (y_i - ybar) / sum(w). Its total
    over each cluster is treated as one draw within its stratum (with
    replacement approximation). Weights are treated as fixed.
    """
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    ybar = (w * y).sum() / w.sum()
    z = w * (y - ybar) / w.sum()
    n = len(y)
    strata = np.zeros(n, int) if strata is None else np.asarray(strata)
    clusters = np.arange(n) if clusters is None else np.asarray(clusters)
    var = 0.0
    for h in np.unique(strata):
        m = strata == h
        _, inv = np.unique(clusters[m], return_inverse=True)
        zc = np.bincount(inv, weights=z[m])
        nh = len(zc)
        if nh > 1:
            var += nh / (nh - 1) * ((zc - zc.mean()) ** 2).sum()
    return float(np.sqrt(var))


@dataclass
class JackknifeResult:
    estimate: float
    se: float
    replicates: np.ndarray


def jackknife(estimator, n: int, n_groups: int = 20, seed=None) -> JackknifeResult:
    """Delete-a-group jackknife (JK1).

    ``estimator(keep)`` receives a boolean mask of the respondents kept and
    must return the estimate computed from them alone, re-running any
    weighting step. Respondents are split at random into ``n_groups``.
    With n_groups = n this is the classical delete-one jackknife.
    """
    rng = np.random.default_rng(seed)
    groups = rng.permutation(np.arange(n) % n_groups)
    full = estimator(np.ones(n, bool))
    reps = np.array([estimator(groups != g) for g in range(n_groups)])
    var = (n_groups - 1) / n_groups * ((reps - reps.mean()) ** 2).sum()
    return JackknifeResult(float(full), float(np.sqrt(var)), reps)


__all__ = ["poststratify", "rake", "RakeResult", "dummies", "pseudo_weights",
           "nonresponse_adjust", "weighted_mean", "kish_deff", "effective_n",
           "linearized_se", "jackknife", "JackknifeResult"]
