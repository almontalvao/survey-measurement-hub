"""Measurement invariance across groups.

Before comparing groups on a scale, check that the scale measures the
same thing in each group:

* configural: same items load on the same factors
* metric:     equal loadings (a one-unit trait change moves items equally)
* scalar:     equal intercepts too (same trait level, same expected answer)

Only under scalar invariance (full or partial) are latent mean
differences comparable. Each step is tested with a chi-square difference
test and the change in CFI (a drop larger than 0.01 is the usual flag).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .cfa import fit_cfa, CFAResult


@dataclass
class InvarianceStep:
    level: str
    fit: CFAResult
    delta_chi2: float = np.nan
    delta_df: int = 0
    p_diff: float = np.nan
    delta_cfi: float = np.nan


def chi2_difference(restricted: CFAResult, free: CFAResult):
    dchi = restricted.chi2 - free.chi2
    ddf = restricted.df - free.df
    return dchi, ddf, float(stats.chi2.sf(dchi, ddf)) if ddf > 0 else np.nan


def invariance_sequence(datasets, pattern, error_cov=()) -> list[InvarianceStep]:
    """Fit configural, metric and scalar models and compare each to the
    previous one."""
    steps = []
    prev = None
    for level in ("configural", "metric", "scalar"):
        fit = fit_cfa(list(datasets), pattern, error_cov=error_cov, level=level)
        step = InvarianceStep(level, fit)
        if prev is not None:
            step.delta_chi2, step.delta_df, step.p_diff = chi2_difference(fit, prev)
            step.delta_cfi = fit.cfi - prev.cfi
        steps.append(step)
        prev = fit
    return steps


def locate_noninvariant_intercepts(datasets, pattern, error_cov=(),
                                   alpha: float = 0.01, max_free=None):
    """Sequential search for partial scalar invariance.

    Starting from the full scalar model, free each still-constrained item
    intercept in turn, keep the one whose release lowers chi-square the
    most, and stop when no release is significant at ``alpha`` after a
    Bonferroni correction for the number of items. Returns (freed items in
    order, final fit, log).
    """
    datasets = list(datasets)
    p = np.asarray(pattern).shape[0]
    freed: list[int] = []
    current = fit_cfa(datasets, pattern, error_cov=error_cov, level="scalar")
    log = []
    limit = p - 1 if max_free is None else max_free
    while len(freed) < limit:
        candidates = [i for i in range(p) if i not in freed]
        best = None
        for i in candidates:
            fit = fit_cfa(datasets, pattern, error_cov=error_cov, level="scalar",
                          free_intercepts=freed + [i])
            drop = current.chi2 - fit.chi2
            if best is None or drop > best[1]:
                best = (i, drop, fit)
        i, drop, fit = best
        pval = float(stats.chi2.sf(drop, 1))
        log.append({"item": i, "delta_chi2": drop, "p": pval})
        if pval * p >= alpha:
            break
        freed.append(i)
        current = fit
    return freed, current, log
