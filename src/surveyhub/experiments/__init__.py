"""Estimators for sensitive questions and survey experiments.

* ``direct_estimate``: share answering yes when asked directly.
* ``list_experiment``: difference in mean item counts between the long
  list (with the sensitive item) and the short list.
* ``list_design_test``: Blair and Imai's (2012) check that adding the
  sensitive item did not change answers to the control items.
* ``randomized_response``: forced-response design estimator.
* ``amce``: average marginal component effects from a randomized
  conjoint, by OLS with respondent-clustered standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .._stats import ols_cluster


@dataclass
class Estimate:
    estimate: float
    se: float

    @property
    def ci(self):
        return (self.estimate - 1.96 * self.se, self.estimate + 1.96 * self.se)


def direct_estimate(answers) -> Estimate:
    y = np.asarray(answers, float)
    p = y.mean()
    return Estimate(float(p), float(np.sqrt(p * (1 - p) / len(y))))


def list_experiment(count, treat) -> Estimate:
    count = np.asarray(count, float)
    t = np.asarray(treat, bool)
    yt, yc = count[t], count[~t]
    se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    return Estimate(float(yt.mean() - yc.mean()), float(se))


@dataclass
class DesignTest:
    pi: np.ndarray            # estimated respondent-type shares, 2 x (J+1)
    se: np.ndarray
    min_pvalue: float         # Bonferroni-adjusted p-value of the most negative share
    design_effect: bool


def list_design_test(count, treat, n_control_items: int, alpha: float = 0.05) -> DesignTest:
    """Under no design effect every respondent-type share is nonnegative:
    pi(y, 1) = P(Yc <= y) - P(Yt <= y)        (has the sensitive trait)
    pi(y, 0) = P(Yt <= y) - P(Yc <= y - 1)    (does not)
    A share significantly below zero signals that the treatment list
    changed how people answered the control items."""
    count = np.asarray(count, int)
    t = np.asarray(treat, bool)
    J = n_control_items
    yt, yc = count[t], count[~t]
    pi = np.zeros((2, J + 1))
    se = np.zeros((2, J + 1))

    def cdf(x, y):
        return (x <= y).mean()

    for y in range(J + 1):
        ft, fc = cdf(yt, y), cdf(yc, y)
        fc1 = cdf(yc, y - 1)
        pi[1, y] = fc - ft
        se[1, y] = np.sqrt(fc * (1 - fc) / len(yc) + ft * (1 - ft) / len(yt))
        pi[0, y] = ft - fc1
        se[0, y] = np.sqrt(fc1 * (1 - fc1) / len(yc) + ft * (1 - ft) / len(yt))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, pi / se, 0.0)
    p = stats.norm.cdf(z)                 # one-sided: share < 0
    p_adj = min(1.0, p.min() * 2)         # Bonferroni over the two families
    return DesignTest(pi, se, float(p_adj), bool(p_adj < alpha))


def randomized_response(answers, p_truth: float, p_yes: float) -> Estimate:
    """Forced response: P(yes) = p_truth * pi + p_yes."""
    y = np.asarray(answers, float)
    lam = y.mean()
    est = (lam - p_yes) / p_truth
    se = np.sqrt(lam * (1 - lam) / len(y)) / p_truth
    return Estimate(float(est), float(se))


@dataclass
class AMCEResult:
    labels: list              # (attribute, level) pairs
    estimate: np.ndarray
    se_cluster: np.ndarray
    se_naive: np.ndarray

    def as_dict(self):
        return {lab: (e, s) for lab, e, s in zip(self.labels, self.estimate, self.se_cluster)}


def amce(levels, chosen, respondent, attributes: dict) -> AMCEResult:
    """Regress the choice indicator on dummies for every non-baseline level
    of every attribute. With attributes randomized independently and
    uniformly, each coefficient is unbiased for that level's AMCE."""
    levels = np.asarray(levels, int)
    names = list(attributes)
    cols, labels = [np.ones(len(chosen))], []
    for c, a in enumerate(names):
        for lev in range(1, len(attributes[a])):
            cols.append((levels[:, c] == lev).astype(float))
            labels.append((a, attributes[a][lev]))
    X = np.column_stack(cols)
    fit_c = ols_cluster(X, chosen, clusters=respondent, kind="cluster")
    fit_n = ols_cluster(X, chosen, kind="classical")
    return AMCEResult(labels, fit_c.coef[1:], fit_c.se[1:], fit_n.se[1:])


__all__ = ["Estimate", "direct_estimate", "list_experiment", "list_design_test",
           "DesignTest", "randomized_response", "amce", "AMCEResult"]
