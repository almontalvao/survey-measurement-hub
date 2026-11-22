"""Differential item functioning (DIF) for dichotomous items.

An item shows DIF when people with the same trait level but in different
groups have different chances of endorsing it. A group difference in the
trait itself (impact) is not DIF, which is why both methods compare
groups only within levels of a matching score.

* Mantel-Haenszel: stratify on the total score, pool the 2x2 tables, and
  report the common odds ratio on the ETS delta scale with A/B/C classes.
  Optional purification re-computes the matching score without the items
  already flagged, since DIF items contaminate it.
* Logistic regression (Swaminathan and Rogers): likelihood-ratio tests
  for uniform (group) and nonuniform (group x score) DIF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .._stats import logistic_irls


@dataclass
class MHResult:
    alpha_mh: np.ndarray        # common odds ratio (reference vs focal)
    delta: np.ndarray           # ETS delta = -2.35 ln(alpha)
    chi2: np.ndarray
    pvalue: np.ndarray
    ets_class: np.ndarray       # "A", "B" or "C"
    flagged: np.ndarray         # bool
    matching_items: np.ndarray  # items used in the final matching score


def _mh_item(x, group, score):
    num = den = sum_a = sum_e = sum_v = 0.0
    for s in np.unique(score):
        m = score == s
        g = group[m]
        y = x[m]
        nR, nF = (g == 0).sum(), (g == 1).sum()
        N = nR + nF
        if nR == 0 or nF == 0 or N < 2:
            continue
        A = ((g == 0) & (y == 1)).sum()
        B = nR - A
        C = ((g == 1) & (y == 1)).sum()
        D = nF - C
        m1 = A + C
        m0 = B + D
        if m1 == 0 or m0 == 0:
            continue
        num += A * D / N
        den += B * C / N
        sum_a += A
        sum_e += nR * m1 / N
        sum_v += nR * nF * m1 * m0 / (N ** 2 * (N - 1))
    alpha = num / den if den > 0 else np.inf
    chi2 = (abs(sum_a - sum_e) - 0.5) ** 2 / sum_v if sum_v > 0 else 0.0
    return alpha, chi2


def mantel_haenszel(X, group, purify: bool = True, alpha: float = 0.05,
                    max_rounds: int = 10) -> MHResult:
    """Mantel-Haenszel DIF for every item of a dichotomous test.

    An item is flagged (ETS class B or C) when its chi-square is significant
    after a Bonferroni correction and |delta| >= 1.
    """
    X = np.asarray(X).astype(int)
    group = np.asarray(group).astype(int)
    n, J = X.shape
    flagged = np.zeros(J, bool)
    for _ in range(max_rounds):
        clean = ~flagged
        res = []
        for j in range(J):
            use = clean.copy()
            use[j] = True                      # studied item stays in the score
            res.append(_mh_item(X[:, j], group, X[:, use].sum(axis=1)))
        a_mh = np.array([r[0] for r in res])
        chi2 = np.array([r[1] for r in res])
        delta = -2.35 * np.log(a_mh)
        p = stats.chi2.sf(chi2, 1)
        sig = p * J < alpha
        new_flag = sig & (np.abs(delta) >= 1.0)
        if not purify or np.array_equal(new_flag, flagged):
            flagged = new_flag
            break
        flagged = new_flag
    cls = np.where(flagged & (np.abs(delta) >= 1.5), "C",
                   np.where(flagged, "B", "A"))
    return MHResult(a_mh, delta, chi2, p, cls, flagged, np.flatnonzero(~flagged))


@dataclass
class LogisticDIF:
    uniform_chi2: np.ndarray
    nonuniform_chi2: np.ndarray
    total_chi2: np.ndarray
    p_total: np.ndarray
    group_coef: np.ndarray       # log-odds shift for the focal group (M1)


def logistic_dif(X, group, matching=None) -> LogisticDIF:
    """Likelihood-ratio DIF tests, one item at a time.

    M0: logit P(x=1) = b0 + b1 score
    M1: + b2 group                 (uniform DIF: M1 vs M0, 1 df)
    M2: + b3 group x score         (nonuniform: M2 vs M1, 1 df; total: 2 df)
    """
    X = np.asarray(X).astype(int)
    g = np.asarray(group, float)
    n, J = X.shape
    score = X.sum(axis=1) if matching is None else np.asarray(matching, float)
    z = (score - score.mean()) / score.std()
    out = np.zeros((4, J))
    for j in range(J):
        y = X[:, j]
        one = np.ones(n)
        m0 = logistic_irls(np.column_stack([one, z]), y)
        m1 = logistic_irls(np.column_stack([one, z, g]), y)
        m2 = logistic_irls(np.column_stack([one, z, g, g * z]), y)
        out[:, j] = [2 * (m1.loglik - m0.loglik), 2 * (m2.loglik - m1.loglik),
                     2 * (m2.loglik - m0.loglik), m1.coef[2]]
    return LogisticDIF(out[0], out[1], out[2], stats.chi2.sf(out[2], 2), out[3])
