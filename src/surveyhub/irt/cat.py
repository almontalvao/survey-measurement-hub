"""Computerized adaptive testing (CAT) simulator.

For each simulated respondent:
1. start from the prior (theta estimate 0);
2. give the unused item with the most Fisher information at the current
   estimate (or pick at random among the top ``top_k``, which spreads
   item exposure);
3. draw the answer from the respondent's true trait level;
4. update the posterior on the quadrature grid and take its mean (EAP);
5. stop when the posterior SD reaches ``se_stop`` or ``max_items`` is hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import category_probs, item_information, quadrature
from ..simulate.items import ItemBank


@dataclass
class CATResult:
    theta_hat: np.ndarray
    se: np.ndarray
    n_items: np.ndarray
    items: list                  # administered item indices per respondent
    exposure: np.ndarray         # share of respondents who saw each item


def simulate_cat(bank: ItemBank, theta_true, true_bank: ItemBank | None = None,
                 se_stop: float = 0.3, max_items: int | None = None,
                 min_items: int = 1, top_k: int = 1, seed=None,
                 n_quad: int = 61) -> CATResult:
    """``bank`` holds the (estimated) parameters the CAT uses to select and
    score items; ``true_bank`` (default: ``bank``) generates the answers."""
    rng = np.random.default_rng(seed)
    true_bank = bank if true_bank is None else true_bank
    theta_true = np.atleast_1d(np.asarray(theta_true, float))
    J = bank.n_items
    max_items = J if max_items is None else min(max_items, J)
    grid, prior = quadrature(n_quad)
    logP = np.log(category_probs(bank.a, bank.d, grid))           # J x Q x K
    info_grid = item_information(bank.a, bank.d, grid)            # J x Q
    true_P = category_probs(true_bank.a, true_bank.d, theta_true)  # J x n x K
    cum_true = np.cumsum(true_P, axis=2)
    n = len(theta_true)
    th_hat = np.zeros(n)
    se = np.zeros(n)
    n_used = np.zeros(n, int)
    used_lists = []
    exposure = np.zeros(J)
    for i in range(n):
        lp = np.log(prior).copy()
        avail = np.ones(J, bool)
        est, sd = 0.0, 1.0
        used = []
        while len(used) < max_items:
            q = int(np.argmin(np.abs(grid - est)))
            info = np.where(avail, info_grid[:, q], -np.inf)
            if top_k > 1:
                cands = np.argsort(info)[::-1][:min(top_k, avail.sum())]
                j = int(rng.choice(cands))
            else:
                j = int(np.argmax(info))
            avail[j] = False
            used.append(j)
            x = int((rng.random() > cum_true[j, i]).sum())
            lp += logP[j, :, x]
            post = np.exp(lp - lp.max())
            post /= post.sum()
            est = float(post @ grid)
            sd = float(np.sqrt(max(post @ grid ** 2 - est ** 2, 0)))
            if len(used) >= min_items and sd <= se_stop:
                break
        th_hat[i], se[i], n_used[i] = est, sd, len(used)
        used_lists.append(used)
        exposure[used] += 1
    return CATResult(th_hat, se, n_used, used_lists, exposure / n)


def fixed_form(bank: ItemBank, n_items: int, theta_range=(-2.0, 2.0)):
    """A short fixed form: the items with the most information summed over
    a range of theta, which is the usual static alternative to a CAT."""
    grid = np.linspace(*theta_range, 41)
    total = item_information(bank.a, bank.d, grid).sum(axis=1)
    return np.sort(np.argsort(total)[::-1][:n_items])
