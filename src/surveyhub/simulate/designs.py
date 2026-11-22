"""Survey-experiment designs with a known truth.

* Sensitive questions: each respondent has a true sensitive status Z.
  Asked directly, some who have Z deny it (social desirability bias).
  The list experiment and randomized response never ask about Z on its
  own, so honest answers do not reveal anything about the individual.
* Conjoint: respondents choose between two randomized job offers; the
  true average marginal component effects (AMCEs) are computed exactly by
  enumerating every pair of profiles under the data-generating model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


# --------------------------------------------------------------- sensitive

@dataclass
class SensitiveSurvey:
    prevalence: float
    direct: np.ndarray             # 0/1 answers, direct-question arm
    list_treat: np.ndarray         # 0/1 list arm assignment (1 = long list)
    list_count: np.ndarray         # item counts reported in the list arm
    n_control_items: int
    rr_answer: np.ndarray          # 0/1 answers, randomized-response arm
    rr_design: dict                # p_truth, p_yes, p_no


def simulate_sensitive(n_direct=1000, n_list=2000, n_rr=1000,
                       prevalence=0.22, admit_rate=0.55,
                       control_probs=(0.65, 0.25, 0.55, 0.35),
                       p_truth=2 / 3, p_yes=1 / 6, seed=None) -> SensitiveSurvey:
    """``admit_rate``: share of true holders who admit Z when asked directly.

    Forced-response randomized response: a private randomizer tells the
    respondent to answer truthfully (p_truth), say "yes" (p_yes), or say
    "no" (1 - p_truth - p_yes).
    """
    rng = np.random.default_rng(seed)
    # Direct question
    z = rng.random(n_direct) < prevalence
    direct = (z & (rng.random(n_direct) < admit_rate)).astype(int)
    # List experiment
    z = rng.random(n_list) < prevalence
    treat = rng.permutation(np.r_[np.ones(n_list // 2, int),
                                  np.zeros(n_list - n_list // 2, int)])
    control = (rng.random((n_list, len(control_probs)))
               < np.asarray(control_probs)).sum(axis=1)
    count = control + treat * z
    # Randomized response
    z = rng.random(n_rr) < prevalence
    u = rng.random(n_rr)
    rr = np.where(u < p_truth, z, u < p_truth + p_yes).astype(int)
    return SensitiveSurvey(prevalence, direct, treat, count, len(control_probs),
                           rr, {"p_truth": p_truth, "p_yes": p_yes,
                                "p_no": 1 - p_truth - p_yes})


# ---------------------------------------------------------------- conjoint

JOB_ATTRIBUTES = {
    "Salary": ["Base", "+10%", "+20%"],
    "Location": ["On-site", "Hybrid", "Remote"],
    "Employer": ["Startup", "Mid-size", "Large firm"],
    "Manager rating": ["3.0 stars", "4.5 stars"],
    "Benefits": ["Standard", "Generous"],
}

# Two respondent types. Utilities are relative to each attribute's first
# (baseline) level. Type weights must sum to 1.
JOB_TYPES = [
    {"weight": 0.6, "utils": {"Salary": [0, 0.55, 1.0], "Location": [0, 0.35, 0.45],
                              "Employer": [0, 0.15, 0.25], "Manager rating": [0, 0.6],
                              "Benefits": [0, 0.4]}},
    {"weight": 0.4, "utils": {"Salary": [0, 0.3, 0.6], "Location": [0, 0.9, 1.7],
                              "Employer": [0, 0.0, -0.2], "Manager rating": [0, 0.5],
                              "Benefits": [0, 0.3]}},
]


@dataclass
class ConjointData:
    respondent: np.ndarray      # row -> respondent id
    task: np.ndarray            # row -> task id (global)
    levels: np.ndarray          # rows x n_attributes, level codes
    chosen: np.ndarray          # 0/1
    attributes: dict


def _profile_utilities(attributes, utils):
    names = list(attributes)
    grid = np.array(list(product(*[range(len(attributes[a])) for a in names])))
    u = np.zeros(len(grid))
    for c, a in enumerate(names):
        u += np.asarray(utils[a], float)[grid[:, c]]
    return grid, u


def true_amce(attributes=None, types=None) -> dict:
    """Exact AMCEs for a forced-choice, two-profile, uniform design.

    AMCE(attr = level) = E[P(chosen) | attr = level] - E[P(chosen) | attr = base],
    averaging over the other attributes, the competing profile and the
    respondent types. Enumerated exactly over every profile pair.
    """
    attributes = JOB_ATTRIBUTES if attributes is None else attributes
    types = JOB_TYPES if types is None else types
    names = list(attributes)
    win = None
    for t in types:
        grid, u = _profile_utilities(attributes, t["utils"])
        m = _sigmoid(u[:, None] - u[None, :]).mean(axis=1)
        win = t["weight"] * m if win is None else win + t["weight"] * m
    out = {}
    for c, a in enumerate(names):
        base = win[grid[:, c] == 0].mean()
        for lev in range(1, len(attributes[a])):
            out[(a, attributes[a][lev])] = float(win[grid[:, c] == lev].mean() - base)
    return out


def simulate_conjoint(n_respondents=1000, n_tasks=5, attributes=None,
                      types=None, seed=None) -> ConjointData:
    attributes = JOB_ATTRIBUTES if attributes is None else attributes
    types = JOB_TYPES if types is None else types
    rng = np.random.default_rng(seed)
    names = list(attributes)
    n_levels = [len(attributes[a]) for a in names]
    w = np.array([t["weight"] for t in types])
    rtype = rng.choice(len(types), n_respondents, p=w / w.sum())
    n_pairs = n_respondents * n_tasks
    lv = np.stack([rng.integers(0, L, size=(n_pairs, 2)) for L in n_levels], axis=2)
    util = np.zeros((n_pairs, 2))
    resp_of_pair = np.repeat(np.arange(n_respondents), n_tasks)
    for c, a in enumerate(names):
        table = np.array([np.asarray(t["utils"][a], float) for t in types])
        util += table[rtype[resp_of_pair]][np.arange(n_pairs)[:, None], lv[:, :, c]]
    p1 = _sigmoid(util[:, 0] - util[:, 1])
    pick1 = rng.random(n_pairs) < p1
    chosen = np.stack([pick1, ~pick1], axis=1).astype(int)
    return ConjointData(
        respondent=np.repeat(resp_of_pair, 2),
        task=np.repeat(np.arange(n_pairs), 2),
        levels=lv.reshape(n_pairs * 2, len(names)),
        chosen=chosen.reshape(-1),
        attributes=attributes,
    )
