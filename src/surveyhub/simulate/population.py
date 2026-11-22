"""A synthetic adult population with a known outcome, and an opt-in
sample drawn from it with a known selection mechanism.

The population plays the role of the census: its margins are the raking
targets and its outcome mean is the truth every estimator is graded on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AGE = ["18-29", "30-44", "45-64", "65+"]
EDU = ["HS or less", "Some college", "BA+"]
REGION = ["Northeast", "Midwest", "South", "West"]
SEX = ["Female", "Male"]


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@dataclass
class Population:
    age: np.ndarray
    edu: np.ndarray
    region: np.ndarray
    sex: np.ndarray
    y: np.ndarray                 # binary outcome (got a flu shot)

    @property
    def N(self) -> int:
        return len(self.y)

    def covariates(self) -> dict:
        return {"age": self.age, "edu": self.edu,
                "region": self.region, "sex": self.sex}

    def margins(self, var: str) -> np.ndarray:
        """Population totals of each level of ``var`` (raking targets)."""
        x = self.covariates()[var]
        return np.bincount(x, minlength=x.max() + 1).astype(float)

    @property
    def truth(self) -> float:
        return float(self.y.mean())


def make_population(N: int = 200_000, seed=None) -> Population:
    rng = np.random.default_rng(seed)
    age = rng.choice(4, N, p=[0.21, 0.25, 0.33, 0.21])
    # education depends a little on age
    edu_p = np.array([[0.36, 0.30, 0.34], [0.33, 0.28, 0.39],
                      [0.38, 0.28, 0.34], [0.45, 0.26, 0.29]])
    edu =(rng.random(N)[:, None] > np.cumsum(edu_p[age], axis=1)).sum(axis=1)
    region = rng.choice(4, N, p=[0.17, 0.21, 0.38, 0.24])
    sex = rng.choice(2, N, p=[0.51, 0.49])
    # Outcome: strongly age-graded, rises with education, small region/sex terms
    lin = (-0.9 + np.array([-0.6, -0.25, 0.25, 1.3])[age]
           + np.array([-0.35, 0.0, 0.45])[edu]
           + np.array([0.15, 0.0, -0.2, 0.05])[region]
           + np.array([0.12, 0.0])[sex])
    y = (rng.random(N) < _sigmoid(lin)).astype(int)
    return Population(age, edu, region, sex, y)


def opt_in_sample(pop: Population, n_target: int = 1500, outcome_effect=0.0,
                  seed=None) -> np.ndarray:
    """Indices of an opt-in sample.

    Selection favors the young and the college-educated. ``outcome_effect``
    adds a log-odds bump for people with y = 1: selection on the outcome
    itself, which no covariate adjustment can remove.
    """
    rng = np.random.default_rng(seed)
    lin = (np.array([0.9, 0.4, -0.2, -0.9])[pop.age]
           + np.array([-0.8, 0.0, 0.7])[pop.edu]
           + np.array([0.1, -0.1, 0.0, 0.2])[pop.region]
           + np.array([0.0, -0.15])[pop.sex]
           + outcome_effect * pop.y)
    w = np.exp(lin)
    p = np.minimum(1.0, w * n_target / w.sum())
    return np.flatnonzero(rng.random(pop.N) < p)
