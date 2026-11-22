"""Item-response data with known item parameters.

Graded response model (Samejima): for item j with discrimination a_j and
ordered intercepts d_j1 > d_j2 > ... > d_j,K-1,

    P(X_j >= k | theta) = logistic(a_j * theta + d_jk),   k = 1..K-1.

With K = 2 this is the two-parameter logistic (2PL) model, and the
difficulty is b_j = -d_j / a_j. Differential item functioning (DIF) is
planted by shifting a focal group's difficulty on chosen items.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@dataclass
class ItemBank:
    a: np.ndarray                 # J discriminations
    d: np.ndarray                 # J x (K-1) intercepts, decreasing in k

    @property
    def n_items(self) -> int:
        return len(self.a)

    @property
    def n_categories(self) -> int:
        return self.d.shape[1] + 1

    @property
    def b(self) -> np.ndarray:
        """Category thresholds on the theta scale (J x (K-1))."""
        return -self.d / self.a[:, None]

    def subset(self, idx) -> "ItemBank":
        idx = np.asarray(idx)
        return ItemBank(self.a[idx].copy(), self.d[idx].copy())


@dataclass
class ResponseData:
    X: np.ndarray                 # n x J integer responses (0..K-1)
    theta: np.ndarray             # n true abilities / trait levels
    bank: ItemBank
    group: np.ndarray | None = None          # 0 reference, 1 focal
    dif_items: list = field(default_factory=list)
    dif_shift: np.ndarray | None = None      # b shift for focal group


def bank_2pl(n_items: int, a_range=(0.8, 2.2), b_range=(-2.0, 2.0),
             seed=None) -> ItemBank:
    rng = np.random.default_rng(seed)
    a = rng.uniform(*a_range, n_items)
    b = rng.uniform(*b_range, n_items)
    return ItemBank(a, (-a * b)[:, None])


def bank_grm(n_items: int, n_categories: int = 5, a_range=(1.0, 2.5),
             center_range=(-1.0, 1.0), spread=(0.6, 1.2), seed=None) -> ItemBank:
    """Polytomous bank: thresholds b_jk spaced around a random center."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(*a_range, n_items)
    K1 = n_categories - 1
    b = np.empty((n_items, K1))
    for j in range(n_items):
        gaps = rng.uniform(*spread, K1)
        t = np.cumsum(gaps)
        b[j] = t - t.mean() + rng.uniform(*center_range)
    return ItemBank(a, -a[:, None] * b)


def simulate_responses(bank: ItemBank, theta, seed=None,
                       d_override: np.ndarray | None = None) -> np.ndarray:
    """Draw graded responses for each (person, item)."""
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta, float)
    d = bank.d if d_override is None else d_override
    if d.ndim == 2:          # same intercepts for everyone
        z = bank.a[None, :, None] * theta[:, None, None] + d[None, :, :]
    else:                    # person-specific intercepts (n x J x K-1)
        z = bank.a[None, :, None] * theta[:, None, None] + d
    pstar = _sigmoid(z)                       # P(X >= k), n x J x K-1
    u = rng.random(pstar.shape[:2])[..., None]
    return (u < pstar).sum(axis=2).astype(int)


def simulate_irt(bank: ItemBank, n: int, theta_mean=0.0, theta_sd=1.0,
                 seed=None) -> ResponseData:
    rng = np.random.default_rng(seed)
    theta = rng.normal(theta_mean, theta_sd, n)
    X = simulate_responses(bank, theta, seed=rng.integers(1 << 31))
    return ResponseData(X, theta, bank)


def simulate_dif(bank: ItemBank, n_ref: int, n_focal: int, dif_items,
                 b_shift, focal_mean=0.0, seed=None) -> ResponseData:
    """Two groups; on ``dif_items`` the focal group finds the item harder
    by ``b_shift`` (uniform DIF), even at the same trait level. The focal
    group may also differ in true mean (impact), which is not DIF."""
    rng = np.random.default_rng(seed)
    dif_items = list(dif_items)
    shift = np.zeros(bank.n_items)
    shift[dif_items] = b_shift
    theta = np.concatenate([rng.normal(0.0, 1.0, n_ref),
                            rng.normal(focal_mean, 1.0, n_focal)])
    group = np.r_[np.zeros(n_ref, int), np.ones(n_focal, int)]
    # b -> b + shift  <=>  d -> d - a * shift
    d_focal = bank.d - (bank.a * shift)[:, None]
    d_person = np.where(group[:, None, None] == 1, d_focal[None], bank.d[None])
    X = simulate_responses(bank, theta, seed=rng.integers(1 << 31),
                           d_override=d_person)
    return ResponseData(X, theta, bank, group, dif_items, shift)
