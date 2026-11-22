"""Linear factor-model data with a known truth.

Every item follows  x = nu + Lambda @ eta + e,  with eta ~ N(kappa, Phi)
and e ~ N(0, Theta). Because Lambda, Phi and Theta are known, so is the
true reliability of any composite, which is what the reliability and
factor-analysis estimators are graded against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FactorData:
    X: np.ndarray            # n x p observed items
    eta: np.ndarray          # n x k latent scores
    Lambda: np.ndarray       # p x k loadings
    Phi: np.ndarray          # k x k latent covariance
    Theta: np.ndarray        # p x p error covariance
    nu: np.ndarray           # p intercepts
    kappa: np.ndarray        # k latent means

    @property
    def Sigma(self) -> np.ndarray:
        """Population covariance of the items."""
        return self.Lambda @ self.Phi @ self.Lambda.T + self.Theta

    @property
    def mu(self) -> np.ndarray:
        """Population mean of the items."""
        return self.nu + self.Lambda @ self.kappa


def simulate_factor_data(n: int, Lambda, Phi=None, Theta=None, nu=None,
                         kappa=None, seed=None) -> FactorData:
    """Draw ``n`` respondents from a linear factor model.

    ``Theta`` may be a vector (uncorrelated errors) or a full matrix
    (correlated errors, e.g. a shared method effect).
    If ``Theta`` is omitted, error variances are set so every item has
    unit variance.
    """
    rng = np.random.default_rng(seed)
    Lambda = np.atleast_2d(np.asarray(Lambda, float))
    if Lambda.shape[0] == 1 and Lambda.shape[1] > 1:
        Lambda = Lambda.T
    p, k = Lambda.shape
    Phi = np.eye(k) if Phi is None else np.atleast_2d(np.asarray(Phi, float))
    if Theta is None:
        common = np.diag(Lambda @ Phi @ Lambda.T)
        if np.any(common >= 1):
            raise ValueError("unit-variance default needs communalities < 1")
        Theta = np.diag(1 - common)
    Theta = np.asarray(Theta, float)
    if Theta.ndim == 1:
        Theta = np.diag(Theta)
    nu = np.zeros(p) if nu is None else np.asarray(nu, float)
    kappa = np.zeros(k) if kappa is None else np.asarray(kappa, float)
    eta = rng.multivariate_normal(kappa, Phi, size=n)
    e = rng.multivariate_normal(np.zeros(p), Theta, size=n)
    X = nu + eta @ Lambda.T + e
    return FactorData(X, eta, Lambda, Phi, Theta, nu, kappa)


def true_composite_reliability(Lambda, Phi, Theta, weights=None) -> float:
    """Share of composite variance due to the common factors.

    For the unit-weighted sum score of a one-factor scale this is the
    population value of McDonald's omega. Correlated errors count as
    error, because they are not variance in the intended construct.
    """
    Lambda = np.atleast_2d(np.asarray(Lambda, float))
    if Lambda.shape[0] == 1 and Lambda.shape[1] > 1:
        Lambda = Lambda.T
    p, k = Lambda.shape
    Phi = np.eye(k) if Phi is None else np.atleast_2d(np.asarray(Phi, float))
    Theta = np.asarray(Theta, float)
    if Theta.ndim == 1:
        Theta = np.diag(Theta)
    w = np.ones(p) if weights is None else np.asarray(weights, float)
    common = w @ Lambda @ Phi @ Lambda.T @ w
    total = common + w @ Theta @ w
    return float(common / total)


def simple_structure(items_per_factor, loadings) -> np.ndarray:
    """Build a p x k loading matrix where each item loads on one factor.

    ``loadings`` is a flat sequence of length sum(items_per_factor).
    """
    loadings = np.asarray(loadings, float)
    k = len(items_per_factor)
    p = int(np.sum(items_per_factor))
    if len(loadings) != p:
        raise ValueError("need one loading per item")
    L = np.zeros((p, k))
    start = 0
    for f, m in enumerate(items_per_factor):
        L[start:start + m, f] = loadings[start:start + m]
        start += m
    return L
