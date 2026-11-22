"""Confirmatory factor analysis by maximum likelihood, from scratch.

Model, for each group g:
    Sigma_g = Lambda_g Phi_g Lambda_g' + Theta_g
    mu_g    = nu_g + Lambda_g kappa_g

Fit function (normal-theory ML with a mean structure):
    F_g = log|Sigma_g| + tr(S_g Sigma_g^-1) - log|S_g| - p
          + (m_g - mu_g)' Sigma_g^-1 (m_g - mu_g)
    F   = sum_g (n_g / N) F_g,       chi-square = N * F_min

The analytic gradient uses dF/dSigma = Sigma^-1 (Sigma - S - d d') Sigma^-1
with d = m - mu, chained to every free parameter. Equality constraints
across groups (metric and scalar invariance) are imposed by letting
several matrix cells point to the same parameter index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------- results

@dataclass
class GroupEstimates:
    Lambda: np.ndarray
    Phi: np.ndarray
    Theta: np.ndarray
    nu: np.ndarray
    kappa: np.ndarray

    @property
    def Sigma(self) -> np.ndarray:
        return self.Lambda @ self.Phi @ self.Lambda.T + self.Theta

    @property
    def mu(self) -> np.ndarray:
        return self.nu + self.Lambda @ self.kappa

    def standardized_loadings(self) -> np.ndarray:
        sd_item = np.sqrt(np.diag(self.Sigma))
        sd_fac = np.sqrt(np.diag(self.Phi))
        return self.Lambda * sd_fac[None, :] / sd_item[:, None]


@dataclass
class CFAResult:
    groups: list
    chi2: float
    df: int
    pvalue: float
    cfi: float
    tli: float
    rmsea: float
    srmr: float
    n_params: int
    n_obs: int
    converged: bool
    level: str = "configural"
    baseline_chi2: float = np.nan
    baseline_df: int = 0
    params: np.ndarray = field(default=None, repr=False)

    @property
    def Lambda(self):
        return self.groups[0].Lambda

    @property
    def Theta(self):
        return self.groups[0].Theta

    @property
    def Phi(self):
        return self.groups[0].Phi

    def summary(self) -> str:
        return (f"chi2({self.df}) = {self.chi2:.2f}, p = {self.pvalue:.3f}, "
                f"CFI = {self.cfi:.3f}, TLI = {self.tli:.3f}, "
                f"RMSEA = {self.rmsea:.3f}, SRMR = {self.srmr:.3f}")


# ------------------------------------------------------------ data inputs

def _moments(data):
    """Accept raw n x p data or a dict with keys S (ML covariance), mean, n."""
    if isinstance(data, dict):
        return (np.asarray(data["S"], float), np.asarray(data["mean"], float),
                int(data["n"]))
    X = np.asarray(data, float)
    n = X.shape[0]
    m = X.mean(axis=0)
    Xc = X - m
    return Xc.T @ Xc / n, m, n


# ------------------------------------------------------------ spec builder

class _Spec:
    def __init__(self):
        self.start: list[float] = []
        self.lower: list[float] = []

    def new(self, start: float, lower: float | None = None) -> int:
        self.start.append(start)
        self.lower.append(lower)
        return len(self.start) - 1


def _build(p, k, moments, pattern, error_cov, level, free_intercepts,
           factor_corr):
    G = len(moments)
    spec = _Spec()
    S_pool = sum(S * n for S, _, n in moments) / sum(n for *_, n in moments)
    sd = np.sqrt(np.diag(S_pool))
    free_int = set(free_intercepts or ())
    lam_idx, phi_idx, phi_fix, th_idx, nu_idx, ka_idx = [], [], [], [], [], []

    shared_lam = None
    shared_nu = None
    for g, (S, m, n) in enumerate(moments):
        # loadings
        if level in ("metric", "scalar") and shared_lam is not None:
            L = shared_lam
        else:
            L = -np.ones((p, k), int)
            for i, f in zip(*np.nonzero(pattern)):
                L[i, f] = spec.new(0.7 * sd[i])
            shared_lam = L
        lam_idx.append(L)
        # factor covariance
        P = -np.ones((k, k), int)
        Pf = np.eye(k)
        for f in range(k):
            if level in ("metric", "scalar") and g > 0:
                P[f, f] = spec.new(1.0, 1e-4)
        if factor_corr:
            for f in range(k):
                for h in range(f):
                    P[f, h] = P[h, f] = spec.new(0.0)
        phi_idx.append(P)
        phi_fix.append(Pf)
        # error covariance
        T = -np.ones((p, p), int)
        for i in range(p):
            T[i, i] = spec.new(0.5 * S[i, i], 1e-6)
        for i, j in error_cov:
            T[i, j] = T[j, i] = spec.new(0.0)
        th_idx.append(T)
        # intercepts
        if level == "scalar" and shared_nu is not None:
            N_ = shared_nu.copy()
            for i in free_int:
                N_[i] = spec.new(m[i])
        else:
            N_ = np.array([spec.new(m[i]) for i in range(p)], int)
            if level == "scalar":
                shared_nu = N_
        nu_idx.append(N_)
        # latent means
        K = -np.ones(k, int)
        if level == "scalar" and g > 0:
            K = np.array([spec.new(0.0) for _ in range(k)], int)
        ka_idx.append(K)

    return spec, dict(lam=lam_idx, phi=phi_idx, phi_fix=phi_fix, th=th_idx,
                      nu=nu_idx, ka=ka_idx)


def _fill(idx, fixed, theta):
    M = np.array(fixed, float, copy=True)
    mask = idx >= 0
    M[mask] = theta[idx[mask]]
    return M


def _matrices(theta, ix, g, p, k):
    L = _fill(ix["lam"][g], np.zeros((p, k)), theta)
    P = _fill(ix["phi"][g], ix["phi_fix"][g], theta)
    T = _fill(ix["th"][g], np.zeros((p, p)), theta)
    nu = _fill(ix["nu"][g], np.zeros(p), theta)
    ka = _fill(ix["ka"][g], np.zeros(k), theta)
    return L, P, T, nu, ka


def _objective(theta, ix, moments, p, k, N):
    F = 0.0
    grad = np.zeros_like(theta)
    for g, (S, m, n) in enumerate(moments):
        L, P, T, nu, ka = _matrices(theta, ix, g, p, k)
        Sig = L @ P @ L.T + T
        try:
            C = np.linalg.cholesky(Sig)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(theta)
        logdet = 2 * np.log(np.diag(C)).sum()
        Si = np.linalg.inv(Sig)
        d = m - (nu + L @ ka)
        _, logdetS = np.linalg.slogdet(S)
        Fg = logdet + np.trace(S @ Si) - logdetS - p + d @ Si @ d
        w = n / N
        F += w * Fg
        Gs = Si @ (Sig - S - np.outer(d, d)) @ Si        # dF/dSigma
        gmu = -2 * Si @ d                                 # dF/dmu
        dL = 2 * Gs @ L @ P + np.outer(gmu, ka)
        dP = L.T @ Gs @ L
        dT = Gs
        dnu = gmu
        dka = L.T @ gmu
        for idx, dM in ((ix["lam"][g], dL), (ix["phi"][g], dP),
                        (ix["th"][g], dT), (ix["nu"][g], dnu),
                        (ix["ka"][g], dka)):
            mask = idx >= 0
            np.add.at(grad, idx[mask], w * dM[mask])
    return F, grad


def fit_cfa(data, pattern, error_cov=(), level: str = "configural",
            free_intercepts=(), factor_corr: bool = True,
            start: np.ndarray | None = None) -> CFAResult:
    """Fit a (multi-group) CFA by maximum likelihood.

    data      : one dataset (n x p array, or dict(S, mean, n)) or a list of them
    pattern   : p x k boolean matrix of free loadings
    error_cov : iterable of (i, j) item pairs whose errors may correlate
    level     : "configural", "metric" or "scalar" (only matters with >1 group)
    free_intercepts : items whose intercepts stay group-specific under
                      scalar invariance (partial invariance)

    Identification: factor variances are fixed to 1 and latent means to 0
    in the first group; under metric (scalar) invariance the other groups'
    factor variances (and latent means) are estimated.
    """
    datasets = data if isinstance(data, (list, tuple)) else [data]
    moments = [_moments(dset) for dset in datasets]
    pattern = np.asarray(pattern, bool)
    if pattern.ndim == 1:
        pattern = pattern[:, None]
    p, k = pattern.shape
    G = len(moments)
    if G == 1:
        level = "configural"
    N = sum(n for *_, n in moments)
    spec, ix = _build(p, k, moments, pattern, list(error_cov), level,
                      free_intercepts, factor_corr and k > 1)
    x0 = np.array(spec.start) if start is None else np.asarray(start, float)
    bounds = [(lo, None) for lo in spec.lower]
    res = optimize.minimize(_objective, x0, args=(ix, moments, p, k, N),
                            jac=True, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 10000, "ftol": 1e-15,
                                     "gtol": 1e-9, "maxcor": 30})
    theta = res.x
    Fmin, _ = _objective(theta, ix, moments, p, k, N)
    chi2 = max(N * Fmin, 0.0)
    n_par = len(theta)
    n_mom = G * (p * (p + 1) // 2 + p)
    df = n_mom - n_par
    # Baseline: independent items, free means and variances
    chi2_0 = N * sum((n / N) * (np.log(np.diag(S)).sum() - np.linalg.slogdet(S)[1])
                     for S, _, n in moments)
    df_0 = G * (p * (p - 1) // 2)
    pval = float(stats.chi2.sf(chi2, df)) if df > 0 else np.nan
    num = max(chi2 - df, 0.0)
    den = max(chi2_0 - df_0, chi2 - df, 0.0)
    cfi = 1.0 - num / den if den > 0 else 1.0
    tli = ((chi2_0 / df_0) - (chi2 / df)) / ((chi2_0 / df_0) - 1) if df > 0 else 1.0
    rmsea = np.sqrt(num / (df * (N - G))) * np.sqrt(G) if df > 0 else 0.0
    groups, srmr_acc = [], 0.0
    for g, (S, m, n) in enumerate(moments):
        L, P, T, nu, ka = _matrices(theta, ix, g, p, k)
        est = GroupEstimates(L, P, T, nu, ka)
        groups.append(est)
        sd = np.sqrt(np.diag(S))
        R = (S - est.Sigma) / np.outer(sd, sd)
        tri = np.tril_indices(p)
        srmr_acc += (n / N) * np.sqrt(np.mean(R[tri] ** 2))
    return CFAResult(groups, float(chi2), int(df), pval, float(cfi), float(tli),
                     float(rmsea), float(srmr_acc), n_par, N,
                     bool(res.success), level, float(chi2_0), int(df_0), theta)
