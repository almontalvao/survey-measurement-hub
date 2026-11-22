"""Known-truth tests: every estimator is graded against a value fixed by
the data-generating process (or by an exact identity), never against its
own output."""

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from surveyhub.simulate import (simulate_factor_data, true_composite_reliability,
                                simple_structure, bank_2pl, bank_grm,
                                simulate_irt, simulate_dif, make_population,
                                opt_in_sample, simulate_sensitive,
                                simulate_conjoint, true_amce)
from surveyhub.reliability import (cronbach_alpha, alpha_from_cov, omega,
                                   spearman_brown, split_half)
from surveyhub.factor import (efa, parallel_analysis, congruence, fit_cfa,
                              invariance_sequence, locate_noninvariant_intercepts)
from surveyhub.irt import (fit_2pl, fit_grm, item_information, mantel_haenszel,
                           logistic_dif, simulate_cat)
from surveyhub.weighting import (rake, poststratify, kish_deff, effective_n,
                                 weighted_mean, linearized_se, jackknife,
                                 pseudo_weights, dummies)
from surveyhub.experiments import (list_experiment, list_design_test,
                                   randomized_response, direct_estimate, amce)

THREE_FACTOR = simple_structure([4, 4, 4], [.8, .7, .6, .75, .7, .65, .8, .6,
                                            .55, .7, .75, .8])
PHI3 = np.array([[1, .4, .3], [.4, 1, .5], [.3, .5, 1]])


# -------------------------------------------------------------- reliability

def test_alpha_equals_reliability_under_tau_equivalence():
    lam = np.full((6, 1), 0.7)
    theta = np.full(6, 0.51)
    Sigma = lam @ lam.T + np.diag(theta)
    truth = true_composite_reliability(lam, None, theta)
    assert abs(alpha_from_cov(Sigma) - truth) < 1e-12


def test_alpha_understates_congeneric_reliability_omega_recovers_it():
    lam = np.array([.9, .8, .7, .5, .4, .3])[:, None]
    d = simulate_factor_data(20000, lam, seed=1)
    truth = true_composite_reliability(lam, None, d.Theta)
    assert alpha_from_cov(d.Sigma) < truth - 0.015          # population alpha is biased
    assert abs(omega(d.X).omega - truth) < 0.01


def test_correlated_errors_inflate_alpha_modeled_omega_is_right():
    lam = np.full((6, 1), 0.6)
    Theta = np.diag(np.full(6, 0.64))
    Theta[0, 1] = Theta[1, 0] = 0.3
    Theta[2, 3] = Theta[3, 2] = 0.3
    d = simulate_factor_data(20000, lam, Theta=Theta, seed=2)
    truth = true_composite_reliability(lam, None, Theta)
    assert alpha_from_cov(d.Sigma) > truth + 0.02
    om = omega(d.X, error_cov=[(0, 1), (2, 3)]).omega
    assert abs(om - truth) < 0.01


def test_spearman_brown_identities():
    assert abs(spearman_brown(0.5, 2) - 2 / 3) < 1e-12
    # shortening then lengthening returns the original reliability
    r = 0.83
    assert abs(spearman_brown(spearman_brown(r, 0.5), 2) - r) < 1e-12


def test_split_half_close_to_truth_for_parallel_items():
    lam = np.full((8, 1), 0.7)
    d = simulate_factor_data(20000, lam, seed=3)
    truth = true_composite_reliability(lam, None, d.Theta)
    assert abs(split_half(d.X) - truth) < 0.01


# ------------------------------------------------------------------ factor

def test_parallel_analysis_finds_three_factors():
    d = simulate_factor_data(1500, THREE_FACTOR, PHI3, seed=4)
    assert parallel_analysis(d.X, n_sims=100, seed=0).n_factors == 3


def test_efa_recovers_loadings_up_to_rotation():
    d = simulate_factor_data(3000, THREE_FACTOR, PHI3, seed=5)
    res = efa(d.X, 3, rotation="promax")
    c, aligned = congruence(res.loadings, THREE_FACTOR)
    assert np.all(c > 0.98)
    assert np.max(np.abs(aligned - THREE_FACTOR)) < 0.1


def test_cfa_on_population_covariance_is_exact():
    d = simulate_factor_data(10, THREE_FACTOR, PHI3, seed=6)
    fit = fit_cfa({"S": d.Sigma, "mean": d.mu, "n": 1000}, THREE_FACTOR != 0)
    assert fit.chi2 < 1e-6
    assert fit.df == 12 * 13 // 2 - (12 + 12 + 3)            # 51
    assert np.max(np.abs(fit.Lambda - THREE_FACTOR)) < 1e-4
    assert np.max(np.abs(fit.Phi - PHI3)) < 1e-4


def test_cfa_sample_recovery_and_fit():
    d = simulate_factor_data(3000, THREE_FACTOR, PHI3, seed=7)
    fit = fit_cfa(d.X, THREE_FACTOR != 0)
    assert fit.converged and fit.cfi > 0.99 and fit.rmsea < 0.03
    assert np.max(np.abs(fit.Lambda - THREE_FACTOR)) < 0.08


def test_misspecified_cfa_fits_badly():
    d = simulate_factor_data(2000, THREE_FACTOR, PHI3, seed=8)
    one_factor = fit_cfa(d.X, np.ones((12, 1), bool))
    assert one_factor.pvalue < 1e-6 and one_factor.cfi < 0.9


def _two_groups(shift_items, shift=0.45, seed=0):
    lam = np.array([.8, .7, .6, .75, .7, .65, .6, .7])[:, None]
    nu2 = np.zeros(8)
    nu2[list(shift_items)] = shift
    g1 = simulate_factor_data(1500, lam, seed=seed)
    g2 = simulate_factor_data(1500, lam, nu=nu2, kappa=[0.25], Phi=[[1.1]],
                              Theta=g1.Theta, seed=seed + 1)
    return g1, g2


def test_invariance_holds_when_nothing_is_planted():
    g1, g2 = _two_groups([], seed=10)
    steps = invariance_sequence([g1.X, g2.X], np.ones((8, 1), bool))
    assert steps[1].p_diff > 0.01 and steps[2].p_diff > 0.01
    assert steps[2].delta_cfi > -0.01
    # scalar model recovers the true latent mean difference
    assert abs(steps[2].fit.groups[1].kappa[0] - 0.25) < 0.08


def test_scalar_invariance_rejected_and_shifted_items_located():
    g1, g2 = _two_groups([1, 5], seed=12)
    steps = invariance_sequence([g1.X, g2.X], np.ones((8, 1), bool))
    assert steps[1].p_diff > 0.01                         # loadings still equal
    assert steps[2].p_diff < 1e-6                         # intercepts are not
    freed, fit, _ = locate_noninvariant_intercepts([g1.X, g2.X], np.ones((8, 1), bool))
    assert sorted(freed) == [1, 5]
    assert abs(fit.groups[1].kappa[0] - 0.25) < 0.08     # partial invariance fixes the mean
    biased = steps[2].fit.groups[1].kappa[0]
    assert biased > fit.groups[1].kappa[0] + 0.05         # full scalar is biased upward


# --------------------------------------------------------------------- IRT

def test_2pl_information_peaks_at_difficulty():
    a, b = 1.7, 0.4
    grid = np.linspace(-4, 4, 8001)
    info = item_information([a], [[-a * b]], grid)[0]
    assert abs(info.max() - a ** 2 / 4) < 1e-9
    assert abs(grid[info.argmax()] - b) < 1e-3


def test_2pl_parameter_recovery():
    bank = bank_2pl(30, seed=1)
    d = simulate_irt(bank, 3000, seed=2)
    m = fit_2pl(d.X)
    assert m.converged
    assert np.corrcoef(m.a, bank.a)[0, 1] > 0.95
    assert np.corrcoef(m.b[:, 0], bank.b[:, 0])[0, 1] > 0.99
    assert np.sqrt(np.mean((m.a - bank.a) ** 2)) < 0.12
    assert np.sqrt(np.mean((m.b - bank.b) ** 2)) < 0.12


def test_grm_parameter_recovery_and_scoring():
    bank = bank_grm(10, 5, seed=3)
    d = simulate_irt(bank, 3000, seed=4)
    m = fit_grm(d.X)
    assert np.corrcoef(m.a, bank.a)[0, 1] > 0.95
    assert np.sqrt(np.mean((m.b - bank.b) ** 2)) < 0.1
    theta_hat, sd = m.eap(d.X)
    assert np.corrcoef(theta_hat, d.theta)[0, 1] > 0.93
    # posterior SDs are calibrated: they match the actual error
    assert abs(np.sqrt(np.mean((theta_hat - d.theta) ** 2)) - np.sqrt(np.mean(sd ** 2))) < 0.03


def test_mantel_haenszel_flags_exactly_the_planted_dif_items():
    bank = bank_2pl(40, a_range=(1.0, 2.2), seed=5)
    planted = [3, 11, 22, 35]
    d = simulate_dif(bank, 1500, 1500, planted, [0.6, 0.7, -0.6, 0.8],
                     focal_mean=-0.4, seed=6)
    mh = mantel_haenszel(d.X, d.group)
    assert sorted(np.flatnonzero(mh.flagged)) == planted
    # sign: items made harder for the focal group have negative delta
    assert mh.delta[3] < 0 and mh.delta[22] > 0


def test_no_dif_means_no_flags_despite_impact():
    bank = bank_2pl(40, a_range=(1.0, 2.2), seed=7)
    d = simulate_dif(bank, 1500, 1500, [], [], focal_mean=-0.5, seed=8)
    assert mantel_haenszel(d.X, d.group).flagged.sum() == 0
    lg = logistic_dif(d.X, d.group)
    assert (lg.p_total * 40 < 0.05).sum() == 0


def test_cat_reaches_precision_target_with_fewer_items():
    bank = bank_2pl(60, a_range=(1.2, 2.5), b_range=(-2.5, 2.5), seed=9)
    theta = np.random.default_rng(10).normal(size=400)
    cat = simulate_cat(bank, theta, se_stop=0.3, seed=11)
    stopped = cat.n_items < 60
    assert np.all(cat.se[stopped] <= 0.3)
    assert cat.n_items.mean() < 25
    rmse = np.sqrt(np.mean((cat.theta_hat - theta) ** 2))
    assert rmse < 0.36                                   # close to the SE target


# --------------------------------------------------------------- weighting

def _margins(pop, idx):
    cov = pop.covariates()
    return {v: cov[v][idx] for v in cov}


def test_raking_matches_every_margin():
    pop = make_population(50_000, seed=1)
    idx = opt_in_sample(pop, 1500, seed=2)
    targets = {v: pop.margins(v) for v in pop.covariates()}
    r = rake(_margins(pop, idx), targets)
    assert r.converged and r.max_margin_error < 1e-8
    assert abs(r.weights.sum() - pop.N) < 1e-4


def test_poststratification_reproduces_cell_totals():
    pop = make_population(50_000, seed=3)
    idx = opt_in_sample(pop, 2000, seed=4)
    cell_pop = pop.age * 3 + pop.edu
    totals = np.bincount(cell_pop, minlength=12).astype(float)
    w = poststratify(cell_pop[idx], totals)
    assert np.allclose(np.bincount(cell_pop[idx], weights=w, minlength=12), totals)


def test_raking_removes_selection_bias_on_average():
    pop = make_population(100_000, seed=5)
    targets = {v: pop.margins(v) for v in pop.covariates()}
    raw, raked = [], []
    for s in range(40):
        idx = opt_in_sample(pop, 1500, seed=100 + s)
        y = pop.y[idx]
        raw.append(y.mean())
        raked.append(weighted_mean(y, rake(_margins(pop, idx), targets).weights))
    assert abs(np.mean(raw) - pop.truth) > 0.025          # the opt-in sample is biased
    assert abs(np.mean(raked) - pop.truth) < 0.008        # raking removes it


def test_pseudo_weights_sum_to_population_and_reduce_bias():
    pop = make_population(100_000, seed=6)
    rng = np.random.default_rng(7)
    ests = []
    for s in range(20):
        idx = opt_in_sample(pop, 1500, seed=200 + s)
        ref = rng.choice(pop.N, 3000, replace=False)
        w = pseudo_weights(dummies(_margins(pop, idx)), dummies(_margins(pop, ref)),
                           np.full(3000, pop.N / 3000))
        ests.append(weighted_mean(pop.y[idx], w))
        assert abs(w.sum() / pop.N - 1) < 0.1
    assert abs(np.mean(ests) - pop.truth) < 0.01


def test_kish_and_variance_identities():
    w = np.ones(500)
    assert kish_deff(w) == pytest.approx(1.0) and effective_n(w) == pytest.approx(500)
    w2 = np.r_[np.ones(250), 3 * np.ones(250)]
    assert kish_deff(w2) == pytest.approx(500 * (250 + 2250) / 1000 ** 2)
    y = np.random.default_rng(8).normal(size=300)
    srs = y.std(ddof=1) / np.sqrt(300)
    assert linearized_se(y, np.ones(300)) == pytest.approx(srs, rel=1e-12)
    assert jackknife(lambda k: y[k].mean(), 300, n_groups=300).se == pytest.approx(srs, rel=1e-10)
    # weighted mean is invariant to rescaling the weights
    assert weighted_mean(y, w2[:300] * 7) == pytest.approx(weighted_mean(y, w2[:300]))


# --------------------------------------------------------------- experiments

def test_sensitive_designs_recover_true_prevalence():
    z_list, z_rr, direct = [], [], []
    for s in range(30):
        sv = simulate_sensitive(prevalence=0.22, admit_rate=0.55, seed=300 + s)
        le = list_experiment(sv.list_count, sv.list_treat)
        rr = randomized_response(sv.rr_answer, **{k: sv.rr_design[k] for k in ("p_truth", "p_yes")})
        z_list.append((le.estimate - 0.22) / le.se)
        z_rr.append((rr.estimate - 0.22) / rr.se)
        direct.append(direct_estimate(sv.direct).estimate)
    assert abs(np.mean(z_list)) < 0.5 and abs(np.mean(z_rr)) < 0.5
    assert np.mean(np.abs(z_list) < 1.96) > 0.85 and np.mean(np.abs(z_rr) < 1.96) > 0.85
    assert np.mean(direct) < 0.15                       # 0.22 * 0.55 = 0.121


def test_list_design_test_quiet_without_design_effect_and_fires_with_one():
    sv = simulate_sensitive(n_list=4000, seed=9)
    assert not list_design_test(sv.list_count, sv.list_treat, 4).design_effect
    # Design effect: in the long list, respondents who hold the trait
    # deflate their control-item count by one to hide it
    count = sv.list_count.copy()
    rng = np.random.default_rng(10)
    hide = (sv.list_treat == 1) & (count >= 2) & (rng.random(len(count)) < 0.35)
    count[hide] -= 1
    assert list_design_test(count, sv.list_treat, 4).design_effect


def test_conjoint_amce_recovers_exact_truth():
    truth = true_amce()
    cj = simulate_conjoint(n_respondents=1500, seed=11)
    res = amce(cj.levels, cj.chosen, cj.respondent, cj.attributes)
    z = np.array([(e - truth[lab]) / s for lab, e, s in
                  zip(res.labels, res.estimate, res.se_cluster)])
    assert np.all(np.abs(z) < 3.0)
    assert np.mean(np.abs(z) < 1.96) >= 0.75
