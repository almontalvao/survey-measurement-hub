"""Build and execute the case-study notebooks without a Jupyter dependency.

Each notebook is defined as a list of (cell_type, source) tuples. Code cells
are executed in a shared namespace; stdout and matplotlib figures are
captured and embedded as real notebook outputs, so the committed .ipynb
files render fully on GitHub.

Usage:  python tools/build_notebooks.py [1|2|3|4|5|6|all]
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_notebook(cells: list[tuple[str, str]], out_path: pathlib.Path) -> None:
    ns: dict = {}
    nb_cells = []
    exec_count = 0
    for kind, src in cells:
        src = src.strip("\n")
        if kind == "md":
            nb_cells.append({"cell_type": "markdown", "metadata": {},
                             "source": src.splitlines(keepends=True)})
            continue
        exec_count += 1
        outputs = []
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, f"<cell {exec_count}>", "exec"), ns)
        text = buf.getvalue()
        if text:
            outputs.append({"output_type": "stream", "name": "stdout",
                            "text": text.splitlines(keepends=True)})
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(b.getvalue()).decode()}})
        plt.close("all")
        nb_cells.append({"cell_type": "code", "execution_count": exec_count,
                         "metadata": {}, "source": src.splitlines(keepends=True),
                         "outputs": outputs})
    nb = {"cells": nb_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10"}},
          "nbformat": 4, "nbformat_minor": 5}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"built {out_path} ({exec_count} code cells)")


SETUP = """
import sys, pathlib, warnings
warnings.filterwarnings("ignore")
_root = pathlib.Path.cwd()
if _root.name == "notebooks":
    _root = _root.parent
sys.path.insert(0, str(_root / "src"))
FIG_DIR = _root / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9, 4.5), "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
# Categorical slots (validated palette), plus ink for the known truth
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#8a8984"
"""


# ===========================================================================
# 01 Reliability: alpha vs omega (HR engagement survey)
# ===========================================================================

def nb01() -> list[tuple[str, str]]:
    return [
        ("md", """
# 01 Reliability: When Cronbach's Alpha Misleads

**Setting:** a six-item employee engagement scale (people analytics).
**Methods:** Cronbach's alpha, McDonald's omega (from a one-factor CFA),
omega with modeled error covariances, item diagnostics, bootstrap CIs.
**Ground truth:** the data come from a factor model with known loadings
and error variances, so the true reliability of the sum score is known
exactly: the share of its variance that comes from the common factor.

Alpha is the default reliability number in almost every survey report.
It equals the true reliability only under two assumptions: every item
loads equally on the trait (tau-equivalence), and item errors are
uncorrelated. This notebook breaks each assumption in turn and shows
which estimator still lands on the truth.
"""),
        ("code", SETUP),
        ("code", """
from surveyhub.simulate import simulate_factor_data, true_composite_reliability
from surveyhub.reliability import (cronbach_alpha, alpha_from_cov, omega,
                                   item_total, alpha_if_deleted, bootstrap_ci)
"""),
        ("md", """
## 1. Three versions of the same scale

* **Equal loadings.** Six items that measure engagement equally well.
* **Unequal loadings.** Two excellent items, two good ones, and two weak
  ones (a common outcome when items are written by committee).
* **Correlated errors.** Equal loadings, but two pairs of items share
  wording ("I feel energized at work" / "I feel energized by my work"),
  so their errors correlate. That shared variance is not engagement.

For each, compare the population value of alpha (computed from the true
covariance matrix, so no sampling noise) with the true reliability.
"""),
        ("code", """
lam_eq = np.full((6, 1), 0.7)
lam_un = np.array([.9, .85, .75, .55, .4, .3])[:, None]
lam_ce = np.full((6, 1), 0.6)
theta_ce = np.diag(np.full(6, 0.64))
theta_ce[0, 1] = theta_ce[1, 0] = 0.3
theta_ce[2, 3] = theta_ce[3, 2] = 0.3

scenarios = {
    "Equal loadings":    dict(Lambda=lam_eq, Theta=np.diag(1 - lam_eq[:, 0] ** 2), error_cov=[]),
    "Unequal loadings":  dict(Lambda=lam_un, Theta=np.diag(1 - lam_un[:, 0] ** 2), error_cov=[]),
    "Correlated errors": dict(Lambda=lam_ce, Theta=theta_ce, error_cov=[(0, 1), (2, 3)]),
}
print(f"{'scenario':<20s} {'true reliability':>17s} {'population alpha':>17s} {'alpha bias':>11s}")
for name, s in scenarios.items():
    Sigma = s["Lambda"] @ s["Lambda"].T + s["Theta"]
    s["truth"] = true_composite_reliability(s["Lambda"], None, s["Theta"])
    a = alpha_from_cov(Sigma)
    print(f"{name:<20s} {s['truth']:17.3f} {a:17.3f} {a - s['truth']:+11.3f}")
"""),
        ("md", """
With infinite data, alpha is exact in the first case, too low in the
second (it treats the weak items as if they were as good as the strong
ones), and too high in the third (it counts the shared wording as
engagement). The direction of the error depends on which assumption
fails, so a high alpha is not reassuring on its own.

## 2. Sampling distributions at a realistic sample size

Draw 300 surveys of 400 employees from each scenario and compute alpha,
omega from a one-factor CFA, and omega from a CFA that also models the
correlated error pairs.
"""),
        ("code", """
R, n = 300, 400
rng = np.random.default_rng(1)
results = {}
for name, s in scenarios.items():
    rows = []
    for r in range(R):
        d = simulate_factor_data(n, s["Lambda"], Theta=s["Theta"], seed=rng.integers(1 << 31))
        a = cronbach_alpha(d.X)
        om = omega(d.X).omega
        om_ce = omega(d.X, error_cov=s["error_cov"]).omega if s["error_cov"] else om
        rows.append((a, om, om_ce))
    results[name] = np.array(rows)

labels = ["alpha", "omega", "omega + error cov"]
print(f"{'scenario':<20s} {'estimator':<18s} {'mean':>7s} {'bias':>7s} {'RMSE':>7s}")
for name, s in scenarios.items():
    for k, lab in enumerate(labels):
        est = results[name][:, k]
        print(f"{name:<20s} {lab:<18s} {est.mean():7.3f} {est.mean() - s['truth']:+7.3f} "
              f"{np.sqrt(np.mean((est - s['truth']) ** 2)):7.3f}")
    print()
"""),
        ("code", """
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
colors = [BLUE, ORANGE, AQUA]
for ax, (name, s) in zip(axes, scenarios.items()):
    est = results[name]
    for k, lab in enumerate(labels):
        lo, mid, hi = np.percentile(est[:, k], [2.5, 50, 97.5])
        ax.plot([k, k], [lo, hi], color=colors[k], lw=2, solid_capstyle="round")
        ax.plot(k, est[:, k].mean(), "o", color=colors[k], ms=8,
                markeredgecolor="white", markeredgewidth=2)
    ax.axhline(s["truth"], color=INK, ls="--", lw=1.2)
    ax.text(2.85, s["truth"], "truth", ha="right", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["alpha", "omega", "omega +\\nerror cov"])
    ax.set_xlim(-0.5, 2.9)
    ax.set_title(name, fontsize=11)
axes[0].set_ylabel("reliability estimate (mean, 95% of samples)")
fig.suptitle("Alpha vs omega against the known reliability (300 samples of n = 400)", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "reliability_alpha_omega.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
        ("md", """
## 3. Item diagnostics on one sample

The classic item table: corrected item-total correlations and alpha if
the item is deleted. In the unequal-loadings scale the weak items show
up clearly. A bootstrap CI conveys how precise the reliability estimate
is with 400 respondents.
"""),
        ("code", """
s = scenarios["Unequal loadings"]
d = simulate_factor_data(400, s["Lambda"], Theta=s["Theta"], seed=7)
it = item_total(d.X)
aid = alpha_if_deleted(d.X)
print(f"alpha (all six items) = {cronbach_alpha(d.X):.3f}\\n")
print(f"{'item':<6s} {'true loading':>13s} {'item-total r':>13s} {'alpha if deleted':>17s}")
for j in range(6):
    print(f"Q{j + 1:<5d} {s['Lambda'][j, 0]:13.2f} {it[j]:13.3f} {aid[j]:17.3f}")

est, lo, hi = bootstrap_ci(lambda X: omega(X).omega, d.X, n_boot=500, seed=3)
print(f"\\nomega = {est:.3f}, 95% bootstrap CI [{lo:.3f}, {hi:.3f}]  (truth {s['truth']:.3f})")
"""),
        ("md", """
## Takeaways

* Alpha is a lower bound on reliability when loadings differ and an
  upper bound when errors correlate. Real scales usually have both
  problems, so its error can go either way.
* Omega from a one-factor CFA recovers the truth when loadings differ.
  When errors correlate it is only right if the error covariances are
  modeled, which takes substantive knowledge of the items (shared
  wording, shared stems, reversed items).
* Report omega with a bootstrap CI, and check the one-factor model's fit
  before trusting any single reliability number.
"""),
    ]


# ===========================================================================
# 02 Factor structure and measurement invariance (two language versions)
# ===========================================================================

def nb02() -> list[tuple[str, str]]:
    return [
        ("md", """
# 02 Factor Structure and Measurement Invariance

**Setting:** a 12-item customer satisfaction survey fielded in English
and Spanish. The business question: are Spanish-speaking customers more
satisfied with the product?
**Methods:** parallel analysis, exploratory factor analysis (principal
axis + promax), confirmatory factor analysis by maximum likelihood,
multi-group invariance testing (configural, metric, scalar), and a
search for partial invariance.
**Ground truth:** three factors with known loadings. The Spanish group
is truly 0.3 SD more satisfied with service and identical on the other
two factors. Two translated items, one about the product and one about
value, are simply easier to agree with (higher intercepts), which is a
measurement artifact, not satisfaction.
"""),
        ("code", SETUP),
        ("code", """
from surveyhub.simulate import simulate_factor_data, simple_structure
from surveyhub.factor import (parallel_analysis, efa, congruence, fit_cfa,
                              invariance_sequence, locate_noninvariant_intercepts)

items = ["Product works as described", "Product is reliable", "Product is well made",
         "Product meets my needs", "Staff are helpful", "Issues resolved quickly",
         "Easy to reach support", "Staff are knowledgeable", "Fair price",
         "Worth what I pay", "Better value than rivals", "Would buy again"]
factors = ["Product", "Service", "Value"]
Lam = simple_structure([4, 4, 4], [.8, .75, .7, .65, .8, .7, .75, .6, .7, .8, .65, .6])
Phi = np.array([[1, .5, .4], [.5, 1, .45], [.4, .45, 1]])

en = simulate_factor_data(1200, Lam, Phi, seed=21)
nu_es = np.zeros(12)
nu_es[[1, 9]] = [0.6, 0.5]               # translation artifact: one product, one value item
true_kappa = np.array([0.0, 0.3, 0.0])  # the real difference: service only
es = simulate_factor_data(1200, Lam, Phi, nu=nu_es, kappa=true_kappa,
                          Theta=en.Theta, seed=22)
print("English n =", len(en.X), "  Spanish n =", len(es.X))
"""),
        ("md", """
## 1. How many factors? Parallel analysis

Keep a factor only if its eigenvalue beats what random, uncorrelated data
of the same size would produce (the 95th percentile over 200 simulated
datasets). This is far more reliable than the "eigenvalue above 1" rule.
"""),
        ("code", """
pa = parallel_analysis(en.X, n_sims=200, seed=0)
print("factors retained:", pa.n_factors)
print("observed eigenvalues :", np.round(pa.observed[:5], 2))
print("random 95th pct      :", np.round(pa.threshold[:5], 2))
"""),
        ("md", """
## 2. Exploratory factor analysis

Principal axis factoring with an oblique (promax) rotation, since
satisfaction factors are expected to correlate. Tucker's congruence
compares each estimated factor with the true one (above 0.95 means the
factors are essentially the same).
"""),
        ("code", """
res = efa(en.X, 3, rotation="promax")
c, aligned = congruence(res.loadings, Lam)
print(f"{'item':<28s}" + "".join(f"{f:>9s}" for f in factors) + "   true")
for i, name in enumerate(items):
    row = "".join(f"{v:9.2f}" for v in aligned[i])
    print(f"{name:<28s}{row}   {Lam[i].max():.2f} on {factors[Lam[i].argmax()]}")
print("\\nTucker congruence per factor:", np.round(c, 3))
"""),
        ("md", """
## 3. Confirmatory factor analysis

Now fix the structure (each item loads only on its factor) and fit by
maximum likelihood. Fit indices: CFI and TLI near 1, RMSEA below 0.05
and SRMR below 0.08 indicate good fit.
"""),
        ("code", """
pattern = Lam != 0
cfa_en = fit_cfa(en.X, pattern)
print("English CFA:", cfa_en.summary())
print("loading error vs truth (max abs):", np.abs(cfa_en.Lambda - Lam).max().round(3))
print("factor correlations:\\n", np.round(cfa_en.Phi, 2))
"""),
        ("md", """
## 4. Invariance testing

Each step adds equality constraints across languages:

| Level | Constrained equal | What it licenses |
|---|---|---|
| Configural | same pattern | same constructs |
| Metric | + loadings | comparing relationships (correlations, slopes) |
| Scalar | + intercepts | comparing means |

A significant chi-square difference or a CFI drop larger than 0.01
means the constraint does not hold.
"""),
        ("code", """
steps = invariance_sequence([en.X, es.X], pattern)
print(f"{'level':<11s} {'chi2':>8s} {'df':>4s} {'CFI':>6s} {'RMSEA':>6s} {'d chi2':>8s} {'d df':>5s} {'p':>9s} {'d CFI':>7s}")
for s in steps:
    print(f"{s.level:<11s} {s.fit.chi2:8.1f} {s.fit.df:4d} {s.fit.cfi:6.3f} {s.fit.rmsea:6.3f} "
          f"{s.delta_chi2:8.1f} {s.delta_df:5d} {s.p_diff:9.2g} {s.delta_cfi:+7.3f}")
"""),
        ("md", """
Metric invariance holds; scalar does not. Before comparing means, find
which intercepts differ: release one at a time, keep the release that
improves fit most, and stop when no release is significant after a
Bonferroni correction.
"""),
        ("code", """
freed, partial, log = locate_noninvariant_intercepts([en.X, es.X], pattern)
for step in log:
    print(f"release '{items[step['item']]}': d chi2 = {step['delta_chi2']:7.1f}, p = {step['p']:.2g}")
print("\\nnon-invariant intercepts found:", [items[i] for i in freed])
print("planted:", [items[i] for i in np.flatnonzero(nu_es)])
print("partial scalar model:", partial.summary())
"""),
        ("md", """
## 5. The business question: who is more satisfied?

Three ways to compare languages, in SD units of the English group:

* **Naive:** difference in each factor's sum score (what a dashboard shows)
* **Full scalar:** latent mean difference with all intercepts equal
* **Partial scalar:** latent mean difference with the two artifact items freed
"""),
        ("code", """
sums_en = np.column_stack([en.X[:, pattern[:, f]].sum(1) for f in range(3)])
sums_es = np.column_stack([es.X[:, pattern[:, f]].sum(1) for f in range(3)])
naive = (sums_es.mean(0) - sums_en.mean(0)) / sums_en.std(0, ddof=1)
full = steps[2].fit.groups[1].kappa
part = partial.groups[1].kappa
print(f"{'factor':<9s} {'truth':>7s} {'naive':>7s} {'full scalar':>12s} {'partial':>8s}")
for f in range(3):
    print(f"{factors[f]:<9s} {true_kappa[f]:7.2f} {naive[f]:7.2f} {full[f]:12.2f} {part[f]:8.2f}")
"""),
        ("code", """
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1, 1.4]})
k = np.arange(1, 13)
ax1.plot(k, pa.observed, "o-", color=BLUE, lw=2, ms=7, label="observed")
ax1.plot(k, pa.threshold, "s--", color=MUTED, lw=1.5, ms=5, label="random data, 95th pct")
ax1.set_xlabel("component")
ax1.set_ylabel("eigenvalue")
ax1.set_title("Parallel analysis keeps 3 factors", fontsize=11)
ax1.legend(frameon=False)

y = np.arange(3)
for off, vals, col, lab in [(-0.2, naive, BLUE, "naive sum scores"),
                            (0.0, full, ORANGE, "full scalar model"),
                            (0.2, part, AQUA, "partial scalar model")]:
    ax2.barh(y + off, vals, height=0.18, color=col, label=lab)
for f in range(3):
    ax2.plot(true_kappa[f], f, "D", color=INK, ms=9, markeredgecolor="white",
             markeredgewidth=1.5, zorder=5)
ax2.plot([], [], "D", color=INK, ms=8, label="truth")
ax2.set_yticks(y)
ax2.set_yticklabels(factors)
ax2.invert_yaxis()
ax2.set_xlabel("Spanish minus English (SD units)")
ax2.set_title("Only the partial invariance model recovers the truth", fontsize=11)
ax2.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "invariance_latent_means.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
        ("md", """
## Takeaways

* Parallel analysis, EFA and CFA recover the three-factor structure from
  1,200 respondents.
* The naive dashboard comparison reports that Spanish-speaking customers
  are happier with the product and its value. They are not: two
  translated items are just easier to agree with. Forcing equal
  intercepts spreads that artifact into the latent means as well.
* Testing invariance, locating the non-invariant intercepts, and freeing
  them recovers the real finding: a service difference of about 0.3 SD
  and nothing else. Test invariance before comparing groups on any
  multi-item scale, especially across languages, modes, or cultures.
* Partial invariance needs most items on each factor to be invariant.
  If half of a factor's items shift, the data cannot tell which half is
  the anchor, and any search will pick one arbitrarily. Plan anchor items
  (carefully translated, pretested) into the questionnaire.
"""),
    ]


# ===========================================================================
# 03 Item response theory (patient-reported outcomes)
# ===========================================================================

def nb03() -> list[tuple[str, str]]:
    return [
        ("md", """
# 03 Item Response Theory for a Patient-Reported Outcome

**Setting:** a patient-reported fatigue item bank (healthcare).
**Methods:** graded response model and 2PL, both fit by marginal maximum
likelihood (Bock-Aitkin EM, written from scratch); EAP scoring; item and
test information; DIF detection with Mantel-Haenszel (with
purification) and logistic regression.
**Ground truth:** item parameters are drawn and fixed before simulation,
and DIF is planted on four known items.

Classical test theory gives one reliability for the whole scale. IRT
describes each item: how sharply it separates people (discrimination,
a) and where on the trait it does so (thresholds, b). That item-level
view is what makes short forms, adaptive tests, and fairness checks
possible.
"""),
        ("code", SETUP),
        ("code", """
from surveyhub.simulate import bank_grm, bank_2pl, simulate_irt, simulate_dif
from surveyhub.irt import (fit_grm, fit_2pl, category_probs, item_information,
                           mantel_haenszel, logistic_dif)
"""),
        ("md", """
## 1. A 10-item short form with five response options

"In the past 7 days, how often did you feel tired?" Never, Rarely,
Sometimes, Often, Always. Samejima's graded response model:

P(X >= k | theta) = logistic(a (theta - b_k)),   k = 1..4

Simulate 2,000 patients and fit the model.
"""),
        ("code", """
grm_true = bank_grm(10, 5, seed=31)
grm_data = simulate_irt(grm_true, 2000, seed=32)
grm = fit_grm(grm_data.X)
print(f"EM converged: {grm.converged} after {grm.n_iter} iterations")
print(f"discrimination: r = {np.corrcoef(grm.a, grm_true.a)[0, 1]:.3f}, "
      f"RMSE = {np.sqrt(np.mean((grm.a - grm_true.a) ** 2)):.3f}")
print(f"thresholds:     r = {np.corrcoef(grm.b.ravel(), grm_true.b.ravel())[0, 1]:.4f}, "
      f"RMSE = {np.sqrt(np.mean((grm.b - grm_true.b) ** 2)):.3f}\\n")
print(f"{'item':<5s} {'a true':>7s} {'a est':>7s}   thresholds true -> est")
for j in range(10):
    tb = " ".join(f"{v:5.2f}" for v in grm_true.b[j])
    eb = " ".join(f"{v:5.2f}" for v in grm.b[j])
    print(f"{j + 1:<5d} {grm_true.a[j]:7.2f} {grm.a[j]:7.2f}   {tb}  ->  {eb}")
"""),
        ("code", """
theta_hat, post_sd = grm.eap(grm_data.X)
print(f"EAP scores vs true trait: r = {np.corrcoef(theta_hat, grm_data.theta)[0, 1]:.3f}")
print(f"actual RMSE {np.sqrt(np.mean((theta_hat - grm_data.theta) ** 2)):.3f} "
      f"vs average posterior SD {np.sqrt(np.mean(post_sd ** 2)):.3f} (calibrated uncertainty)")

grid = np.linspace(-4, 4, 161)
info = grm.test_info(grid)
for t in (-2, -1, 0, 1, 2):
    q = np.argmin(np.abs(grid - t))
    print(f"theta = {t:+d}: test information {info[q]:5.2f}, SE {1 / np.sqrt(info[q]):.2f}")
"""),
        ("md", """
Precision varies along the trait: the short form measures moderate
fatigue well and the extremes less well. A single reliability
coefficient hides this.

## 2. A 40-item symptom checklist and differential item functioning

Yes/no items, 2PL model. Patients under 65 (reference) and 65 and over
(focal); the older group truly has more fatigue (impact, which is fine).
On four items, older patients at the same fatigue level answer
differently: for example "I had to limit my social activity" may be
endorsed more by older patients for reasons unrelated to fatigue. That
is DIF, and it biases group comparisons.
"""),
        ("code", """
bank40 = bank_2pl(40, a_range=(1.0, 2.2), seed=33)
planted = [4, 13, 21, 33]
dif_data = simulate_dif(bank40, 1500, 1500, planted, [0.6, -0.7, 0.6, -0.8],
                        focal_mean=0.4, seed=34)
m2pl = fit_2pl(dif_data.X)
print(f"2PL on the pooled sample: a r = {np.corrcoef(m2pl.a, bank40.a)[0, 1]:.3f}, "
      f"b r = {np.corrcoef(m2pl.b[:, 0], bank40.b[:, 0])[0, 1]:.4f}")

raw_gap = dif_data.X[dif_data.group == 1].mean() - dif_data.X[dif_data.group == 0].mean()
print(f"older patients endorse {raw_gap * 40:+.1f} more of the 40 items on average "
      f"(true impact: +0.4 SD more fatigue)")
"""),
        ("code", """
mh = mantel_haenszel(dif_data.X, dif_data.group, purify=True)
lr = logistic_dif(dif_data.X, dif_data.group)
print("Mantel-Haenszel flags (with purification):", [int(j) for j in np.flatnonzero(mh.flagged)])
print("logistic-regression flags (Bonferroni):    ", [int(j) for j in np.flatnonzero(lr.p_total * 40 < 0.05)])
print("planted DIF items:                         ", planted, "\\n")
print(f"{'item':<5s} {'b shift':>8s} {'MH delta':>9s} {'ETS':>4s} {'LR chi2':>8s}")
for j in sorted(set(planted) | set(np.flatnonzero(mh.flagged))):
    print(f"{j:<5d} {dif_data.dif_shift[j]:+8.2f} {mh.delta[j]:+9.2f} {mh.ets_class[j]:>4s} {lr.total_chi2[j]:8.1f}")
print(f"\\nlargest |delta| among the 36 clean items: {np.abs(np.delete(mh.delta, planted)).max():.2f}")
"""),
        ("md", """
ETS classes: A (negligible), B (moderate), C (large, |delta| >= 1.5).
Negative delta means the item is harder to endorse for the focal group
at the same trait level.
"""),
        ("code", """
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
ax = axes[0, 0]
ax.scatter(bank40.a, m2pl.a, s=30, color=BLUE, edgecolor="white", linewidth=1, label="2PL checklist")
ax.scatter(grm_true.a, grm.a, s=40, marker="s", color=ORANGE, edgecolor="white", linewidth=1, label="GRM short form")
lims = [0.8, 2.6]
ax.plot(lims, lims, color=INK, lw=1, ls="--")
ax.set_xlabel("true discrimination a")
ax.set_ylabel("estimated a")
ax.set_title("Discrimination recovered", fontsize=11)
ax.legend(frameon=False)

ax = axes[0, 1]
ax.scatter(bank40.b[:, 0], m2pl.b[:, 0], s=30, color=BLUE, edgecolor="white", linewidth=1, label="2PL checklist")
ax.scatter(grm_true.b.ravel(), grm.b.ravel(), s=40, marker="s", color=ORANGE, edgecolor="white", linewidth=1, label="GRM thresholds")
lims = [-3, 3]
ax.plot(lims, lims, color=INK, lw=1, ls="--")
ax.set_xlabel("true threshold b")
ax.set_ylabel("estimated b")
ax.set_title("Thresholds recovered", fontsize=11)
ax.legend(frameon=False)

ax = axes[1, 0]
P = category_probs(grm.a[:1], grm.d[:1], grid)[0]
Pt = category_probs(grm_true.a[:1], grm_true.d[:1], grid)[0]
cat_names = ["Never", "Rarely", "Sometimes", "Often", "Always"]
shades = ["#c7dcf5", "#8cb8eb", "#2a78d6", "#1d5aa3", "#123a6b"]
for k in range(5):
    ax.plot(grid, P[:, k], color=shades[k], lw=2, label=cat_names[k])
    ax.plot(grid, Pt[:, k], color=INK, lw=0.8, ls=":")
ax.set_xlabel("fatigue (theta)")
ax.set_ylabel("P(response)")
ax.set_title("Item 1 category curves (dotted: truth)", fontsize=11)
ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))

ax = axes[1, 1]
cols = [ORANGE if j in planted else BLUE for j in range(40)]
ax.bar(np.arange(40), mh.delta, color=cols, width=0.7)
for v in (-1.5, 1.5):
    ax.axhline(v, color=INK, lw=0.8, ls="--")
ax.axhspan(-1, 1, color=MUTED, alpha=0.12)
ax.set_xlabel("item")
ax.set_ylabel("Mantel-Haenszel delta")
ax.set_title("DIF: the four planted items (orange) stand out", fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "irt_recovery_dif.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
        ("md", """
## Takeaways

* The EM implementation recovers discriminations and thresholds for both
  the polytomous short form and the dichotomous checklist, and its EAP
  posterior SDs match the actual scoring error.
* Test information shows where on the trait a scale is precise, which
  is the basis for choosing short-form items and for adaptive testing
  (notebook 04).
* DIF methods separate a real group difference (impact) from items that
  behave differently across groups. Mantel-Haenszel with purification
  flags exactly the four planted items and none of the 36 clean ones.
"""),
    ]


# ===========================================================================
# 04 Computerized adaptive testing
# ===========================================================================

def nb04() -> list[tuple[str, str]]:
    return [
        ("md", """
# 04 Computerized Adaptive Testing: Same Precision, Fewer Questions

**Setting:** a 60-item calibrated symptom bank (healthcare). Every
extra question costs patient time and raises dropout.
**Methods:** 2PL calibration by EM, a CAT simulator (maximum-information
item selection, EAP updating, stop at posterior SD 0.3), randomesque
exposure control, and an optimized fixed short form for comparison.
**Ground truth:** each simulated patient's true trait level.

A CAT asks the most informative remaining question given the answers so
far. Patients with mild symptoms never see items about severe ones, and
the test stops once the score is precise enough.
"""),
        ("code", SETUP),
        ("code", """
from surveyhub.simulate import bank_2pl, simulate_irt, simulate_responses
from surveyhub.irt import fit_2pl, eap_scores, simulate_cat, fixed_form
"""),
        ("md", """
## 1. Calibrate the bank

Item parameters are estimated from a calibration sample of 3,000. The
CAT then uses these estimates, not the true values, as it would in
practice. Answers are always generated from the true parameters.
"""),
        ("code", """
true_bank = bank_2pl(60, a_range=(1.2, 2.5), b_range=(-2.5, 2.5), seed=41)
calib = simulate_irt(true_bank, 3000, seed=42)
model = fit_2pl(calib.X)
est_bank = model.bank
print(f"calibration: a r = {np.corrcoef(model.a, true_bank.a)[0, 1]:.3f}, "
      f"b r = {np.corrcoef(model.b[:, 0], true_bank.b[:, 0])[0, 1]:.4f}")
"""),
        ("md", """
## 2. Four ways to score 1,000 new patients

* **Full bank:** all 60 items.
* **CAT:** stop when the posterior SD is at most 0.3 (reliability about 0.91).
* **CAT, randomesque:** pick at random among the 5 most informative
  items, so no single item is overexposed.
* **Fixed short form:** the most informative items over the trait range,
  with as many items as the CAT used on average.
"""),
        ("code", """
rng = np.random.default_rng(43)
theta = rng.normal(size=1000)
X_full = simulate_responses(true_bank, theta, seed=44)
th_full, se_full = eap_scores(est_bank.a, est_bank.d, X_full)

cat = simulate_cat(est_bank, theta, true_bank=true_bank, se_stop=0.3, seed=45)
cat_r = simulate_cat(est_bank, theta, true_bank=true_bank, se_stop=0.3, top_k=5, seed=46)
L = int(round(cat.n_items.mean()))
form = fixed_form(est_bank, L)
th_fix, se_fix = eap_scores(est_bank.a[form], est_bank.d[form], X_full[:, form])

def rmse(x):
    return np.sqrt(np.mean((x - theta) ** 2))

print(f"{'method':<22s} {'items (mean)':>13s} {'RMSE':>6s} {'corr':>6s} {'max exposure':>13s}")
print(f"{'full bank':<22s} {60:13.1f} {rmse(th_full):6.3f} {np.corrcoef(th_full, theta)[0, 1]:6.3f} {1.0:13.2f}")
print(f"{'CAT (SE 0.3)':<22s} {cat.n_items.mean():13.1f} {rmse(cat.theta_hat):6.3f} "
      f"{np.corrcoef(cat.theta_hat, theta)[0, 1]:6.3f} {cat.exposure.max():13.2f}")
print(f"{'CAT randomesque':<22s} {cat_r.n_items.mean():13.1f} {rmse(cat_r.theta_hat):6.3f} "
      f"{np.corrcoef(cat_r.theta_hat, theta)[0, 1]:6.3f} {cat_r.exposure.max():13.2f}")
print(f"{'fixed short form':<22s} {L:13.1f} {rmse(th_fix):6.3f} {np.corrcoef(th_fix, theta)[0, 1]:6.3f} {1.0:13.2f}")
print(f"\\nCAT stopped by the SE rule for {np.mean(cat.n_items < 60):.1%} of patients; "
      f"{np.mean(cat.n_items <= 15):.0%} needed 15 items or fewer")
"""),
        ("md", """
## 3. Where the precision goes

Average error is only part of the story. The fixed form spends its items
near the middle of the trait, so patients at the extremes are measured
poorly. The CAT keeps precision roughly flat across the range.
"""),
        ("code", """
edges = np.arange(-2.75, 2.76, 0.5)
mids = (edges[:-1] + edges[1:]) / 2
def cond_rmse(est):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (theta >= lo) & (theta < hi)
        out.append(np.sqrt(np.mean((est[m] - theta[m]) ** 2)) if m.sum() >= 10 else np.nan)
    return np.array(out)

cr_full, cr_cat, cr_fix = cond_rmse(th_full), cond_rmse(cat.theta_hat), cond_rmse(th_fix)
print(f"{'theta bin':>10s} {'full':>6s} {'CAT':>6s} {'fixed':>6s}")
for m_, a, b, c in zip(mids, cr_full, cr_cat, cr_fix):
    print(f"{m_:+10.2f} {a:6.3f} {b:6.3f} {c:6.3f}")
"""),
        ("code", """
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
ax1.plot(mids, cr_fix, "o-", color=ORANGE, lw=2, ms=7, label=f"fixed form ({L} items)")
ax1.plot(mids, cr_cat, "o-", color=BLUE, lw=2, ms=7, label=f"CAT (mean {cat.n_items.mean():.1f} items)")
ax1.plot(mids, cr_full, "o-", color=AQUA, lw=2, ms=7, label="full bank (60 items)")
ax1.axhline(0.3, color=INK, ls="--", lw=1)
ax1.text(mids[0], 0.31, "stopping target 0.3", fontsize=9, color=INK)
ax1.set_xlabel("true trait level (theta)")
ax1.set_ylabel("RMSE of the score")
ax1.set_title("CAT keeps precision flat across the trait", fontsize=11)
ax1.legend(frameon=False)

ax2.hist(cat.n_items, bins=np.arange(0.5, 61.5, 2), color=BLUE, edgecolor="white")
ax2.axvline(60, color=INK, lw=1, ls="--")
ax2.text(59, ax2.get_ylim()[1] * 0.9, "full bank ", ha="right", fontsize=9)
ax2.set_xlabel("items administered")
ax2.set_ylabel("patients")
ax2.set_title("Items needed per patient", fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "cat_efficiency.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
        ("md", """
## Takeaways

* A CAT meets its precision target with about a fifth of the bank. The
  full 60-item test is more precise still, but at five times the
  respondent burden.
* A fixed short form of the same average length is fine in the middle
  of the trait and three to four times worse at the extremes, which is
  often where clinical decisions are made.
* Pure maximum-information selection overuses the best items.
  Randomesque selection caps exposure at a small cost in length.
* A CAT is only as good as its calibration. Here the bank was
  calibrated on 3,000 respondents and checked for fit first.
"""),
    ]


# ===========================================================================
# 05 Survey weighting (opt-in online poll)
# ===========================================================================

def nb05() -> list[tuple[str, str]]:
    return [
        ("md", """
# 05 Weighting an Opt-In Online Poll

**Setting:** an online opt-in health survey estimating the share of
adults who got a flu shot this season. Opt-in panels skew young and
college-educated, and both traits predict vaccination.
**Methods:** post-stratification, raking (iterative proportional
fitting) with and without trimming, propensity pseudo-weights against a
reference sample, Kish design effects, Taylor-linearization and
delete-a-group jackknife standard errors.
**Ground truth:** a synthetic population of 200,000 adults (the
"census") with a known vaccination rate. Each poll is drawn from it with
a known selection mechanism.

Two scenarios:
1. **Selection on observables:** who joins depends on age, education,
   region and sex. Weighting on those variables should work.
2. **Selection on the outcome:** people who got the shot are more
   interested in a health survey. No demographic weight can fix that.
"""),
        ("code", SETUP),
        ("code", """
from surveyhub.simulate import make_population, opt_in_sample, AGE, EDU
from surveyhub.weighting import (rake, poststratify, pseudo_weights, dummies,
                                 weighted_mean, kish_deff, effective_n,
                                 linearized_se, jackknife)

pop = make_population(200_000, seed=51)
cov = pop.covariates()
targets = {v: pop.margins(v) for v in cov}
print(f"population: N = {pop.N:,}, true vaccination rate = {pop.truth:.3f}")
"""),
        ("md", """
## 1. One poll, up close
"""),
        ("code", """
idx = opt_in_sample(pop, 1500, seed=52)
y = pop.y[idx]
m = {v: cov[v][idx] for v in cov}
print(f"respondents: {len(idx)}\\n")
print(f"{'group':<14s} {'population':>11s} {'poll':>7s}")
for v, names in (("age", AGE), ("edu", EDU)):
    for lev, name in enumerate(names):
        print(f"{name:<14s} {np.mean(cov[v] == lev):11.1%} {np.mean(m[v] == lev):7.1%}")
"""),
        ("code", """
cells_pop = pop.age * 3 + pop.edu
w_ps = poststratify(cells_pop[idx], np.bincount(cells_pop, minlength=12).astype(float))
r = rake(m, targets)
rt = rake(m, targets, trim=(0.3, 3.0))
rng = np.random.default_rng(53)
ref = rng.choice(pop.N, 2000, replace=False)
w_pw = pseudo_weights(dummies(m), dummies({v: cov[v][ref] for v in cov}),
                      np.full(2000, pop.N / 2000))
print(f"raking converged in {r.n_iter} sweeps, worst margin error {r.max_margin_error:.1e}\\n")
print(f"{'method':<26s} {'estimate':>9s} {'error':>7s} {'deff':>6s} {'eff n':>7s} {'SE (lin)':>9s}")
print(f"{'unweighted':<26s} {y.mean():9.3f} {y.mean() - pop.truth:+7.3f} {1:6.2f} {len(y):7.0f} "
      f"{linearized_se(y, np.ones(len(y))):9.4f}")
for name, w in [("post-strat (age x edu)", w_ps), ("raked (4 margins)", r.weights),
                ("raked + trimmed", rt.weights), ("propensity pseudo-weights", w_pw)]:
    e = weighted_mean(y, w)
    print(f"{name:<26s} {e:9.3f} {e - pop.truth:+7.3f} {kish_deff(w):6.2f} {effective_n(w):7.0f} "
          f"{linearized_se(y, w):9.4f}")
"""),
        ("md", """
Weighting moves the estimate toward the truth and costs precision: the
design effect from unequal weights shrinks 1,500 respondents to an
effective sample of a few hundred. Trimming extreme weights buys some of
that back at the cost of a little bias.

The jackknife re-runs the raking on each replicate, so its standard
error includes the variability of the weighting step itself.
"""),
        ("code", """
def raked_estimate(keep):
    w = rake({v: m[v][keep] for v in m}, targets).weights
    return weighted_mean(y[keep], w)

jk = jackknife(raked_estimate, len(y), n_groups=20, seed=54)
print(f"raked estimate {jk.estimate:.3f}: linearized SE {linearized_se(y, r.weights):.4f}, "
      f"jackknife SE {jk.se:.4f}")
"""),
        ("md", """
## 2. Repeat the poll 200 times, in both scenarios

For each method, record bias, the spread of estimates, and whether the
95% CI covers the truth. For raking, compare linearized and jackknife
CIs.
"""),
        ("code", """
def run(outcome_effect, R=200, seed=0):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["unweighted", "post-stratified", "raked", "raked + trimmed", "pseudo-weights"]}
    cover = {k: [] for k in out}
    cover_jk = []
    for _ in range(R):
        idx = opt_in_sample(pop, 1500, outcome_effect=outcome_effect, seed=rng.integers(1 << 31))
        y = pop.y[idx]
        m = {v: cov[v][idx] for v in cov}
        ref = rng.choice(pop.N, 2000, replace=False)
        ws = {
            "unweighted": np.ones(len(y)),
            "post-stratified": poststratify(cells_pop[idx], np.bincount(cells_pop, minlength=12).astype(float)),
            "raked": rake(m, targets).weights,
            "raked + trimmed": rake(m, targets, trim=(0.3, 3.0)).weights,
            "pseudo-weights": pseudo_weights(dummies(m), dummies({v: cov[v][ref] for v in cov}),
                                             np.full(2000, pop.N / 2000)),
        }
        for k, w in ws.items():
            e, se = weighted_mean(y, w), linearized_se(y, w)
            out[k].append(e)
            cover[k].append(abs(e - pop.truth) < 1.96 * se)
        jk = jackknife(lambda keep: weighted_mean(y[keep], rake({v: m[v][keep] for v in m}, targets).weights),
                       len(y), n_groups=20, seed=rng.integers(1 << 31))
        cover_jk.append(abs(jk.estimate - pop.truth) < 1.96 * jk.se)
    return ({k: np.array(v) for k, v in out.items()},
            {k: np.mean(v) for k, v in cover.items()}, np.mean(cover_jk))

sims = {}
for label, eff in [("Selection on observables", 0.0), ("Selection on the outcome", 0.8)]:
    est, cov_rate, cov_jk = run(eff, seed=55 if eff == 0 else 56)
    sims[label] = (est, cov_rate, cov_jk)
    print(label)
    print(f"  {'method':<18s} {'bias':>7s} {'SD':>6s} {'RMSE':>6s} {'95% CI coverage':>16s}")
    for k, e in est.items():
        print(f"  {k:<18s} {e.mean() - pop.truth:+7.3f} {e.std():6.3f} "
              f"{np.sqrt(np.mean((e - pop.truth) ** 2)):6.3f} {cov_rate[k]:16.1%}")
    print(f"  raked with jackknife CI: coverage {cov_jk:.1%}\\n")
"""),
        ("code", """
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
for ax, (label, (est, cov_rate, cov_jk)) in zip(axes, sims.items()):
    names = list(est)
    for i, k in enumerate(names):
        lo, hi = np.percentile(est[k], [2.5, 97.5])
        col = MUTED if k == "unweighted" else BLUE
        ax.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
        ax.plot(est[k].mean(), i, "o", color=col, ms=8, markeredgecolor="white", markeredgewidth=2)
    ax.axvline(pop.truth, color=INK, ls="--", lw=1.2)
    ax.text(pop.truth + 0.003, len(names) - 0.55, "truth", fontsize=9, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{k}\\n{cov_rate[k]:.0%} CIs cover" for k in names], fontsize=9)
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("estimated vaccination rate (mean, 95% of polls)")
fig.tight_layout()
fig.savefig(FIG_DIR / "weighting_poll.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
        ("md", """
## Takeaways

* When selection depends only on variables with known population totals,
  raking, post-stratification, and propensity weights all remove the
  bias of the raw opt-in sample, and their CIs cover close to 95%.
* Weighting is not free: design effects of 2 or more halve the
  effective sample size. Trimming trades a little bias for precision.
* When selection depends on the outcome itself, every demographic weight
  leaves the bias in place (here it even grows a little, because the
  weights up-weight the groups whose few volunteers are the most
  self-selected), and the CIs become confidently wrong. No amount of weighting can test for this from inside the
  sample; it calls for a benchmark question with a known answer, a
  probability sample, or a sensitivity analysis.
"""),
    ]


# ===========================================================================
# 06 Sensitive questions and conjoint
# ===========================================================================

def nb06() -> list[tuple[str, str]]:
    return [
        ("md", """
# 06 Sensitive Questions and Survey Experiments

**Setting:** an employee survey asking about workplace misconduct
(falsifying time records), followed by a conjoint experiment on job
offers.
**Methods:** list experiment (item count technique) with a design-effect
test, forced-response randomized response, and AMCE estimation for a
forced-choice conjoint with respondent-clustered standard errors.
**Ground truth:** 22% of employees truly engaged in the behavior; only
55% of them admit it when asked directly. Conjoint AMCEs are computed
exactly by enumerating all 108 x 108 profile pairs under the
data-generating model.
"""),
        ("code", SETUP),
        ("code", """
from surveyhub.simulate import simulate_sensitive, simulate_conjoint, true_amce
from surveyhub.experiments import (direct_estimate, list_experiment, list_design_test,
                                   randomized_response, amce)
TRUE_PREV = 0.22
"""),
        ("md", """
## 1. Three ways to ask a sensitive question

Respondents are randomized into arms:

* **Direct (n = 1,000):** "Have you ever falsified a time record?"
* **List experiment (n = 2,000):** "How many of these have you done?"
  The control half sees four innocuous items; the treatment half sees
  the same four plus the sensitive one. Nobody reveals which items.
* **Randomized response (n = 1,000):** a private die roll tells the
  respondent to answer truthfully (4 in 6), say "yes" (1 in 6), or say
  "no" (1 in 6). A "yes" never identifies anyone.
"""),
        ("code", """
sv = simulate_sensitive(prevalence=TRUE_PREV, admit_rate=0.55, seed=61)
est = {
    "Direct question": direct_estimate(sv.direct),
    "List experiment": list_experiment(sv.list_count, sv.list_treat),
    "Randomized response": randomized_response(sv.rr_answer, sv.rr_design["p_truth"], sv.rr_design["p_yes"]),
}
print(f"{'design':<21s} {'estimate':>9s} {'SE':>7s}   95% CI           (truth {TRUE_PREV})")
for k, e in est.items():
    lo, hi = e.ci
    print(f"{k:<21s} {e.estimate:9.3f} {e.se:7.3f}   [{lo:.3f}, {hi:.3f}]")
dt = list_design_test(sv.list_count, sv.list_treat, sv.n_control_items)
print(f"\\nlist design-effect test: min adjusted p = {dt.min_pvalue:.2f} "
      f"-> {'design effect detected' if dt.design_effect else 'no evidence of a design effect'}")
"""),
        ("md", """
Privacy has a price: for the same total sample the indirect designs
carry much larger standard errors, because each answer is mixed with
noise on purpose. The next cell repeats the survey 500 times.
"""),
        ("code", """
rows = {k: [] for k in est}
for s in range(500):
    sv_ = simulate_sensitive(prevalence=TRUE_PREV, admit_rate=0.55, seed=1000 + s)
    rows["Direct question"].append(direct_estimate(sv_.direct))
    rows["List experiment"].append(list_experiment(sv_.list_count, sv_.list_treat))
    rows["Randomized response"].append(randomized_response(sv_.rr_answer, sv_.rr_design["p_truth"],
                                                           sv_.rr_design["p_yes"]))
print(f"{'design':<21s} {'mean':>7s} {'bias':>7s} {'SD':>6s} {'coverage':>9s}")
for k, lst in rows.items():
    e = np.array([r.estimate for r in lst])
    c = np.mean([abs(r.estimate - TRUE_PREV) < 1.96 * r.se for r in lst])
    print(f"{k:<21s} {e.mean():7.3f} {e.mean() - TRUE_PREV:+7.3f} {e.std():6.3f} {c:9.1%}")
"""),
        ("md", """
## 2. A conjoint experiment on job offers

Each of 1,500 respondents sees five pairs of job offers with randomized
salary, work location, employer type, manager rating and benefits, and
picks one. The average marginal component effect (AMCE) of a level is
the change in the probability that an offer is chosen when the
attribute moves from its baseline to that level, averaged over
everything else. Two respondent types differ in how much they value
remote work; the AMCE averages over them.
"""),
        ("code", """
truth = true_amce()
cj = simulate_conjoint(n_respondents=1500, n_tasks=5, seed=62)
res = amce(cj.levels, cj.chosen, cj.respondent, cj.attributes)
print(f"{'attribute: level':<30s} {'truth':>7s} {'AMCE':>7s} {'SE (cl)':>8s} {'SE (naive)':>11s} {'z':>6s}")
for lab, e, s_c, s_n in zip(res.labels, res.estimate, res.se_cluster, res.se_naive):
    print(f"{lab[0] + ': ' + lab[1]:<30s} {truth[lab]:7.3f} {e:7.3f} {s_c:8.4f} {s_n:11.4f} "
          f"{(e - truth[lab]) / s_c:+6.2f}")
"""),
        ("code", """
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.35]})
names = list(est)
for i, k in enumerate(names):
    e = est[k]
    lo, hi = e.ci
    ax1.plot([lo, hi], [i, i], color=BLUE, lw=2.5, solid_capstyle="round")
    ax1.plot(e.estimate, i, "o", color=BLUE, ms=8, markeredgecolor="white", markeredgewidth=2)
ax1.axvline(TRUE_PREV, color=INK, ls="--", lw=1.2)
ax1.text(TRUE_PREV + 0.003, -0.6, "true prevalence", ha="left", fontsize=9)
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names)
ax1.set_ylim(len(names) - 0.5, -0.9)
ax1.set_xlabel("estimated prevalence (95% CI)")
ax1.set_title("Indirect designs recover what people won't admit", fontsize=11)

labels = [f"{a}: {l}" for a, l in res.labels]
yy = np.arange(len(labels))
ax2.errorbar(res.estimate, yy, xerr=1.96 * res.se_cluster, fmt="o", color=BLUE, ms=7,
             mec="white", mew=1.5, elinewidth=2, capsize=0, label="estimated AMCE (95% CI)")
ax2.plot([truth[l] for l in res.labels], yy, "|", color=INK, ms=16, mew=2, label="exact truth")
ax2.axvline(0, color=MUTED, lw=0.8)
ax2.set_yticks(yy)
ax2.set_yticklabels(labels, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel("change in probability the offer is chosen")
ax2.set_title("Conjoint AMCEs vs the exact truth", fontsize=11)
ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "sensitive_conjoint.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
        ("md", """
## Takeaways

* The direct question underestimates the behavior by almost half and
  its CI is confidently wrong. Both indirect designs are unbiased and
  their CIs cover the truth, at the cost of a much larger standard
  error, so they need bigger samples.
* Check the list experiment's assumptions: the design-effect test looks
  for signs that adding the sensitive item changed answers to the
  control items. Choose control items that avoid floor and ceiling
  effects.
* Conjoint AMCEs from a simple linear regression recover the exact
  truth because the attributes are randomized. Cluster standard errors
  by respondent, since each person contributes many choices.
"""),
    ]


BUILDERS = {
    "1": ("01_reliability_engagement_survey.ipynb", nb01),
    "2": ("02_factor_structure_invariance.ipynb", nb02),
    "3": ("03_irt_patient_reported_outcomes.ipynb", nb03),
    "4": ("04_adaptive_testing.ipynb", nb04),
    "5": ("05_weighting_opt_in_poll.ipynb", nb05),
    "6": ("06_sensitive_questions_conjoint.ipynb", nb06),
}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = BUILDERS.keys() if which == "all" else [which]
    for k in keys:
        fname, builder = BUILDERS[k]
        run_notebook(builder(), ROOT / "notebooks" / fname)
