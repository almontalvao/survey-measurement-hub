# survey-measurement-hub

Survey measurement methods implemented **from scratch** (numpy, with scipy
only for a generic optimizer and probability distributions), and every
estimator **validated against a known truth**: data are simulated from a
model whose reliability, item parameters, population mean, prevalence or
effects are fixed in advance.

Companion repo to the field guide at
[arnaldomontalvao.com/survey-measurement](https://arnaldomontalvao.com/survey-measurement).

## What's inside

| Topic | Methods (from scratch) | Case study | Field | Ground truth |
|---|---|---|---|---|
| Reliability | Cronbach's alpha, McDonald's omega, split-half, item diagnostics, bootstrap CIs | Employee engagement scale | People analytics | Reliability implied by known loadings and error covariances |
| Factor structure | Parallel analysis, principal axis EFA (varimax, promax), ML CFA with fit indices | Customer satisfaction survey | Customer research | Known loading matrix and factor correlations |
| Measurement invariance | Multi-group CFA (configural, metric, scalar), partial invariance search | Same survey in English and Spanish | Cross-cultural research | Planted intercept shifts and latent mean differences |
| Item response theory | Graded response model and 2PL by Bock-Aitkin EM, EAP scoring, information, Mantel-Haenszel and logistic DIF | Patient-reported fatigue item bank | Healthcare | Known item parameters and planted DIF items |
| Adaptive testing | CAT simulator (max information, EAP, SE stopping, randomesque exposure control) | Shortening a 60-item symptom bank | Healthcare | Each simulated patient's true trait level |
| Weighting and inference | Post-stratification, raking with trimming, propensity pseudo-weights, Kish deff, Taylor linearization, delete-a-group jackknife | Opt-in online health poll | Public opinion and health | A synthetic census with a known outcome mean |
| Sensitive questions | List experiment with design-effect test, forced-response randomized response | Workplace misconduct survey | Organizational research | Known true prevalence |
| Survey experiments | Conjoint AMCEs by OLS with respondent-clustered SEs | Job-offer choice experiment | Labor markets | AMCEs computed exactly over all 108 x 108 profile pairs |

## Results (from the executed notebooks)

- **Reliability.** With unequal loadings, alpha understates the true
  reliability (0.787 vs 0.807); with correlated errors from shared
  wording it overstates it (0.800 vs 0.720). Omega recovers the truth in
  both cases (bias under 0.002) once the error covariances are modeled.
- **Factor structure and invariance.** Parallel analysis retains exactly
  3 factors, EFA congruence with the true loadings is 0.996 or higher, and
  the CFA fits (CFI 1.000). Across languages, scalar invariance is
  rejected (chi-square difference 480 on 9 df), the search frees exactly
  the two planted items, and the partial model recovers the true latent
  mean differences (0.27 vs 0.30 for service, near zero elsewhere) that a
  naive sum-score comparison gets wrong.
- **IRT.** EM recovers GRM discriminations (r = 0.99) and thresholds
  (RMSE 0.055); EAP posterior SDs match the actual scoring error (0.337 vs
  0.336). Mantel-Haenszel with purification flags exactly the four planted
  DIF items out of 40, with no false positives.
- **Adaptive testing.** The CAT reaches its precision target with 12 of
  60 items on average (RMSE 0.29); a fixed form of the same length is
  three to four times less precise at the extremes of the trait.
- **Weighting.** An opt-in sample that skews young and educated misses
  the vaccination rate by 3.4 points and its CIs cover the truth 20% of
  the time. Raking, post-stratification and propensity weights cut bias
  to 0.2 points or less with 94 to 98% coverage, at a design effect near
  2.5. When people select on the outcome itself, every weight fails
  (0% coverage).
- **Sensitive questions and conjoint.** Asked directly, only 12% admit a
  behavior that 22% engaged in. The list experiment (22.4%) and
  randomized response (22.0%) recover it over 500 replications with 95%
  CI coverage, at about 2x (randomized response) to 4x (list experiment)
  the standard error of the direct question. Conjoint AMCEs
  match the exact enumerated truth.

## Layout

```
src/surveyhub/
├── simulate/      data generators with a planted truth (factor models,
│                  item banks, a synthetic population, survey experiments)
├── reliability/   alpha, omega, split-half, item diagnostics, bootstrap
├── factor/        efa.py, cfa.py (multi-group ML with analytic gradient),
│                  invariance.py
├── irt/           models.py (GRM/2PL by EM), dif.py, cat.py
├── weighting/     post-stratification, raking, propensity weights, variance
└── experiments/   list experiment, randomized response, conjoint AMCE
notebooks/         6 executed case studies (render on GitHub)
docs/              MEASUREMENT_GUIDE.md, the method-selection field guide
tests/             26 known-truth tests
tools/             build_notebooks.py (rebuilds and executes the notebooks)
```

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests/           # every estimator vs its known truth
python tools/build_notebooks.py   # rebuild + re-execute all notebooks
```

Requires Python 3.10 or newer. Dependencies: numpy, scipy, matplotlib
(pytest to run the tests).

## Why from scratch?

Production survey work should use mature software (lavaan, mirt, the
R survey package, and their Python counterparts). This repo exists to make
the machinery visible: each estimator is a short, readable function, and
each one is checked against a truth you can change and re-run.

## License

MIT
