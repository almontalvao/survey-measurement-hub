# Survey Measurement: A Field Guide

A survey number is only as good as the chain behind it: the construct you
meant to measure, the questions you wrote, the people who answered, and
the model that turned answers into estimates. This guide walks that chain
and points to the notebook in this repo where each method is run against
a known truth.

## 1. Foundations

**Constructs and operationalization.** Satisfaction, engagement, fatigue
and trust cannot be observed directly. A scale is a bet that a set of
answers moves with the construct. Write the construct definition first;
every item should trace back to it.

**The true-score model.** Observed score = true score + error.
Reliability is the share of observed variance that is true-score
variance. It caps every correlation you can observe: two measures with
reliabilities 0.7 and 0.8 can correlate at most sqrt(0.56) = 0.75, even
if the constructs are identical.

**Validity** asks whether the score measures the intended construct:
content (do the items cover it?), structural (does the factor structure
match the theory?), convergent and discriminant (does it correlate with
what it should and not with what it should not?), and criterion (does it
predict outcomes?). Reliability is necessary for validity, never
sufficient.

**Total Survey Error.** Error enters on two sides. Measurement:
validity, measurement error, processing error. Representation: coverage,
sampling, nonresponse, adjustment. Precision (variance) and accuracy
(bias) trade off within each.

## 2. Questionnaire design

- **Wording.** One idea per item (no double-barreled questions), plain
  words, a clear reference period ("in the past 7 days"), and no leading
  premises.
- **Response scales.** Five to seven fully labeled points are a good
  default for attitudes. Include a midpoint when neutrality is real, and
  "don't know" when not knowing is plausible.
- **Order and context.** Earlier questions prime later ones; general
  questions asked after specific ones change meaning. Randomize item
  order within blocks when the order carries no logic.
- **Response styles.** Acquiescence (agreeing regardless of content),
  extreme responding, and satisficing (straight-lining, speeding)
  contaminate scores. Balanced keying and attention checks help, at the
  cost of the correlated-error problem in notebook 01.
- **Mode effects.** Interviewers raise social desirability; web surveys
  raise satisficing. Do not compare across modes without testing
  invariance.
- **Pretesting.** Cognitive interviews ("what does this question mean to
  you?") catch problems no statistic can, then a pilot for item
  statistics.

## 3. Reliability and factor structure (notebooks 01, 02)

| Question | Method |
|---|---|
| How consistent is my multi-item score? | Omega total from a one-factor CFA (alpha only if loadings are similar and errors independent) |
| Is it stable over time? | Test-retest correlation or ICC |
| Do two raters agree? | Cohen's kappa (categories), ICC (continuous) |
| How many dimensions? | Parallel analysis, then EFA with an oblique rotation |
| Does my theorized structure fit? | CFA: CFI/TLI near 0.95 or above, RMSEA under 0.06, SRMR under 0.08 |
| Can I compare groups? | Multi-group CFA: configural, metric, then scalar invariance |

Alpha is biased low with unequal loadings and biased high with
correlated errors (notebook 01). Group mean comparisons need scalar
invariance; a few non-invariant items can be freed (partial invariance),
but only if most items on each factor stay invariant (notebook 02).

## 4. Item response theory (notebooks 03, 04)

- **Models.** Rasch/1PL (equal discriminations), 2PL (item-specific
  discriminations), 3PL (adds guessing, for tests), graded response
  model (ordered categories).
- **Information.** Each item is precise near its thresholds; test
  information is the sum, and the score's SE is 1/sqrt(information).
  Precision varies along the trait, which one reliability number hides.
- **DIF.** An item functions differently when equally able (or equally
  affected) people in different groups answer it differently.
  Mantel-Haenszel with purification and logistic regression are the
  workhorses; flag on both significance and effect size.
- **CAT.** Choose the most informative next item, update the score, stop
  at a target SE. Needs a calibrated, unidimensional bank and exposure
  control.
- **Linking.** Put two forms or two calibrations on one scale with anchor
  items (mean-sigma, Stocking-Lord).

## 5. Sampling, weighting and inference (notebook 05)

- **Probability samples** know each unit's selection chance; base
  weights are their inverses. Stratification lowers variance;
  clustering raises it (design effect 1 + (m - 1) rho).
- **Nonprobability samples** (opt-in panels) have unknown selection.
  Weighting assumes selection is ignorable given the weighting
  variables, which cannot be tested from inside the sample.
- **Adjustments.** Post-stratification (needs cell totals), raking
  (needs only margins), propensity pseudo-weights (needs a reference
  sample), multilevel regression and post-stratification (MRP, for small
  areas and sparse cells).
- **The cost.** Unequal weights inflate variance: Kish deff = n sum(w^2) /
  sum(w)^2. Trim extreme weights when the deff is large.
- **Variance.** Taylor linearization for smooth statistics; replicate
  methods (jackknife, bootstrap, BRR) re-run the weighting on each
  replicate and handle complex estimators.

## 6. Sensitive questions and survey experiments (notebook 06)

- **List experiment.** Difference in mean item counts between long and
  short lists. Needs no design effect and no floor or ceiling effects;
  costs a lot of precision.
- **Randomized response.** A private randomizer adds known noise.
  Unbiased, but respondents must trust and follow the procedure.
- **Endorsement experiments.** Randomize whether a policy is attributed
  to a sensitive group.
- **Conjoint.** Randomize attributes of profiles; AMCEs come from a
  linear regression with SEs clustered by respondent.
- **Vignettes and factorial surveys.** Randomize features of a scenario
  to estimate how each shapes judgments.

## 7. Which method, when

| Situation | Start with | Watch out for |
|---|---|---|
| New multi-item scale | Cognitive pretest, pilot, EFA, then CFA on new data | Reusing one sample for both EFA and CFA |
| Reporting reliability | Omega with a bootstrap CI | Alpha with unequal loadings or shared wording |
| Comparing groups on a scale | Invariance testing | Translated or mode-specific items |
| Long instrument, burden matters | IRT calibration, then CAT or an information-based short form | Multidimensional banks |
| Fairness of items across groups | DIF (Mantel-Haenszel and logistic) | Treating impact as DIF |
| Skewed or opt-in sample | Raking or propensity weights, with replicate SEs | Selection on the outcome itself |
| Sensitive behavior | List experiment or randomized response | Small samples (large SEs) |
| Trade-offs between features | Conjoint | Implausible attribute combinations |
