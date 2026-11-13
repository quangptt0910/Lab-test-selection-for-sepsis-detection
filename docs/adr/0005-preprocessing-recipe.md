# ADR-0005: Preprocessing recipe

Status: accepted (M0-6). Settings in `configs/preprocessing.yaml`.

## Context
M0-3 fixed the cleaning rules (`configs/cleaning.yaml`), M0-4 compared hospitals A and B on dev patients (`docs/eda/README.md`), and M0-5 reviewed four top solutions (`docs/research/top4-solutions.md`). The feature builder (M2) and the baselines (M3) need one fixed order of steps, and a clear list of what is fitted and where.

Two facts shape the choice:
- **Missing labs are not missing at random.** Over measured hours, never-septic patients have a higher median lactate than septic patients' label-1 hours (1.80 vs 1.70 in A, 1.94 vs 1.84 in B; EDA finding 7). A lab is ordered when a clinician is worried, so the patients with a value are not typical of those without one.
- **The hospitals differ in charting and in values.** DBP is missing for 36.6% of A patients and 0.1% of B (finding 3); MAP is 78.7 in A vs 86.3 in B (finding 9). Anything fitted on one hospital carries that hospital's level.

## Decision
1. **Order of steps, for every model:** M0-3 cleaning → reveal mask (ADR-0006) → forward-fill within each patient, in row order, over revealed values only, starting at each value's reveal row → NaN before the first value. This is the rank #1 recipe (ADR-0001), also used by M0-5's #0 team; the two best-scoring solutions in the review both keep NaN before the first value.
2. **Not done:** no linear interpolation (it uses later values unless cut at t; rejected in M0-5), no mean fill before the first value for trees (M0-5; finding 7), no missing flags or counts for labs (ADR-0002).
3. **Trees (B2, B3):** no scaling, log transform or PCA. Monotone transforms do not change tree splits, and M0-5's #2 team found no gain from log + standardising (0.3931 vs 0.3933). PCA would mix revealed and unrevealed labs into one value. LightGBM handles NaN.
4. **B1 logistic** needs complete, scaled inputs:
   - Labs never revealed: a **fixed normal adult value** per lab (`b1.lab_fill`), not fitted on data. A mean or median of measured values would give untested patients the values of patients someone was worried about (finding 7).
   - Vitals before their first value: the median of the training side.
   - `log1p` for right-skewed labs (`b1.log_labs`), then a standard scaler fitted on the training side. M0-5's #2 team log-transformed and standardised its inputs; it brought nothing for trees, but a linear model needs it. The list holds labs that are bounded at zero with a long upper tail (enzymes, bilirubin, kidney markers, lactate, troponin, glucose, WBC). It follows from clinical knowledge, not a skew measurement; `provenance: project_assumption`.
   - Correlated pairs (Hct–Hgb, Bilirubin_direct–Bilirubin_total; finding 8) make coefficients unstable. Which to drop is a feature decision for M2; the panels keep both labs.
5. **Fitted on the training side only** (ADR-0004): vital fill medians, the B1 scaler, the alarm threshold and hyperparameters. The cleaning limits, the lab normals and the panel windows are fixed constants, the same in every setup.
6. **Windows count rows and end at t, t included.** A window of w rows at row t covers rows t − w + 1 … t. M0-5 found four different conventions among the teams; this fixes ours.

## Consequences
- The lab normals are an assumption (`provenance: project_assumption`). For a patient whose lab was never revealed, B1 behaves as if it were normal. That matches how a clinician reads an untested patient, but B1 cannot tell "normal" from "not tested", and must not: that difference is the ordering signal ADR-0002 excludes.
- The vital medians differ between A → B and B → A, so B1 can fill the same patient differently per setup. This is the intended training-side behaviour.
- LightGBM can still learn "this vital is missing" through its NaN branch, and DBP missingness marks hospital A (finding 3). That is handled by the M2 feature choice (proposal: use MAP and SBP, not DBP), not by preprocessing.
- The cleaning limits and panel windows were chosen from A-dev and B-dev together. They are not fitted per setup, but A → B and B → A results should be read knowing the evaluated hospital informed them (as in `configs/cleaning.yaml`).
