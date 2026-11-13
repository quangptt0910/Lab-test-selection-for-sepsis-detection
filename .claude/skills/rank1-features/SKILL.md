---
name: rank1-features
description: "Formulas and conventions for risk-model features and the U1-U0 training target, adapted from the PhysioNet 2019 winning team. Use when writing or changing code in src/sepsis/features, src/sepsis/labels or risk-model configs."
---

# Rank #1 features, adapted to replay

Source: Morrill et al., "Utilisation of the Signature Method to Identify the Early Onset of Sepsis..." (rank #1, PhysioNet/CinC 2019). This project's changes are marked **[ours]**.

## Input columns
Vitals: `HR O2Sat Temp SBP MAP DBP Resp EtCO2`
Labs: `BaseExcess HCO3 FiO2 pH PaCO2 SaO2 AST BUN Alkalinephos Calcium Chloride Creatinine Bilirubin_direct Glucose Lactate Magnesium Phosphate Potassium Bilirubin_total TroponinI Hct Hgb PTT WBC Fibrinogen Platelets`
Static/time: `Age Gender Unit1 Unit2 HospAdmTime ICULOS`. Target column: `SepsisLabel`.

## 1. Imputation
- Forward-fill within each patient, in time order.
- Leave NaN before the first observed value. Do not mean-fill for LightGBM.
- **[ours]** Labs forward-fill only from *revealed* values in the current view, starting at each value's reveal row (request row + 1), never at its recorded row. Panels, look-back windows and the reveal delay are in `configs/panels.yaml` (ADR-0006). No "component known" or other revealed/missing flags for labs (ADR-0002); `hours_since_<panel>` being NaN already says "never revealed".
- **[ours]** B1 logistic only: fixed clinical normals for unrevealed labs, training-side medians for vitals, log1p + scaler fitted on the training side (`configs/preprocessing.yaml`, ADR-0005). Trees get no fill, scaling or log.
- **[ours]** Every window counts rows and ends at t, t included: a window of w rows covers t − w + 1 … t (CONTEXT.md).

## 2. Hand-crafted features
| Feature | Definition |
|---|---|
| `shock_index` | HR / SBP (NaN if SBP missing or 0) |
| `bun_cr` | BUN / Creatinine |
| `partial_sofa` | Sum of the four component scores below. A missing component scores 0. |
| `partial_sofa_delta_24h` | partial_sofa(t) − min(partial_sofa over [t−24, t]) |
| `<vital>_max_w`, `<vital>_min_w` | Rolling max/min over a look-back window (window from config) |
| `hours_since_<panel>` | **[ours]** Rows since the most recent revealed value of the panel was recorded (not since it was requested), NaN if none (ADR-0006). This replaces rank #1's measurement counts (ADR-0002). |

### PartialSOFA component thresholds (standard SOFA)
| Component | Column (unit) | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| Coagulation | Platelets (×10³/µL) | ≥150 | <150 | <100 | <50 | <20 |
| Liver | Bilirubin_total (mg/dL) | <1.2 | 1.2–1.9 | 2.0–5.9 | 6.0–11.9 | ≥12 |
| Cardiovascular | MAP (mmHg) | ≥70 | <70 | n/a | n/a | n/a |
| Renal | Creatinine (mg/dL) | <1.2 | 1.2–1.9 | 2.0–3.4 | 3.5–4.9 | ≥5 |

Cardiovascular scores 2–4 need vasopressor data, which this dataset doesn't have. Compare against the rank #1 code (`CanIgetyoursignature.zip`) before claiming parity.

## 3. Signatures (B3 only)
- Library: `iisignature` (`uv sync --extra signatures`).
- Sliding window ending at t (length from config, try 6–12 h), depth 2–3 (tune).
- Rank #1 used: signatures of `[partial_sofa, MAP, bun_cr]` with a time channel; and signatures of non-stationary vitals after **cumulative sum then lead-lag**.
- Pad windows shorter than the window length from the first available row. Never use rows > t.

## 4. Target: U1 − U0
Constants (official scorer): dt_early = −12, dt_optimal = −6, dt_late = 3, max_u_tp = 1, min_u_fn = −2, u_fp = −0.05, u_tn = 0.

t_sepsis = index of the first `SepsisLabel == 1` row + 6. dt = t − t_sepsis.

| Case | U1 (alarm) | U0 (no alarm) |
|---|---|---|
| Non-septic patient | −0.05 | 0 |
| Septic, dt ≤ −6 | max(dt/6 + 2, −0.05) | 0 |
| Septic, −6 < dt ≤ 3 | −dt/9 + 1/3 | −(2/9)·dt − 4/3 |
| Septic, dt > 3 | not scored: target 0, sample weight 0 | |

Required unit-test values for U1 − U0:
- dt −15 → −0.05
- dt −12 → 0
- dt −9 → 0.5
- dt −6 → 1.0
- dt −3 → 1.333
- dt 0 → 1.667
- dt 3 → 2.0
- non-septic → −0.05

Also cross-check against the official scorer: for a toy patient, flipping one hour's prediction must change the scorer's raw utility by exactly U1 − U0 for that hour.

Train LightGBM as a regressor (MSE) on this target. Alarm = prediction > threshold, tuned on validation with the official scorer, then frozen.

## 5. Tests every feature change needs
- **No-future test:** perturb all rows > t; features at t are unchanged.
- **Masking test:** a lab not in the view never affects any feature.
- **Hand-computed example:** one tiny patient table with expected values written out.
