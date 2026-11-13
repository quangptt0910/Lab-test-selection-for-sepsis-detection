# Exploratory analysis: hospital A vs B (dev patients)

<!-- generated:start (python -m sepsis.eda.run; do not edit by hand) -->
Dev patients only, after the M0-3 cleaning (A: 16268 patients, 632531 patient-hours, B: 16000 patients, 610409 patient-hours). Aggregates only: no patient rows. Settings in `configs/eda.yaml`.

## 1. Cohort

| hospital | patients | patient_hours | septic_pct | hours_p25 | hours_median | hours_p75 | onset_hour_median | septic_on_arrival_pct | iculos_start_after_1_pct |
|---|---|---|---|---|---|---|---|---|---|
| A | 16268 | 632531 | 8.8 | 25.0 | 39.0 | 47.0 | 34.0 | 11.2 | 36.7 |
| B | 16000 | 610409 | 5.7 | 23.0 | 38.0 | 47.0 | 34.0 | 19.7 | 6.4 |

Hours are rows. `onset_hour_median` is t_sepsis (first label row + 6) of septic patients; `septic_on_arrival_pct` is the share of septic patients whose label is 1 within the first 1 row(s).

![cohort](figures/cohort.png)

## 2. Missingness and time between measurements

`pct_hours`: % of hours with a value. `pct_patients`: % of patients with at least one value. `median_hours`: median rows between two values of the same patient. Describes how often labs were ordered; never a feature (ADR-0002).

| column | pct_hours_A | pct_hours_B | pct_patients_A | pct_patients_B | median_hours_A | median_hours_B |
|---|---|---|---|---|---|---|
| HR | 92.3 | 87.9 | 100.0 | 100.0 | 1.0 | 1.0 |
| O2Sat | 87.9 | 85.8 | 99.9 | 100.0 | 1.0 | 1.0 |
| Temp | 33.7 | 33.8 | 98.9 | 99.8 | 2.0 | 2.0 |
| SBP | 84.8 | 86.0 | 98.7 | 99.9 | 1.0 | 1.0 |
| MAP | 89.7 | 85.2 | 100.0 | 99.5 | 1.0 | 1.0 |
| DBP | 51.5 | 86.0 | 63.4 | 99.9 | 1.0 | 1.0 |
| Resp | 90.2 | 78.9 | 99.9 | 99.8 | 1.0 | 1.0 |
| BaseExcess | 10.4 | 0.2 | 62.1 | 2.8 | 2.0 | 1.0 |
| HCO3 | 8.0 | 0.2 | 97.4 | 2.1 | 8.0 | 1.0 |
| FiO2 | 14.1 | 2.2 | 58.6 | 28.6 | 3.0 | 4.0 |
| pH | 11.5 | 2.2 | 64.7 | 28.3 | 2.0 | 3.0 |
| PaCO2 | 8.8 | 2.2 | 61.8 | 28.4 | 3.0 | 3.0 |
| SaO2 | 5.0 | 1.9 | 39.2 | 25.2 | 2.0 | 3.0 |
| AST | 1.5 | 1.7 | 28.9 | 42.4 | 6.0 | 18.0 |
| BUN | 8.1 | 5.5 | 98.0 | 92.3 | 8.0 | 14.0 |
| Alkalinephos | 1.4 | 1.8 | 28.0 | 42.4 | 6.0 | 17.0 |
| Calcium | 5.0 | 5.4 | 81.5 | 92.0 | 12.0 | 15.0 |
| Chloride | 8.3 | 0.6 | 97.4 | 7.9 | 8.0 | 3.0 |
| Creatinine | 6.6 | 5.5 | 97.8 | 92.3 | 12.0 | 14.0 |
| Bilirubin_direct | 0.1 | 0.2 | 2.9 | 7.2 | 5.0 | 15.0 |
| Glucose | 12.1 | 22.0 | 98.0 | 94.3 | 4.0 | 3.0 |
| Lactate | 3.4 | 1.9 | 37.8 | 23.6 | 2.0 | 3.0 |
| Magnesium | 7.8 | 4.8 | 93.2 | 82.3 | 8.0 | 13.0 |
| Phosphate | 5.0 | 2.9 | 82.2 | 58.3 | 12.0 | 19.0 |
| Potassium | 10.8 | 7.7 | 97.9 | 93.0 | 5.0 | 7.0 |
| Bilirubin_total | 1.2 | 1.8 | 28.3 | 42.5 | 11.0 | 17.0 |
| TroponinI | 0.1 | 1.8 | 2.4 | 32.8 | 8.0 | 7.0 |
| Hct | 11.7 | 5.8 | 98.3 | 90.4 | 5.0 | 11.0 |
| Hgb | 8.8 | 5.9 | 97.5 | 90.5 | 6.0 | 11.0 |
| PTT | 4.8 | 1.0 | 77.6 | 21.9 | 8.0 | 7.0 |
| WBC | 7.5 | 5.3 | 97.0 | 90.2 | 8.0 | 15.0 |
| Fibrinogen | 0.8 | 0.5 | 12.6 | 9.7 | 4.0 | 7.0 |
| Platelets | 6.5 | 5.3 | 97.2 | 90.2 | 12.0 | 14.0 |

![missingness](figures/missingness.png)

![lab gaps](figures/lab_gaps.png)

## 3. Labs measured together (candidate panels)

Labs clustered on 1 - (how often they share an hour), average linkage, distance threshold 0.6. Input for `configs/panels.yaml` (M0-6).

| hospital | size | labs |
|---|---|---|
| A | 18 | HCO3, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Magnesium, Phosphate, Potassium, Bilirubin_total, Hct, Hgb, PTT, WBC, Platelets |
| A | 5 | BaseExcess, pH, PaCO2, SaO2, Lactate |
| A | 1 | FiO2 |
| A | 1 | Fibrinogen |
| A | 1 | TroponinI |
| B | 15 | AST, BUN, Alkalinephos, Calcium, Creatinine, Bilirubin_direct, Glucose, Magnesium, Phosphate, Potassium, Bilirubin_total, Hct, Hgb, WBC, Platelets |
| B | 8 | BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, Chloride, Lactate |
| B | 2 | PTT, Fibrinogen |
| B | 1 | TroponinI |

![co-measurement](figures/co_measurement.png)

## 4. Septic vs non-septic

Label-1 hours of septic patients against all hours of patients who never become septic. Top 10 per hospital by |standardized mean difference|, among variables with at least 200 values on each side:

| hospital | column | septic_median | nonseptic_median | smd | n_septic | n_nonseptic |
|---|---|---|---|---|---|---|
| B | HR | 90.00 | 82.00 | 0.42 | 7780 | 486571 |
| B | BUN | 26.00 | 17.00 | 0.41 | 705 | 30220 |
| B | FiO2 | 0.50 | 0.40 | 0.39 | 745 | 10509 |
| A | Resp | 20.00 | 18.00 | 0.39 | 12685 | 492803 |
| B | Temp | 37.20 | 36.90 | 0.36 | 3252 | 185962 |
| A | HR | 90.00 | 83.00 | 0.35 | 12856 | 504604 |
| A | BUN | 23.00 | 17.00 | 0.33 | 1333 | 43593 |
| B | WBC | 12.00 | 9.40 | 0.32 | 693 | 28809 |
| A | Temp | 37.30 | 37.00 | 0.28 | 4392 | 185564 |
| B | Bilirubin_total | 1.20 | 0.80 | 0.26 | 348 | 9467 |
| B | SaO2 | 97.10 | 97.50 | -0.25 | 665 | 8865 |
| B | Resp | 19.00 | 18.00 | 0.23 | 6496 | 438769 |
| A | Bilirubin_total | 1.20 | 0.80 | 0.21 | 303 | 6237 |
| B | DBP | 61.00 | 65.00 | -0.20 | 7660 | 475824 |
| A | Creatinine | 1.10 | 0.90 | 0.19 | 1036 | 35863 |
| B | Creatinine | 1.31 | 0.97 | 0.19 | 705 | 30251 |
| A | Hgb | 10.20 | 10.50 | -0.15 | 1420 | 47804 |
| A | WBC | 11.60 | 10.80 | 0.14 | 1207 | 40362 |
| A | PTT | 35.00 | 32.10 | 0.13 | 793 | 25896 |
| A | HCO3 | 24.00 | 24.00 | -0.10 | 1325 | 42946 |

![septic vs non-septic](figures/septic_vs_nonseptic.png)

Onset-aligned medians with IQR band, each patient's last value carried forward (earlier hours only) (48 h before to 6 h after t_sepsis, rows; points with fewer than 20 patients hidden). Dashed: median of never-septic patients over hours where the value was measured (not carried forward), so for labs it reflects the patients who got that lab ordered.

![trajectories](figures/trajectories.png)

## 5. Correlation and redundant variables

Pairs with |Spearman r| >= 0.9, among pairs measured together in at least 500 hours:

| hospital | a | b | r |
|---|---|---|---|
| A | Hct | Hgb | 0.96 |
| A | Bilirubin_direct | Bilirubin_total | 0.95 |
| B | BaseExcess | HCO3 | 0.98 |
| B | Hct | Hgb | 0.97 |

![correlation A](figures/correlation_A.png)

![correlation B](figures/correlation_B.png)

PCA of per-patient medians, as a density (plot only, never a model input):

![pca](figures/pca.png)

## 6. Hospital A vs B

Variables with |standardized mean difference| >= 0.2, largest first (values: means over hours with a value):

| column | mean_A | mean_B | smd | pct_hours_A | pct_hours_B |
|---|---|---|---|---|---|
| SaO2 | 91.29 | 96.59 | -0.59 | 4.95 | 1.86 |
| Bilirubin_direct | 3.14 | 1.03 | 0.55 | 0.15 | 0.23 |
| MAP | 78.73 | 86.30 | -0.48 | 89.75 | 85.19 |
| DBP | 59.92 | 66.18 | -0.47 | 51.48 | 85.97 |
| BaseExcess | -0.69 | -2.59 | 0.45 | 10.43 | 0.22 |
| SBP | 120.89 | 126.52 | -0.24 | 84.78 | 86.00 |
| Bilirubin_total | 2.69 | 1.64 | 0.24 | 1.21 | 1.76 |
| HCO3 | 24.08 | 23.30 | 0.20 | 8.02 | 0.18 |
| AST | 346.82 | 175.27 | 0.20 | 1.46 | 1.75 |

![hospital shift](figures/hospital_shift.png)
<!-- generated:end -->

## Findings

Hand-written after reading the tables and figures above: finding -> what it means for the project. Kept when the analysis is re-run.
