# ADR-0006: Lab panels and reveal rules

Status: accepted (M0-6). Settings in `configs/panels.yaml`, checked by `sepsis.data.panels.validate_panels`.

## Context
The action model orders panels, and the replay reveals their labs. Both need a fixed mapping from the 26 lab columns to panels, and a rule for which recorded values a request may reveal and when they become visible. The rule must not let a request see values recorded after it.

Evidence from dev patients (cleaned, both hospitals):
- **Co-measurement** (EDA section 3, finding 4): the blood-gas group (BaseExcess, pH, PaCO2, SaO2, Lactate) clusters in both hospitals. HCO3 and Chloride cluster with the chemistry labs in A and with the blood gas in B. Checked further for this ADR: 100% of the HCO3 and Chloride hours also have BUN, Creatinine or Potassium in **both** hospitals (in B also 99% with pH or PaCO2).
- **FiO2:** in A only 33% of its hours also have pH or PaCO2 (it is charted like a ventilator setting); in B 84%.
- **Glucose:** in B it is drawn about every 3 rows, often alone: it is in 96% of B's chemistry hours, BUN in 24%.
- **Time between values** (EDA section 2 and finding 5, `p90_hours`, rows, A / B):
  - Lactate 12 / 12.
  - pH, PaCO2, SaO2: 9–10 in A, 14–15 in B. BaseExcess 9 / 2 and FiO2 5 / 15.
  - The core routine labs (BUN, Creatinine, WBC, Platelets, Hgb, PTT, Magnesium, Calcium, Phosphate, AST, Alkalinephos, Bilirubin_total): 23–27 in each hospital.
  - Exceptions:
    - drawn more often: Glucose 17 / 6, Potassium 19 / 24, Chloride 24 / 12, HCO3 24 / 3, Hct 18 / 25, Fibrinogen 13 / 17;
    - drawn less often: Bilirubin_direct 22 / 34.
  - TroponinI 18 / 16.
- **TroponinI** is measured for 2.4% of A patients and 32.8% of B (finding 3).

## Decision
1. **Seven clinical panels, one mapping for both hospitals.** Nothing is chosen per hospital, because that would fit the mapping on the evaluated hospital (ADR-0004). The clustering is used as a check, not as the source.

   | Panel | Labs | Look-back (rows) |
   |---|---|---|
   | LACTATE | Lactate | 12 |
   | ABG | pH, PaCO2, BaseExcess, SaO2, FiO2 | 12 |
   | CHEM | BUN, Creatinine, Glucose, Potassium, Chloride, HCO3, Calcium, Magnesium, Phosphate | 24 |
   | CBC | WBC, Hgb, Hct, Platelets | 24 |
   | LFT | AST, Alkalinephos, Bilirubin_total, Bilirubin_direct | 24 |
   | COAG | PTT, Fibrinogen | 24 |
   | TROPONIN | TroponinI | 24 |

   - HCO3 and Chloride go to CHEM: they come with the chemistry core in both hospitals.
   - FiO2 is a lab column and goes to ABG, so it stays hidden until revealed (finding 4). Making it always visible would need a new ADR.
   - Glucose stays in CHEM so the panel list in `CONTEXT.md` does not change. Because a request looks back over a window, B's frequent glucose draws do not stop CHEM from returning the last BUN or Creatinine.
2. **Look-back window.** A request for panel P at row t reveals every recorded value of P's labs in rows t − lookback_P + 1 … t (the window ends at t, t included).
   - 24 rows for CHEM, CBC, LFT, COAG and TROPONIN: about the p90 of their core labs, 23–27 in each hospital alone, so A-dev alone and B-dev alone give the same choice (finding 5).
   - 12 rows for LACTATE (p90 12 in both) and ABG. For ABG this is deliberately shorter than B's p90 (14–15) and longer than A's (9–10): blood gas changes fast, so a result older than 12 rows should not be returned as a new one. It is a fixed choice, not a fit to either hospital.
3. **No look-forward.** Values recorded after row t are never revealed by a request at t, even when the record holds a result one or two rows later. Using them would let the request see the future. This is the same reason M0-5 rejects interpolation (it uses later values unless cut at t).
4. **Reveal row.** A request at row t is made after the risk at t (the M5-1 loop: view → risk → alarm → policy → reveal). Its values get the **reveal row** t + `reveal_delay_rows` = t + 1. `reveal_delay_rows` must be at least 1.
5. **What the view holds at row s.** A revealed value is in the view at row s only if its reveal row ≤ s. It is never written back into rows before its reveal row: features at rows already scored are never recomputed, and forward-fill carries a revealed value forward from its reveal row, not from its recorded row. Its age still comes from the recorded row (point 7). Once revealed, it stays revealed. M2-1 and M5-1 must test that a value recorded at r and revealed at t + 1 does not change any feature at rows ≤ t.
6. **Unavailable request.** If no lab of P has a value in the window, the request is logged as unavailable and nothing is imputed (CONTEXT.md). EDA finding 7 expects this often for patients nobody tested.
7. **`hours_since_<panel>`** is the number of rows since the most recent revealed value of any lab in P was **recorded**, not since it was requested; NaN if none. A value revealed from 20 rows back reads as 20 rows old. It describes the value's age, so the model can discount stale values: with 24-row windows, values can be up to 23 rows old (finding 5). ADR-0002 names `hours_since_<panel>` as the allowed lab-timing feature. Under the "as recorded" training mask, its value also reflects when a clinician last ordered the panel. That is the price of giving the age of the value, and it is accepted: it carries one number per panel, never counts or missing/present runs.
8. **Training mask.** The "as recorded" mask (M2-1) treats every recorded value as if it had been requested at its own row, with the same `reveal_delay_rows`: a value recorded at row r joins the view from r + 1. So `hours_since_<panel>` is at least 1 in training as in replay, and is never a "measured this hour" flag (ADR-0002). Random panel masking is applied on top (ADR-0002, `lab_masking_p`).

## Consequences
- The TROPONIN panel can only be judged in B; in A → B and B → A it has almost no training or evaluation signal from A.
- In A, FiO2 behaves like a setting but is only seen after an ABG request, so the model sees less FiO2 in A than the record holds.
- A 24-row window means a CHEM request can return a value up to 23 rows old. `hours_since_<panel>` carries that age, so the model can discount it.
- Panel costs and the WAIT threshold are decided in M6, not here.
- The windows were read from A-dev and B-dev; they are fixed constants in every setup. The 24-row choice holds for each hospital alone; the ABG choice of 12 lies between the two hospitals' p90 values.
- Bilirubin_direct (p90 34 in B) and Glucose in A (17) fall outside the typical gap of their panel. Bilirubin_direct is rare in both hospitals and a proposal to drop it from features is in finding 8.
