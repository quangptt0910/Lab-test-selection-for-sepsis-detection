# ADR-0001: Adopt the PhysioNet 2019 rank #1 recipe as the risk-model baseline

Status: accepted. Superseded in part by ADR-0004: B3 is now kept only if it improves the mix setup without hurting A → B or B → A, and the evaluation protocol is defined there.

## Context
The challenge winner (Morrill et al., "Can I get your signature?") scored 0.360 utility on the hidden test set. Their ablation on the training data shows where the gains came from:

| Features | Utility (CV) |
|---|---|
| ICULOS only | 0.282 |
| + original 40 variables | 0.389 |
| + hand-crafted features | 0.422 |
| + signatures | 0.434 |

Most of the gain came from simple parts: forward-fill, hand-crafted clinical features, training on U1 − U0, and LightGBM with a large hyperparameter search. Signatures added +0.012. The model failed on hospital C (−0.123), which the authors attribute to features that depend on local ordering habits.

## Decision
- B2 = forward-fill (NaN before first value) + ICULOS + raw values + PartialSOFA (+24 h change) + shock index + BUN/Cr + rolling vital max/min + `hours_since_<panel>`, trained with LightGBM on U1 − U0.
- B3 = B2 + signature features, kept only if it beats B2 in both hospital-held-out directions.
- Measurement-count features are excluded (ADR-0002).
- Note (M3-1): B2 also includes the SIRS count, which rank #1 did not use, so B2 is exactly the `core` section of `configs/features.yaml` and "B2 beats B0" compares models with the same rule inputs. Its LightGBM parameters start from rank #1's regularised random-search values (`docs/research/1_…md`). Other clinical scores (criterion scores, NEWS2, qSOFA) are tested as an ablation, not as part of B2.

## Consequences
- The baseline is strong and well understood, so later gains are measurable.
- Hospital-held-out evaluation is mandatory, because generalization failure is the known weakness of this recipe.
