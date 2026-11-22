# ADR-0002: No lab measurement-count or missingness-pattern features

Status: accepted

## Context
The top three challenge teams all used clinician measurement behaviour as signal: rank 1 counted measurements per window, rank 2 used missing flags, rank 3 encoded runs of missing/present values. In the original data this means "a clinician was worried enough to order a test".

In our replay, which labs are present is decided by **our own action model**. The same features would then describe our policy rather than the patient, and the risk model would be trained on a different meaning than it sees at replay time. The rank 1 authors also link their hospital C failure to features that depend on local ordering practice.

## Decision
- Lab features may use: last revealed value, `hours_since_<panel>` (NaN if never revealed), and values derived from revealed labs (PartialSOFA, BUN/Cr).
- No counts of lab measurements, no missing/present run features, no "was measured this hour" flags for labs.
- No differences of lab values, forward-filled or between the last two revealed values: the first is non-zero only when a new draw arrives, the second is NaN until a second draw (M2-2). Lab trends come from PartialSOFA's 24-row change.
- Vitals are exempt: they are recorded every hour independent of our policy.
- Signature channels (B3, M4-1) are vitals, `shock_index` and `partial_sofa` only, and a cumulative-sum signature takes vitals only. A forward-filled lab changes only when a new draw is revealed, so a lab channel's path steps are lab differences, and its level-2 terms say when in the window the draw arrived. `partial_sofa` is allowed for the same reason as its 24-row change; `bun_cr`, FiO2 ratios and other labs are not. `sepsis.features.signatures` refuses any other channel.
- The risk model is trained with random lab masking so it learns to work on partial views.

## Consequences
- B2 may score below the published rank 1 numbers. That is expected and should be stated in the report.
- An ablation "with vs. without count features" on the full-information setting is allowed as an analysis, never as the deployed model.
