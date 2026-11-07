"""Clinical onset time t_sepsis, defined exactly as the official scorer does."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

# SepsisLabel is already shifted 6 h early. The official scorer sets
# t_sepsis = argmax(labels) - dt_optimal with dt_optimal = -6
# (third_party/evaluate_sepsis_score.py).
# This is the challenge's definition, not a tunable setting.
LABEL_LEAD_HOURS = 6


def t_sepsis(labels: Sequence[int] | pd.Series) -> float:
    """Return the onset row: first SepsisLabel == 1 row index + 6, or NaN if never septic.

    Counts rows (0-based, in file order), not ICULOS. Never shift SepsisLabel again.
    """
    hits = np.flatnonzero(np.asarray(labels) == 1)
    return float(hits[0] + LABEL_LEAD_HOURS) if hits.size else math.nan
