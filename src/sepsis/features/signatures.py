"""Signature features (M4-1, B3): log-signatures of short paths ending at the current row.

Rank #1's idea (docs/research/1_CanIgetyoursignature_*.md), computed here with NumPy to depth 3
instead of `iisignature`. For each row t and channel, the window of the last `window_rows` rows
(ADR-0005: rows t - w + 1 ... t, same patient) becomes a 2-D path:
- `time`: (row position, value). Level 1 is the net change; the level-2 area says whether the
  change came early or late in the window.
- `cumsum_lead_lag`: the running sum from 0 (so every value counts), then lead-lag. The level-2
  area of a lead-lag path is half the sum of squared steps; vitals only (ADR-0002).

The features are the log-signature coordinates at the Lyndon words (`1, 2, 12, 112, 122` at
depth 3). They are the tensor-log coefficients, an invertible linear map of iisignature's Lyndon
basis, so they carry the same information.

ADR-0002: a forward-filled lab changes only when a new draw is revealed, so a signature of a lab
channel would be a lab difference. Channels are therefore vitals, `shock_index` (HR / SBP) and
`partial_sofa` (allowed like its 24-row change, `partial_sofa_delta`); anything else is refused.
Windows look back only, within the patient: nothing reads a later row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from sepsis.features import clinical
from sepsis.features._frame import require_sorted

# Vitals the feature rows show: DBP is in `raw.exclude` (its missingness marks hospital A, EDA
# finding 3) and EtCO2 is dropped by cleaning.
VITAL_CHANNELS = ("HR", "O2Sat", "Temp", "SBP", "MAP", "Resp")
DERIVED_CHANNELS = ("shock_index", "partial_sofa")
PATHS = ("time", "cumsum_lead_lag")
MAX_DEPTH = 3


# --- windows -----------------------------------------------------------------------------------


def windows(values: pd.DataFrame, channel: pd.Series, window_rows: int) -> np.ndarray:
    """`(rows, window_rows)`: for each row t, the channel at rows t - w + 1 ... t of its patient.

    Rows before the stay start repeat the first row (rank #1's `fixed` padding). A NaN inside the
    window takes the latest earlier value in the window, and leading NaNs the first value in it
    (both at rows <= t). A window with no value is all NaN.
    """
    require_sorted(values)
    w = int(window_rows)
    if w < 2:
        raise ValueError("window_rows must be at least 2 (a path needs two points)")
    pid = values["patient_id"].to_numpy()
    n = len(pid)
    rows = np.arange(n)
    is_start = np.ones(n, dtype=bool)
    is_start[1:] = pid[1:] != pid[:-1]
    start = np.maximum.accumulate(np.where(is_start, rows, 0))
    index = np.maximum(start[:, None], rows[:, None] - (w - 1) + np.arange(w)[None, :])
    window = channel.to_numpy(dtype=float)[index]

    present = ~np.isnan(window)
    position = np.where(present, np.arange(w)[None, :], 0)
    np.maximum.accumulate(position, axis=1, out=position)
    window = np.take_along_axis(window, position, axis=1)  # forward fill inside the window
    first = np.argmax(present, axis=1)
    leading = np.arange(w)[None, :] < first[:, None]
    first_value = np.take_along_axis(window, first[:, None], axis=1)
    return np.where(leading, first_value, window)


# --- paths -------------------------------------------------------------------------------------


def time_path(window: np.ndarray) -> np.ndarray:
    """`(rows, w, 2)`: (row position 0 ... w - 1, value)."""
    time = np.broadcast_to(np.arange(window.shape[1], dtype=float), window.shape)
    return np.stack([time, window], axis=-1)


def lead_lag_path(window: np.ndarray) -> np.ndarray:
    """`(rows, 2w - 1, 2)`: (x0, x0), (x1, x0), (x1, x1), (x2, x1), ... (lead first)."""
    doubled = np.repeat(window, 2, axis=1)
    return np.stack([doubled[:, 1:], doubled[:, :-1]], axis=-1)


def cumsum_lead_lag_path(window: np.ndarray) -> np.ndarray:
    """Lead-lag of the running sum, starting from 0 so every value of the window counts."""
    zero = np.zeros((window.shape[0], 1))
    return lead_lag_path(np.concatenate([zero, np.cumsum(window, axis=1)], axis=1))


# --- signatures --------------------------------------------------------------------------------


def _outer(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tensor product per row: `(n, *a) x (n, *b) -> (n, *a, *b)`."""
    n = x.shape[0]
    return (x.reshape(n, -1, 1) * y.reshape(n, 1, -1)).reshape(n, *x.shape[1:], *y.shape[1:])


def signature(path: np.ndarray, depth: int) -> tuple[np.ndarray, ...]:
    """The signature levels 1 ... depth of each path `(n, points, d)`, by Chen's identity.

    Each straight step Δ multiplies the signature by exp(Δ) = 1 + Δ + Δ⊗Δ/2 + Δ⊗Δ⊗Δ/6.
    """
    if not 1 <= int(depth) <= MAX_DEPTH:
        raise ValueError(f"depth must be 1 to {MAX_DEPTH}")
    n, _, d = path.shape
    s1, s2, s3 = np.zeros((n, d)), np.zeros((n, d, d)), np.zeros((n, d, d, d))
    for step in np.diff(path, axis=1).transpose(1, 0, 2):
        step2 = _outer(step, step)
        if depth >= 3:
            s3 = s3 + _outer(s2, step) + _outer(s1, step2) / 2 + _outer(step2, step) / 6
        if depth >= 2:
            s2 = s2 + _outer(s1, step) + step2 / 2
        s1 = s1 + step
    return (s1, s2, s3)[: int(depth)]


def lyndon_words(letters: int, depth: int) -> list[tuple[int, ...]]:
    """Lyndon words over 1 ... letters up to length `depth`, by length, then lexicographically."""
    words, word = [], [1]
    while word:  # Duval's algorithm
        if len(word) <= depth:
            words.append(tuple(word))
        word = (word * (depth // len(word) + 1))[:depth]
        while word and word[-1] == letters:
            word.pop()
        if word:
            word[-1] += 1
    return sorted(words, key=lambda w: (len(w), w))


def log_signature(path: np.ndarray, depth: int) -> np.ndarray:
    """`(n, words)`: the tensor log of the signature at the Lyndon words (`lyndon_words`)."""
    levels = signature(path, depth)
    s1 = levels[0]
    logs = [s1]
    if depth >= 2:
        s2 = levels[1]
        logs.append(s2 - _outer(s1, s1) / 2)
    if depth >= 3:
        s3 = levels[2]
        logs.append(s3 - (_outer(s1, s2) + _outer(s2, s1)) / 2 + _outer(_outer(s1, s1), s1) / 3)
    words = lyndon_words(path.shape[2], depth)
    rows = np.arange(path.shape[0])
    return np.stack(
        [logs[len(w) - 1][(rows, *(letter - 1 for letter in w))] for w in words], axis=1
    )


# --- feature block -----------------------------------------------------------------------------


def _check(channels: Sequence[str], path: str) -> None:
    if path not in PATHS:
        raise ValueError(f"unknown signature path {path!r}; use one of {PATHS}")
    refused = [c for c in channels if c not in VITAL_CHANNELS + DERIVED_CHANNELS]
    if refused:
        raise ValueError(
            f"signature channels {refused} are not allowed (ADR-0002): use vitals "
            f"{list(VITAL_CHANNELS)} or {list(DERIVED_CHANNELS)}"
        )
    if path == "cumsum_lead_lag" and not set(channels) <= set(VITAL_CHANNELS):
        raise ValueError("a cumulative-sum signature takes vitals only (ADR-0002)")


def _channel(values: pd.DataFrame, name: str, sofa: Mapping[str, Any]) -> pd.Series:
    if name == "shock_index":
        return clinical.shock_index(values)["shock_index"].set_axis(values.index)
    if name == "partial_sofa":
        return clinical.partial_sofa(values, sofa)["partial_sofa"].set_axis(values.index)
    if name in values.columns:
        return values[name]
    return pd.Series(np.nan, index=values.index)


def signature_block(
    values: pd.DataFrame, settings: Mapping[str, Any], sofa: Mapping[str, Any], *, name: str
) -> pd.DataFrame:
    """`<name>_<channel>_<word>`: the log-signature of each channel's window, channel by channel.

    `settings`: `channels`, `path` (`time` or `cumsum_lead_lag`), `depth` (1-3), `window_rows`.
    """
    channels, path = list(settings["channels"]), str(settings["path"])
    _check(channels, path)
    depth, w = int(settings["depth"]), int(settings["window_rows"])
    make_path = time_path if path == "time" else cumsum_lead_lag_path
    words = ["".join(map(str, word)) for word in lyndon_words(2, depth)]
    out = {}
    for channel in channels:
        terms = log_signature(make_path(windows(values, _channel(values, channel, sofa), w)), depth)
        for k, word in enumerate(words):
            out[f"{name}_{channel}_{word}"] = terms[:, k]
    return pd.DataFrame(out, index=values.index)
