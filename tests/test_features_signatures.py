"""Signature features (M4-1, B3): the maths on hand-checked toy paths, the windows, the ADR-0002
channel guard and the blocks through the builder. No real patient data.

The leakage suite (tests/test_feature_leakage.py) and the extra-draws test
(tests/test_features_forbidden.py) cover the signature blocks automatically once registered.
"""

import numpy as np
import pandas as pd
import pytest

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.features.signatures import (
    cumsum_lead_lag_path,
    lead_lag_path,
    log_signature,
    lyndon_words,
    signature,
    signature_block,
    time_path,
    windows,
)

NAN = np.nan
SOFA = load_yaml(FEATURES_CONFIG)["thresholds"]["sofa"]


def _path(*points: tuple[float, ...]) -> np.ndarray:
    """One path as a batch of one: shape (1, points, dims)."""
    return np.asarray(points, dtype=float)[None]


# --- signature maths ---------------------------------------------------------------------------


def test_level_one_is_the_net_change() -> None:
    s1, _, _ = signature(_path((0, 1), (2, 5), (3, 2)), depth=3)
    np.testing.assert_allclose(s1[0], [3.0, 1.0])


def test_level_two_of_an_l_shaped_path_is_hand_computed() -> None:
    # right 1, then up 1: S^{12} = 1 (x is 1 while y rises), S^{21} = 0, S^{ii} = 1/2.
    _, s2 = signature(_path((0, 0), (1, 0), (1, 1)), depth=2)
    np.testing.assert_allclose(s2[0], [[0.5, 1.0], [0.0, 0.5]])


def test_shuffle_identity() -> None:
    rng = np.random.default_rng(0)
    path = np.cumsum(rng.normal(size=(4, 9, 2)), axis=1)
    s1, s2, s3 = signature(path, depth=3)
    # S(i)·S(j) = S(ij) + S(ji)
    np.testing.assert_allclose(s1[:, :, None] * s1[:, None, :], s2 + s2.transpose(0, 2, 1))
    # S(i)·S(jk) = S(ijk) + S(jik) + S(jki)
    lhs = s1[:, :, None, None] * s2[:, None, :, :]
    rhs = s3 + s3.transpose(0, 2, 1, 3) + s3.transpose(0, 3, 1, 2)  # at [i,j,k]: ijk, jik, jki
    np.testing.assert_allclose(lhs, rhs, atol=1e-9)


def test_chen_identity_joins_two_paths() -> None:
    rng = np.random.default_rng(1)
    path = np.cumsum(rng.normal(size=(3, 11, 2)), axis=1)
    a, b, whole = path[:, :6], path[:, 5:], path
    a1, a2, a3 = signature(a, depth=3)
    b1, b2, b3 = signature(b, depth=3)
    w1, w2, w3 = signature(whole, depth=3)

    def outer(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        flat = x.reshape(n, -1, 1) * y.reshape(n, 1, -1)
        return flat.reshape(n, *x.shape[1:], *y.shape[1:])

    np.testing.assert_allclose(w1, a1 + b1)
    np.testing.assert_allclose(w2, a2 + b2 + outer(a1, b1), atol=1e-9)
    np.testing.assert_allclose(w3, a3 + b3 + outer(a2, b1) + outer(a1, b2), atol=1e-9)


def test_lyndon_words_of_two_letters_to_depth_three() -> None:
    assert lyndon_words(2, 3) == [(1,), (2,), (1, 2), (1, 1, 2), (1, 2, 2)]
    assert lyndon_words(2, 2) == [(1,), (2,), (1, 2)]


def test_log_signature_of_two_unit_steps_matches_bch() -> None:
    # Steps a = e1, b = e2. BCH: log = a + b + [a,b]/2 + ([a,[a,b]] + [b,[b,a]])/12, whose
    # tensor coefficients at the Lyndon words 1, 2, 12, 112, 122 are 1, 1, 1/2, 1/12, 1/12.
    out = log_signature(_path((0, 0), (1, 0), (1, 1)), depth=3)
    np.testing.assert_allclose(out[0], [1.0, 1.0, 0.5, 1 / 12, 1 / 12], atol=1e-12)


def test_log_signature_of_a_straight_line_has_only_level_one() -> None:
    line = _path(*[(k, 2.0 + 0.5 * k) for k in range(8)])
    np.testing.assert_allclose(log_signature(line, depth=3)[0], [7.0, 3.5, 0, 0, 0], atol=1e-9)


def test_log_signature_area_of_a_lead_lag_path_is_half_the_squared_steps() -> None:
    x = np.array([[3.0, 4.0, 2.0, 2.0, 5.0]])
    out = log_signature(lead_lag_path(x), depth=2)
    assert out[0, 2] == pytest.approx(0.5 * np.sum(np.diff(x[0]) ** 2))


def test_depth_outside_one_to_three_is_refused() -> None:
    with pytest.raises(ValueError, match="depth"):
        signature(_path((0, 0), (1, 1)), depth=4)


# --- paths -------------------------------------------------------------------------------------


def test_time_path_pairs_the_row_position_with_the_value() -> None:
    np.testing.assert_allclose(time_path(np.array([[5.0, 7.0, 6.0]]))[0], [[0, 5], [1, 7], [2, 6]])


def test_lead_lag_path_points() -> None:
    got = lead_lag_path(np.array([[1.0, 3.0, 2.0]]))[0]
    np.testing.assert_allclose(got, [[1, 1], [3, 1], [3, 3], [2, 3], [2, 2]])


def test_cumsum_lead_lag_starts_from_zero_so_every_value_counts() -> None:
    got = cumsum_lead_lag_path(np.array([[1.0, 3.0]]))[0]
    np.testing.assert_allclose(got, [[0, 0], [1, 0], [1, 1], [4, 1], [4, 4]])


# --- windows -----------------------------------------------------------------------------------


def _two_patients() -> pd.DataFrame:
    a = pd.DataFrame({"patient_id": "A_1", "hour": range(5), "HR": [NAN, 80.0, 90.0, 100.0, 70.0]})
    b = pd.DataFrame({"patient_id": "B_1", "hour": range(3), "HR": [200.0, 60.0, 65.0]})
    return pd.concat([a, b], ignore_index=True)


def test_windows_end_at_the_row_and_pad_from_the_stay_start() -> None:
    values = _two_patients()
    got = windows(values, values["HR"], window_rows=3)
    # A_1 row 0 has no value at all: NaN. Row 1: padded with its first value (row 0 is NaN, so
    # filled from inside the window).
    assert np.isnan(got[0]).all()
    np.testing.assert_allclose(got[1], [80, 80, 80])
    np.testing.assert_allclose(got[3], [80, 90, 100])
    np.testing.assert_allclose(got[4], [90, 100, 70])
    # B_1 never sees A_1's rows.
    np.testing.assert_allclose(got[5], [200, 200, 200])
    np.testing.assert_allclose(got[7], [200, 60, 65])


def test_windows_fill_a_gap_from_earlier_rows_of_the_window() -> None:
    values = pd.DataFrame({"patient_id": "A_1", "hour": range(4), "HR": [70.0, NAN, 90.0, NAN]})
    np.testing.assert_allclose(windows(values, values["HR"], window_rows=4)[3], [70, 70, 90, 90])


def test_windows_never_read_a_later_row() -> None:
    values = _two_patients()
    changed = values.copy()
    changed.loc[3:4, "HR"] = [1.0, 2.0]
    before = windows(values, values["HR"], window_rows=3)
    after = windows(changed, changed["HR"], window_rows=3)
    np.testing.assert_array_equal(before[:3], after[:3])


def test_windows_refuse_unsorted_rows() -> None:
    values = _two_patients().iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        windows(values, values["HR"], window_rows=3)


# --- blocks and the ADR-0002 guard -------------------------------------------------------------


def _vitals(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": "A_1",
            "hour": range(n),
            "HR": np.linspace(80, 105, n),
            "SBP": [120.0, 118, NAN, 110, 105, 100][:n],
            "MAP": [80.0, 78, 75, 74, 70, 66][:n],
            "Platelets": [NAN, NAN, 120.0, 120.0, 90.0, 90.0][:n],
            "Bilirubin_total": NAN,
            "Creatinine": NAN,
            "BUN": [NAN, 20.0, 20.0, 20.0, 25.0, 25.0][:n],
        }
    )


def _settings(**over: object) -> dict:
    return {"channels": ["MAP"], "path": "time", "depth": 3, "window_rows": 4, **over}


def test_block_columns_are_named_by_block_channel_and_word() -> None:
    out = signature_block(_vitals(), _settings(channels=["MAP", "HR"]), SOFA, name="sig_vitals")
    words = ["1", "2", "12", "112", "122"]
    assert list(out.columns) == [f"sig_vitals_{c}_{w}" for c in ("MAP", "HR") for w in words]
    assert out.index.equals(_vitals().index)


def test_block_matches_the_log_signature_of_each_window() -> None:
    values = _vitals()
    out = signature_block(values, _settings(), SOFA, name="sig_vitals")
    expected = log_signature(time_path(windows(values, values["MAP"], window_rows=4)), depth=3)
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_derived_channels_are_computed_from_the_view() -> None:
    values = _vitals()
    out = signature_block(
        values, _settings(channels=["shock_index", "partial_sofa"]), SOFA, name="sig_x"
    )
    assert {"sig_x_shock_index_2", "sig_x_partial_sofa_2"} <= set(out.columns)
    # Row 2 has no SBP but the window still holds earlier shock-index values.
    assert np.isfinite(out["sig_x_shock_index_2"].iloc[2])


@pytest.mark.parametrize("channel", ["bun_cr", "BUN", "Lactate", "FiO2", "DBP", "EtCO2", "SaO2"])
def test_lab_and_excluded_channels_are_refused(channel: str) -> None:
    with pytest.raises(ValueError, match="ADR-0002"):
        signature_block(_vitals(), _settings(channels=[channel]), SOFA, name="sig_x")


def test_a_cumulative_sum_takes_vitals_only() -> None:
    settings = _settings(channels=["partial_sofa"], path="cumsum_lead_lag")
    with pytest.raises(ValueError, match="vitals only"):
        signature_block(_vitals(), settings, SOFA, name="sig_x")


def test_unknown_path_is_refused() -> None:
    with pytest.raises(ValueError, match="path"):
        signature_block(_vitals(), _settings(path="spline"), SOFA, name="sig_x")


def test_signature_blocks_in_the_config_are_off_and_use_allowed_channels() -> None:
    config = load_yaml(FEATURES_CONFIG)
    blocks = config["signatures"]
    assert set(blocks) == {"sig_vitals", "sig_vitals_cumsum", "sig_partial_sofa"}
    assert all(not block["enabled"] for block in blocks.values())
    for name, block in blocks.items():  # each block is valid: no refusal
        signature_block(_vitals(), block, SOFA, name=name)


# --- optional cross-check ----------------------------------------------------------------------


def test_signature_matches_iisignature_if_installed() -> None:
    iisignature = pytest.importorskip("iisignature")
    rng = np.random.default_rng(2)
    path = np.cumsum(rng.normal(size=(5, 8, 2)), axis=1)
    s1, s2, s3 = signature(path, depth=3)
    ours = np.concatenate([s1.reshape(5, -1), s2.reshape(5, -1), s3.reshape(5, -1)], axis=1)
    np.testing.assert_allclose(ours, iisignature.sig(path, 3), atol=1e-9)
