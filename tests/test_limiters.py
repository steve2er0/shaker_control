"""Numerical regressions for the voltage waveform limiters; no DAQ required."""

from functools import partial

import numpy as np
import pytest

from rv_controller import apply_safety_limiters, crest_factor_limit, rms_limit


def measured_rms(signal):
    """Independent RMS measurement, scaled to handle extreme amplitudes."""
    peak = np.max(np.abs(signal)) if signal.size else 0.0
    return peak * np.sqrt(np.mean((signal / peak) ** 2)) if peak else 0.0


def measured_cf(signal):
    peak = np.max(np.abs(signal)) if signal.size else 0.0
    return 1.0 / np.sqrt(np.mean((signal / peak) ** 2)) if peak else 0.0


@pytest.mark.parametrize("max_cf", [1.0, 2.0, 5.0])
@pytest.mark.parametrize("soft_knee", [0.0, 0.8, 1.0])
def test_peaks_are_limited_against_output_rms(max_cf, soft_knee):
    signal = np.random.default_rng(42).normal(size=4096)
    signal[0], signal[1] = 100.0, -80.0
    original = signal.copy()

    limited, cf_before, cf_after, active = crest_factor_limit(
        signal, max_cf=max_cf, soft_knee=soft_knee
    )

    assert active
    assert measured_cf(limited) <= max_cf + 1e-12
    assert cf_before == pytest.approx(measured_cf(signal))
    assert cf_after == pytest.approx(measured_cf(limited))
    assert np.all(np.abs(limited) <= np.abs(signal) + 1e-12)
    assert measured_rms(limited) <= measured_rms(signal)
    np.testing.assert_array_equal(signal, original)


def test_infeasible_sparse_impulse_returns_zero_and_truthful_statistics():
    signal = np.zeros(1024)
    signal[0] = 1.0

    limited, cf_before, cf_after, active = crest_factor_limit(signal, max_cf=5.0)

    assert cf_before == pytest.approx(32.0)
    assert active
    np.testing.assert_array_equal(limited, np.zeros_like(signal))
    assert cf_after == measured_cf(limited) == 0.0


@pytest.mark.parametrize("signal", [
    np.array([]),
    np.zeros(16),
    np.full(16, 2.0),
    np.sin(2.0 * np.pi * np.arange(1024) / 32.0),
])
def test_acceptable_waveforms_are_unchanged(signal):
    limited, cf_before, cf_after, active = crest_factor_limit(signal, max_cf=5.0)

    np.testing.assert_array_equal(limited, signal)
    assert not active
    assert cf_before == pytest.approx(measured_cf(signal))
    assert cf_after == pytest.approx(measured_cf(limited))


@pytest.mark.parametrize("amplitude", [1e-300, 1e300])
def test_crest_limiting_is_independent_of_amplitude(amplitude):
    signal = np.tile(np.array([-1.0, 1.0]), 512)
    signal[0] = 100.0
    signal *= amplitude

    limited, _, cf_after, active = crest_factor_limit(signal, max_cf=3.0)

    assert active
    assert np.all(np.isfinite(limited))
    assert np.any(limited)
    assert measured_cf(limited) <= 3.0 + 1e-12
    assert cf_after == pytest.approx(measured_cf(limited))


@pytest.mark.parametrize("max_cf", [0.0, 0.99, -1.0, np.nan, np.inf])
def test_invalid_crest_limit_is_rejected(max_cf):
    with pytest.raises(ValueError):
        crest_factor_limit(np.ones(8), max_cf=max_cf)


@pytest.mark.parametrize("soft_knee", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_soft_knee_is_rejected(soft_knee):
    with pytest.raises(ValueError):
        crest_factor_limit(np.ones(8), soft_knee=soft_knee)


@pytest.mark.parametrize("limiter", [
    crest_factor_limit, partial(rms_limit, max_rms=1.4), apply_safety_limiters,
])
@pytest.mark.parametrize("signal", [np.array([np.nan]), np.array([np.inf]), np.ones((2, 8))])
def test_invalid_waveform_is_rejected(limiter, signal):
    with pytest.raises(ValueError):
        limiter(signal)


def test_rms_limit_includes_dc_component():
    signal = np.full(1024, 10.0)

    limited, rms_before, rms_after, active = rms_limit(signal, 1.4, headroom=0.9)

    assert active
    assert rms_before == pytest.approx(10.0)
    assert measured_rms(limited) == pytest.approx(1.26)
    assert rms_after == pytest.approx(measured_rms(limited))


@pytest.mark.parametrize("signal", [np.array([]), np.zeros(16), np.full(16, 0.1)])
def test_rms_limit_preserves_acceptable_waveforms(signal):
    limited, rms_before, rms_after, active = rms_limit(signal, 1.4)

    np.testing.assert_array_equal(limited, signal)
    assert not active
    assert rms_before == pytest.approx(measured_rms(signal))
    assert rms_after == pytest.approx(measured_rms(limited))


@pytest.mark.parametrize("max_rms,headroom", [
    (-1.0, 0.9), (np.nan, 0.9), (np.inf, 0.9),
    (1.4, -0.1), (1.4, 1.1), (1.4, np.nan), (1.4, np.inf),
])
def test_invalid_rms_settings_are_rejected(max_rms, headroom):
    with pytest.raises(ValueError):
        rms_limit(np.ones(8), max_rms, headroom)


def test_combined_statistics_describe_final_output_after_both_limiters():
    signal = np.random.default_rng(7).normal(size=4096)
    signal[0], signal[1] = 200.0, -160.0

    limited, stats = apply_safety_limiters(signal, max_cf=3.0, max_rms=0.5)

    assert stats['rms_limiting'] and stats['cf_limiting'] and stats['any_limiting']
    assert measured_rms(limited) <= 0.5
    assert measured_cf(limited) <= 3.0 + 1e-12
    assert stats['rms_before'] == pytest.approx(measured_rms(signal))
    assert stats['rms_after'] == pytest.approx(measured_rms(limited))
    assert stats['cf_after'] == pytest.approx(measured_cf(limited))


def test_combined_zero_block_has_zero_metrics():
    limited, stats = apply_safety_limiters(np.zeros(128))

    assert not stats['any_limiting']
    assert stats['rms_after'] == stats['cf_after'] == 0.0
    np.testing.assert_array_equal(limited, np.zeros(128))


def test_sparse_waveform_at_attainable_boundary_retains_nonzero_output():
    signal = np.zeros(100)
    signal[:4] = [1.0, -2.0, 3.0, -4.0]

    limited, _, cf_after, active = crest_factor_limit(signal, max_cf=5.0)

    assert active
    assert np.count_nonzero(limited) == 4
    assert measured_cf(limited) == pytest.approx(5.0)
    assert cf_after == pytest.approx(measured_cf(limited))
    np.testing.assert_array_equal(np.sign(limited), np.sign(signal))
