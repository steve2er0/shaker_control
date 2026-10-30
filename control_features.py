"""Numerical response protection and bounded sine amplitude feedback."""
from __future__ import annotations

import math
import numpy as np


FEATURE_DEFAULTS = {
    'response_limits_enabled': True, 'response_max_rms_g': 10., 'response_max_peak_g': 30.,
    'response_target_ratio': 2., 'response_max_backlog_seconds': 2.,
    'feedback_min_rms_g': .001, 'feedback_min_drive_v': .0005,
    'feedback_loss_seconds': 3., 'feedback_startup_seconds': 3.,
    'sine_closed_loop': True, 'sine_pilot_vpk': .001,
    'sine_correction_rate': .5, 'sine_settle_seconds': 1.,
    'sine_settle_blocks': 2, 'sine_max_step_seconds': 45.,
    'sine_scale_response_stops': True,
    'tracking_tolerance_percent': 5., 'report_min_seconds': 10., 'report_settle_seconds': 2., 'report_enabled': True,
    'calibration_run': False, 'calibration_equipment': '', 'calibration_amp_setting': '',
    'calibration_sensor_id': '', 'calibration_min_hz': 0., 'calibration_max_hz': 0.,
    'calibration_level_peak_g': 0., 'calibration_backend': '',
}


class ResponseTrip(RuntimeError):
    def __init__(self, reason, details):
        super().__init__(reason)
        self.details = details


class ResponseGuard:
    """Check acquired blocks before the controller may react to their contents.

    Absolute limits apply to every enabled sensor. Feedback loss applies only
    to the control sensor, only while driven, and after the startup allowance.
    These are block checks, not an independent hardware interlock.
    """
    def __init__(self, settings):
        self.settings = settings
        self.elapsed = 0.
        self.missing_seconds = 0.

    def check(self, data, drive_rms, expected_rms, backlog_seconds=0., feedback_rms=None):
        values = np.atleast_2d(data)
        if not np.all(np.isfinite(values)):
            raise ResponseTrip('Acquired acceleration contains NaN or infinity', {'invalid_input': True})
        rms = np.std(values, axis=1)
        peak = np.max(np.abs(values), axis=1)
        details = {'channel_rms_g': rms.tolist(), 'channel_raw_peak_g': peak.tolist(),
                   'input_backlog_seconds': float(backlog_seconds)}
        s = self.settings
        self.elapsed += s.block_seconds
        if not s.response_limits_enabled:
            return details
        def trip(message):
            raise ResponseTrip(message, details)
        for index, (r, p) in enumerate(zip(rms, peak)):
            label = s.input_channel_labels[index]
            if p > s.response_max_peak_g:
                trip(f'Measured peak limit: {label} reached {p:.5g} g (limit {s.response_max_peak_g:g} g).')
            if r > s.response_max_rms_g:
                trip(f'Measured RMS limit: {label} reached {r:.5g} g RMS (limit {s.response_max_rms_g:g}).')
        if rms[0] > s.target_rms_g * s.response_target_ratio:
            trip('Control acceleration exceeded the configured multiple of the full target.')
        if backlog_seconds > s.response_max_backlog_seconds:
            trip('Unread input exceeded the configured data-age limit.')
        observed = rms[0] if feedback_rms is None else feedback_rms
        threshold = max(s.feedback_min_rms_g, expected_rms * .02)
        missing = (drive_rms >= s.feedback_min_drive_v and expected_rms > 0
                   and observed < threshold and self.elapsed > s.feedback_startup_seconds)
        self.missing_seconds = self.missing_seconds + s.block_seconds if missing else 0.
        details['feedback_missing_seconds'] = self.missing_seconds
        if self.missing_seconds >= s.feedback_loss_seconds:
            trip('Control sensor feedback was lost while output was driven.')
        return details


def measure_tone(data, frequency, fs):
    """Fit the fundamental independently of phase; report noise/harmonics apart.

    A small least-squares system avoids treating broadband noise as sine peak.
    The residual contains harmonics, noise and transition content, not THD alone.
    """
    values = np.asarray(data, dtype=float)
    if frequency <= 0:
        return 0., float(np.std(values))
    phase = np.arange(values.size) * (2 * np.pi * frequency / fs)
    # Solve only a 3x3 normal system, without allocating a tall design matrix.
    sin, cos = np.sin(phase), np.cos(phase)
    n = len(values)
    matrix = np.array([[sin @ sin, sin @ cos, sin.sum()],
                       [sin @ cos, cos @ cos, cos.sum()], [sin.sum(), cos.sum(), n]])
    coefficient = np.linalg.solve(matrix, [sin @ values, cos @ values, values.sum()])
    residual = values - coefficient[0] * sin - coefficient[1] * cos - coefficient[2]
    return float(np.hypot(*coefficient[:2])), float(np.std(residual))


class SineFeedback:
    """Estimate gain from confirmed applied drive; bound every upward change.

    No error integral accumulates during clipping, missing feedback or queue
    delay. Corrections are absolute estimates referenced to the applied block.
    """
    def __init__(self, settings):
        self.settings = settings
        self.gains = {}
        self.last_gain = None
        self.last_queued_peak = 0.
        self.last_frequency = None
        self.at_frequency_seconds = 0.
        self.in_tolerance = 0
        self.ready = False

    def command(self, frequency, feedforward_per_g, target_peak):
        s = self.settings
        gain = self.gains.get(frequency, self.last_gain)
        desired = target_peak / gain if gain else feedforward_per_g * target_peak
        pilot = s.sine_pilot_vpk
        previous = self.last_queued_peak or min(pilot, desired)
        if gain is None:
            desired = min(desired, pilot)
        ceiling = min(s.ao_volt_limit, s.max_rms_volts * math.sqrt(2) * s.rms_limit_headroom)
        commanded = min(desired, previous * (1 + s.sine_correction_rate * s.block_seconds), ceiling)
        self.last_queued_peak = max(0., commanded)
        return self.last_queued_peak

    def observe(self, frequency, tone_peak, residual_rms, applied_peak, level, limited, target_peak):
        s = self.settings
        if frequency != self.last_frequency:
            self.last_frequency = frequency
            self.at_frequency_seconds = 0.
            self.in_tolerance = 0
        self.at_frequency_seconds += s.block_seconds
        # A fitted tone averages broadband noise over the block. Keep a user
        # floor as well as a conservative residual-based resolution estimate.
        fit_floor = 6 * residual_rms / math.sqrt(max(4, int(s.fs * s.block_seconds)))
        resolved = (applied_peak > 1e-9 and tone_peak > max(s.feedback_min_rms_g * math.sqrt(2), fit_floor))
        self.ready = resolved and not limited
        if self.ready:
            measured_gain = tone_peak / applied_peak
            prior = self.gains.get(frequency, measured_gain)
            gain = .4 * measured_gain + .6 * prior
            self.gains[frequency] = self.last_gain = max(gain, 1e-9)
        within = (target_peak > 0 and abs(tone_peak / target_peak - 1) <= s.tracking_tolerance_percent / 100)
        if self.ready and level >= .999999 and within and self.at_frequency_seconds >= s.sine_settle_seconds:
            self.in_tolerance += 1
        else:
            self.in_tolerance = 0
        return self.in_tolerance >= s.sine_settle_blocks
