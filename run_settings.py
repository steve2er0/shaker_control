"""Validated immutable settings captured once for each controller run."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from app_paths import resolve_recording_directory

NUMERIC_FIELDS = ('fs', 'block_seconds', 'buf_seconds', 'welch_nperseg',
    'ao_volt_limit', 'max_rms_volts', 'max_crest_factor', 'crest_soft_knee',
    'rms_limit_headroom', 'kp', 'ki', 'initial_level_fraction',
    'max_level_fraction_rate', 'eq_num_bands', 'eq_adapt_rate', 'eq_smooth_factor',
    'eq_adapt_level_threshold', 'eq_adapt_level_power', 'eq_adapt_min_weight',
    'sine_start_hz', 'sine_end_hz', 'sine_g_level', 'sine_octaves_per_min',
    'sine_initial_level', 'sine_max_level_rate', 'sine_initial_settle',
    'sine_points_per_octave', 'sine_step_dwell', 'sine_default_vpk', 'sine_drive_scale',
    'accel_mv_per_g', 'accel_excitation_amps', 'sim_plant_gain', 'sim_noise_level',
    'sim_delay_samples', 'sim_nonlinearity', 'realtime_psd_update_stride')
INTEGER_FIELDS = ('welch_nperseg', 'eq_num_bands', 'sim_delay_samples', 'realtime_psd_update_stride')
BOOLEAN_FIELDS = ('simulation_mode', 'data_log_enabled', 'sine_g_level_is_rms', 'sine_repeat', 'sync_ao_clock')


class ConfigurationError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__('; '.join(errors.values()))


def parse_points(text):
    """Parse pasted frequency/value pairs without silently dropping bad rows."""
    points = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.replace(',', ' ').split()
        if len(parts) != 2:
            raise ValueError(f'Line {line_number}: enter frequency and value')
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError:
            raise ValueError(f'Line {line_number}: both entries must be numbers') from None
    return points


def format_points(points):
    # Python's shortest round-trip float text preserves imported precision
    # without showing binary representation tails such as 0.0040000000000000001.
    return '\n'.join(f'{float(frequency)}, {float(value)}' for frequency, value in points)


def _profile_rms(points):
    variance = 0.0
    for (f0, s0), (f1, s1) in zip(points, points[1:]):
        ratio_log = math.log(f1) - math.log(f0)
        exponent = (math.log(s1) - math.log(s0)) / ratio_log
        power = exponent + 1
        variance += (s0*f0*ratio_log if abs(power) < 1e-10 else
                     (s1*f1-s0*f0)/power)
    return math.sqrt(variance)


def validate_settings(values):
    errors = {}
    for key in NUMERIC_FIELDS:
        if key in values:
            try:
                if not math.isfinite(float(values[key])):
                    errors[key] = 'Enter a finite number.'
            except (TypeError, ValueError):
                errors[key] = 'Enter a finite number.'
    for key in BOOLEAN_FIELDS:
        if key in values and not isinstance(values[key], bool):
            errors[key] = 'Use a true/false boolean value.'

    def number(key, lower=None, upper=None, positive=False):
        try:
            value = float(values[key])
            valid = math.isfinite(value)
        except (KeyError, TypeError, ValueError):
            valid, value = False, 0.0
        if not valid:
            errors[key] = 'Enter a finite number.'
            value = 0.0
        elif positive and value <= 0:
            errors[key] = 'Must be greater than zero.'
        elif lower is not None and value < lower:
            errors[key] = f'Must be at least {lower:g}.'
        elif upper is not None and value > upper:
            errors[key] = f'Must be at most {upper:g}.'
        return value

    def points(key, minimum, allow_zero=False):
        try:
            data = [(float(f), float(v)) for f, v in values.get(key, [])]
            valid = len(data) >= minimum and all(
                math.isfinite(f) and f > 0 and math.isfinite(v)
                and (v >= 0 if allow_zero else v > 0)
                for f, v in data
            ) and all(data[i][0] < data[i + 1][0] for i in range(len(data) - 1))
        except (ValueError, TypeError):
            data, valid = [], False
        if not valid:
            errors[key] = f'Enter at least {minimum} ordered frequency/value pairs with positive frequencies and valid levels.'
        return data

    mode = values.get('test_mode')
    if mode not in ('random', 'sine_sweep'):
        errors['test_mode'] = 'Choose random vibration or sine sweep.'
    fs = number('fs', positive=True)
    block = number('block_seconds', positive=True)
    buffer = number('buf_seconds', positive=True)
    segment = number('welch_nperseg', lower=2)
    if segment != int(segment):
        errors['welch_nperseg'] = 'Segment length must be an integer.'
    if not math.isfinite(fs * block) or fs * block < segment:
        errors['welch_nperseg'] = 'PSD segment must fit in one processing block.'
    if buffer < block:
        errors['buf_seconds'] = 'Buffer duration must be at least the processing block duration.'
    ao_limit = number('ao_volt_limit', positive=True)
    rms_limit = number('max_rms_volts', lower=0)
    if rms_limit > ao_limit:
        errors['max_rms_volts'] = 'RMS voltage limit cannot exceed the absolute voltage limit.'
    number('max_crest_factor', lower=1)
    number('crest_soft_knee', lower=0, upper=1)
    number('rms_limit_headroom', lower=0, upper=1)

    if mode == 'random':
        profile = points('target_psd_points', 2)
        if profile and profile[-1][0] >= .95 * fs / 2:
            errors['target_psd_points'] = 'Random target must be below 95% of Nyquist (0.475 × sample rate) for the equalizer.'
        if 'target_psd_points' not in errors:
            try:
                rms = _profile_rms(profile)
                if not math.isfinite(rms) or rms <= 0:
                    raise ValueError()
            except (ValueError, OverflowError, ZeroDivisionError):
                errors['target_psd_points'] = 'Target PSD must integrate to a finite positive RMS.'
            if fs > 0 and segment >= 2:
                spacing = fs/segment
                if math.floor(profile[-1][0]/spacing) < math.ceil(profile[0][0]/spacing):
                    errors['welch_nperseg'] = 'PSD has no bins in the target band. Increase segment length or lower sample rate.'
        number('initial_level_fraction', lower=0, upper=2)
        number('max_level_fraction_rate', lower=0)
        number('kp', lower=0)
        number('ki', lower=0)
        bands = number('eq_num_bands', lower=1, upper=64)
        if bands != int(bands):
            errors['eq_num_bands'] = 'Band count must be an integer.'
        number('eq_adapt_rate', lower=0, upper=1)
        number('eq_smooth_factor', lower=0, upper=1)
    elif mode == 'sine_sweep':
        start = number('sine_start_hz', positive=True)
        end = number('sine_end_hz', positive=True)
        if max(start, end) >= fs / 2:
            errors['sine_end_hz'] = 'All sweep frequencies must be below half the sample rate.'
        number('sine_g_level', positive=True)
        number('sine_octaves_per_min', positive=True)
        number('sine_initial_level', lower=0, upper=1)
        number('sine_max_level_rate', lower=0)
        number('sine_initial_settle', lower=0)
        number('sine_points_per_octave', lower=1)
        number('sine_step_dwell', positive=True)
        number('sine_default_vpk', lower=0)
        number('sine_drive_scale', lower=0)
        if values.get('sine_drive_table'):
            points('sine_drive_table', 1, allow_zero=True)

    if values.get('simulation_mode'):
        number('sim_plant_gain', positive=True)
        number('sim_noise_level', lower=0)
        number('sim_delay_samples', lower=0)
        number('sim_nonlinearity', lower=0)
        try:
            resonances = [(float(f), float(q), float(g)) for f, q, g in values['sim_resonances']]
            if any(not all(math.isfinite(v) for v in row) or row[0] <= 0 or row[1] <= 0 or row[2] < 0 for row in resonances):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            errors['sim_resonances'] = 'Simulation resonances require finite frequency, positive Q, and nonnegative gain.'
    else:
        for key in ('device_ai', 'device_ao'):
            if not str(values.get(key, '')).strip():
                errors[key] = 'Enter the physical DAQ channel.'
        number('accel_mv_per_g', positive=True)
        number('accel_excitation_amps', lower=0)
    labels = values.get('input_channel_labels', [])
    if not isinstance(labels, (list, tuple)) or not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        errors['input_channel_labels'] = 'Enter at least one nonempty channel label.'
    if isinstance(labels, (list, tuple)) and len(set(map(str, labels))) != len(labels):
        errors['input_channel_labels'] = 'Channel labels must be unique.'
    number('realtime_psd_update_stride', lower=1)
    if mode == 'random':
        number('eq_adapt_level_threshold', lower=0)
        number('eq_adapt_level_power', lower=0)
        number('eq_adapt_min_weight', lower=0, upper=1)
        try:
            low, high = map(float, values['eq_gain_limits'])
            if not math.isfinite(low) or not math.isfinite(high) or not 0 < low <= high:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            errors['eq_gain_limits'] = 'Equalizer gain limits must be two ordered positive numbers.'
    if values.get('data_log_enabled') and not str(values.get('data_log_dir', '')).strip():
        errors['data_log_dir'] = 'Choose a logging directory.'
    return errors


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class RunSettings:
    values: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, 'values', _freeze(dict(self.values)))

    @classmethod
    def capture(cls, values):
        values = dict(values)
        if not isinstance(values.get('input_channel_labels'), (list, tuple)):
            raise ConfigurationError({'input_channel_labels': 'Channel labels must be a list.'})
        labels = list(values.get('input_channel_labels', []))
        if not values.get('simulation_mode'):
            labels = labels[:1]
        values['input_channel_labels'] = labels
        values['num_input_channels'] = len(labels)
        errors = validate_settings(values)
        if errors:
            raise ConfigurationError(errors)
        for key in NUMERIC_FIELDS:
            if key in values:
                values[key] = float(values[key])
        for key in INTEGER_FIELDS:
            if key in values:
                if values[key] != int(values[key]):
                    raise ConfigurationError({key: 'Must be an integer.'})
                values[key] = int(values[key])
        for key in ('target_psd_points', 'sine_drive_table'):
            if key in values:
                try:
                    values[key] = [(float(f), float(v)) for f, v in values[key]]
                    if any(not math.isfinite(f) or not math.isfinite(v) for f, v in values[key]):
                        raise ValueError()
                except (TypeError, ValueError):
                    raise ConfigurationError({key: 'Frequency/value pairs must be finite numbers.'}) from None
        values['input_channel_labels'] = [str(label) for label in labels]
        for key in ('device_ai', 'device_ao', 'data_log_dir', 'calibration_source', 'preset_source'):
            if key in values and not isinstance(values[key], str):
                raise ConfigurationError({key: 'Must be text.'})
        if values.get('data_log_dir'):
            try:
                values['data_log_dir'] = str(resolve_recording_directory(values['data_log_dir']))
            except (OSError, ValueError, RuntimeError) as error:
                raise ConfigurationError({'data_log_dir': str(error)}) from None
        for key in ('eq_gain_limits', 'sim_resonances'):
            try:
                if key == 'sim_resonances':
                    values[key] = [tuple(float(v) for v in row) for row in values[key]]
                else:
                    values[key] = tuple(float(v) for v in values[key])
            except (KeyError, ValueError, TypeError):
                raise ConfigurationError({key: 'Enter numeric model settings.'}) from None
        try:
            json.dumps(values, allow_nan=False)
        except (ValueError, TypeError):
            raise ConfigurationError({'settings': 'All captured settings must contain finite numbers and JSON-compatible values.'}) from None
        return cls(_freeze(values))

    def __getattr__(self, key):
        try:
            return self.values[key]
        except KeyError:
            raise AttributeError(key) from None

    def as_dict(self):
        return _plain(self.values)

    @property
    def config_id(self):
        payload = json.dumps(self.as_dict(), sort_keys=True, allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def backend(self):
        return 'Simulation' if self.simulation_mode else 'Hardware'

    @property
    def target_rms_g(self):
        if self.test_mode == 'sine_sweep':
            return self.sine_g_level if self.sine_g_level_is_rms else self.sine_g_level / math.sqrt(2)
        return _profile_rms(self.target_psd_points)

    @property
    def target_summary(self):
        if self.test_mode == 'sine_sweep':
            unit = 'g RMS' if self.sine_g_level_is_rms else 'g peak'
            drive = 'drive table' if self.sine_drive_table else 'default drive'
            return f'{self.sine_start_hz:g}–{self.sine_end_hz:g} Hz · {self.sine_g_level:g} {unit} · {drive}'
        return f'{self.target_psd_points[0][0]:g}–{self.target_psd_points[-1][0]:g} Hz · {self.target_rms_g:.3f} g RMS'
