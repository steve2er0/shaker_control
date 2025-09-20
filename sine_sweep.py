"""Utilities for generating logarithmic sine sweeps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np


@dataclass
class SweepStatus:
    """Metadata describing the most recent sweep block."""

    freq_start: float
    freq_end: float
    cycle_completed: bool


class LogSweepGenerator:
    """Generate a logarithmic sine sweep with constant octaves-per-minute rate."""

    def __init__(
        self,
        fs: float,
        start_freq: float,
        end_freq: float,
        octaves_per_minute: float,
        repeat: bool = True,
    ) -> None:
        if fs <= 0:
            raise ValueError("Sample rate must be positive")
        if start_freq <= 0 or end_freq <= 0:
            raise ValueError("Frequencies must be positive")

        self.fs = float(fs)
        self.start_freq = float(start_freq)
        self.end_freq = float(end_freq)
        self.repeat = bool(repeat)

        self._log_start = math.log(self.start_freq)
        self._log_end = math.log(self.end_freq)
        self._log_delta = self._log_end - self._log_start

        self._phase = 0.0
        self._elapsed = 0.0
        self._completed_cycle = False
        self._cycle_count = 0

        oct_rate = float(octaves_per_minute)
        if oct_rate <= 0.0 or math.isclose(self.start_freq, self.end_freq):
            # Treat as constant frequency sweep
            self._duration = None
        else:
            total_octaves = math.log(self.end_freq / self.start_freq, 2.0)
            rate_oct_per_sec = abs(oct_rate) / 60.0
            if rate_oct_per_sec == 0.0:
                self._duration = None
            else:
                self._duration = abs(total_octaves) / rate_oct_per_sec

    def current_frequency(self) -> float:
        """Return the instantaneous frequency for the next sample."""
        if not self._has_valid_duration():
            return self.start_freq

        if not self.repeat and self._completed_cycle:
            progress = 1.0
        else:
            progress = min(max(self._elapsed / self._duration, 0.0), 1.0)

        return float(math.exp(self._log_start + progress * self._log_delta))

    def is_finished(self) -> bool:
        """Return True when a non-repeating sweep has completed."""
        return (not self.repeat) and self._completed_cycle

    def generate_block(
        self,
        num_samples: int,
        return_cosine: bool = False,
    ) -> Tuple[np.ndarray, SweepStatus] | Tuple[np.ndarray, np.ndarray, SweepStatus]:
        """Generate a block of unit-amplitude sine samples and sweep metadata.

        Args:
            num_samples: Number of samples to generate.
            return_cosine: When True, also return the cosine reference matching the sine
                block for use with quadrature demodulation.
        """
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("Number of samples must be positive")

        sin_block = np.empty(num_samples, dtype=np.float64)
        cos_block = np.empty(num_samples, dtype=np.float64) if return_cosine else None
        freq_start = self.current_frequency()
        cycle_completed = False

        for idx in range(num_samples):
            freq = self.current_frequency()
            self._phase += 2.0 * math.pi * freq / self.fs
            # Keep phase bounded to avoid precision issues on long runs
            if self._phase > math.pi:
                self._phase = math.fmod(self._phase + math.pi, 2.0 * math.pi) - math.pi
            sin_block[idx] = math.sin(self._phase)
            if return_cosine:
                cos_block[idx] = math.cos(self._phase)

            if self._has_valid_duration():
                self._elapsed += 1.0 / self.fs
                if self.repeat:
                    if self._elapsed >= self._duration:
                        cycle_completed = True
                        self._cycle_count += 1
                        while self._elapsed >= self._duration:
                            self._elapsed -= self._duration
                        # Preserve phase continuity, but avoid unbounded growth
                        self._phase = math.fmod(self._phase, 2.0 * math.pi)
                else:
                    if self._elapsed >= self._duration:
                        cycle_completed = True
                        if not self._completed_cycle:
                            self._cycle_count += 1
                        self._elapsed = self._duration
                        self._completed_cycle = True
            else:
                self._elapsed += 1.0 / self.fs

        freq_end = self.current_frequency()
        status = SweepStatus(freq_start=freq_start, freq_end=freq_end, cycle_completed=cycle_completed)
        if return_cosine:
            return sin_block, cos_block, status
        return sin_block, status

    def _has_valid_duration(self) -> bool:
        return self._duration is not None and self._duration > 0.0

    def cycles_completed(self) -> int:
        """Return the number of completed sweep cycles."""
        return self._cycle_count


class LogSweepStepper:
    """Generate logarithmically spaced frequency steps between start and end."""

    def __init__(
        self,
        start_freq: float,
        end_freq: float,
        points_per_octave: float,
        repeat: bool = False,
    ) -> None:
        if start_freq <= 0 or end_freq <= 0:
            raise ValueError("Frequencies must be positive")
        if points_per_octave <= 0:
            raise ValueError("points_per_octave must be positive")

        self.start_freq = float(start_freq)
        self.end_freq = float(end_freq)
        self.repeat = bool(repeat)
        self._increasing = self.end_freq >= self.start_freq
        step_ratio = 2.0 ** (1.0 / float(points_per_octave))
        self._ratio = step_ratio if self._increasing else 1.0 / step_ratio
        self._current = self.start_freq
        self._completed = False

    def current_frequency(self) -> float:
        return self._current

    def advance(self) -> bool:
        """Advance to the next frequency. Returns False when sweep is finished."""
        if self._completed and not self.repeat:
            return False

        next_freq = self._current * self._ratio
        if self._increasing:
            if next_freq > self.end_freq:
                if self.repeat:
                    self._current = self.start_freq
                    self._completed = True
                    return True
                self._completed = True
                return False
        else:
            if next_freq < self.end_freq:
                if self.repeat:
                    self._current = self.start_freq
                    self._completed = True
                    return True
                self._completed = True
                return False

        self._current = next_freq
        return True

    def is_finished(self) -> bool:
        return self._completed and not self.repeat


class SineOscillator:
    """Numerically-controlled oscillator for sine/cosine generation."""

    def __init__(self, fs: float, initial_freq: float) -> None:
        if fs <= 0:
            raise ValueError("Sample rate must be positive")
        self.fs = float(fs)
        self.phase = 0.0
        self.set_frequency(initial_freq)

    def set_frequency(self, freq: float) -> None:
        freq = float(freq)
        if freq <= 0:
            raise ValueError("Frequency must be positive")
        self.freq = freq
        self._phase_inc = 2.0 * math.pi * self.freq / self.fs

    def generate(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("Number of samples must be positive")

        idx = np.arange(num_samples, dtype=float)
        phase = self.phase + idx * self._phase_inc
        sin_block = np.sin(phase)
        cos_block = np.cos(phase)
        self.phase = float((phase[-1] + self._phase_inc) % (2.0 * math.pi))
        return sin_block, cos_block


def target_g_rms(g_level: float, level_is_rms: bool) -> float:
    """Convert configured g-level into RMS units for control."""
    g_level = float(g_level)
    if g_level <= 0:
        raise ValueError("g-level must be positive")
    if level_is_rms:
        return g_level
    return g_level / math.sqrt(2.0)


def build_drive_lookup(
    points: Iterable[Tuple[float, float]],
    default_vpk: float,
) -> Callable[[float], float]:
    """Return a function that provides peak volts vs frequency using log interpolation."""

    cleaned = []
    for freq, vpk in points:
        freq = float(freq)
        vpk = float(vpk)
        if freq > 0 and vpk >= 0:
            cleaned.append((freq, vpk))

    if not cleaned:
        default_vpk = max(0.0, float(default_vpk))
        return lambda _: default_vpk

    cleaned.sort(key=lambda item: item[0])
    freqs = np.array([item[0] for item in cleaned], dtype=float)
    volts = np.array([item[1] for item in cleaned], dtype=float)
    log_freqs = np.log(freqs)

    def lookup(freq_hz: float) -> float:
        freq_hz = float(max(freq_hz, 1e-9))
        log_f = math.log(freq_hz)
        if log_f <= log_freqs[0]:
            return float(volts[0])
        if log_f >= log_freqs[-1]:
            return float(volts[-1])
        idx = np.searchsorted(log_freqs, log_f)
        f0, f1 = log_freqs[idx - 1], log_freqs[idx]
        v0, v1 = volts[idx - 1], volts[idx]
        weight = (log_f - f0) / max(f1 - f0, 1e-12)
        return float(v0 + weight * (v1 - v0))

    return lookup
