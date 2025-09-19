"""Utilities for generating logarithmic sine sweeps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

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

    def generate_block(self, num_samples: int) -> Tuple[np.ndarray, SweepStatus]:
        """Generate a block of unit-amplitude sine samples and sweep metadata."""
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("Number of samples must be positive")

        block = np.empty(num_samples, dtype=np.float64)
        freq_start = self.current_frequency()
        cycle_completed = False

        for idx in range(num_samples):
            freq = self.current_frequency()
            self._phase += 2.0 * math.pi * freq / self.fs
            # Keep phase bounded to avoid precision issues on long runs
            if self._phase > math.pi:
                self._phase = math.fmod(self._phase + math.pi, 2.0 * math.pi) - math.pi
            block[idx] = math.sin(self._phase)

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
        return block, status

    def _has_valid_duration(self) -> bool:
        return self._duration is not None and self._duration > 0.0

    def cycles_completed(self) -> int:
        """Return the number of completed sweep cycles."""
        return self._cycle_count


def target_g_rms(g_level: float, level_is_rms: bool) -> float:
    """Convert configured g-level into RMS units for control."""
    g_level = float(g_level)
    if g_level <= 0:
        raise ValueError("g-level must be positive")
    if level_is_rms:
        return g_level
    return g_level / math.sqrt(2.0)
