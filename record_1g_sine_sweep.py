#!/usr/bin/env python3
"""Utility to calibrate sine sweep drive levels for 1 g-peak response."""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

import config
from sine_sweep import build_drive_lookup
from simulation import create_simulation_system

try:
    import nidaqmx
    from nidaqmx.constants import (
        AcquisitionType,
        AccelSensitivityUnits,
        AccelUnits,
        Coupling,
        ExcitationSource,
        RegenerationMode,
        TerminalConfiguration,
        VoltageUnits,
        WAIT_INFINITELY,
    )
    from nidaqmx.errors import DaqError
    from nidaqmx.stream_readers import AnalogSingleChannelReader
    from nidaqmx.stream_writers import AnalogSingleChannelWriter
    NIDAQMX_AVAILABLE = True
except ImportError:  # pragma: no cover
    nidaqmx = None
    DaqError = None
    AnalogSingleChannelReader = None
    AnalogSingleChannelWriter = None
    WAIT_INFINITELY = None
    NIDAQMX_AVAILABLE = False


def _build_frequency_list(
    explicit: Iterable[float] | None,
    start_hz: float,
    end_hz: float,
    points_per_octave: float,
) -> List[float]:
    if explicit:
        cleaned = sorted({float(f) for f in explicit if f and f > 0})
        if not cleaned:
            raise ValueError("No valid positive frequencies supplied.")
        return cleaned

    start_hz = float(max(start_hz, 1e-6))
    end_hz = float(max(end_hz, start_hz))
    points_per_octave = float(max(points_per_octave, 1.0))

    ratio = 2.0 ** (1.0 / points_per_octave)
    freqs = [start_hz]
    current = start_hz
    while True:
        next_freq = current * ratio
        if next_freq > end_hz * (1.0 + 1e-6):
            break
        if not math.isclose(next_freq, freqs[-1]):
            freqs.append(next_freq)
        current = next_freq
    if not math.isclose(freqs[-1], end_hz):
        freqs.append(end_hz)
    return freqs


class MeasurementSession:
    """Base interface for acquiring shaker response data."""

    def __init__(self, fs: float, block_samples: int) -> None:
        self.fs = float(fs)
        self.block_samples = int(block_samples)

    def write_and_capture(self, command_block: np.ndarray, timeout: float | None) -> np.ndarray:
        raise NotImplementedError

    def clear_output(self) -> None:
        pass

    def close(self) -> None:  # pragma: no cover - override as needed
        pass


class SimulationSession(MeasurementSession):
    def __init__(
        self,
        fs: float,
        block_samples: int,
        num_input_channels: int,
    ) -> None:
        super().__init__(fs, block_samples)
        (
            self._ao_writer,
            self._ai_reader,
            self._ai_task,
            self._ao_task,
        ) = create_simulation_system(
            fs=fs,
            sim_plant_gain=config.SIM_PLANT_GAIN,
            sim_resonances=config.SIM_RESONANCES,
            sim_noise_level=config.SIM_NOISE_LEVEL,
            sim_delay_samples=config.SIM_DELAY_SAMPLES,
            sim_nonlinearity=config.SIM_NONLINEARITY,
            num_input_channels=num_input_channels,
        )
        self._timeout = None

    def write_and_capture(self, command_block: np.ndarray, timeout: float | None) -> np.ndarray:
        self._ao_writer.write_many_sample(command_block)
        data = np.empty(self.block_samples, dtype=np.float64)
        self._ai_reader.read_many_sample(
            data,
            number_of_samples_per_channel=self.block_samples,
        )
        return data

    def clear_output(self) -> None:
        zeros = np.zeros(self.block_samples, dtype=np.float64)
        for _ in range(2):  # drive to zero for good measure
            self.write_and_capture(zeros, None)

    def close(self) -> None:
        try:
            self._ao_task.stop()
        except Exception:
            pass
        try:
            self._ao_task.close()
        except Exception:
            pass
        try:
            self._ai_task.stop()
        except Exception:
            pass
        try:
            self._ai_task.close()
        except Exception:
            pass


class HardwareSession(MeasurementSession):
    def __init__(
        self,
        fs: float,
        block_samples: int,
        buf_seconds: float,
        ao_limit: float,
        num_input_channels: int,
    ) -> None:
        if not NIDAQMX_AVAILABLE:
            raise RuntimeError("nidaqmx package not available; cannot create hardware session")

        super().__init__(fs, block_samples)
        self._buf_samples = max(int(fs * buf_seconds), block_samples)
        self._ao_limit = float(max(ao_limit, 0.1))
        self._volts_to_g = None

        self._ai_task = nidaqmx.Task()
        self._ao_task = nidaqmx.Task()

        excitation_current = max(0.0, float(getattr(config, "ACCEL_EXCITATION_AMPS", 0.0)))
        excitation_source = ExcitationSource.INTERNAL if excitation_current > 0 else ExcitationSource.NONE

        try:
            ai_channel = self._ai_task.ai_channels.add_ai_accel_chan(
                physical_channel=config.DEVICE_AI,
                name_to_assign_to_channel="control_accel",
                terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                min_val=-5.0,
                max_val=5.0,
                units=AccelUnits.G,
                sensitivity=getattr(config, "ACCEL_MV_PER_G", 100.0),
                sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
                current_excit_source=excitation_source,
                current_excit_val=excitation_current,
            )
            try:
                ai_channel.ai_coupling = Coupling.AC
            except Exception:  # pragma: no cover - hardware specific
                pass
            self._volts_to_g = None
        except Exception as accel_err:
            if DaqError is not None and not isinstance(accel_err, DaqError):
                raise
            print(f"Falling back to voltage input for control accelerometer ({accel_err}).")
            self._ai_task.close()
            self._ai_task = nidaqmx.Task()
            voltage_range = max(2.0, self._ao_limit * 2.0)
            self._ai_task.ai_channels.add_ai_voltage_chan(
                physical_channel=config.DEVICE_AI,
                name_to_assign_to_channel="control_volts",
                terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                min_val=-voltage_range,
                max_val=voltage_range,
                units=VoltageUnits.VOLTS,
            )
            accel_mV_per_g = float(max(getattr(config, "ACCEL_MV_PER_G", 100.0), 1e-6))
            self._volts_to_g = 1.0 / (accel_mV_per_g / 1000.0)

        self._ai_task.timing.cfg_samp_clk_timing(
            rate=fs,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self._buf_samples,
        )

        try:
            desired_ai_buf = max(self._buf_samples * 4, self._ai_task.in_stream.input_buf_size)
            self._ai_task.in_stream.input_buf_size = desired_ai_buf
        except Exception:  # pragma: no cover - hardware specific
            pass

        self._ao_task.ao_channels.add_ao_voltage_chan(
            physical_channel=config.DEVICE_AO,
            name_to_assign_to_channel="drive",
            min_val=-self._ao_limit,
            max_val=self._ao_limit,
            units=VoltageUnits.VOLTS,
        )
        self._ao_task.timing.cfg_samp_clk_timing(
            rate=fs,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self._buf_samples,
        )
        try:
            desired_ao_buf = max(self._buf_samples * 4, self._ao_task.out_stream.output_buf_size)
            self._ao_task.out_stream.output_buf_size = desired_ao_buf
        except Exception:  # pragma: no cover - hardware specific
            pass
        try:
            self._ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        except Exception:  # pragma: no cover - hardware specific
            pass

        self._reader = AnalogSingleChannelReader(self._ai_task.in_stream)
        self._writer = AnalogSingleChannelWriter(self._ao_task.out_stream, auto_start=False)

        preload = np.zeros(self._buf_samples, dtype=np.float64)
        self._writer.write_many_sample(preload)
        self._ai_task.start()
        self._ao_task.start()

        self._timeout = WAIT_INFINITELY if WAIT_INFINITELY is not None else max(1.0, self.block_samples / fs * 2.0)

    def write_and_capture(self, command_block: np.ndarray, timeout: float | None) -> np.ndarray:
        timeout = self._timeout if timeout is None else timeout
        self._writer.write_many_sample(command_block, timeout=timeout)
        data = np.empty(self.block_samples, dtype=np.float64)
        self._reader.read_many_sample(
            data,
            number_of_samples_per_channel=self.block_samples,
            timeout=timeout,
        )
        if self._volts_to_g is not None:
            data *= self._volts_to_g
        return data

    def clear_output(self) -> None:
        zero_block = np.zeros(self.block_samples, dtype=np.float64)
        for _ in range(4):
            try:
                self._writer.write_many_sample(zero_block, timeout=self._timeout)
            except Exception:
                break

    def close(self) -> None:
        try:
            self._ao_task.stop()
        except Exception:
            pass
        try:
            self._ai_task.stop()
        except Exception:
            pass
        try:
            self._ao_task.close()
        except Exception:
            pass
        try:
            self._ai_task.close()
        except Exception:
            pass


def _generate_command_block(freq: float, amplitude_vpk: float, fs: float, num_samples: int) -> np.ndarray:
    t = np.arange(num_samples, dtype=np.float64) / fs
    return amplitude_vpk * np.sin(2.0 * math.pi * freq * t)


def _measure_peak_g(
    session: MeasurementSession,
    freq: float,
    amplitude_vpk: float,
    settle_blocks: int,
    measure_blocks: int,
    timeout: float | None,
    fs: float,
    block_samples: int,
    max_volt: float,
) -> float:
    amplitude_vpk = float(max(amplitude_vpk, 0.0))
    command_block = _generate_command_block(freq, amplitude_vpk, fs, block_samples)
    max_cmd = float(np.max(np.abs(command_block)))
    if max_cmd > max_volt + 1e-9:
        scale = max_volt / max_cmd
        command_block *= scale
        max_cmd = max_volt
    for _ in range(max(0, settle_blocks)):
        session.write_and_capture(command_block, timeout)
    peaks = []
    for _ in range(max(1, measure_blocks)):
        data = session.write_and_capture(command_block, timeout)
        data = np.asarray(data)
        if data.ndim > 1:
            data = data[0]
        centered = data - np.mean(data)
        rms = math.sqrt(float(np.mean(np.square(centered))))
        peak = rms * math.sqrt(2.0)
        peaks.append(peak)
    return float(np.mean(peaks))


def calibrate_frequency(
    session: MeasurementSession,
    freq: float,
    target_peak_g: float,
    tolerance_g: float,
    max_iterations: int,
    initial_guess_vpk: float,
    settle_blocks: int,
    measure_blocks: int,
    timeout: float | None,
    fs: float,
    block_samples: int,
    min_volt: float,
    max_volt: float,
) -> Tuple[float, float, int]:
    amplitude = float(max(initial_guess_vpk, min_volt))
    amplitude = min(amplitude, max_volt)
    last_peak = 0.0
    for attempt in range(1, max_iterations + 1):
        measured_peak = _measure_peak_g(
            session,
            freq,
            amplitude,
            settle_blocks,
            measure_blocks,
            timeout,
            fs,
            block_samples,
            max_volt,
        )
        last_peak = measured_peak
        error = measured_peak - target_peak_g
        print(
            f"    iter {attempt}: drive={amplitude:.4f} Vpk -> meas={measured_peak:.4f} g (err={error:+.4f} g)",
            flush=True,
        )
        if math.isnan(measured_peak) or not math.isfinite(measured_peak):
            raise RuntimeError("Measured response is invalid (NaN/Inf).")
        if abs(error) <= tolerance_g:
            corrected = amplitude
            if measured_peak > 1e-6:
                corrected = min(max_volt, max(min_volt, amplitude * (target_peak_g / measured_peak)))
            return corrected, measured_peak, attempt
        if measured_peak < 1e-6:
            amplitude = min(max_volt, max(amplitude * 2.0, min_volt))
            continue
        scale = target_peak_g / measured_peak
        scale = float(np.clip(scale, 0.4, 2.5))
        amplitude = min(max_volt, max(min_volt, amplitude * scale))
    raise RuntimeError(
        f"Failed to reach target at {freq:.2f} Hz within {max_iterations} iterations. "
        f"Last measured peak: {last_peak:.4f} g"
    )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate sine sweep drive voltages for 1 g-peak response",
    )
    parser.add_argument("--frequencies", nargs="*", type=float, help="Explicit list of frequencies in Hz")
    parser.add_argument("--start-hz", type=float, default=getattr(config, "SINE_SWEEP_START_HZ", 20.0))
    parser.add_argument("--end-hz", type=float, default=getattr(config, "SINE_SWEEP_END_HZ", 2000.0))
    parser.add_argument(
        "--points-per-octave",
        type=float,
        default=getattr(config, "SINE_SWEEP_POINTS_PER_OCTAVE", 3.0),
        help="Number of calibration points per octave when frequencies are generated",
    )
    parser.add_argument("--target-g", type=float, default=1.0, help="Desired amplitude (g units)")
    parser.add_argument(
        "--target-is-rms",
        action="store_true",
        help="Interpret --target-g as RMS instead of peak",
    )
    parser.add_argument(
        "--tolerance-g",
        type=float,
        default=0.05,
        help="Acceptable absolute peak error in g",
    )
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--settle-blocks", type=int, default=1)
    parser.add_argument("--measure-blocks", type=int, default=1)
    parser.add_argument("--fs", type=float, default=getattr(config, "FS", 8192.0))
    parser.add_argument(
        "--block-seconds",
        type=float,
        default=getattr(config, "BLOCK_SECONDS", 0.5),
        help="Length of each write/read block in seconds",
    )
    parser.add_argument(
        "--buf-seconds",
        type=float,
        default=getattr(config, "BUF_SECONDS", 4.0),
        help="Buffer size passed to the DAQ tasks (seconds)",
    )
    parser.add_argument("--min-volt", type=float, default=0.02)
    parser.add_argument("--max-volt", type=float, default=getattr(config, "AO_VOLT_LIMIT", 1.8))
    parser.add_argument(
        "--outfile",
        type=Path,
        default=Path("1g_sine_sweep_config.txt"),
        help="Output file for the calibrated drive table",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output file")
    parser.add_argument("--simulation", action="store_true", help="Force simulation mode regardless of config")
    parser.add_argument("--hardware", action="store_true", help="Force hardware mode and fail if unavailable")
    parser.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="Optional pause (seconds) between frequency points",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if args.hardware and args.simulation:
        print("Cannot force both hardware and simulation modes.", file=sys.stderr)
        return 2

    fs = float(args.fs)
    block_samples = max(64, int(round(fs * args.block_seconds)))

    try:
        frequencies = _build_frequency_list(args.frequencies, args.start_hz, args.end_hz, args.points_per_octave)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2

    target_peak = float(args.target_g)
    if args.target_is_rms:
        target_peak *= math.sqrt(2.0)
    tolerance_g = max(float(args.tolerance_g), 1e-4)

    default_vpk = getattr(config, "SINE_SWEEP_DEFAULT_VPK", 0.4)
    table_points = getattr(config, "SINE_SWEEP_DRIVE_TABLE", [])
    drive_lookup = build_drive_lookup(table_points, default_vpk)

    use_simulation = args.simulation or getattr(config, "SIMULATION_MODE", False)
    if args.hardware:
        use_simulation = False
    if not args.hardware and not NIDAQMX_AVAILABLE:
        use_simulation = True

    num_input_channels = 1

    session: MeasurementSession
    if use_simulation:
        print("Running in simulation mode.")
        session = SimulationSession(fs, block_samples, num_input_channels=num_input_channels)
    else:
        print("Running with hardware (nidaqmx).")
        try:
            session = HardwareSession(
                fs=fs,
                block_samples=block_samples,
                buf_seconds=args.buf_seconds,
                ao_limit=args.max_volt,
                num_input_channels=num_input_channels,
            )
        except Exception as err:
            print(f"Failed to initialize hardware session: {err}", file=sys.stderr)
            return 1

    if args.measure_blocks <= 0:
        print("measure_blocks must be positive", file=sys.stderr)
        return 2

    results: List[Tuple[float, float, float]] = []
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Calibrating {len(frequencies)} frequencies to reach {target_peak:.3f} g-peak (tolerance {tolerance_g:.3f} g).")

    try:
        for idx, freq in enumerate(frequencies, start=1):
            print(f"[{idx}/{len(frequencies)}] {freq:.2f} Hz")
            initial_guess = min(args.max_volt, max(args.min_volt, drive_lookup(freq)))
            try:
                calibrated_vpk, measured_peak, attempts = calibrate_frequency(
                    session=session,
                    freq=freq,
                    target_peak_g=target_peak,
                    tolerance_g=tolerance_g,
                    max_iterations=args.max_iterations,
                    initial_guess_vpk=initial_guess,
                    settle_blocks=max(args.settle_blocks, 0),
                    measure_blocks=max(args.measure_blocks, 1),
                    timeout=None,
                    fs=fs,
                    block_samples=block_samples,
                    min_volt=args.min_volt,
                    max_volt=args.max_volt,
                )
            except Exception as err:
                print(f"  Calibration failed: {err}", file=sys.stderr)
                session.clear_output()
                return 1

            session.clear_output()
            recorded_vpk = min(args.max_volt, max(args.min_volt, calibrated_vpk))
            print(
                f"  -> recorded {recorded_vpk:.4f} Vpk for {freq:.2f} Hz (reached {measured_peak:.4f} g in {attempts} iterations)"
            )
            results.append((freq, recorded_vpk, measured_peak))
            if args.wait > 0:
                time.sleep(args.wait)
    finally:
        session.clear_output()
        session.close()

    outfile = Path(args.outfile)
    if outfile.exists() and not args.overwrite:
        print(f"Error: {outfile} already exists. Use --overwrite to replace it.", file=sys.stderr)
        return 2

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="ascii") as f:
        f.write("# 1 g sine sweep calibration drive table\n")
        f.write(f"# Generated on {start_time}\n")
        f.write("# Format: frequency_hz, volts_peak\n")
        for freq, volts, _ in results:
            f.write(f"{freq:.6f}, {volts:.6f}\n")

    print(f"Saved drive table with {len(results)} entries to {outfile}.")
    print("Use these (frequency, Vpk) pairs in config.SINE_SWEEP_DRIVE_TABLE for 1 g-peak targeting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
