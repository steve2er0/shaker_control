#!/usr/bin/env python3
"""Shaker control entry point, acquisition worker, and recorded run data.

The native run workspace is implemented in workspace_ui.py. Each worker uses
validated settings captured before acquisition starts.
"""

import sys
import csv
import copy
import json
import uuid
import numpy as np
import time
import math
from threading import Event
from pathlib import Path
from datetime import datetime
from scipy.signal import welch
try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

# GUI imports
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Signal, QObject, QThread, Slot

# Optional hardware imports
try:
    import nidaqmx
    from nidaqmx.constants import (AcquisitionType, TerminalConfiguration, ExcitationSource,
                                   AccelUnits, Coupling, VoltageUnits, AccelSensitivityUnits)
    from nidaqmx.errors import DaqError
    from nidaqmx.stream_writers import AnalogSingleChannelWriter
    from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogMultiChannelReader
    NIDAQMX_AVAILABLE = True
except ImportError:
    nidaqmx = None
    DaqError = None
    NIDAQMX_AVAILABLE = False

# Import our existing modules
import config
from app_paths import resolve_recording_directory
from run_settings import RunSettings
from rv_controller import MultiBandEqualizer, RandomVibrationController, create_bandpass_filter, make_bandlimited_noise, apply_safety_limiters
from simulation import create_simulation_system
from sine_sweep import (
    LogSweepStepper,
    SineOscillator,
    build_drive_lookup,
    target_g_rms,
)


def create_hardware_input_task(settings):
    """Configure all inputs, rebuilding the task if an acceleration channel needs voltage scaling.

    A failed channel add may leave a partial channel in NI's task. Rebuilding
    preserves the exact requested order without duplicate or missing inputs.
    No task is started here.
    """
    channels = settings.hardware_input_channels
    voltage_channels = set()
    while True:
        task = nidaqmx.Task()
        scales = np.ones(len(channels))
        modes = []
        try:
            for index, spec in enumerate(channels):
                physical = spec['physical_channel']
                if index in voltage_channels:
                    voltage_range = max(2.0, settings.ao_volt_limit * 2.0)
                    task.ai_channels.add_ai_voltage_chan(
                        physical_channel=physical, name_to_assign_to_channel=f'accel_{index}',
                        terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                        min_val=-voltage_range, max_val=voltage_range, units=VoltageUnits.VOLTS)
                    scales[index] = 1000.0 / spec['sensitivity_mv_per_g']
                    modes.append(f'{physical}: Voltage input scaled to g (no IEPE)')
                    continue
                current = spec['excitation_amps']
                try:
                    channel = task.ai_channels.add_ai_accel_chan(
                        physical_channel=physical, name_to_assign_to_channel=f'accel_{index}',
                        terminal_config=TerminalConfiguration.PSEUDO_DIFF, min_val=-5.0, max_val=5.0,
                        units=AccelUnits.G, sensitivity=spec['sensitivity_mv_per_g'],
                        sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
                        current_excit_source=ExcitationSource.INTERNAL if current > 0 else ExcitationSource.NONE,
                        current_excit_val=current)
                except Exception as error:
                    if DaqError is None or not isinstance(error, DaqError):
                        raise
                    print(f'{physical}: acceleration channel unavailable: {error}. Using voltage scaling.')
                    voltage_channels.add(index)
                    break
                try:
                    channel.ai_coupling = Coupling.AC
                except DaqError:
                    pass
                power = f'{current * 1000:g} mA IEPE' if current > 0 else 'external power'
                modes.append(f'{physical}: Acceleration input in g ({power})')
            else:
                return task, scales, '; '.join(modes)
        except BaseException:
            task.close()
            raise
        task.close()


def scale_input_block(data, scales):
    """Convert every voltage fallback channel to g, retaining channel order."""
    if scales is not None:
        data *= np.asarray(scales)[:, None] if data.ndim == 2 else float(np.asarray(scales)[0])
    if not np.all(np.isfinite(data)):
        raise RuntimeError('Acquired acceleration contains NaN or infinity')
    return data[0] if data.ndim == 2 else data


class DataLogger:
    """Handle raw block logging and summary exports."""

    def __init__(self, shared_config, mode, fs, block_samples, num_channels):
        self.enabled = bool(getattr(shared_config, 'data_log_enabled', False))
        self.mode = mode
        self.fs = float(fs)
        self.block_samples = int(block_samples)
        self.config_channels = max(1, int(num_channels))
        self.welch_nperseg = int(getattr(shared_config, 'welch_nperseg', 2048))

        if not self.enabled:
            self.h5 = None
            return
        if h5py is None:
            raise RuntimeError('Recording requested but h5py is unavailable')

        base_dir = resolve_recording_directory(getattr(shared_config, 'data_log_dir', None))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:6]
        self.run_dir = base_dir / f"{timestamp}_{mode}"
        self.run_dir.mkdir(parents=True, exist_ok=False)

        (self.run_dir / 'settings.json').write_text(json.dumps(shared_config.as_dict(), indent=2, allow_nan=False) + '\n', encoding='utf-8')
        self.h5 = h5py.File(self.run_dir / 'data.h5', 'w')
        self.h5.attrs['schema_version'] = 2
        self.h5.attrs['backend'] = shared_config.backend
        self.h5.attrs['ao_units'] = 'V'
        self.h5.attrs['ai_units'] = 'g'
        self.h5.attrs['sat_fraction_definition'] = 'Fraction of command samples at or above 98% of absolute AO voltage limit; distinct from clipped samples.'
        self.h5.attrs['config_id'] = shared_config.config_id
        self.h5.attrs['settings_json'] = json.dumps(shared_config.as_dict(), allow_nan=False)
        self.h5.attrs['mode'] = mode
        self.h5.attrs['fs'] = self.fs
        self.h5.attrs['block_samples'] = self.block_samples
        self.h5.attrs['configured_channels'] = self.config_channels
        self.h5.attrs['input_channel_labels'] = json.dumps(list(shared_config.input_channel_labels))
        if not shared_config.simulation_mode:
            self.h5.attrs['input_physical_channels'] = json.dumps(
                [spec['physical_channel'] for spec in shared_config.hardware_input_channels])

        self.ao_ds = self.h5.create_dataset(
            'ao',
            shape=(0, self.block_samples),
            maxshape=(None, self.block_samples),
            dtype='f4',
            chunks=(1, self.block_samples),
        )
        self.ai_ds = self.h5.create_dataset(
            'ai',
            shape=(0, self.config_channels, self.block_samples),
            maxshape=(None, self.config_channels, self.block_samples),
            dtype='f4',
            chunks=(1, self.config_channels, self.block_samples),
        )

        self.block_index = 0
        self.timestamps = []
        self.level_fraction = []
        self.sat_fraction = []

        self.last_ai_block = None

        if mode == 'sine_sweep':
            self.sine_block_values = []
            self.sine_freqs = []
            self.sine_peak_meas = []
            self.sine_peak_target = []

    def _prepare_ai_block(self, ai_block):
        ai = np.asarray(ai_block, dtype=np.float32)
        if ai.ndim == 1:
            ai = ai[np.newaxis, :]
        block_len = ai.shape[1]
        if block_len != self.block_samples:
            if block_len > self.block_samples:
                ai = ai[:, : self.block_samples]
            else:
                padded = np.zeros((ai.shape[0], self.block_samples), dtype=np.float32)
                padded[:, :block_len] = ai
                ai = padded
        if ai.shape[0] < self.config_channels:
            padded = np.zeros((self.config_channels, self.block_samples), dtype=np.float32)
            padded[: ai.shape[0], :] = ai
            ai = padded
        elif ai.shape[0] > self.config_channels:
            ai = ai[: self.config_channels, :]
        return ai

    def _append_block(self, ao_block, ai_block):
        if not self.enabled:
            return
        ao = np.asarray(ao_block, dtype=np.float32)
        if ao.ndim > 1:
            ao = ao.ravel()
        if ao.size != self.block_samples:
            if ao.size > self.block_samples:
                ao = ao[: self.block_samples]
            else:
                padded = np.zeros(self.block_samples, dtype=np.float32)
                padded[: ao.size] = ao
                ao = padded
        ai = self._prepare_ai_block(ai_block)

        self.ao_ds.resize(self.block_index + 1, axis=0)
        self.ao_ds[self.block_index, :] = ao
        self.ai_ds.resize(self.block_index + 1, axis=0)
        self.ai_ds[self.block_index, :, :] = ai
        self.last_ai_block = ai.copy()
        self.block_index += 1

    def log_random_block(self, ao_block, ai_block, level_fraction, sat_frac):
        if not self.enabled:
            return
        self._append_block(ao_block, ai_block)
        self.timestamps.append(time.time())
        self.level_fraction.append(float(level_fraction))
        self.sat_fraction.append(float(sat_frac))

    def log_sine_block(self, ao_block, ai_block, level_fraction, sat_frac, freq_hz, peak_meas, peak_target):
        if not self.enabled:
            return
        self._append_block(ao_block, ai_block)
        self.timestamps.append(time.time())
        self.level_fraction.append(float(level_fraction))
        self.sat_fraction.append(float(sat_frac))
        self.sine_block_values.append((float(freq_hz), float(peak_meas), float(peak_target)))
        freq_val = float(freq_hz)
        peak_meas_val = float(peak_meas)
        peak_target_val = float(peak_target)
        if self.sine_freqs and math.isclose(self.sine_freqs[-1], freq_val, rel_tol=1e-6, abs_tol=1e-3):
            self.sine_peak_meas[-1] = max(self.sine_peak_meas[-1], peak_meas_val)
            self.sine_peak_target[-1] = max(self.sine_peak_target[-1], peak_target_val)
        else:
            self.sine_freqs.append(freq_val)
            self.sine_peak_meas.append(peak_meas_val)
            self.sine_peak_target.append(peak_target_val)

    def _write_random_preview(self):
        if self.last_ai_block is None:
            return
        freq, _ = welch(
            self.last_ai_block[0], fs=self.fs, nperseg=min(self.welch_nperseg, self.block_samples)
        )
        psd_channels = []
        for ch_idx in range(self.last_ai_block.shape[0]):
            _, psd = welch(
                self.last_ai_block[ch_idx],
                fs=self.fs,
                nperseg=min(self.welch_nperseg, self.block_samples),
            )
            psd_channels.append(psd)

        preview_path = self.run_dir / 'preview_psd.csv'
        with preview_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            header = ['frequency_hz'] + [f'ch{idx}_psd' for idx in range(self.last_ai_block.shape[0])]
            writer.writerow(header)
            for i in range(len(freq)):
                row = [freq[i]] + [psd_channels[ch][i] for ch in range(len(psd_channels))]
                writer.writerow(row)

    def _write_sine_preview(self):
        if not self.sine_freqs:
            return
        summary_path = self.run_dir / 'sine_summary.csv'
        with summary_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['frequency_hz', 'g_peak_measured', 'g_peak_target'])
            for freq, meas, target in zip(self.sine_freqs, self.sine_peak_meas, self.sine_peak_target):
                writer.writerow([freq, meas, target])

    def close(self):
        if not self.enabled or self.h5 is None:
            return

        meta_grp = self.h5.create_group('metadata')
        meta_grp.create_dataset('timestamp', data=np.array(self.timestamps, dtype=float))
        meta_grp.create_dataset('level_fraction', data=np.array(self.level_fraction, dtype=float))
        meta_grp.create_dataset('sat_fraction', data=np.array(self.sat_fraction, dtype=float))

        if self.mode == 'sine_sweep':
            sine_values = np.asarray(self.sine_block_values, dtype=float).reshape(-1, 3)
            meta_grp.create_dataset('frequency_hz', data=sine_values[:, 0])
            meta_grp.create_dataset('g_peak_measured', data=sine_values[:, 1])
            meta_grp.create_dataset('g_peak_target', data=sine_values[:, 2])

        self.h5.flush()
        self.h5.close()
        self.h5 = None

        if self.mode.startswith('random'):
            self._write_random_preview()
        elif self.mode == 'sine_sweep':
            self._write_sine_preview()


class SharedConfig(QObject):
    """Shared configuration object with signals for updates"""
    config_updated = Signal()
    
    def __init__(self):
        super().__init__()
        self.revision = 0
        self.load_from_config()
    
    def load_from_config(self):
        """Load configuration from config.py"""
        # System parameters
        self.fs = config.FS
        self.buf_seconds = config.BUF_SECONDS
        self.block_seconds = config.BLOCK_SECONDS
        self.welch_nperseg = config.WELCH_NPERSEG
        
        # Target PSD
        self.target_psd_points = list(config.TARGET_PSD_POINTS)

        # Test mode and sine sweep configuration
        test_mode = str(getattr(config, 'TEST_MODE', 'random')).strip().lower()
        if test_mode not in {"random", "sine_sweep"}:
            test_mode = "random"
        self.test_mode = test_mode
        self.sine_start_hz = float(getattr(config, 'SINE_SWEEP_START_HZ', 20.0))
        self.sine_end_hz = float(getattr(config, 'SINE_SWEEP_END_HZ', 2000.0))
        self.sine_g_level = float(getattr(config, 'SINE_SWEEP_G_LEVEL', 3.0))
        self.sine_g_level_is_rms = bool(getattr(config, 'SINE_SWEEP_G_LEVEL_IS_RMS', False))
        self.sine_octaves_per_min = float(getattr(config, 'SINE_SWEEP_OCTAVES_PER_MIN', 1.0))
        self.sine_repeat = bool(getattr(config, 'SINE_SWEEP_REPEAT', True))
        # Use random vibration defaults if sine sweep overrides are not defined
        sine_initial_default = getattr(config, 'SINE_SWEEP_INITIAL_LEVEL', config.INITIAL_LEVEL_FRACTION)
        sine_rate_default = getattr(config, 'SINE_SWEEP_MAX_LEVEL_RATE', config.MAX_LEVEL_FRACTION_RATE)
        self.sine_points_per_octave = float(getattr(config, 'SINE_SWEEP_POINTS_PER_OCTAVE', 12.0))
        self.sine_step_dwell = float(max(0.05, getattr(config, 'SINE_SWEEP_STEP_DWELL', 0.5)))
        self.sine_default_vpk = float(max(0.0, getattr(config, 'SINE_SWEEP_DEFAULT_VPK', 0.4)))
        self.sine_drive_scale = float(getattr(config, 'SINE_SWEEP_DRIVE_SCALE', 1.0))
        table_raw = getattr(config, 'SINE_SWEEP_DRIVE_TABLE', [])
        try:
            self.sine_drive_table = [(float(f), float(v)) for f, v in table_raw]
        except Exception:
            self.sine_drive_table = []

        # Data logging
        self.data_log_enabled = bool(getattr(config, 'DATA_LOG_ENABLED', False))
        self.data_log_dir = str(resolve_recording_directory(getattr(config, 'DATA_LOG_DIR', None)))

        # Control parameters
        self.initial_level_fraction = config.INITIAL_LEVEL_FRACTION
        self.max_level_fraction_rate = config.MAX_LEVEL_FRACTION_RATE
        self.kp = config.KP
        self.ki = config.KI

        self.sine_initial_level = float(sine_initial_default)
        self.sine_max_level_rate = float(sine_rate_default)
        self.sine_initial_settle = float(max(0.0, getattr(config, 'SINE_SWEEP_INITIAL_SETTLE', 0.0)))
        
        # Equalizer parameters
        self.eq_num_bands = config.EQ_NUM_BANDS
        self.eq_adapt_rate = config.EQ_ADAPT_RATE
        self.eq_smooth_factor = config.EQ_SMOOTH_FACTOR
        self.eq_adapt_level_threshold = float(getattr(config, 'EQ_ADAPT_LEVEL_THRESHOLD', 0.4))
        self.eq_adapt_level_power = float(getattr(config, 'EQ_ADAPT_LEVEL_POWER', 1.0))
        self.eq_adapt_min_weight = float(getattr(config, 'EQ_ADAPT_MIN_WEIGHT', 0.0))
        
        # Safety limiters
        self.max_crest_factor = config.MAX_CREST_FACTOR
        self.max_rms_volts = config.MAX_RMS_VOLTS
        self.ao_volt_limit = config.AO_VOLT_LIMIT
        self.accel_excitation_amps = getattr(config, 'ACCEL_EXCITATION_AMPS', 0.0)
        self.sync_ao_clock = bool(getattr(config, 'AO_SYNC_WITH_AI', False))
        
        # Simulation mode
        self.simulation_mode = config.SIMULATION_MODE
        self.sim_plant_gain = config.SIM_PLANT_GAIN
        self.sim_noise_level = config.SIM_NOISE_LEVEL

        self.device_ai = config.DEVICE_AI
        self.device_ao = config.DEVICE_AO
        self.accel_mv_per_g = config.ACCEL_MV_PER_G
        self.response_channels = copy.deepcopy(getattr(config, 'RESPONSE_CHANNELS', []))
        self.sim_resonances = list(config.SIM_RESONANCES)
        self.sim_delay_samples = config.SIM_DELAY_SAMPLES
        self.sim_nonlinearity = config.SIM_NONLINEARITY
        self.crest_soft_knee = config.CREST_SOFT_KNEE
        self.rms_limit_headroom = config.RMS_LIMIT_HEADROOM
        self.eq_gain_limits = tuple(config.EQ_GAIN_LIMITS)
        self.calibration_source = str(getattr(config, 'SINE_SWEEP_CALIBRATION_SOURCE', ''))
        self.preset_source = ''

        # Input channel metadata
        labels = getattr(config, "INPUT_CHANNEL_LABELS", ["Control Accel"])
        self.input_channel_labels = list(labels)
        if not self.input_channel_labels:
            self.input_channel_labels = ["Control Accel"]
        self.num_input_channels = len(self.input_channel_labels) if self.simulation_mode else 1 + len(self.response_channels)

        # Real-time viewer tuning
        self.realtime_psd_update_stride = int(getattr(config, "REALTIME_PSD_UPDATE_STRIDE", 5))
        self._fields = tuple(key for key, value in vars(self).items()
                             if not key.startswith('_') and isinstance(value, (str, bool, int, float, list, tuple, dict)))

    def as_dict(self):
        return {key: copy.deepcopy(getattr(self, key)) for key in self._fields}

    def snapshot(self):
        return RunSettings.capture(self.as_dict())

    def apply_values(self, values):
        # Validate the whole draft before mutating any accepted setting.
        labels = [str(label) for label in values['input_channel_labels']]
        values = RunSettings.capture(values).as_dict()
        values['input_channel_labels'] = labels
        for key in self._fields:
            if key in values and key != 'revision':
                setattr(self, key, copy.deepcopy(values[key]))
        self.revision += 1
        self.num_input_channels = len(self.input_channel_labels) if self.simulation_mode else 1 + len(self.response_channels)
        self.config_updated.emit()



class ControllerWorker(QThread):
    """Acquisition thread; its Qt object stays owned by the GUI thread.

    No QObject is moved to or destroyed in the acquisition thread. This avoids
    PySide disconnect/destruction lock inversion while GUI plots are updating.
    """
    
    # Signals for thread-safe communication
    psd_data_ready = Signal(object)  # frequency, measured, target (or None), averaged
    metrics_data_ready = Signal(object)  # Unit-qualified metric dictionary
    eq_data_ready = Signal(object)  # (freq_centers, gains)
    realtime_data_ready = Signal(object)  # streaming block of acceleration data
    status_message = Signal(str)
    state_changed = Signal(object)
    report_ready = Signal(object)
    
    def __init__(self, shared_config):
        super().__init__()
        self.shared_config = shared_config if isinstance(shared_config, RunSettings) else shared_config.snapshot()
        self.actual_backend = None
        self.input_mode = None
        self.clock_status = 'Not initialized'
        self.log_dir = None
        self.blocks = 0
        self.last_metrics = None
        self.terminal_state = "stopped"
        self.error_message = ""
        self.should_stop = Event()
        self.is_running = False
        
        # PSD averaging
        self.psd_averaged = None
        self.psd_alpha = 0.1  # Exponential averaging factor
    
    def _announce(self, state, message=''):
        self.state_changed.emit({'state': state, 'backend': self.actual_backend,
                                 'requested_backend': self.shared_config.backend, 'message': message,
                                 'input_mode': self.input_mode, 'clock_status': self.clock_status,
                                 'log_dir': self.log_dir})
        if message:
            self.status_message.emit(message)

    def _publish_metrics(self, metrics):
        self.blocks += 1
        metrics['elapsed_s'] = time.monotonic() - self.monotonic_start
        metrics['block_index'] = self.blocks
        self.last_metrics = metrics
        self.metrics_data_ready.emit(metrics)

    @Slot()
    def run(self):
        self.run_controller()

    @Slot()
    def run_controller(self):
        """Main controller loop running in separate thread"""
        ao_task = None
        ai_task = None
        ai_reader = None
        ao_writer = None
        volts_to_g_scale = None
        data_logger = None
        started_tasks = []
        try:
            self.is_running = True
            self.start_time = time.time()
            self.monotonic_start = time.monotonic()
            self.started_at = datetime.now().astimezone().isoformat()
            self._announce('starting', 'Initializing the selected backend…')
            if self.should_stop.is_set():
                return
            
            # Initialize system based on current config
            fs = self.shared_config.fs
            block_samples = int(fs * self.shared_config.block_seconds)
            
            test_mode = self.shared_config.test_mode
            if test_mode == 'random':
                # Create controller
                controller = RandomVibrationController(
                    target_psd_points=self.shared_config.target_psd_points,
                    fs=fs,
                    Kp=self.shared_config.kp,
                    Ki=self.shared_config.ki,
                    max_level_fraction_rate=self.shared_config.max_level_fraction_rate,
                    welch_nperseg=self.shared_config.welch_nperseg
                )
                # Use the captured piecewise PSD integral for both command
                # scaling and the target reported by setup/results.
                controller.a_rms_target = self.shared_config.target_rms_g
            
                # Create equalizer
                f1 = self.shared_config.target_psd_points[0][0]
                f2 = self.shared_config.target_psd_points[-1][0]
            
                equalizer = MultiBandEqualizer(
                    f1=f1, f2=f2,
                    num_bands=self.shared_config.eq_num_bands,
                    fs=fs,
                    gain_limits=self.shared_config.eq_gain_limits,
                    adapt_rate=self.shared_config.eq_adapt_rate,
                    smooth_factor=self.shared_config.eq_smooth_factor
                )
            
                # Create bandpass filter
                bp = create_bandpass_filter(f1, f2, fs)

            # Initialize DAQ or simulation backend
            if self.should_stop.is_set():
                return
            simulation_mode = bool(self.shared_config.simulation_mode)
            mode_label = "SIMULATION"

            if not simulation_mode and not NIDAQMX_AVAILABLE:
                raise RuntimeError('NI-DAQ Python bindings are unavailable. Select Simulation explicitly to run a simulation.')

            if simulation_mode:
                ao_writer, ai_reader, ai_task, ao_task = create_simulation_system(
                    fs, self.shared_config.sim_plant_gain, self.shared_config.sim_resonances,
                    self.shared_config.sim_noise_level, self.shared_config.sim_delay_samples,
                    self.shared_config.sim_nonlinearity, num_input_channels=self.shared_config.num_input_channels
                )
                self.input_mode = 'Synthetic acceleration'
                self.clock_status = 'Synthetic sample clock'
                plant_gain_g_per_V = self.shared_config.sim_plant_gain * 0.8
                mode_label = "SIMULATION"
            else:
                try:
                    device_ao = self.shared_config.device_ao
                    ao_volt_limit = self.shared_config.ao_volt_limit
                    buf_samples = int(self.shared_config.fs * self.shared_config.buf_seconds)
                    ai_task, volts_to_g_scale, self.input_mode = create_hardware_input_task(self.shared_config)
                    ao_task = nidaqmx.Task()
                    print(self.input_mode)
                    if np.any(volts_to_g_scale != 1.0) or 'Voltage input' in self.input_mode:
                        self.status_message.emit('Some inputs use voltage scaling; see Results for input modes.')

                    ai_task.timing.cfg_samp_clk_timing(
                        rate=fs,
                        sample_mode=AcquisitionType.CONTINUOUS,
                        samps_per_chan=buf_samples,
                    )
                    try:
                        desired_ai_buf = max(buf_samples * 4, ai_task.in_stream.input_buf_size)
                        ai_task.in_stream.input_buf_size = desired_ai_buf
                    except Exception:
                        pass

                    ao_task.ao_channels.add_ao_voltage_chan(
                        physical_channel=device_ao,
                        name_to_assign_to_channel="drive",
                        min_val=-ao_volt_limit,
                        max_val=ao_volt_limit,
                        units=VoltageUnits.VOLTS,
                    )
                    ao_task.timing.cfg_samp_clk_timing(
                        rate=fs,
                        sample_mode=AcquisitionType.CONTINUOUS,
                        samps_per_chan=buf_samples,
                    )
                    try:
                        desired_ao_buf = max(buf_samples * 4, ao_task.out_stream.output_buf_size)
                        ao_task.out_stream.output_buf_size = desired_ao_buf
                    except Exception:
                        pass

                    if self.shared_config.sync_ao_clock:
                        try:
                            ao_task.timing.samp_clk_src = ai_task.timing.samp_clk_term
                            ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                                ai_task.triggers.start_trigger.term
                            )
                            self.clock_status = 'Shared AI clock configured'
                        except Exception as sync_err:
                            self.clock_status = 'Requested shared clock could not be configured'
                            raise RuntimeError(f'Requested AI/AO clock sharing failed: {sync_err}. Select independent clocks in setup if intended.') from sync_err
                    else:
                        self.clock_status = 'Independent AI/AO clocks configured'
                        print("AO synchronization disabled by configuration; using onboard clock.")

                    reader_class = AnalogMultiChannelReader if self.shared_config.num_input_channels > 1 else AnalogSingleChannelReader
                    ai_reader = reader_class(ai_task.in_stream)
                    ao_writer = AnalogSingleChannelWriter(ao_task.out_stream, auto_start=False)

                    ao_writer.write_many_sample(np.zeros(buf_samples))
                    ai_task.start()
                    started_tasks.append(ai_task)
                    ao_task.start()
                    started_tasks.append(ao_task)

                    plant_gain_g_per_V = 1.0
                    mode_label = "HARDWARE"
                except Exception as hw_err:
                    raise RuntimeError(f'Hardware initialization failed: {hw_err}') from hw_err

            if ao_writer is None or ai_reader is None:
                raise RuntimeError("Failed to initialize data acquisition path")

            self.actual_backend = mode_label.title()
            print(f"Controller worker running in {mode_label.lower()} mode.")

            test_mode = getattr(self.shared_config, 'test_mode', 'random') or 'random'

            if getattr(self.shared_config, 'data_log_enabled', False):
                mode_tag = 'sine_sweep' if test_mode.lower() == 'sine_sweep' else 'random_vibration'
                try:
                    data_logger = DataLogger(
                        self.shared_config,
                        mode_tag,
                        fs,
                        block_samples,
                        self.shared_config.num_input_channels,
                    )
                    self.log_dir = str(data_logger.run_dir)
                    data_logger.h5.attrs['input_mode'] = self.input_mode
                    data_logger.h5.attrs['clock_status'] = self.clock_status
                except Exception as log_err:
                    raise RuntimeError(f'Recording could not start: {log_err}') from log_err

            self._announce('running', f'{self.actual_backend} run active')

            if test_mode.lower() == 'sine_sweep':
                self._run_sine_sweep_loop(
                    ao_writer=ao_writer,
                    ai_reader=ai_reader,
                    simulation_mode=simulation_mode,
                    volts_to_g_scale=volts_to_g_scale,
                    plant_gain_initial=plant_gain_g_per_V,
                    fs=fs,
                    block_samples=block_samples,
                    data_logger=data_logger,
                )
                return

            # Control loop variables
            level_fraction = self.shared_config.initial_level_fraction
            block_duration = block_samples / max(fs, 1e-9)
            io_timeout = max(1.0, block_duration * 2.0 + 0.5)
            adapt_threshold = float(getattr(self.shared_config, 'eq_adapt_level_threshold', 0.4))
            adapt_power = float(max(0.0, getattr(self.shared_config, 'eq_adapt_level_power', 1.0)))
            adapt_min_weight = float(np.clip(getattr(self.shared_config, 'eq_adapt_min_weight', 0.0), 0.0, 1.0))

            
            loop_count = 0
            while not self.should_stop.is_set():
                loop_count += 1
                loop_start = time.perf_counter()
                current_time = time.time() - self.start_time
                
                # Debug logging every 10 seconds
                if loop_count % 100 == 0:  # ~10 seconds at 0.1s sleep
                    print(f"Debug: Control loop running at {current_time:.1f}s, loop {loop_count}")
                
                # Generate drive signal
                drive = make_bandlimited_noise(block_samples, bp, equalizer)
                if np.std(drive) > 1e-12:
                    drive = drive / np.std(drive)
                
                # Calculate target voltage
                command_level = level_fraction
                target_rms_block = controller.a_rms_target * command_level
                volts_block = (target_rms_block / max(plant_gain_g_per_V, 1e-6)) * drive
                
                # Apply minimum drive
                min_drive_rms = 0.05
                current_rms = np.std(volts_block)
                if current_rms < min_drive_rms:
                    volts_block = volts_block * (min_drive_rms / max(current_rms, 1e-12))
                
                # Apply safety limiters
                volts_block, limiter_stats = apply_safety_limiters(
                    volts_block, self.shared_config.max_crest_factor, 
                    self.shared_config.max_rms_volts, self.shared_config.crest_soft_knee, self.shared_config.rms_limit_headroom)
                
                # Clip to limits
                clipping_fraction = float(np.mean(np.abs(volts_block) > self.shared_config.ao_volt_limit))
                volts_block = np.clip(volts_block, -self.shared_config.ao_volt_limit, 
                                    self.shared_config.ao_volt_limit)
                
                # Simulate plant response
                if simulation_mode:
                    ao_writer.write_many_sample(volts_block)
                else:
                    ao_writer.write_many_sample(volts_block, timeout=io_timeout)
                if self.shared_config.num_input_channels > 1:
                    data = np.empty((self.shared_config.num_input_channels, block_samples), dtype=np.float64)
                else:
                    data = np.empty(block_samples, dtype=np.float64)
                if simulation_mode:
                    ai_reader.read_many_sample(data, number_of_samples_per_channel=block_samples)
                else:
                    ai_reader.read_many_sample(data, number_of_samples_per_channel=block_samples, timeout=io_timeout)

                accel_data = scale_input_block(data, volts_to_g_scale)
                
                # Estimate PSD
                f, Pxx, metrics = controller.estimate_psd(accel_data)
                if metrics is None:
                    print(f"Debug: PSD estimation returned None at {time.time() - self.start_time:.1f}s")
                    continue
                
                S_avg_meas, S_avg_target, a_rms_meas = metrics
                
                # Update plant gain
                cmd_rms = np.std(volts_block)
                if cmd_rms > 1e-9:
                    new_gain = a_rms_meas / cmd_rms
                    plant_gain_g_per_V = 0.8*plant_gain_g_per_V + 0.2*new_gain
                
                # Update equalizer
                level_fraction_clamped = float(np.clip(level_fraction, 0.0, 1.0))
                if adapt_threshold >= 1.0:
                    adapt_weight = 1.0 if level_fraction_clamped >= 1.0 else 0.0
                else:
                    adapt_weight = (level_fraction_clamped - adapt_threshold) / max(1.0 - adapt_threshold, 1e-6)
                adapt_weight = float(np.clip(adapt_weight, 0.0, 1.0))
                if adapt_power > 0.0 and adapt_weight > 0.0:
                    adapt_weight = adapt_weight ** adapt_power
                if adapt_weight < adapt_min_weight:
                    adapt_weight = adapt_min_weight

                equalizer.update_gains(f, Pxx, controller.target_psd_func, adapt_weight=adapt_weight)
                
                # Update control
                level_fraction = controller.update_control(S_avg_meas, S_avg_target, level_fraction)
                
                # Calculate saturation
                sat_frac = np.mean(np.abs(volts_block) >= (0.98*self.shared_config.ao_volt_limit))
                
                # Emit data for GUI updates
                psd_target = controller.target_psd_func(f)
                
                # Calculate averaged PSD (exponential smoothing)
                if self.psd_averaged is None:
                    self.psd_averaged = Pxx.copy()
                else:
                    self.psd_averaged = self.psd_alpha * Pxx + (1 - self.psd_alpha) * self.psd_averaged
                
                self.psd_data_ready.emit((f, Pxx, psd_target, self.psd_averaged))
                
                output_rms = float(np.sqrt(np.mean(volts_block ** 2)))
                self._publish_metrics({'accel_rms_g': float(a_rms_meas),
                    'target_rms_g': controller.a_rms_target * command_level,
                    'level_fraction': command_level, 'next_level_fraction': level_fraction,
                    'saturation_fraction': float(sat_frac), 'clipping_fraction': clipping_fraction,
                    'plant_gain_g_per_v': float(plant_gain_g_per_V),
                    'output_rms_v': output_rms, 'output_peak_v': float(np.max(np.abs(volts_block))),
                    'output_cf': float(np.max(np.abs(volts_block)) / output_rms) if output_rms else 0.0,
                    'limiter_active': bool(limiter_stats['any_limiting'])})
                
                eq_info = equalizer.get_band_info()
                self.eq_data_ready.emit((eq_info['centers'], eq_info['gains']))
                
                # Emit current block for real-time visualization
                self.realtime_data_ready.emit(np.array(data, copy=True))

                if data_logger:
                    data_logger.log_random_block(
                        volts_block,
                        np.array(data, copy=True),
                        command_level,
                        sat_frac,
                    )

                # Debug logging for data emission
                if loop_count % 50 == 0:  # Every ~5 seconds
                    print(f"Debug: Emitting data at {current_time:.1f}s, RMS={np.std(accel_data):.4f}")
                
                # Processing consumes part of the block's wall-clock budget.
                # Both backends wait only for the remainder, interruptibly.
                remaining = block_duration - (time.perf_counter() - loop_start)
                if remaining > 0:
                    self.should_stop.wait(remaining)
                
        except Exception as e:
            cancelled_read = (self.should_stop.is_set() and DaqError is not None
                              and isinstance(e, DaqError) and e.error_code in (-200284, -200474))
            if not cancelled_read:
                self.terminal_state = 'fault'
                self.error_message = str(e)
                self._announce('fault', self.error_message)
        finally:
            cleanup_errors = []
            if data_logger:
                try:
                    data_logger.close()
                except Exception as error:
                    cleanup_errors.append(f'Recording finalization failed: {error}')
            for task in (ao_task, ai_task):
                if task is None:
                    continue
                operations = (task.stop, task.close) if task in started_tasks else (task.close,)
                for operation in operations:
                    try:
                        operation()
                    except Exception as error:
                        cleanup_errors.append(f'DAQ cleanup: {error}')
            if cleanup_errors:
                self.terminal_state = 'fault'
                self.error_message = '; '.join(filter(None, [self.error_message] + cleanup_errors))
            self.is_running = False
            report = {'outcome': self.terminal_state, 'backend': self.actual_backend,
                'input_mode': self.input_mode, 'clock_status': self.clock_status,
                'requested_backend': self.shared_config.backend, 'error': self.error_message,
                'started_at': getattr(self, 'started_at', ''),
                'duration_s': time.monotonic() - getattr(self, 'monotonic_start', time.monotonic()),
                'blocks': self.blocks, 'latest_metrics': self.last_metrics,
                'settings': self.shared_config.as_dict(), 'config_id': self.shared_config.config_id,
                'log_dir': str(data_logger.run_dir) if data_logger and data_logger.enabled else None}
            if report['log_dir']:
                try:
                    (Path(report['log_dir']) / 'run-summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
                except Exception as error:
                    report['outcome'] = self.terminal_state = 'fault'
                    report['error'] = self.error_message = f'{self.error_message} Run summary could not be saved: {error}'.strip()
            self._announce(self.terminal_state, self.error_message or ('Sweep complete' if self.terminal_state == 'complete' else 'Run stopped'))
            self.report_ready.emit(report)

    def _run_sine_sweep_loop(self, ao_writer, ai_reader, simulation_mode, volts_to_g_scale,
                              plant_gain_initial, fs, block_samples, data_logger=None):
        """Execute a sine sweep control loop using current configuration."""
        level_fraction = float(np.clip(self.shared_config.sine_initial_level, 0.0, 1.0))
        start_hz = max(0.01, float(self.shared_config.sine_start_hz))
        end_hz = max(0.01, float(self.shared_config.sine_end_hz))
        max_level_rate = max(0.0, float(self.shared_config.sine_max_level_rate))

        block_duration = block_samples / max(fs, 1e-9)
        io_timeout = max(1.0, block_duration * 2.0 + 0.5)
        plant_gain = max(float(plant_gain_initial), 1e-6)
        last_level_update = time.time()
        t_last_print = time.time()

        points_per_octave = max(1.0, float(self.shared_config.sine_points_per_octave))
        stepper = LogSweepStepper(
            start_freq=start_hz,
            end_freq=end_hz,
            points_per_octave=points_per_octave,
            repeat=bool(self.shared_config.sine_repeat),
        )
        oscillator = SineOscillator(fs, stepper.current_frequency())
        octaves_per_min = float(self.shared_config.sine_octaves_per_min)
        if abs(octaves_per_min) > 0.0:
            dwell_from_rate = 60.0 / (abs(octaves_per_min) * points_per_octave)
        else:
            dwell_from_rate = 0.0
        dwell_seconds = max(
            self.shared_config.sine_step_dwell,
            block_duration,
            dwell_from_rate,
        )
        lookup_drive = build_drive_lookup(
            self.shared_config.sine_drive_table,
            self.shared_config.sine_default_vpk,
        )

        target_g_rms_value = target_g_rms(
            self.shared_config.sine_g_level,
            self.shared_config.sine_g_level_is_rms,
        )
        target_g_peak = target_g_rms_value * math.sqrt(2.0)

        self.eq_data_ready.emit(None)
        self.psd_averaged = None
        first_step = True

        while not self.should_stop.is_set():
            freq = stepper.current_frequency()
            oscillator.set_frequency(freq)
            extra_settle = self.shared_config.sine_initial_settle if first_step else 0.0
            dwell_remaining = dwell_seconds + max(0.0, float(extra_settle))

            drive_vpk = max(0.0, lookup_drive(freq) * self.shared_config.sine_drive_scale)
            preview_target_peak_g = target_g_peak * max(level_fraction, 0.0)
            preview_target_vpk = drive_vpk * preview_target_peak_g

            print(
                f"Sine sweep step: {freq:.1f} Hz, command ~= {preview_target_vpk:.3f} Vpk "
                f"(target ~= {preview_target_peak_g:.3f} gpk)"
            )

            while dwell_remaining > 1e-9 and not self.should_stop.is_set():
                loop_start = time.perf_counter()
                sin_block, _ = oscillator.generate(block_samples)

                command_level = level_fraction
                target_peak_g_current = target_g_peak * max(command_level, 0.0)
                target_vpk = drive_vpk * target_peak_g_current
                command_block = target_vpk * sin_block

                command_block, limiter_stats = apply_safety_limiters(
                    command_block,
                    self.shared_config.max_crest_factor,
                    self.shared_config.max_rms_volts,
                    self.shared_config.crest_soft_knee,
                    self.shared_config.rms_limit_headroom,
                )
                clipping_fraction = float(np.mean(np.abs(command_block) > self.shared_config.ao_volt_limit))
                command_block = np.clip(
                    command_block,
                    -self.shared_config.ao_volt_limit,
                    self.shared_config.ao_volt_limit,
                )

                if simulation_mode:
                    ao_writer.write_many_sample(command_block)
                else:
                    ao_writer.write_many_sample(command_block, timeout=io_timeout)

                if self.shared_config.num_input_channels > 1:
                    data = np.empty((self.shared_config.num_input_channels, block_samples), dtype=np.float64)
                else:
                    data = np.empty(block_samples, dtype=np.float64)

                if simulation_mode:
                    ai_reader.read_many_sample(data, number_of_samples_per_channel=block_samples)
                else:
                    ai_reader.read_many_sample(
                        data,
                        number_of_samples_per_channel=block_samples,
                        timeout=io_timeout,
                    )

                accel_data = scale_input_block(data, volts_to_g_scale)
                cmd_rms = float(np.std(command_block))
                meas_rms = float(np.std(accel_data))
                meas_peak = meas_rms * math.sqrt(2.0)
                a_rms_meas = meas_rms

                if cmd_rms > 1e-9:
                    new_gain = meas_rms / cmd_rms
                    plant_gain = 0.8 * plant_gain + 0.2 * new_gain

                now = time.time()
                dt = max(now - last_level_update, 1e-3)
                last_level_update = now
                if level_fraction < 1.0 and max_level_rate > 0.0:
                    level_fraction = min(1.0, level_fraction + max_level_rate * dt)

                sat_frac = float(
                    np.mean(np.abs(command_block) >= (0.98 * self.shared_config.ao_volt_limit))
                )
                if sat_frac > 0.2:
                    level_fraction = max(0.0, level_fraction * 0.9)
                    self.status_message.emit("AO near saturation; backing off level.")

                nperseg = int(self.shared_config.welch_nperseg)
                if nperseg > len(accel_data):
                    nperseg = len(accel_data)
                f, Pxx = welch(accel_data, fs=fs, nperseg=nperseg)

                if self.psd_averaged is None or len(self.psd_averaged) != len(Pxx):
                    self.psd_averaged = Pxx.copy()
                else:
                    self.psd_averaged = (
                        self.psd_alpha * Pxx + (1 - self.psd_alpha) * self.psd_averaged
                    )

                self.psd_data_ready.emit((f, Pxx, None, self.psd_averaged))
                output_rms = float(np.sqrt(np.mean(command_block ** 2)))
                progress = (abs(math.log(freq / start_hz) / math.log(end_hz / start_hz)) if start_hz != end_hz else 1.0)
                self._publish_metrics({'accel_rms_g': a_rms_meas,
                    'accel_peak_g': meas_peak, 'target_peak_g': target_peak_g_current,
                    'target_rms_g': target_peak_g_current / math.sqrt(2),
                    'frequency_hz': freq, 'sweep_progress': min(1.0, progress),
                    'level_fraction': command_level, 'next_level_fraction': level_fraction,
                    'saturation_fraction': sat_frac, 'clipping_fraction': clipping_fraction,
                    'plant_gain_g_per_v': plant_gain,
                    'output_rms_v': output_rms, 'output_peak_v': float(np.max(np.abs(command_block))),
                    'output_cf': float(np.max(np.abs(command_block)) / output_rms) if output_rms else 0.0,
                    'limiter_active': bool(limiter_stats['any_limiting'])})
                self.realtime_data_ready.emit(np.array(data, copy=True))

                if data_logger:
                    data_logger.log_sine_block(
                        command_block,
                        np.array(data, copy=True),
                        command_level,
                        sat_frac,
                        freq,
                        meas_peak,
                        target_peak_g_current,
                    )

                dwell_remaining -= block_duration

                if getattr(config, 'CONSOLE_UPDATE_INTERVAL', 5.0) > 0:
                    pass  # optional additional logging removed for simplicity

                remaining = block_duration - (time.perf_counter() - loop_start)
                if remaining > 0:
                    self.should_stop.wait(remaining)

            first_step = False
            if not stepper.advance():
                if not self.shared_config.sine_repeat:
                    self.terminal_state = "complete"
                    break

    def stop(self):
        """Stop the controller loop"""
        self.should_stop.set()


from workspace_ui import WorkspaceWindow, SettingsPanel, RunView, ChannelView

# Keep the public desktop entry point; the native workspace owns presentation.
ConfigurationTab = SettingsPanel
ControllerTab = RunView
RealTimeDataTab = ChannelView


class MainWindow(WorkspaceWindow):
    def __init__(self):
        super().__init__(SharedConfig(), ControllerWorker)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Random Vibration Control")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Vibration Testing Systems")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
