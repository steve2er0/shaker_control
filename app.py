#!/usr/bin/env python3
"""
Random Vibration Shaker Control - Desktop GUI Application

A professional desktop application for random vibration testing with:
- Configuration tab for all system parameters
- Controller tab with start/stop controls and response visualization
- Real-time data tab with time domain and PSD plots

Built with PySide6 and PyQtGraph for high-performance plotting.
"""

import sys
import csv
import numpy as np
import time
import math
from collections import deque
from threading import Thread, Event
from pathlib import Path
from datetime import datetime
from scipy.signal import welch
try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

# GUI imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                              QHBoxLayout, QWidget, QFormLayout, QDoubleSpinBox, 
                              QSpinBox, QPushButton, QCheckBox, QLabel, QGroupBox,
                              QComboBox, QScrollArea, QStackedWidget)
from PySide6.QtCore import QTimer, Signal, QObject, QThread, Qt
from PySide6.QtGui import QFont

# Plotting imports
import pyqtgraph as pg

# Optional hardware imports
try:
    import nidaqmx
    from nidaqmx.constants import (AcquisitionType, TerminalConfiguration, ExcitationSource,
                                   AccelUnits, Coupling, VoltageUnits, AccelSensitivityUnits,
                                   RegenerationMode, WAIT_INFINITELY)
    from nidaqmx.errors import DaqError
    from nidaqmx.stream_writers import AnalogSingleChannelWriter
    from nidaqmx.stream_readers import AnalogSingleChannelReader
    NIDAQMX_AVAILABLE = True
except ImportError:
    nidaqmx = None
    DaqError = None
    NIDAQMX_AVAILABLE = False
    WAIT_INFINITELY = None

# Import our existing modules
import config
from rv_controller import MultiBandEqualizer, RandomVibrationController, create_bandpass_filter, make_bandlimited_noise, apply_safety_limiters
from simulation import create_simulation_system
from sine_sweep import (
    LogSweepStepper,
    SineOscillator,
    build_drive_lookup,
)


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
            print("Warning: h5py not available; data logging disabled.")
            self.enabled = False
            self.h5 = None
            return

        base_dir = Path(getattr(shared_config, 'data_log_dir', 'data_logs')).expanduser()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = base_dir / f"{timestamp}_{mode}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.h5 = h5py.File(self.run_dir / 'data.h5', 'w')
        self.h5.attrs['mode'] = mode
        self.h5.attrs['fs'] = self.fs
        self.h5.attrs['block_samples'] = self.block_samples
        self.h5.attrs['configured_channels'] = self.config_channels

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
        self.sine_freqs.append(float(freq_hz))
        self.sine_peak_meas.append(float(peak_meas))
        self.sine_peak_target.append(float(peak_target))

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
        with preview_path.open('w', newline='') as csv_file:
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
        with summary_path.open('w', newline='') as csv_file:
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
            meta_grp.create_dataset('frequency_hz', data=np.array(self.sine_freqs, dtype=float))
            meta_grp.create_dataset('g_peak_measured', data=np.array(self.sine_peak_meas, dtype=float))
            meta_grp.create_dataset('g_peak_target', data=np.array(self.sine_peak_target, dtype=float))

        self.h5.flush()
        self.h5.close()

        if self.mode.startswith('random'):
            self._write_random_preview()
        elif self.mode == 'sine_sweep':
            self._write_sine_preview()


class SharedConfig(QObject):
    """Shared configuration object with signals for updates"""
    config_updated = Signal()
    
    def __init__(self):
        super().__init__()
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
        self.data_log_dir = getattr(config, 'DATA_LOG_DIR', 'data_logs')

        # Control parameters
        self.initial_level_fraction = config.INITIAL_LEVEL_FRACTION
        self.max_level_fraction_rate = config.MAX_LEVEL_FRACTION_RATE
        self.kp = config.KP
        self.ki = config.KI

        self.sine_initial_level = float(sine_initial_default)
        self.sine_max_level_rate = float(sine_rate_default)
        
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

        # Input channel metadata
        labels = getattr(config, "INPUT_CHANNEL_LABELS", ["Control Accel"])
        self.input_channel_labels = list(labels)
        if not self.input_channel_labels:
            self.input_channel_labels = ["Control Accel"]
        if not self.simulation_mode:
            # Hardware mode currently drives a single control accelerometer channel
            self.input_channel_labels = self.input_channel_labels[:1]
            if not self.input_channel_labels:
                self.input_channel_labels = ["Control Accel"]
        self.num_input_channels = len(self.input_channel_labels)

        # Real-time viewer tuning
        self.realtime_psd_update_stride = int(getattr(config, "REALTIME_PSD_UPDATE_STRIDE", 5))


class ConfigurationTab(QWidget):
    """Configuration tab with form inputs for all parameters"""
    
    def __init__(self, shared_config):
        super().__init__()
        self.shared_config = shared_config
        self.init_ui()
        self.load_values()
    
    def init_ui(self):
        outer_layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # System Parameters Group
        system_group = QGroupBox("System Parameters")
        system_layout = QFormLayout()

        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(1000, 100000)
        self.fs_spin.setSuffix(" Hz")
        system_layout.addRow("Sample Rate:", self.fs_spin)
        
        self.buf_seconds_spin = QDoubleSpinBox()
        self.buf_seconds_spin.setRange(0.1, 10.0)
        self.buf_seconds_spin.setSuffix(" s")
        system_layout.addRow("Buffer Duration:", self.buf_seconds_spin)
        
        self.block_seconds_spin = QDoubleSpinBox()
        self.block_seconds_spin.setRange(0.1, 5.0)
        self.block_seconds_spin.setSuffix(" s")
        system_layout.addRow("Block Duration:", self.block_seconds_spin)
        
        self.welch_nperseg_spin = QSpinBox()
        self.welch_nperseg_spin.setRange(256, 8192)
        system_layout.addRow("Welch Segment Length:", self.welch_nperseg_spin)
        
        self.ao_volt_limit_spin = QDoubleSpinBox()
        self.ao_volt_limit_spin.setRange(0.5, 10.0)
        self.ao_volt_limit_spin.setSuffix(" V")
        system_layout.addRow("AO Voltage Limit:", self.ao_volt_limit_spin)

        system_group.setLayout(system_layout)

        # Test Mode Selection
        mode_group = QGroupBox("Test Mode")
        mode_layout = QFormLayout()
        self.test_mode_combo = QComboBox()
        self.test_mode_combo.addItem("Random Vibration", "random")
        self.test_mode_combo.addItem("Sine Sweep", "sine_sweep")
        mode_layout.addRow("Mode:", self.test_mode_combo)
        mode_group.setLayout(mode_layout)

        # Data logging toggle
        self.data_log_check = QCheckBox("Enable Data Logging")
        mode_layout.addRow("Logging:", self.data_log_check)

        # Control Parameters Group
        control_group = QGroupBox("Control Parameters")
        control_layout = QFormLayout()

        self.kp_spin = QDoubleSpinBox()
        self.kp_spin.setRange(0.1, 10.0)
        self.kp_spin.setDecimals(2)
        control_layout.addRow("Proportional Gain (Kp):", self.kp_spin)
        
        self.ki_spin = QDoubleSpinBox()
        self.ki_spin.setRange(0.01, 5.0)
        self.ki_spin.setDecimals(2)
        control_layout.addRow("Integral Gain (Ki):", self.ki_spin)
        
        self.initial_level_spin = QDoubleSpinBox()
        self.initial_level_spin.setRange(0.1, 5.0)
        self.initial_level_spin.setDecimals(2)
        control_layout.addRow("Initial Level Fraction:", self.initial_level_spin)
        
        self.max_level_rate_spin = QDoubleSpinBox()
        self.max_level_rate_spin.setRange(0.1, 2.0)
        self.max_level_rate_spin.setDecimals(2)
        control_layout.addRow("Max Level Rate:", self.max_level_rate_spin)

        control_group.setLayout(control_layout)

        # Sine sweep parameters
        self.sine_group = QGroupBox("Sine Sweep Parameters")
        sine_layout = QFormLayout()

        self.sine_start_spin = QDoubleSpinBox()
        self.sine_start_spin.setRange(0.1, 20000.0)
        self.sine_start_spin.setDecimals(2)
        self.sine_start_spin.setSuffix(" Hz")
        sine_layout.addRow("Start Frequency:", self.sine_start_spin)

        self.sine_end_spin = QDoubleSpinBox()
        self.sine_end_spin.setRange(0.1, 20000.0)
        self.sine_end_spin.setDecimals(2)
        self.sine_end_spin.setSuffix(" Hz")
        sine_layout.addRow("End Frequency:", self.sine_end_spin)

        self.sine_g_level_spin = QDoubleSpinBox()
        self.sine_g_level_spin.setRange(0.01, 30.0)
        self.sine_g_level_spin.setDecimals(3)
        self.sine_g_level_spin.setSuffix(" g")
        sine_layout.addRow("Target g-Level:", self.sine_g_level_spin)

        self.sine_g_level_is_rms_check = QCheckBox("g-Level is RMS (unchecked = peak)")
        sine_layout.addRow("", self.sine_g_level_is_rms_check)

        self.sine_octaves_spin = QDoubleSpinBox()
        self.sine_octaves_spin.setRange(0.01, 10.0)
        self.sine_octaves_spin.setDecimals(3)
        sine_layout.addRow("Octaves per Minute:", self.sine_octaves_spin)

        self.sine_repeat_check = QCheckBox("Repeat Sweep When Complete")
        sine_layout.addRow("", self.sine_repeat_check)

        self.sine_initial_level_spin = QDoubleSpinBox()
        self.sine_initial_level_spin.setRange(0.0, 2.0)
        self.sine_initial_level_spin.setDecimals(2)
        sine_layout.addRow("Initial Level Fraction:", self.sine_initial_level_spin)

        self.sine_max_level_rate_spin = QDoubleSpinBox()
        self.sine_max_level_rate_spin.setRange(0.0, 5.0)
        self.sine_max_level_rate_spin.setDecimals(2)
        sine_layout.addRow("Max Level Rate:", self.sine_max_level_rate_spin)

        self.sine_group.setLayout(sine_layout)

        # Equalizer Parameters Group
        eq_group = QGroupBox("Equalizer Parameters")
        eq_layout = QFormLayout()

        self.eq_num_bands_spin = QSpinBox()
        self.eq_num_bands_spin.setRange(4, 64)
        eq_layout.addRow("Number of Bands:", self.eq_num_bands_spin)
        
        self.eq_adapt_rate_spin = QDoubleSpinBox()
        self.eq_adapt_rate_spin.setRange(0.01, 1.0)
        self.eq_adapt_rate_spin.setDecimals(3)
        eq_layout.addRow("Adaptation Rate:", self.eq_adapt_rate_spin)
        
        self.eq_smooth_factor_spin = QDoubleSpinBox()
        self.eq_smooth_factor_spin.setRange(0.1, 0.99)
        self.eq_smooth_factor_spin.setDecimals(2)
        eq_layout.addRow("Smoothing Factor:", self.eq_smooth_factor_spin)
        
        eq_group.setLayout(eq_layout)
        
        # Safety Parameters Group
        safety_group = QGroupBox("Safety Limiters")
        safety_layout = QFormLayout()
        
        self.max_crest_factor_spin = QDoubleSpinBox()
        self.max_crest_factor_spin.setRange(2.0, 10.0)
        self.max_crest_factor_spin.setDecimals(1)
        safety_layout.addRow("Max Crest Factor:", self.max_crest_factor_spin)
        
        self.max_rms_volts_spin = QDoubleSpinBox()
        self.max_rms_volts_spin.setRange(0.1, 5.0)
        self.max_rms_volts_spin.setDecimals(2)
        self.max_rms_volts_spin.setSuffix(" V")
        safety_layout.addRow("Max RMS Voltage:", self.max_rms_volts_spin)
        
        safety_group.setLayout(safety_layout)
        
        # Simulation Parameters Group
        sim_group = QGroupBox("Simulation Parameters")
        sim_layout = QFormLayout()
        
        self.simulation_mode_check = QCheckBox("Enable Simulation Mode")
        sim_layout.addRow("", self.simulation_mode_check)
        
        self.sim_plant_gain_spin = QDoubleSpinBox()
        self.sim_plant_gain_spin.setRange(0.1, 20.0)
        self.sim_plant_gain_spin.setSuffix(" g/V")
        sim_layout.addRow("Plant Gain:", self.sim_plant_gain_spin)
        
        self.sim_noise_level_spin = QDoubleSpinBox()
        self.sim_noise_level_spin.setRange(0.001, 0.1)
        self.sim_noise_level_spin.setDecimals(4)
        self.sim_noise_level_spin.setSuffix(" g RMS")
        sim_layout.addRow("Noise Level:", self.sim_noise_level_spin)
        
        sim_group.setLayout(sim_layout)
        
        # Apply button
        self.apply_button = QPushButton("Apply Configuration")
        self.apply_button.clicked.connect(self.apply_config)

        # Add all groups to main layout
        layout.addWidget(system_group)
        layout.addWidget(mode_group)
        layout.addWidget(control_group)
        layout.addWidget(self.sine_group)
        layout.addWidget(eq_group)
        layout.addWidget(safety_group)
        layout.addWidget(sim_group)
        layout.addWidget(self.apply_button)
        layout.addStretch()

        self.scroll_area.setWidget(content_widget)
        outer_layout.addWidget(self.scroll_area)

        self.setLayout(outer_layout)
        self.test_mode_combo.currentIndexChanged.connect(self._update_mode_group_state)

    def load_values(self):
        """Load current values from shared config"""
        self.fs_spin.setValue(self.shared_config.fs)
        self.buf_seconds_spin.setValue(self.shared_config.buf_seconds)
        self.block_seconds_spin.setValue(self.shared_config.block_seconds)
        self.welch_nperseg_spin.setValue(self.shared_config.welch_nperseg)
        self.ao_volt_limit_spin.setValue(self.shared_config.ao_volt_limit)
        
        self.kp_spin.setValue(self.shared_config.kp)
        self.ki_spin.setValue(self.shared_config.ki)
        self.initial_level_spin.setValue(self.shared_config.initial_level_fraction)
        self.max_level_rate_spin.setValue(self.shared_config.max_level_fraction_rate)
        
        self.eq_num_bands_spin.setValue(self.shared_config.eq_num_bands)
        self.eq_adapt_rate_spin.setValue(self.shared_config.eq_adapt_rate)
        self.eq_smooth_factor_spin.setValue(self.shared_config.eq_smooth_factor)
        
        self.max_crest_factor_spin.setValue(self.shared_config.max_crest_factor)
        self.max_rms_volts_spin.setValue(self.shared_config.max_rms_volts)

        self.simulation_mode_check.setChecked(self.shared_config.simulation_mode)
        self.sim_plant_gain_spin.setValue(self.shared_config.sim_plant_gain)
        self.sim_noise_level_spin.setValue(self.shared_config.sim_noise_level)

        mode_index = self.test_mode_combo.findData(self.shared_config.test_mode)
        if mode_index != -1:
            self.test_mode_combo.setCurrentIndex(mode_index)
        else:
            self.test_mode_combo.setCurrentIndex(0)

        self.sine_start_spin.setValue(self.shared_config.sine_start_hz)
        self.sine_end_spin.setValue(self.shared_config.sine_end_hz)
        self.sine_g_level_spin.setValue(self.shared_config.sine_g_level)
        self.sine_g_level_is_rms_check.setChecked(self.shared_config.sine_g_level_is_rms)
        self.sine_octaves_spin.setValue(self.shared_config.sine_octaves_per_min)
        self.sine_repeat_check.setChecked(self.shared_config.sine_repeat)
        self.sine_initial_level_spin.setValue(self.shared_config.sine_initial_level)
        self.sine_max_level_rate_spin.setValue(self.shared_config.sine_max_level_rate)

        self.data_log_check.setChecked(self.shared_config.data_log_enabled)

        self._update_mode_group_state()

    def apply_config(self):
        """Apply form values to shared config"""
        self.shared_config.fs = self.fs_spin.value()
        self.shared_config.buf_seconds = self.buf_seconds_spin.value()
        self.shared_config.block_seconds = self.block_seconds_spin.value()
        self.shared_config.welch_nperseg = int(self.welch_nperseg_spin.value())
        self.shared_config.ao_volt_limit = self.ao_volt_limit_spin.value()
        
        self.shared_config.kp = self.kp_spin.value()
        self.shared_config.ki = self.ki_spin.value()
        self.shared_config.initial_level_fraction = self.initial_level_spin.value()
        self.shared_config.max_level_fraction_rate = self.max_level_rate_spin.value()
        
        self.shared_config.eq_num_bands = int(self.eq_num_bands_spin.value())
        self.shared_config.eq_adapt_rate = self.eq_adapt_rate_spin.value()
        self.shared_config.eq_smooth_factor = self.eq_smooth_factor_spin.value()
        
        self.shared_config.max_crest_factor = self.max_crest_factor_spin.value()
        self.shared_config.max_rms_volts = self.max_rms_volts_spin.value()
        
        self.shared_config.simulation_mode = self.simulation_mode_check.isChecked()
        self.shared_config.sim_plant_gain = self.sim_plant_gain_spin.value()
        self.shared_config.sim_noise_level = self.sim_noise_level_spin.value()

        self.shared_config.test_mode = self.test_mode_combo.currentData()
        self.shared_config.sine_start_hz = self.sine_start_spin.value()
        self.shared_config.sine_end_hz = self.sine_end_spin.value()
        self.shared_config.sine_g_level = self.sine_g_level_spin.value()
        self.shared_config.sine_g_level_is_rms = self.sine_g_level_is_rms_check.isChecked()
        self.shared_config.sine_octaves_per_min = self.sine_octaves_spin.value()
        self.shared_config.sine_repeat = self.sine_repeat_check.isChecked()
        self.shared_config.sine_initial_level = self.sine_initial_level_spin.value()
        self.shared_config.sine_max_level_rate = self.sine_max_level_rate_spin.value()

        self.shared_config.data_log_enabled = self.data_log_check.isChecked()

        # Emit signal that config was updated
        self.shared_config.config_updated.emit()

        # Visual feedback
        self.apply_button.setText("Applied!")
        QTimer.singleShot(1000, lambda: self.apply_button.setText("Apply Configuration"))

    def _update_mode_group_state(self):
        """Enable sine sweep controls only when relevant."""
        is_sine = self.test_mode_combo.currentData() == "sine_sweep"
        self.sine_group.setEnabled(is_sine)


class ControllerTab(QWidget):
    """Controller tab with start/stop controls and PyQtGraph plots"""
    
    def __init__(self, shared_config):
        super().__init__()
        self.shared_config = shared_config
        self.init_ui()
        
        # Data storage
        self.max_points = 200
        self.time_data = deque(maxlen=self.max_points)
        self.rms_data = deque(maxlen=self.max_points)
        self.savg_meas_data = deque(maxlen=self.max_points)
        self.savg_target_data = deque(maxlen=self.max_points)
        self.level_data = deque(maxlen=self.max_points)
        self.sat_data = deque(maxlen=self.max_points)
        self.plant_gain_data = deque(maxlen=self.max_points)
        
        self.update_counter = 0
        self.is_running = False
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Controller")
        self.stop_button = QPushButton("Stop Controller")
        self.stop_button.setEnabled(False)
        
        # Status label
        self.status_label = QLabel("Status: Stopped")
        font = QFont()
        font.setBold(True)
        self.status_label.setFont(font)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.status_label)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Create plot widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        
        # PSD plot with linear axes
        self.psd_plot = self.plot_widget.addPlot(title="Power Spectral Density")
        self.psd_plot.setLogMode(x=False, y=True)
        self.psd_plot.setLabel('left', 'PSD [g²/Hz]')
        self.psd_plot.setLabel('bottom', 'Frequency [Hz]')
        self.psd_plot.showGrid(x=True, y=True)
        # Lock PSD x-axis to 20–2000 Hz and prevent x auto-ranging
        self.psd_plot.setXRange(20.0, 2000.0, padding=0)
        vb_psd = self.psd_plot.getViewBox()
        vb_psd.setLimits(xMin=20.0, xMax=2000.0)
        vb_psd.setAutoVisible(x=False, y=True)
        self.psd_plot.enableAutoRange(x=False, y=True)
        
        # PSD legend before curves so items auto-register
        self.psd_legend = self.psd_plot.addLegend(offset=(10, 10))
        try:
            self.psd_legend.setBrush(pg.mkBrush(30, 30, 30, 160))
        except Exception:
            pass
        
        # PSD curves
        self.psd_measured_curve = self.psd_plot.plot(pen='b', name='Measured PSD')
        self.psd_target_curve = self.psd_plot.plot(pen='r', style='--', name='Target PSD')
        self.psd_averaged_curve = self.psd_plot.plot(pen='g', name='Averaged PSD')

        # Next row - Control metrics
        self.plot_widget.nextRow()
        
        # Control metrics plot
        self.metrics_plot = self.plot_widget.addPlot(title="Control Metrics")
        self.metrics_plot.setLabel('left', 'Value')
        self.metrics_plot.setLabel('bottom', 'Update Index')
        self.metrics_plot.showGrid(x=True, y=True)
        self.metrics_legend = self.metrics_plot.addLegend(offset=(10, 10))
        try:
            self.metrics_legend.setBrush(pg.mkBrush(30, 30, 30, 160))
        except Exception:
            pass

        # Control metric curves
        self.rms_curve = self.metrics_plot.plot(pen='g', name='a_rms [g]')
        self.level_curve = self.metrics_plot.plot(pen='b', name='Level Fraction')
        self.sat_curve = self.metrics_plot.plot(pen='r', name='Saturation %')
        self.plant_gain_curve = self.metrics_plot.plot(pen='m', name='Plant Gain [g/V]')
        
        # Next row - Equalizer gains
        self.plot_widget.nextRow()
        
        # Equalizer plot
        self.eq_plot = self.plot_widget.addPlot(title="Equalizer Gains")
        self.eq_plot.setLogMode(x=False, y=False)
        self.eq_plot.setLabel('left', 'Gain')
        self.eq_plot.setLabel('bottom', 'Frequency [Hz]')
        self.eq_plot.showGrid(x=True, y=True)
        self.eq_plot.setYRange(0.1, 10.0)

        # HARD LOCK the visible x-range to 20–2000 Hz
        self.eq_xmin, self.eq_xmax = 20.0, 2000.0
        self.eq_plot.setXRange(self.eq_xmin, self.eq_xmax, padding=0)
        vb = self.eq_plot.getViewBox()
        vb.setLimits(xMin=self.eq_xmin, xMax=self.eq_xmax)   # prevent pans/zooms beyond limits
        vb.setAutoVisible(x=False, y=False)                  # disable auto-rescale

        # Legend and reference line for equalizer plot
        self.eq_plot.addLegend(offset=(10, 10))
        unity_pen = pg.mkPen(color='k', style=Qt.DashLine, width=1)
        self.eq_unity_curve = self.eq_plot.plot(pen=unity_pen, name='Unity Gain')
        self.eq_unity_curve.setData([])

        # Equalizer bar graph (will be created with data)
        self.eq_bargraph = None
        self.eq_bar_in_legend = False
        
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
    
    def start_controller(self):
        """Update UI state when controller starts"""
        self.is_running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Running")
    
    def stop_controller(self):
        """Update UI state when controller stops"""
        self.is_running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped")
    
    def update_plots(self, psd_data=None, metrics_data=None, eq_data=None):
        """Update all plots with new data"""
        
        # Update PSD plot
        if psd_data:
            if len(psd_data) == 4:  # New format with averaged PSD
                f, psd_measured, psd_target, psd_averaged = psd_data
            else:  # Legacy format without averaged PSD
                f, psd_measured, psd_target = psd_data
                psd_averaged = None

            # Sanitize frequency axis for log scale: finite and > 0
            f = np.asarray(f, dtype=float)
            mask_base = np.isfinite(f) & (f > 0)

            # Clip to display band [20, 2000] Hz
            f = f[mask_base]
            if f.size == 0:
                return
            band_mask = (f >= 20.0) & (f <= 2000.0)

            # Apply masks to arrays with matching length
            def _apply_mask(arr, mask):
                arr = np.asarray(arr)
                if arr.shape == f.shape:
                    return arr[mask]
                return arr

            f_plot = f[band_mask]
            psd_measured_plot = _apply_mask(psd_measured[mask_base], band_mask)
            psd_target_plot   = _apply_mask(psd_target[mask_base] if psd_target is not None else None, band_mask) if psd_target is not None else None
            psd_avg_plot      = _apply_mask(psd_averaged[mask_base], band_mask) if psd_averaged is not None else None

            # For log-y plots, only plot positive finite PSD values
            valid_meas = np.isfinite(psd_measured_plot) & (psd_measured_plot > 0)
            self.psd_measured_curve.setData(f_plot[valid_meas], psd_measured_plot[valid_meas])
            if psd_avg_plot is not None:
                valid_avg = np.isfinite(psd_avg_plot) & (psd_avg_plot > 0)
                self.psd_averaged_curve.setData(f_plot[valid_avg], psd_avg_plot[valid_avg])
            if psd_target_plot is not None:
                valid_target = np.isfinite(psd_target_plot) & (psd_target_plot > 0)
                if np.any(valid_target):
                    self.psd_target_curve.setData(f_plot[valid_target], psd_target_plot[valid_target])

            # Re-enforce fixed x-range after updates
            self.psd_plot.setXRange(20.0, 2000.0, padding=0)
        
        # Update metrics plot
        if metrics_data:
            if len(metrics_data) < 6:
                return
            self.update_counter += 1
            self.time_data.append(self.update_counter)

            a_rms, s_avg_meas, s_avg_target, level_fraction, sat_frac, plant_gain = metrics_data[:6]
            freq_hz = metrics_data[6] if len(metrics_data) >= 7 else None
            peak_g = metrics_data[7] if len(metrics_data) >= 8 else None
            target_peak_g = metrics_data[8] if len(metrics_data) >= 9 else None

            self.rms_data.append(a_rms)
            self.level_data.append(level_fraction)
            self.sat_data.append(sat_frac * 100)  # Convert to percentage
            self.plant_gain_data.append(plant_gain)

            # Update curves
            self.rms_curve.setData(list(self.time_data), list(self.rms_data))
            self.level_curve.setData(list(self.time_data), list(self.level_data))
            self.sat_curve.setData(list(self.time_data), list(self.sat_data))
            self.plant_gain_curve.setData(list(self.time_data), list(self.plant_gain_data))

            if (
                freq_hz is not None
                and peak_g is not None
                and self.shared_config.test_mode == "sine_sweep"
            ):
                if target_peak_g is not None:
                    self.status_label.setText(
                        f"Status: Running | f={freq_hz:.1f} Hz | gpk={peak_g:.3f} (target {target_peak_g:.3f})"
                    )
                else:
                    self.status_label.setText(
                        f"Status: Running | f={freq_hz:.1f} Hz | gpk={peak_g:.3f}"
                    )
        
        # Update equalizer plot
        if eq_data is not None:
            try:
                freq_centers, gains = eq_data
            except (TypeError, ValueError):
                freq_centers, gains = ([], [])

            if freq_centers is None or len(freq_centers) == 0:
                self._clear_eq_plot()
                return

            # Sanitize: keep only finite, >0 freqs; clip to [20, 2000] Hz
            freq_centers = np.asarray(freq_centers, dtype=float)
            gains = np.asarray(gains, dtype=float)

            mask = np.isfinite(freq_centers) & (freq_centers > 0) & np.isfinite(gains)
            freq_centers = freq_centers[mask]
            gains = gains[mask]

            # Convert rad/s -> Hz if values are clearly too large
            if freq_centers.size and np.nanmedian(freq_centers) > 1e5:
                freq_centers = freq_centers / (2*np.pi)

            # Clip to display band
            freq_centers = np.clip(freq_centers, self.eq_xmin, self.eq_xmax)

            if freq_centers.size == 0:
                self._clear_eq_plot()
                return

            # Bar widths proportional to center frequency
            num_bands = len(freq_centers)
            if num_bands > 20:
                width_factor = 0.10
            elif num_bands > 12:
                width_factor = 0.20
            else:
                width_factor = 0.30
            bar_widths = freq_centers * width_factor

            if self.eq_bargraph is None:
                self.eq_bargraph = pg.BarGraphItem(x=freq_centers, height=gains,
                                                   width=bar_widths, brush='b', name='EQ Band Gain')
                self.eq_plot.addItem(self.eq_bargraph)
            else:
                self.eq_bargraph.setOpts(x=freq_centers, height=gains, width=bar_widths)

            if (self.eq_plot.legend is not None) and (not self.eq_bar_in_legend):
                try:
                    self.eq_plot.legend.addItem(self.eq_bargraph, 'EQ Band Gain')
                    self.eq_bar_in_legend = True
                except Exception:
                    pass

            # Keep the x-range locked
            self.eq_plot.setXRange(self.eq_xmin, self.eq_xmax, padding=0)

            # Update unity gain reference line
            self.eq_unity_curve.setData([self.eq_xmin, self.eq_xmax], [1.0, 1.0])
        else:
            self._clear_eq_plot()

    def _clear_eq_plot(self):
        """Remove equalizer bars when not in use."""
        if self.eq_bargraph is not None:
            try:
                self.eq_plot.removeItem(self.eq_bargraph)
            except Exception:
                pass
            self.eq_bargraph = None
            self.eq_bar_in_legend = False
        self.eq_unity_curve.setData([])


class RealTimeDataTab(QWidget):
    """Real-time data tab with mode-aware visualization."""

    def __init__(self, shared_config):
        super().__init__()
        self.shared_config = shared_config
        self.fs = float(self.shared_config.fs)
        self.channel_labels = list(self.shared_config.input_channel_labels)
        self.num_channels = max(1, self.shared_config.num_input_channels)
        self.welch_nperseg = int(getattr(self.shared_config, 'welch_nperseg', config.WELCH_NPERSEG))
        self.psd_update_stride = max(1, int(getattr(self.shared_config, 'realtime_psd_update_stride', 5)))

        # Random-vibration data stores
        self.history_length = 300
        self.update_indices = deque(maxlen=self.history_length)
        self.rms_history = [deque(maxlen=self.history_length) for _ in range(self.num_channels)]
        self.update_count = 0
        self.psd_target = None
        self.psd_latest = [None] * self.num_channels
        self._psd_dirty = False

        # Sine-sweep data stores
        self.sine_freq_history = deque(maxlen=2000)
        self.sine_peak_history = deque(maxlen=2000)
        self.sine_target_history = deque(maxlen=2000)
        self.latest_block = None
        self.latest_time_axis = None

        self._build_ui()
        self.shared_config.config_updated.connect(self._handle_config_update)

        self.is_streaming = False
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_streaming_plot)
        self.update_timer.setInterval(33)
        self._update_mode_view()

    def _build_ui(self):
        layout = QVBoxLayout()

        self.options_container = QWidget()
        options_layout = QHBoxLayout(self.options_container)
        self.autoscale_check = QCheckBox("Enable Autoscaling")
        self.autoscale_check.setChecked(True)
        options_layout.addWidget(QLabel("Display Options:"))
        options_layout.addWidget(self.autoscale_check)
        options_layout.addStretch()
        layout.addWidget(self.options_container)

        self.stack = QStackedWidget()
        self.stack_random = self._create_random_view()
        self.stack_sine = self._create_sine_view()
        self.stack.addWidget(self.stack_random)
        self.stack.addWidget(self.stack_sine)
        layout.addWidget(self.stack)

        self.setLayout(layout)

    def _create_random_view(self) -> QWidget:
        widget = pg.GraphicsLayoutWidget()
        self.rms_plot = widget.addPlot(title="Channel g-RMS History")
        self.rms_plot.setLabel('left', 'g-RMS [g]')
        self.rms_plot.setLabel('bottom', 'Update Index')
        self.rms_plot.showGrid(x=True, y=True)
        self.rms_plot.addLegend()
        self.rms_curves = []
        for idx, label in enumerate(self.channel_labels):
            color = pg.intColor(idx, hues=max(1, self.num_channels))
            pen = pg.mkPen(color=color, width=2)
            curve = self.rms_plot.plot(pen=pen, name=label)
            self.rms_curves.append(curve)

        widget.nextRow()
        self.psd_plot = widget.addPlot(title="Averaged PSD from Controller")
        self.psd_plot.setLogMode(x=False, y=True)
        self.psd_plot.setLabel('left', 'PSD [g²/Hz]')
        self.psd_plot.setLabel('bottom', 'Frequency [Hz]')
        self.psd_plot.showGrid(x=True, y=True)
        vb_psd = self.psd_plot.getViewBox()
        vb_psd.setAutoVisible(x=False, y=True)
        self.psd_plot.enableAutoRange(x=False, y=True)
        self.psd_plot.setLimits(xMin=20.0, xMax=2000.0)
        self.psd_plot.setRange(xRange=(20.0, 2000.0), padding=0)
        self.psd_plot.addLegend()
        self.psd_curves = []
        for idx, label in enumerate(self.channel_labels):
            color = pg.intColor(idx, hues=max(1, self.num_channels))
            pen = pg.mkPen(color=color, width=1.8)
            curve = self.psd_plot.plot(pen=pen, name=f"{label} PSD")
            self.psd_curves.append(curve)
        self.psd_target_curve = self.psd_plot.plot(pen=pg.mkPen('r', style=Qt.DashLine, width=2), name='Target PSD')
        return widget

    def _create_sine_view(self) -> QWidget:
        widget = pg.GraphicsLayoutWidget()
        self.freq_plot = widget.addPlot(title="Control Accel g-peak vs Frequency")
        self.freq_plot.setLabel('bottom', 'Frequency [Hz]')
        self.freq_plot.setLabel('left', 'g-peak [g]')
        self.freq_plot.setLogMode(x=False, y=False)
        self.freq_plot.showGrid(x=True, y=True)
        self.freq_scatter = pg.ScatterPlotItem(size=7, brush=pg.mkBrush(50, 200, 255, 180))
        self.freq_plot.addItem(self.freq_scatter)
        self.freq_plot.setRange(xRange=(self.shared_config.sine_start_hz, self.shared_config.sine_end_hz))
        self.freq_plot.setLimits(xMin=self.shared_config.sine_start_hz * 0.5,
                                 xMax=self.shared_config.sine_end_hz * 2.0)
        self.freq_target_curve = self.freq_plot.plot(pen=pg.mkPen('r', style=Qt.DashLine, width=2))

        widget.nextRow()
        self.time_plot = widget.addPlot(title="Control Acceleration (latest block)")
        self.time_plot.setLabel('bottom', 'Time [s]')
        self.time_plot.setLabel('left', 'Accel [g]')
        self.time_plot.showGrid(x=True, y=True)
        self.time_curve = self.time_plot.plot(pen=pg.mkPen('y', width=1.5))
        return widget

    def _is_sine_mode(self) -> bool:
        return getattr(self.shared_config, 'test_mode', '') == "sine_sweep"

    def _update_mode_view(self) -> None:
        if self._is_sine_mode():
            self.stack.setCurrentWidget(self.stack_sine)
            self.options_container.hide()
        else:
            self.stack.setCurrentWidget(self.stack_random)
            self.options_container.show()

    def _handle_config_update(self):
        self._update_mode_view()

    def start_streaming(self):
        """Start real-time data streaming"""
        self.is_streaming = True
        self._update_mode_view()
        if not self._is_sine_mode():
            self.update_indices.clear()
            for history in self.rms_history:
                history.clear()
            self.update_count = 0
            self.psd_latest = [None] * self.num_channels
            self._psd_dirty = False
            self.psd_target = None
        else:
            self.sine_freq_history.clear()
            self.sine_peak_history.clear()
            self.sine_target_history.clear()
            self.latest_block = None
            self.latest_time_axis = None

        # Start update timer (must be called from main thread)
        if not self.update_timer.isActive():
            self.update_timer.start()
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.is_streaming = False
        
        # Stop update timer
        self.update_timer.stop()
    
    def add_data_point(self, acceleration_value):
        """Add new block of data to the streaming plot"""
        if not self.is_streaming:
            return

        block = np.asarray(acceleration_value, dtype=float)
        if block.ndim == 1:
            block = block[np.newaxis, :]
        if block.shape[0] != self.num_channels:
            print(f"Debug: Received block with {block.shape[0]} channels, expected {self.num_channels}")
            return

        n_samples = block.shape[1]
        if n_samples == 0:
            return

        if self._is_sine_mode():
            control_block = block[0]
            self.latest_block = control_block.astype(float, copy=True)
            self.latest_time_axis = np.arange(n_samples, dtype=float) / self.fs
        else:
            rms_values = np.sqrt(np.mean(block ** 2, axis=1))
            self.update_count += 1
            self.update_indices.append(self.update_count)
            for ch_idx, value in enumerate(rms_values):
                self.rms_history[ch_idx].append(float(value))

            compute_psd = (self.update_count == 1) or (self.update_count % self.psd_update_stride == 0)
            if compute_psd:
                for ch_idx in range(1, self.num_channels):
                    try:
                        f_vals, Pxx = welch(block[ch_idx], fs=self.fs, nperseg=self.welch_nperseg)
                        self.psd_latest[ch_idx] = (f_vals, Pxx)
                        self._psd_dirty = True
                    except Exception:
                        continue

    def update_streaming_plot(self):
        """Update the streaming plot display"""
        if not self.is_streaming:
            return

        if self._is_sine_mode():
            if self.sine_freq_history:
                self.freq_scatter.setData(list(self.sine_freq_history), list(self.sine_peak_history))
                if self.sine_target_history:
                    self.freq_target_curve.setData(
                        list(self.sine_freq_history), list(self.sine_target_history)
                    )
                else:
                    self.freq_target_curve.clear()
            else:
                self.freq_scatter.setData([], [])
                self.freq_target_curve.clear()

            if self.latest_block is not None and self.latest_time_axis is not None:
                self.time_curve.setData(self.latest_time_axis, self.latest_block)
            else:
                self.time_curve.clear()
            return

        if not self.update_indices:
            return

        x_vals = list(self.update_indices)
        for curve, channel_history in zip(self.rms_curves, self.rms_history):
            curve.setData(x_vals, list(channel_history))

        if self._psd_dirty and any(entry is not None for entry in self.psd_latest):
            try:
                for ch_idx, curve in enumerate(self.psd_curves):
                    psd_entry = self.psd_latest[ch_idx]
                    if psd_entry is None:
                        curve.clear()
                        continue
                    f_vals, Pxx = psd_entry
                    f_vals = np.asarray(f_vals, dtype=float)
                    Pxx = np.asarray(Pxx, dtype=float)
                    mask = np.isfinite(f_vals) & (f_vals > 0)
                    f_vals = f_vals[mask]
                    Pxx = Pxx[mask]
                    if f_vals.size == 0:
                        curve.clear()
                        continue
                    band_mask = (f_vals >= 20.0) & (f_vals <= 2000.0)
                    f_band = f_vals[band_mask]
                    p_band = Pxx[band_mask]
                    if f_band.size == 0:
                        curve.clear()
                        continue
                    valid = np.isfinite(p_band) & (p_band > 0)
                    if np.any(valid):
                        curve.setData(f_band[valid], p_band[valid])
                    else:
                        curve.clear()

                if self.psd_target is not None:
                    f_target, S_target = self.psd_target
                    f_target = np.asarray(f_target, dtype=float)
                    S_target = np.asarray(S_target, dtype=float)
                    mask = np.isfinite(f_target) & (f_target > 0) & np.isfinite(S_target) & (S_target > 0)
                    if np.any(mask):
                        self.psd_target_curve.setData(f_target[mask], S_target[mask])
                    else:
                        self.psd_target_curve.clear()

                self.psd_plot.setRange(xRange=(20.0, 2000.0), padding=0)

            except Exception:
                pass
            finally:
                self._psd_dirty = False

        if self.autoscale_check.isChecked():
            populated = [np.asarray(history, dtype=float) for history in self.rms_history if history]
            if populated:
                y_min = min(arr.min() for arr in populated)
                y_max = max(arr.max() for arr in populated)
            else:
                y_min, y_max = -1.0, 1.0
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
            self.rms_plot.setYRange(y_min - margin, y_max + margin)
            self.rms_plot.setXRange(max(0, self.update_indices[0]), self.update_indices[-1])
    
    def update_psd_data(self, psd_data):
        """Update PSD data from controller"""
        if self._is_sine_mode():
            return
        if len(psd_data) == 4:  # New format with averaged PSD
            f, psd_measured, psd_target, psd_averaged = psd_data
            self.psd_latest[0] = (f, psd_averaged)
            self.psd_target = (f, psd_target)
        else:  # Legacy format without averaged PSD
            f, psd_measured, psd_target = psd_data
            self.psd_latest[0] = (f, psd_measured)
            self.psd_target = (f, psd_target)
        self._psd_dirty = True

    def update_sine_metrics(self, freq_hz: float, peak_g: float, target_peak_g: float):
        if not self._is_sine_mode() or not self.is_streaming:
            return
        self.sine_freq_history.append(float(freq_hz))
        self.sine_peak_history.append(float(peak_g))
        self.sine_target_history.append(float(target_peak_g))


class ControllerWorker(QObject):
    """Worker thread for running the control loop"""
    
    # Signals for thread-safe communication
    psd_data_ready = Signal(object)  # (f, psd_measured, psd_target)
    metrics_data_ready = Signal(object)  # (a_rms, s_avg_meas, s_avg_target, level_fraction, sat_frac, plant_gain)
    eq_data_ready = Signal(object)  # (freq_centers, gains)
    realtime_data_ready = Signal(object)  # streaming block of acceleration data
    status_message = Signal(str)
    
    def __init__(self, shared_config):
        super().__init__()
        self.shared_config = shared_config
        self.should_stop = Event()
        self.is_running = False
        
        # PSD averaging
        self.psd_averaged = None
        self.psd_alpha = 0.1  # Exponential averaging factor
    
    def run_controller(self):
        """Main controller loop running in separate thread"""
        ao_task = None
        ai_task = None
        ai_reader = None
        ao_writer = None
        volts_to_g_scale = None
        data_logger = None
        try:
            self.is_running = True
            self.start_time = time.time()
            self.status_message.emit("Initializing controller...")
            
            # Initialize system based on current config
            fs = self.shared_config.fs
            block_samples = int(fs * self.shared_config.block_seconds)
            
            # Create controller
            controller = RandomVibrationController(
                target_psd_points=self.shared_config.target_psd_points,
                fs=fs,
                Kp=self.shared_config.kp,
                Ki=self.shared_config.ki,
                max_level_fraction_rate=self.shared_config.max_level_fraction_rate,
                welch_nperseg=self.shared_config.welch_nperseg
            )
            
            # Create equalizer
            f1 = self.shared_config.target_psd_points[0][0]
            f2 = self.shared_config.target_psd_points[-1][0]
            
            equalizer = MultiBandEqualizer(
                f1=f1, f2=f2, 
                num_bands=self.shared_config.eq_num_bands, 
                fs=fs,
                gain_limits=(0.1, 10.0), 
                adapt_rate=self.shared_config.eq_adapt_rate,
                smooth_factor=self.shared_config.eq_smooth_factor
            )
            
            # Create bandpass filter
            bp = create_bandpass_filter(f1, f2, fs)

            # Initialize DAQ or simulation backend
            simulation_mode = bool(self.shared_config.simulation_mode)
            mode_label = "SIMULATION"

            if not simulation_mode and not NIDAQMX_AVAILABLE:
                self.status_message.emit("nidaqmx package not available; using simulation mode")
                print("nidaqmx not available - falling back to simulation mode")
                simulation_mode = True
                self.shared_config.simulation_mode = True

            if simulation_mode:
                ao_writer, ai_reader, ai_task, ao_task = create_simulation_system(
                    fs, self.shared_config.sim_plant_gain, config.SIM_RESONANCES,
                    self.shared_config.sim_noise_level, config.SIM_DELAY_SAMPLES,
                    config.SIM_NONLINEARITY, num_input_channels=self.shared_config.num_input_channels
                )
                plant_gain_g_per_V = self.shared_config.sim_plant_gain * 0.8
                mode_label = "SIMULATION"
            else:
                try:
                    device_ai = config.DEVICE_AI
                    device_ao = config.DEVICE_AO
                    accel_mV_per_g = config.ACCEL_MV_PER_G
                    ao_volt_limit = self.shared_config.ao_volt_limit
                    buf_samples = int(self.shared_config.fs * self.shared_config.buf_seconds)
                    excitation_current = max(0.0, float(self.shared_config.accel_excitation_amps))
                    excitation_source = ExcitationSource.INTERNAL if excitation_current > 0 else ExcitationSource.NONE

                    ai_task = nidaqmx.Task()
                    ao_task = nidaqmx.Task()

                    try:
                        ai_ch = ai_task.ai_channels.add_ai_accel_chan(
                            physical_channel=device_ai,
                            name_to_assign_to_channel="accel",
                            terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                            min_val=-5.0,
                            max_val=5.0,
                            units=AccelUnits.G,
                            sensitivity=accel_mV_per_g,
                            sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
                            current_excit_source=excitation_source,
                            current_excit_val=excitation_current,
                        )
                        volts_to_g_scale = None
                        if excitation_current > 0:
                            print(f"Hardware input configured for IEPE accelerometer mode ({excitation_current*1000:.2f} mA).")
                        else:
                            print("Hardware input configured for accelerometer without IEPE excitation.")
                        try:
                            ai_ch.ai_coupling = Coupling.AC
                        except Exception:
                            pass
                    except Exception as accel_err:
                        if DaqError is not None and not isinstance(accel_err, DaqError):
                            raise
                        print(f"Accelerometer channel init failed: {accel_err}. Falling back to voltage mode.")
                        self.status_message.emit("Accel channel unavailable; using voltage mode")
                        try:
                            ai_task.close()
                        except Exception:
                            pass
                        ai_task = nidaqmx.Task()
                        voltage_range = max(2.0, ao_volt_limit * 2.0)
                        ai_task.ai_channels.add_ai_voltage_chan(
                            physical_channel=device_ai,
                            name_to_assign_to_channel="accel_volts",
                            terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                            min_val=-voltage_range,
                            max_val=voltage_range,
                            units=VoltageUnits.VOLTS,
                        )
                        volts_to_g_scale = 1.0 / max(accel_mV_per_g / 1000.0, 1e-9)
                        print(f"Scaling voltage input by {volts_to_g_scale:.3f} to convert to g.")

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
                        except Exception as sync_err:
                            if DaqError is not None and not isinstance(sync_err, DaqError):
                                raise
                            print(f"AO sync configuration failed: {sync_err}. Using onboard clock.")
                            self.status_message.emit("AO using onboard clock (not synchronized)")
                            try:
                                ao_task.triggers.start_trigger.disable()
                            except Exception:
                                pass
                    else:
                        print("AO synchronization disabled by configuration; using onboard clock.")

                    ai_reader = AnalogSingleChannelReader(ai_task.in_stream)
                    ao_writer = AnalogSingleChannelWriter(ao_task.out_stream, auto_start=False)

                    ao_writer.write_many_sample(np.zeros(buf_samples))
                    ai_task.start()
                    ao_task.start()

                    plant_gain_g_per_V = 1.0
                    mode_label = "HARDWARE"
                except Exception as hw_err:
                    print(f"Hardware initialization failed: {hw_err}")
                    self.status_message.emit(f"Hardware init failed: {hw_err}")
                    if ao_task is not None:
                        try:
                            ao_task.stop()
                        except Exception:
                            pass
                        try:
                            ao_task.close()
                        except Exception:
                            pass
                    if ai_task is not None:
                        try:
                            ai_task.stop()
                        except Exception:
                            pass
                        try:
                            ai_task.close()
                        except Exception:
                            pass
                    ao_task = None
                    ai_task = None
                    volts_to_g_scale = None
                    simulation_mode = True
                    self.shared_config.simulation_mode = True
                    ao_writer, ai_reader, ai_task, ao_task = create_simulation_system(
                        fs, self.shared_config.sim_plant_gain, config.SIM_RESONANCES,
                        self.shared_config.sim_noise_level, config.SIM_DELAY_SAMPLES,
                        config.SIM_NONLINEARITY, num_input_channels=self.shared_config.num_input_channels
                    )
                    plant_gain_g_per_V = self.shared_config.sim_plant_gain * 0.8
                    mode_label = "SIMULATION"

            if ao_writer is None or ai_reader is None:
                raise RuntimeError("Failed to initialize data acquisition path")

            self.status_message.emit(f"Controller running ({mode_label.lower()})...")
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
                except Exception as log_err:
                    print(f"Data logger init failed: {log_err}")
                    data_logger = None

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
            io_timeout = WAIT_INFINITELY if not simulation_mode else max(1.0, block_duration * 2.0)
            loop_sleep = block_duration if simulation_mode else 0.0
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
                target_rms_block = controller.a_rms_target * level_fraction
                volts_block = (target_rms_block / max(plant_gain_g_per_V, 1e-6)) * drive
                
                # Apply minimum drive
                min_drive_rms = 0.05
                current_rms = np.std(volts_block)
                if current_rms < min_drive_rms:
                    volts_block = volts_block * (min_drive_rms / max(current_rms, 1e-12))
                
                # Apply safety limiters
                volts_block, limiter_stats = apply_safety_limiters(
                    volts_block, self.shared_config.max_crest_factor, 
                    self.shared_config.max_rms_volts, 0.8, 0.9)
                
                # Clip to limits
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

                if volts_to_g_scale is not None:
                    if data.ndim == 2:
                        data[0, :] = data[0, :] * volts_to_g_scale
                        accel_data = data[0]
                    else:
                        data *= volts_to_g_scale
                        accel_data = data
                else:
                    accel_data = data[0] if data.ndim == 2 else data
                
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
                
                self.metrics_data_ready.emit((a_rms_meas, S_avg_meas, S_avg_target, 
                                           level_fraction, sat_frac, plant_gain_g_per_V))
                
                eq_info = equalizer.get_band_info()
                self.eq_data_ready.emit((eq_info['centers'], eq_info['gains']))
                
                # Emit current block for real-time visualization
                self.realtime_data_ready.emit(np.array(data, copy=True))

                if data_logger:
                    data_logger.log_random_block(
                        volts_block,
                        np.array(data, copy=True),
                        level_fraction,
                        sat_frac,
                    )

                # Debug logging for data emission
                if loop_count % 50 == 0:  # Every ~5 seconds
                    print(f"Debug: Emitting data at {current_time:.1f}s, RMS={np.std(accel_data):.4f}")
                
                # Pace the loop based on execution environment
                if simulation_mode:
                    if loop_sleep > 0:
                        time.sleep(loop_sleep)
                else:
                    remaining = block_duration - (time.perf_counter() - loop_start)
                    if remaining > 0:
                        time.sleep(remaining)
                
        except Exception as e:
            print(f"Debug: Exception in control loop at {time.time() - self.start_time:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            self.status_message.emit(f"Error: {e}")
        finally:
            if data_logger:
                try:
                    data_logger.close()
                except Exception:
                    pass
            for task in (ao_task, ai_task):
                if task is None:
                    continue
                try:
                    task.stop()
                except Exception:
                    pass
                try:
                    task.close()
                except Exception:
                    pass
            self.is_running = False
            self.status_message.emit("Controller stopped")

    def _run_sine_sweep_loop(self, ao_writer, ai_reader, simulation_mode, volts_to_g_scale,
                              plant_gain_initial, fs, block_samples, data_logger=None):
        """Execute a sine sweep control loop using current configuration."""
        level_fraction = float(np.clip(self.shared_config.sine_initial_level, 0.0, 1.0))
        start_hz = max(0.01, float(self.shared_config.sine_start_hz))
        end_hz = max(0.01, float(self.shared_config.sine_end_hz))
        max_level_rate = max(0.0, float(self.shared_config.sine_max_level_rate))

        block_duration = block_samples / max(fs, 1e-9)
        if not simulation_mode and WAIT_INFINITELY is not None:
            io_timeout = WAIT_INFINITELY
        else:
            io_timeout = max(1.0, block_duration * 2.0)
        loop_sleep = block_duration if simulation_mode else 0.0
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
        dwell_seconds = max(self.shared_config.sine_step_dwell, block_duration)
        lookup_drive = build_drive_lookup(
            self.shared_config.sine_drive_table,
            self.shared_config.sine_default_vpk,
        )

        self.eq_data_ready.emit(None)
        self.psd_averaged = None

        while not self.should_stop.is_set():
            freq = stepper.current_frequency()
            oscillator.set_frequency(freq)
            dwell_remaining = dwell_seconds

            drive_vpk = lookup_drive(freq) * self.shared_config.sine_drive_scale
            drive_vpk = max(0.0, drive_vpk)

            print(f"Sine sweep step: {freq:.1f} Hz, command ≈ {drive_vpk:.3f} Vpk")

            while dwell_remaining > 1e-9 and not self.should_stop.is_set():
                loop_start = time.perf_counter()
                sin_block, cos_block = oscillator.generate(block_samples)

                target_vpk = drive_vpk * max(level_fraction, 0.0)
                command_block = target_vpk * sin_block

                command_block, limiter_stats = apply_safety_limiters(
                    command_block,
                    self.shared_config.max_crest_factor,
                    self.shared_config.max_rms_volts,
                    0.8,
                    0.9,
                )
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

                if volts_to_g_scale is not None:
                    if data.ndim == 2:
                        data[0, :] *= volts_to_g_scale
                        accel_data = data[0]
                    else:
                        data *= volts_to_g_scale
                        accel_data = data
                else:
                    accel_data = data[0] if data.ndim == 2 else data

                scale = 2.0 / float(len(command_block))
                cmd_peak = scale * float(np.hypot(np.dot(command_block, sin_block), np.dot(command_block, cos_block)))
                meas_peak = scale * float(np.hypot(np.dot(accel_data, sin_block), np.dot(accel_data, cos_block)))
                a_rms_meas = meas_peak / np.sqrt(2.0)

                if cmd_peak > 1e-9:
                    new_gain = meas_peak / cmd_peak
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

                nperseg = min(len(accel_data), max(256, int(self.shared_config.welch_nperseg)))
                if nperseg > len(accel_data):
                    nperseg = len(accel_data)
                f, Pxx = welch(accel_data, fs=fs, nperseg=nperseg)

                if self.psd_averaged is None or len(self.psd_averaged) != len(Pxx):
                    self.psd_averaged = Pxx.copy()
                else:
                    self.psd_averaged = (
                        self.psd_alpha * Pxx + (1 - self.psd_alpha) * self.psd_averaged
                    )

                target_rms_block = target_vpk / np.sqrt(2.0)
                target_peak = target_rms_block * np.sqrt(2.0)
                self.psd_data_ready.emit((f, Pxx, None, self.psd_averaged))
                self.metrics_data_ready.emit(
                    (
                        a_rms_meas,
                        a_rms_meas,
                        target_rms_block,
                        level_fraction,
                        sat_frac,
                        plant_gain,
                        freq,
                        meas_peak,
                        target_peak,
                    )
                )
                self.realtime_data_ready.emit(np.array(data, copy=True))

                if data_logger:
                    data_logger.log_sine_block(
                        command_block,
                        np.array(data, copy=True),
                        level_fraction,
                        sat_frac,
                        freq,
                        meas_peak,
                        target_peak,
                    )

                dwell_remaining -= block_duration

                if getattr(config, 'CONSOLE_UPDATE_INTERVAL', 5.0) > 0:
                    pass  # optional additional logging removed for simplicity

                if simulation_mode:
                    if loop_sleep > 0:
                        time.sleep(loop_sleep)
                else:
                    remaining = block_duration - (time.perf_counter() - loop_start)
                    if remaining > 0:
                        time.sleep(remaining)

            if not stepper.advance():
                if not self.shared_config.sine_repeat:
                    self.status_message.emit("Sine sweep complete")
                    break

    def stop(self):
        """Stop the controller loop"""
        self.should_stop.set()


class MainWindow(QMainWindow):
    """Main application window with tabbed interface"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # Shared configuration
        self.shared_config = SharedConfig()
        
        # Create tabs
        self.config_tab = ConfigurationTab(self.shared_config)
        self.controller_tab = ControllerTab(self.shared_config)
        self.realtime_tab = RealTimeDataTab(self.shared_config)
        
        # Add tabs to widget
        self.tab_widget.addTab(self.config_tab, "Configuration")
        self.tab_widget.addTab(self.controller_tab, "Controller")
        self.tab_widget.addTab(self.realtime_tab, "Real-Time Data")
        
        # Initialize thread objects (will be created fresh on each start)
        self.controller_thread = None
        self.controller_worker = None
        
        # Connect controller tab buttons to start/stop methods
        self.controller_tab.start_button.clicked.connect(self.start_controller)
        self.controller_tab.stop_button.clicked.connect(self.stop_controller)
    
    def init_ui(self):
        self.setWindowTitle("Random Vibration Shaker Control")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and tab widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
    
    def handle_psd_data(self, data):
        """Handle PSD data from controller worker"""
        self.controller_tab.update_plots(psd_data=data)
    
    def handle_metrics_data(self, data):
        """Handle metrics data from controller worker"""
        self.controller_tab.update_plots(metrics_data=data)
        if (
            self.shared_config.test_mode == "sine_sweep"
            and isinstance(data, (list, tuple))
            and len(data) >= 9
        ):
            try:
                freq_hz = float(data[6])
                peak_g = float(data[7])
                target_peak_g = float(data[8])
            except (TypeError, ValueError):
                return
            self.realtime_tab.update_sine_metrics(freq_hz, peak_g, target_peak_g)
    
    def handle_eq_data(self, data):
        """Handle equalizer data from controller worker"""
        self.controller_tab.update_plots(eq_data=data)
    
    def start_controller(self):
        """Start the controller in a separate thread"""
        if self.controller_worker is None or not self.controller_worker.is_running:
            # Create new thread each time (Qt requirement)
            self.controller_thread = QThread()
            self.controller_worker = ControllerWorker(self.shared_config)
            self.controller_worker.moveToThread(self.controller_thread)
            
            # Reconnect signals for new worker
            self.controller_worker.psd_data_ready.connect(self.handle_psd_data)
            self.controller_worker.psd_data_ready.connect(self.realtime_tab.update_psd_data)
            self.controller_worker.metrics_data_ready.connect(self.handle_metrics_data)
            self.controller_worker.eq_data_ready.connect(self.handle_eq_data)
            self.controller_worker.realtime_data_ready.connect(
                self.realtime_tab.add_data_point)
            
            # Connect thread start to worker
            self.controller_thread.started.connect(self.controller_worker.run_controller)
            
            # Start thread
            self.controller_thread.start()
            
            # Update controller tab UI
            self.controller_tab.start_controller()
            # Start real-time data viewer from main thread
            self.realtime_tab.start_streaming()
    
    def stop_controller(self):
        """Stop the controller"""
        if hasattr(self, 'controller_worker') and self.controller_worker.is_running:
            self.controller_worker.stop()
            
            # Stop real-time data viewer first (from main thread)
            self.realtime_tab.stop_streaming()
            
            # Wait for thread to finish
            if hasattr(self, 'controller_thread'):
                self.controller_thread.quit()
                self.controller_thread.wait(2000)  # Wait up to 2 seconds
            
            # Update controller tab UI
            self.controller_tab.stop_controller()
    
    def closeEvent(self, event):
        """Handle application close"""
        # Stop real-time data viewer first
        self.realtime_tab.stop_streaming()
        
        # Stop controller worker
        if self.controller_worker and self.controller_worker.is_running:
            self.controller_worker.stop()
            if self.controller_thread:
                self.controller_thread.quit()
                self.controller_thread.wait(3000)  # Wait up to 3 seconds
        
        event.accept()


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
