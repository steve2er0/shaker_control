#!/usr/bin/env python3
"""
Random Vibration Shaker Control - Desktop GUI Application

A professional desktop application for random vibration testing with:
- Configuration tab for all system parameters
- Controller response visualization
- Real-time data streaming and monitoring

Built with PySide6 and PyQtGraph for high-performance plotting.
"""

import sys
import numpy as np
import time
from collections import deque
from threading import Thread, Event
import copy

# GUI imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                              QHBoxLayout, QWidget, QFormLayout, QDoubleSpinBox, 
                              QSpinBox, QPushButton, QCheckBox, QLabel, QGroupBox,
                              QGridLayout, QComboBox)
from PySide6.QtCore import QTimer, Signal, QObject, QThread
from PySide6.QtGui import QFont

# Plotting imports
import pyqtgraph as pg

# Import our existing modules
import config
from rv_controller import MultiBandEqualizer, RandomVibrationController, create_bandpass_filter, make_bandlimited_noise, apply_safety_limiters
from simulation import create_simulation_system


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
        
        # Control parameters
        self.initial_level_fraction = config.INITIAL_LEVEL_FRACTION
        self.max_level_fraction_rate = config.MAX_LEVEL_FRACTION_RATE
        self.kp = config.KP
        self.ki = config.KI
        
        # Equalizer parameters
        self.eq_num_bands = config.EQ_NUM_BANDS
        self.eq_adapt_rate = config.EQ_ADAPT_RATE
        self.eq_smooth_factor = config.EQ_SMOOTH_FACTOR
        
        # Safety limiters
        self.max_crest_factor = config.MAX_CREST_FACTOR
        self.max_rms_volts = config.MAX_RMS_VOLTS
        self.ao_volt_limit = config.AO_VOLT_LIMIT
        
        # Simulation mode
        self.simulation_mode = config.SIMULATION_MODE
        self.sim_plant_gain = config.SIM_PLANT_GAIN
        self.sim_noise_level = config.SIM_NOISE_LEVEL


class ConfigurationTab(QWidget):
    """Configuration tab with form inputs for all parameters"""
    
    def __init__(self, shared_config):
        super().__init__()
        self.shared_config = shared_config
        self.init_ui()
        self.load_values()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create form layout
        form_layout = QFormLayout()
        
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
        layout.addWidget(control_group)
        layout.addWidget(eq_group)
        layout.addWidget(safety_group)
        layout.addWidget(sim_group)
        layout.addWidget(self.apply_button)
        layout.addStretch()
        
        self.setLayout(layout)
    
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
        
        # Emit signal that config was updated
        self.shared_config.config_updated.emit()
        
        # Visual feedback
        self.apply_button.setText("Applied!")
        QTimer.singleShot(1000, lambda: self.apply_button.setText("Apply Configuration"))


class ControllerResponseTab(QWidget):
    """Controller response tab with PyQtGraph plots"""
    
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
        self.eq_gains_data = deque(maxlen=self.max_points)
        
        self.update_counter = 0
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create plot widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        
        # PSD plot
        self.psd_plot = self.plot_widget.addPlot(title="Power Spectral Density")
        self.psd_plot.setLogMode(x=True, y=True)
        self.psd_plot.setLabel('left', 'PSD', units='g²/Hz')
        self.psd_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.psd_plot.showGrid(x=True, y=True)
        
        # PSD curves
        self.psd_measured_curve = self.psd_plot.plot(pen='b', name='Measured PSD')
        self.psd_target_curve = self.psd_plot.plot(pen='r', style='--', name='Target PSD')
        
        # Add legend
        self.psd_plot.addLegend()
        
        # Next row - Control metrics
        self.plot_widget.nextRow()
        
        # Control metrics plot
        self.metrics_plot = self.plot_widget.addPlot(title="Control Metrics")
        self.metrics_plot.setLabel('left', 'Value')
        self.metrics_plot.setLabel('bottom', 'Update Index')
        self.metrics_plot.showGrid(x=True, y=True)
        
        # Control metric curves
        self.rms_curve = self.metrics_plot.plot(pen='g', name='a_rms [g]')
        self.level_curve = self.metrics_plot.plot(pen='b', name='Level Fraction')
        self.sat_curve = self.metrics_plot.plot(pen='r', name='Saturation %')
        self.plant_gain_curve = self.metrics_plot.plot(pen='m', name='Plant Gain [g/V]')
        
        self.metrics_plot.addLegend()
        
        # Next row - Equalizer gains
        self.plot_widget.nextRow()
        
        # Equalizer plot
        self.eq_plot = self.plot_widget.addPlot(title="Equalizer Gains")
        self.eq_plot.setLogMode(x=True, y=False)
        self.eq_plot.setLabel('left', 'Gain')
        self.eq_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.eq_plot.showGrid(x=True, y=True)
        self.eq_plot.setYRange(0.1, 10.0)
        
        # Equalizer bar graph (will be updated with data)
        self.eq_bargraph = None
        
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
    
    def update_plots(self, psd_data=None, metrics_data=None, eq_data=None):
        """Update all plots with new data"""
        
        # Update PSD plot
        if psd_data:
            f, psd_measured, psd_target = psd_data
            self.psd_measured_curve.setData(f, psd_measured)
            if psd_target is not None:
                self.psd_target_curve.setData(f, psd_target)
        
        # Update metrics plot
        if metrics_data:
            self.update_counter += 1
            self.time_data.append(self.update_counter)
            
            a_rms, s_avg_meas, s_avg_target, level_fraction, sat_frac, plant_gain = metrics_data
            
            self.rms_data.append(a_rms)
            self.level_data.append(level_fraction)
            self.sat_data.append(sat_frac * 100)  # Convert to percentage
            self.plant_gain_data.append(plant_gain)
            
            # Update curves
            self.rms_curve.setData(list(self.time_data), list(self.rms_data))
            self.level_curve.setData(list(self.time_data), list(self.level_data))
            self.sat_curve.setData(list(self.time_data), list(self.sat_data))
            self.plant_gain_curve.setData(list(self.time_data), list(self.plant_gain_data))
        
        # Update equalizer plot
        if eq_data:
            freq_centers, gains = eq_data
            if self.eq_bargraph is None:
                # Create bar graph
                self.eq_bargraph = pg.BarGraphItem(x=freq_centers, height=gains, width=0.1, brush='b')
                self.eq_plot.addItem(self.eq_bargraph)
            else:
                # Update existing bar graph
                self.eq_bargraph.setOpts(height=gains)


class RealTimeDataTab(QWidget):
    """Real-time data tab with streaming plots and controls"""
    
    def __init__(self, shared_config):
        super().__init__()
        self.shared_config = shared_config
        self.init_ui()
        
        # Data storage for streaming
        self.max_time_window = 10.0  # 10 seconds
        self.streaming_data = deque()
        self.streaming_time = deque()
        self.start_time = None
        
        # Streaming state
        self.is_streaming = False
        
        # Timer for plot updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_streaming_plot)
        self.update_timer.setInterval(33)  # ~30 fps
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_streaming)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_streaming)
        self.stop_button.setEnabled(False)
        
        self.autoscale_check = QCheckBox("Enable Autoscaling")
        self.autoscale_check.setChecked(True)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.autoscale_check)
        button_layout.addStretch()
        
        # Status label
        self.status_label = QLabel("Status: Stopped")
        button_layout.addWidget(self.status_label)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget(title="Real-Time Acceleration Data")
        self.plot_widget.setLabel('left', 'Acceleration', units='g')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True)
        
        # Plot curve
        self.realtime_curve = self.plot_widget.plot(pen='g', name='Acceleration')
        
        layout.addLayout(button_layout)
        layout.addWidget(self.plot_widget)
        
        self.setLayout(layout)
    
    def start_streaming(self):
        """Start real-time data streaming"""
        self.is_streaming = True
        self.start_time = time.time()
        self.streaming_data.clear()
        self.streaming_time.clear()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Streaming")
        
        # Start update timer
        self.update_timer.start()
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.is_streaming = False
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped")
        
        # Stop update timer
        self.update_timer.stop()
    
    def add_data_point(self, acceleration_value):
        """Add new data point to streaming plot"""
        if not self.is_streaming:
            return
        
        current_time = time.time() - self.start_time
        
        # Add new data
        self.streaming_time.append(current_time)
        self.streaming_data.append(acceleration_value)
        
        # Remove old data outside time window
        while self.streaming_time and (current_time - self.streaming_time[0]) > self.max_time_window:
            self.streaming_time.popleft()
            self.streaming_data.popleft()
    
    def update_streaming_plot(self):
        """Update the streaming plot display"""
        if not self.is_streaming or not self.streaming_time:
            return
        
        # Update plot data
        self.realtime_curve.setData(list(self.streaming_time), list(self.streaming_data))
        
        # Autoscaling
        if self.autoscale_check.isChecked() and self.streaming_data:
            y_min = min(self.streaming_data)
            y_max = max(self.streaming_data)
            margin = (y_max - y_min) * 0.1
            self.plot_widget.setYRange(y_min - margin, y_max + margin)
            
            if self.streaming_time:
                self.plot_widget.setXRange(max(0, self.streaming_time[-1] - self.max_time_window), 
                                          self.streaming_time[-1])


class ControllerWorker(QObject):
    """Worker thread for running the control loop"""
    
    # Signals for thread-safe communication
    psd_data_ready = Signal(object)  # (f, psd_measured, psd_target)
    metrics_data_ready = Signal(object)  # (a_rms, s_avg_meas, s_avg_target, level_fraction, sat_frac, plant_gain)
    eq_data_ready = Signal(object)  # (freq_centers, gains)
    realtime_data_ready = Signal(float)  # acceleration value
    status_message = Signal(str)
    
    def __init__(self, shared_config):
        super().__init__()
        self.shared_config = shared_config
        self.should_stop = Event()
        self.is_running = False
    
    def run_controller(self):
        """Main controller loop running in separate thread"""
        try:
            self.is_running = True
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
            
            # Create simulation system
            ao_writer, ai_reader, ai_task, ao_task = create_simulation_system(
                fs, self.shared_config.sim_plant_gain, config.SIM_RESONANCES, 
                self.shared_config.sim_noise_level, config.SIM_DELAY_SAMPLES, 
                config.SIM_NONLINEARITY
            )
            
            self.status_message.emit("Controller running...")
            
            # Control loop variables
            level_fraction = self.shared_config.initial_level_fraction
            plant_gain_g_per_V = self.shared_config.sim_plant_gain * 0.8
            
            while not self.should_stop.is_set():
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
                ao_writer.write_many_sample(volts_block)
                data = np.empty(block_samples, dtype=np.float64)
                ai_reader.read_many_sample(data, number_of_samples_per_channel=block_samples)
                
                # Estimate PSD
                f, Pxx, metrics = controller.estimate_psd(data)
                if metrics is None:
                    continue
                
                S_avg_meas, S_avg_target, a_rms_meas = metrics
                
                # Update plant gain
                cmd_rms = np.std(volts_block)
                if cmd_rms > 1e-9:
                    new_gain = a_rms_meas / cmd_rms
                    plant_gain_g_per_V = 0.8*plant_gain_g_per_V + 0.2*new_gain
                
                # Update equalizer
                equalizer.update_gains(f, Pxx, controller.target_psd_func)
                
                # Update control
                level_fraction = controller.update_control(S_avg_meas, S_avg_target, level_fraction)
                
                # Calculate saturation
                sat_frac = np.mean(np.abs(volts_block) >= (0.98*self.shared_config.ao_volt_limit))
                
                # Emit data for GUI updates
                psd_target = controller.target_psd_func(f)
                self.psd_data_ready.emit((f, Pxx, psd_target))
                
                self.metrics_data_ready.emit((a_rms_meas, S_avg_meas, S_avg_target, 
                                           level_fraction, sat_frac, plant_gain_g_per_V))
                
                eq_info = equalizer.get_band_info()
                self.eq_data_ready.emit((eq_info['centers'], eq_info['gains']))
                
                # Emit real-time data point (use RMS of current block)
                self.realtime_data_ready.emit(np.std(data))
                
                # Small delay to prevent overwhelming the GUI
                time.sleep(0.1)
                
        except Exception as e:
            self.status_message.emit(f"Error: {e}")
        finally:
            self.is_running = False
            self.status_message.emit("Controller stopped")
    
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
        self.controller_tab = ControllerResponseTab(self.shared_config)
        self.realtime_tab = RealTimeDataTab(self.shared_config)
        
        # Add tabs to widget
        self.tab_widget.addTab(self.config_tab, "Configuration")
        self.tab_widget.addTab(self.controller_tab, "Controller Response")
        self.tab_widget.addTab(self.realtime_tab, "Real-Time Data")
        
        # Controller worker thread
        self.controller_thread = QThread()
        self.controller_worker = ControllerWorker(self.shared_config)
        self.controller_worker.moveToThread(self.controller_thread)
        
        # Connect signals
        self.controller_worker.psd_data_ready.connect(
            lambda data: self.controller_tab.update_plots(psd_data=data))
        self.controller_worker.metrics_data_ready.connect(
            lambda data: self.controller_tab.update_plots(metrics_data=data))
        self.controller_worker.eq_data_ready.connect(
            lambda data: self.controller_tab.update_plots(eq_data=data))
        self.controller_worker.realtime_data_ready.connect(
            self.realtime_tab.add_data_point)
        
        # Start controller automatically
        self.start_controller()
    
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
    
    def start_controller(self):
        """Start the controller in a separate thread"""
        if not self.controller_worker.is_running:
            self.controller_thread.started.connect(self.controller_worker.run_controller)
            self.controller_thread.start()
    
    def closeEvent(self, event):
        """Handle application close"""
        # Stop controller worker
        if self.controller_worker.is_running:
            self.controller_worker.stop()
            self.controller_thread.quit()
            self.controller_thread.wait(3000)  # Wait up to 3 seconds
        
        event.accept()


class RealTimeDataTab(QWidget):
    """Real-time data tab with streaming plots and controls"""
    
    def __init__(self, shared_config):
        super().__init__()
        self.shared_config = shared_config
        self.init_ui()
        
        # Data storage for streaming
        self.max_time_window = 10.0  # 10 seconds
        self.streaming_data = deque()
        self.streaming_time = deque()
        self.start_time = None
        
        # Streaming state
        self.is_streaming = False
        
        # Timer for plot updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_streaming_plot)
        self.update_timer.setInterval(33)  # ~30 fps
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_streaming)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_streaming)
        self.stop_button.setEnabled(False)
        
        self.autoscale_check = QCheckBox("Enable Autoscaling")
        self.autoscale_check.setChecked(True)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.autoscale_check)
        button_layout.addStretch()
        
        # Status label
        self.status_label = QLabel("Status: Stopped")
        button_layout.addWidget(self.status_label)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget(title="Real-Time Acceleration Data")
        self.plot_widget.setLabel('left', 'Acceleration', units='g')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True)
        
        # Plot curve
        self.realtime_curve = self.plot_widget.plot(pen='g', name='Acceleration')
        
        layout.addLayout(button_layout)
        layout.addWidget(self.plot_widget)
        
        self.setLayout(layout)
    
    def start_streaming(self):
        """Start real-time data streaming"""
        self.is_streaming = True
        self.start_time = time.time()
        self.streaming_data.clear()
        self.streaming_time.clear()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Streaming")
        
        # Start update timer
        self.update_timer.start()
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.is_streaming = False
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped")
        
        # Stop update timer
        self.update_timer.stop()
    
    def add_data_point(self, acceleration_value):
        """Add new data point to streaming plot"""
        if not self.is_streaming:
            return
        
        current_time = time.time() - self.start_time
        
        # Add new data
        self.streaming_time.append(current_time)
        self.streaming_data.append(acceleration_value)
        
        # Remove old data outside time window
        while self.streaming_time and (current_time - self.streaming_time[0]) > self.max_time_window:
            self.streaming_time.popleft()
            self.streaming_data.popleft()
    
    def update_streaming_plot(self):
        """Update the streaming plot display"""
        if not self.is_streaming or not self.streaming_time:
            return
        
        # Update plot data
        self.realtime_curve.setData(list(self.streaming_time), list(self.streaming_data))
        
        # Autoscaling
        if self.autoscale_check.isChecked() and self.streaming_data:
            y_min = min(self.streaming_data)
            y_max = max(self.streaming_data)
            margin = (y_max - y_min) * 0.1
            self.plot_widget.setYRange(y_min - margin, y_max + margin)
            
            if self.streaming_time:
                self.plot_widget.setXRange(max(0, self.streaming_time[-1] - self.max_time_window), 
                                          self.streaming_time[-1])


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
