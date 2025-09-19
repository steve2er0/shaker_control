import numpy as np
from scipy.signal import welch, firwin, lfilter, butter, sosfilt, iirfilter
import time
from collections import deque

# Conditional import of DAQ modules (only needed for hardware mode)
try:
    import nidaqmx
    from nidaqmx.constants import (AcquisitionType, TerminalConfiguration, ExcitationSource,
                                   AccelUnits, Coupling, VoltageUnits)
    from nidaqmx.stream_writers import AnalogSingleChannelWriter
    from nidaqmx.stream_readers import AnalogSingleChannelReader
    NIDAQMX_AVAILABLE = True
except ImportError:
    print("Warning: nidaqmx not available - simulation mode only")
    NIDAQMX_AVAILABLE = False

# ---- NEW: plotting imports ----
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ---------------- User parameters ----------------
device_ai = "Dev1/ai0"
device_ao = "Dev1/ao0"

fs = 51200.0             # Sample rate [Hz]
buf_seconds = 5.0        # Buffer size for AO/AI streaming
block_seconds = 1.0      # Processing block duration (Welch windows fit inside)

# Target PSD points: list of (frequency [Hz], PSD level [g^2/Hz]) pairs
# Set to achievable levels that won't cause saturation
target_psd_points = [
    (20.0, 0.0025),     # 8e-3 g^2/Hz
    (80.0, 0.01),     # 1.5e-2 g^2/Hz  
    (800.0, 0.01),    # 1.5e-2 g^2/Hz
    (2000.0, 0.0025)    # 8e-3 g^2/Hz
]

# Derived frequency band limits
f1 = target_psd_points[0][0]    # First frequency point
f2 = target_psd_points[-1][0]   # Last frequency point
accel_mV_per_g = 100.0   # Accelerometer sensitivity [mV/g]
ao_volt_limit = 2.0      # Absolute max AO voltage magnitude [V]
initial_level_fraction = 0.10  # Start at 100% of target to avoid immediate saturation
max_level_fraction_rate = 0.5 # Faster level changes to reach target quicker
welch_nperseg = 2048     # Shorter segments for faster response (power of 2)
welch_noverlap = welch_nperseg // 2

# Control gains (tune cautiously)
Kp = 2.0  # Higher gain to reach target faster
Ki = 0.5  # Faster integral to overcome steady-state error

# Equalizer parameters
eq_num_bands = 12           # Number of frequency bands for equalizer
eq_gain_limits = (0.1, 10.0)  # Min/max gain per band
eq_adapt_rate = 0.1      # Slower adaptation to avoid oscillation
eq_smooth_factor = 0.9     # More smoothing for stability

# Safety limiters
max_crest_factor = 6.0     # Higher crest factor to allow more peak voltage
max_rms_volts = 1.8        # Higher RMS limit to allow target achievement
crest_soft_knee = 0.8      # Soft limiting starts at this fraction of max_crest_factor
rms_limit_headroom = 0.9   # Scale factor when RMS limiting is active

# Simulation mode (set to True to run without DAQ hardware)
SIMULATION_MODE = True

# Safety check: force simulation mode if nidaqmx is not available
if not SIMULATION_MODE and not NIDAQMX_AVAILABLE:
    print("ERROR: nidaqmx not available but SIMULATION_MODE=False")
    print("Installing nidaqmx: pip install nidaqmx")
    print("Or set SIMULATION_MODE=True to run simulation")
    print("Forcing SIMULATION_MODE=True for now...")
    SIMULATION_MODE = True

# Simulation parameters
sim_plant_gain = 4.0      # Higher base plant gain [g/V] for realistic PSD levels
sim_resonances = [         # List of (frequency, Q, gain_multiplier) for resonances
    (150.0, 15.0, 3.0),    # 150 Hz resonance, Q=15, 3x gain
    (800.0, 25.0, 2.5),    # 800 Hz resonance, Q=25, 2.5x gain
]
sim_noise_level = 0.02     # Measurement noise level [g RMS]
sim_delay_samples = 5      # System delay in samples
sim_nonlinearity = 0.05    # Nonlinearity factor (0 = linear)

# --------------- Derived parameters --------------
buf_samples = int(fs * buf_seconds)
block_samples = int(fs * block_seconds)

# FIR bandpass to create band-limited noise
numtaps = 1025  # longer = sharper band edges; ensure manageable CPU
bp = firwin(numtaps, [f1, f2], pass_zero=False, fs=fs)

# Create target PSD interpolation function
def create_target_psd_function(target_points):
    """Create interpolation function for target PSD from list of (freq, PSD) points"""
    freqs, psds = zip(*target_points)
    freqs = np.array(freqs)
    psds = np.array(psds)
    
    # Use linear interpolation in log-log space for smoother transitions
    log_freqs = np.log10(freqs)
    log_psds = np.log10(psds)
    
    # Create interpolator (extrapolate with boundary values)
    interp_func = interp1d(log_freqs, log_psds, kind='linear', 
                          bounds_error=False, fill_value=(log_psds[0], log_psds[-1]))
    
    def target_psd(f):
        """Return target PSD for given frequency array"""
        result = np.full_like(f, np.nan, dtype=float)
        valid_mask = (f >= freqs[0]) & (f <= freqs[-1])
        if np.any(valid_mask):
            log_f_valid = np.log10(f[valid_mask])
            result[valid_mask] = 10**interp_func(log_f_valid)
        return result
    
    return target_psd

# Create target PSD function
target_psd_func = create_target_psd_function(target_psd_points)

# Calculate expected band RMS at target by integrating the PSD profile
def calculate_target_rms(target_points):
    """Calculate target RMS by integrating the PSD profile"""
    try:
        # Create a fine frequency grid for integration
        f_fine = np.logspace(np.log10(target_points[0][0]), np.log10(target_points[-1][0]), 1000)
        psd_fine = target_psd_func(f_fine)
        
        # Check for NaN in PSD values
        if np.any(~np.isfinite(psd_fine)):
            print("Debug: NaN in target PSD function, using simple estimate")
            # Fallback: simple average of target points
            freqs, psds = zip(*target_points)
            avg_psd = np.mean(psds)
            freq_range = target_points[-1][0] - target_points[0][0]
            return np.sqrt(avg_psd * freq_range)
        
        # Integrate PSD to get variance (RMS^2)
        # Using trapezoidal rule: variance = integral of PSD over frequency
        df = np.diff(f_fine)
        psd_avg = (psd_fine[:-1] + psd_fine[1:]) / 2
        variance = np.sum(psd_avg * df)
        
        result = np.sqrt(variance)
        if not np.isfinite(result):
            print("Debug: NaN in RMS calculation, using fallback")
            # Fallback calculation
            freqs, psds = zip(*target_points)
            avg_psd = np.mean(psds)
            freq_range = target_points[-1][0] - target_points[0][0]
            result = np.sqrt(avg_psd * freq_range)
        
        return result
    except Exception as e:
        print(f"Debug: Error in calculate_target_rms: {e}")
        # Emergency fallback
        freqs, psds = zip(*target_points)
        avg_psd = np.mean(psds)
        return np.sqrt(avg_psd * 100)  # Assume 100 Hz bandwidth

a_rms_target = calculate_target_rms(target_psd_points)

# --------------- Multi-Band Equalizer ---------------
class MultiBandEqualizer:
    def __init__(self, f1, f2, num_bands, fs, gain_limits=(0.1, 10.0), 
                 adapt_rate=0.05, smooth_factor=0.8):
        self.f1, self.f2 = f1, f2
        self.num_bands = num_bands
        self.fs = fs
        self.gain_limits = gain_limits
        self.adapt_rate = adapt_rate
        self.smooth_factor = smooth_factor
        
        # Create logarithmically spaced band edges
        self.band_edges = np.logspace(np.log10(f1), np.log10(f2), num_bands + 1)
        self.band_centers = np.sqrt(self.band_edges[:-1] * self.band_edges[1:])  # Geometric mean
        
        # Initialize gains to unity
        self.gains = np.ones(num_bands)
        
        # Create bandpass filters for each band
        self._create_bandpass_filters()
        
        print(f"Equalizer bands: {[f'{self.band_edges[i]:.1f}-{self.band_edges[i+1]:.1f} Hz' for i in range(num_bands)]}")
        
    def _create_bandpass_filters(self):
        """Create bandpass filters for each frequency band"""
        self.band_filters = []
        nyquist = self.fs / 2
        
        for i in range(self.num_bands):
            low_freq = max(self.band_edges[i], 1.0)  # Avoid DC
            high_freq = min(self.band_edges[i + 1], nyquist * 0.95)  # Avoid Nyquist
            
            # Design 4th order Butterworth bandpass filter
            sos = butter(4, [low_freq, high_freq], btype='bandpass', fs=self.fs, output='sos')
            self.band_filters.append(sos)
    
    def apply_equalization(self, signal):
        """Apply frequency-dependent gains to input signal"""
        if len(signal) == 0:
            return signal
        
        # Check input signal for NaN/inf
        if np.any(~np.isfinite(signal)):
            print("Debug: NaN in equalizer input signal")
            return signal  # Return original if input is bad
            
        # Filter signal into bands and apply gains
        equalized_signal = np.zeros_like(signal)
        
        for i, (sos, gain) in enumerate(zip(self.band_filters, self.gains)):
            try:
                # Filter signal through this band
                band_signal = sosfilt(sos, signal)
                
                # Check for NaN in band signal
                if np.any(~np.isfinite(band_signal)):
                    print(f"Debug: NaN in band {i} filter output")
                    continue  # Skip this band
                
                # Apply gain and add to output
                equalized_signal += gain * band_signal
                
            except Exception as e:
                print(f"Debug: Error in equalizer band {i}: {e}")
                continue  # Skip problematic band
        
        # Final check for NaN in output
        if np.any(~np.isfinite(equalized_signal)):
            print("Debug: NaN in final equalizer output, returning input")
            return signal  # Return original signal if output is bad
            
        return equalized_signal
    
    def update_gains(self, f_measured, psd_measured, target_psd_func):
        """Update equalizer gains based on measured vs target PSD"""
        target_psd = target_psd_func(f_measured)
        
        # Calculate gain adjustments for each band
        new_gains = self.gains.copy()
        
        for i in range(self.num_bands):
            # Find frequencies in this band
            band_mask = (f_measured >= self.band_edges[i]) & (f_measured < self.band_edges[i + 1])
            
            if not np.any(band_mask):
                continue
                
            # Get measured and target PSD in this band
            psd_meas_band = psd_measured[band_mask]
            psd_target_band = target_psd[band_mask]
            
            # Skip if no valid target data in this band
            valid_target = ~np.isnan(psd_target_band)
            if not np.any(valid_target):
                continue
                
            # Calculate average ratio (target/measured) for this band
            avg_measured = np.mean(psd_meas_band[valid_target])
            avg_target = np.mean(psd_target_band[valid_target])
            
            if avg_measured > 1e-12:  # Avoid division by zero
                desired_gain_adjustment = np.sqrt(avg_target / avg_measured)
                
                # Debug equalizer occasionally (uncomment for debugging)
                # if i == 0 and np.random.random() < 0.1:  # 10% of time, first band only
                #     print(f"EQ Debug band {i}: target={avg_target:.2e}, meas={avg_measured:.2e}, "
                #           f"ratio={desired_gain_adjustment:.3f}, gain={self.gains[i]:.3f}")
                
                # Apply adaptation rate
                gain_update = 1.0 + self.adapt_rate * (desired_gain_adjustment - 1.0)
                new_gains[i] *= gain_update
                
        # Apply smoothing and limits
        self.gains = self.smooth_factor * self.gains + (1 - self.smooth_factor) * new_gains
        self.gains = np.clip(self.gains, self.gain_limits[0], self.gain_limits[1])
    
    def get_band_info(self):
        """Return band information for plotting/debugging"""
        return {
            'centers': self.band_centers,
            'edges': self.band_edges, 
            'gains': self.gains
        }

# Create equalizer instance
equalizer = MultiBandEqualizer(f1=f1, f2=f2, num_bands=eq_num_bands, fs=fs,
                              gain_limits=eq_gain_limits, adapt_rate=eq_adapt_rate,
                              smooth_factor=eq_smooth_factor)

# --------------- Plant Simulator ---------------
class ShakerPlantSimulator:
    def __init__(self, fs, base_gain=2.0, resonances=None, noise_level=0.02, 
                 delay_samples=5, nonlinearity=0.05):
        self.fs = fs
        self.base_gain = base_gain
        self.resonances = resonances or []
        self.noise_level = noise_level
        self.delay_samples = delay_samples
        self.nonlinearity = nonlinearity
        
        # Create delay buffer
        self.delay_buffer = np.zeros(max(delay_samples, 1))
        
        # Create resonance filters
        self.resonance_filters = []
        self.resonance_states = []
        
        for freq, Q, gain_mult in self.resonances:
            # Create second-order resonant filter using bandpass approach
            nyquist = fs / 2
            if freq < nyquist * 0.95:  # Ensure frequency is below Nyquist
                # Create a narrow bandpass filter around the resonance
                bandwidth = freq / Q  # Bandwidth from Q factor
                low_freq = max(1.0, freq - bandwidth/2)
                high_freq = min(nyquist * 0.95, freq + bandwidth/2)
                
                # Design bandpass filter
                sos = butter(2, [low_freq, high_freq], btype='bandpass', fs=fs, output='sos')
                self.resonance_filters.append(sos)
                self.resonance_states.append(np.zeros((sos.shape[0], 2)))
        
        print(f"Simulator: {len(self.resonance_filters)} resonances, "
              f"gain={base_gain:.1f} g/V, noise={noise_level:.3f} g RMS")
    
    def process(self, voltage_input):
        """
        Simulate shaker plant response to voltage input
        Returns acceleration in g
        """
        if len(voltage_input) == 0:
            return np.array([])
        
        # Apply delay
        delayed_input = np.zeros_like(voltage_input)
        for i, v in enumerate(voltage_input):
            # Shift delay buffer
            self.delay_buffer[1:] = self.delay_buffer[:-1]
            self.delay_buffer[0] = v
            delayed_input[i] = self.delay_buffer[-1]
        
        # Base linear response
        acceleration = delayed_input * self.base_gain
        
        # Add nonlinearity (soft saturation)
        if self.nonlinearity > 0:
            sat_level = 5.0  # Saturation starts at 5V
            nonlin_gain = self.nonlinearity
            acceleration += nonlin_gain * np.tanh(delayed_input / sat_level) * np.abs(delayed_input)
        
        # Ensure minimum response level for small signals (avoid numerical issues)
        if np.max(np.abs(acceleration)) < 1e-6:
            acceleration = acceleration + np.random.randn(len(acceleration)) * 1e-4
        
        # Apply simple resonances (frequency-dependent gain without filtering for now)
        # This avoids filter initialization issues while still providing realistic response
        for freq, Q, gain_mult in self.resonances:
            # Add a simple sinusoidal component at the resonance frequency
            t = np.arange(len(delayed_input)) / self.fs
            # Make resonance effects more pronounced
            resonant_component = np.sin(2 * np.pi * freq * t) * delayed_input * (gain_mult - 1.0) * 0.5
            acceleration += resonant_component * self.base_gain
        
        # Add measurement noise
        if self.noise_level > 0:
            noise = np.random.randn(len(acceleration)) * self.noise_level
            acceleration += noise
        
        # Final safety check: ensure no NaN or inf values
        if np.any(~np.isfinite(acceleration)):
            # Silently replace with noise to avoid spam
            acceleration = np.random.randn(len(acceleration)) * self.noise_level
        
        return acceleration

# Create plant simulator (only used in simulation mode)
if SIMULATION_MODE:
    plant_sim = ShakerPlantSimulator(fs=fs, base_gain=sim_plant_gain, 
                                   resonances=sim_resonances, noise_level=sim_noise_level,
                                   delay_samples=sim_delay_samples, nonlinearity=sim_nonlinearity)

# --------------- Mock DAQ Interface (for simulation) ---------------
class MockAnalogSingleChannelWriter:
    def __init__(self):
        self.last_data = None
    
    def write_many_sample(self, data):
        self.last_data = data.copy()
        # Simulate write timing
        time.sleep(0.001)

class MockAnalogSingleChannelReader:
    def __init__(self, plant_simulator):
        self.plant_sim = plant_simulator
    
    def read_many_sample(self, data_array, number_of_samples_per_channel, timeout=10.0):
        # Get the last written AO data and simulate plant response
        if hasattr(mock_ao_writer, 'last_data') and mock_ao_writer.last_data is not None:
            simulated_response = self.plant_sim.process(mock_ao_writer.last_data)
            # Copy to the provided array
            data_array[:] = simulated_response[:len(data_array)]
        else:
            # No AO data yet, return realistic background noise to avoid NaN issues
            # Use higher noise level to ensure meaningful startup data
            data_array[:] = np.random.randn(len(data_array)) * 0.01  # 10x higher initial noise
        # Simulate read timing
        time.sleep(0.001)

class MockTask:
    def __init__(self):
        self.channels = MockChannels()
        self.timing = MockTiming()
        self.triggers = MockTriggers()
        
    def start(self):
        pass
        
    def stop(self):
        pass
        
    def close(self):
        pass

class MockChannels:
    def __init__(self):
        pass
        
    def create_ai_accel_chan(self, physical_channel, name_to_assign_to_channel, 
                           sensitivity, sensitivity_units, units):
        return MockChannel()
        
    def create_ao_voltage_chan(self, physical_channel, name_to_assign_to_channel,
                             min_val, max_val, units):
        return MockChannel()

class MockChannel:
    def __init__(self):
        self.ai_coupling = None
        self.ai_excit_src = None
        self.ai_excit_val = None
        self.ai_term_cfg = None

class MockTiming:
    def __init__(self):
        self.samp_clk_src = None
        self.samp_clk_term = "/Dev1/ai/SampleClock"
        
    def cfg_samp_clk_timing(self, rate, sample_mode, samps_per_chan):
        pass

class MockTriggers:
    def __init__(self):
        self.start_trigger = MockStartTrigger()

class MockStartTrigger:
    def __init__(self):
        self.term = "/Dev1/ai/StartTrigger"
        
    def cfg_dig_edge_start_trig(self, trigger_source):
        pass

# Create mock objects for simulation
if SIMULATION_MODE:
    mock_ao_writer = MockAnalogSingleChannelWriter()
    mock_ai_reader = MockAnalogSingleChannelReader(plant_sim)

# --------------- Plotting helper ----------------
class LivePSDPlotter:
    def __init__(self, f1, f2, target_psd_func, target_psd_points, a_rms_target, equalizer, smooth_alpha=0.25, update_every=1):
        self.f1, self.f2 = f1, f2
        self.target_psd_func = target_psd_func
        self.target_psd_points = target_psd_points
        self.a_rms_target = a_rms_target
        self.equalizer = equalizer
        self.smooth_alpha = smooth_alpha
        self.update_every = update_every
        self._cnt = 0
        self._f_prev = None
        self._Pxx_smooth = None

        plt.ion()
        self.fig = plt.figure(figsize=(12, 9))
        gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        self.ax_psd = self.fig.add_subplot(gs[0, 0])
        self.ax_dash = self.fig.add_subplot(gs[1, 0])
        self.ax_eq = self.fig.add_subplot(gs[2, 0])

        # PSD axes setup
        self.ax_psd.set_xscale('log')
        self.ax_psd.set_yscale('log')
        self.ax_psd.set_xlabel('Frequency [Hz]')
        self.ax_psd.set_ylabel('PSD [g^2/Hz]')
        self.ax_psd.grid(True, which='both', alpha=0.3)

        # Band shading
        self.band_span = self.ax_psd.axvspan(self.f1, self.f2, color='gray', alpha=0.08, label='Target band')

        # Placeholders for lines
        self.line_meas, = self.ax_psd.plot([], [], label='Measured PSD', color='C0', lw=1.3)
        self.line_meas_smooth, = self.ax_psd.plot([], [], label='Measured (smoothed)', color='C1', lw=2, alpha=0.8)
        self.line_target, = self.ax_psd.plot([], [], label='Target PSD', color='C3', lw=2, ls='--')

        self.ax_psd.legend(loc='best')

        # Dashboard time series (last N points)
        self.Ndash = 200
        self.t_hist = deque(maxlen=self.Ndash)
        self.rms_hist = deque(maxlen=self.Ndash)
        self.Savg_hist = deque(maxlen=self.Ndash)
        self.Savg_target_hist = deque(maxlen=self.Ndash)
        self.level_hist = deque(maxlen=self.Ndash)
        self.sat_hist = deque(maxlen=self.Ndash)
        self.cf_hist = deque(maxlen=self.Ndash)
        self.rms_limit_hist = deque(maxlen=self.Ndash)
        self.plant_gain_hist = deque(maxlen=self.Ndash)
        self.target_rms_hist = deque(maxlen=self.Ndash)

        self.line_rms, = self.ax_dash.plot([], [], label='a_rms [g]', color='C0', lw=2)
        self.line_Savg, = self.ax_dash.plot([], [], label='S_avg meas', color='C1', lw=2)
        self.line_Savg_target, = self.ax_dash.plot([], [], label='S_avg target', color='C1', ls='--', lw=2)
        self.line_level, = self.ax_dash.plot([], [], label='level_fraction', color='C2', lw=2)
        self.line_sat, = self.ax_dash.plot([], [], label='AO sat %', color='C3', lw=2)
        self.line_cf, = self.ax_dash.plot([], [], label='Crest factor', color='C5', ls=':', lw=1)
        self.line_plant_gain, = self.ax_dash.plot([], [], label='Plant gain [g/V]', color='C6', ls='-.', lw=1)
        self.line_rms_limit, = self.ax_dash.plot([], [], label='Limiter active', color='red', marker='o', markersize=3, alpha=0.8, linestyle='None')
        self.ax_dash.set_xlabel('Update index')
        self.ax_dash.grid(True, alpha=0.3)
        self.ax_dash.legend(loc='best')
        self._tick = 0

        # Equalizer gains plot
        eq_info = self.equalizer.get_band_info()
        self.ax_eq.set_xscale('log')
        self.ax_eq.set_xlabel('Frequency [Hz]')
        self.ax_eq.set_ylabel('EQ Gain')
        self.ax_eq.grid(True, alpha=0.3)
        self.ax_eq.set_ylim([0.1, 10.0])
        
        # Create bar plot for equalizer gains
        self.eq_bars = self.ax_eq.bar(eq_info['centers'], eq_info['gains'], 
                                     width=eq_info['centers'] * 0.3, alpha=0.7, color='C4')
        
        # Add horizontal line at unity gain
        self.ax_eq.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Unity gain')
        self.ax_eq.legend(loc='best')
        self.ax_eq.set_title('Multi-Band Equalizer Gains')

        self._update_title()

        self.fig.tight_layout()

    def _update_title(self):
        # Create title showing target points
        points_str = ", ".join([f"{f:.0f}Hz:{psd:.3g}" for f, psd in self.target_psd_points])
        self.ax_psd.set_title(
            f"Target PSD points: {points_str} g^2/Hz, "
            f"a_rms,target={self.a_rms_target:.4g} g"
        )

    def _make_target_curve(self, f):
        """Create target PSD curve using the interpolation function"""
        return self.target_psd_func(f)

    def update(self, f, Pxx, S_avg_meas, S_avg_target, a_rms_meas, level_fraction, sat_frac, plant_gain, limiter_stats=None):
        # Only update every self.update_every calls to save CPU
        self._cnt += 1
        if (self._cnt % self.update_every) != 0:
            return

        # Initialize frequency-dependent state on first call or if f changed
        if (self._f_prev is None) or (len(self._f_prev) != len(f)) or (not np.allclose(self._f_prev, f)):
            self._f_prev = f.copy()
            self._Pxx_smooth = Pxx.copy()
            target = self._make_target_curve(f)
            self.line_target.set_data(f, target)
            # Adjust x limits based on band
            xmin = max(0.8, self.f1 * 0.8)
            xmax = max(self.f2 * 1.2, self.f1 * 2.0)
            self.ax_psd.set_xlim([xmin, xmax])

        # Exponential smoothing of measured PSD
        alpha = self.smooth_alpha
        self._Pxx_smooth = alpha * Pxx + (1 - alpha) * self._Pxx_smooth

        # Update PSD lines
        self.line_meas.set_data(f, Pxx)
        self.line_meas_smooth.set_data(f, self._Pxx_smooth)
        
        # Always update target curve to ensure it's visible
        target = self._make_target_curve(f)
        self.line_target.set_data(f, target)

        # Y limits: adapt gently based on smoothed PSD inside band
        band = (f >= self.f1) & (f <= self.f2)
        if np.any(band):
            valid_psd = self._Pxx_smooth[band]
            valid_psd = valid_psd[~np.isnan(valid_psd)]
            if len(valid_psd) > 0:
                y_min = max(1e-8, np.min(valid_psd) * 0.1)
                y_max = np.max(valid_psd) * 10.0
            if y_max > y_min:
                    self.ax_psd.set_ylim([y_min, y_max])
            else:
                # Default range when no valid data - use target PSD range
                target_range = [p[1] for p in self.target_psd_points]
                y_min = min(target_range) * 0.1
                y_max = max(target_range) * 10.0
                self.ax_psd.set_ylim([y_min, y_max])

        # Update dashboard histories
        self._tick += 1
        self.t_hist.append(self._tick)
        self.rms_hist.append(a_rms_meas)
        self.Savg_hist.append(S_avg_meas)
        self.Savg_target_hist.append(S_avg_target)
        self.level_hist.append(level_fraction)
        self.sat_hist.append(sat_frac * 100)  # Convert to percentage
        self.plant_gain_hist.append(plant_gain)
        
        # Add limiter information if provided
        if limiter_stats:
            self.cf_hist.append(limiter_stats.get('cf_after', 0.0))
            self.rms_limit_hist.append(self._tick if limiter_stats.get('any_limiting', False) else np.nan)
        else:
            self.cf_hist.append(0.0)
            self.rms_limit_hist.append(np.nan)

        # Push dashboard traces
        self.line_rms.set_data(self.t_hist, self.rms_hist)
        self.line_Savg.set_data(self.t_hist, self.Savg_hist)
        self.line_Savg_target.set_data(self.t_hist, self.Savg_target_hist)
        self.line_level.set_data(self.t_hist, self.level_hist)
        self.line_sat.set_data(self.t_hist, self.sat_hist)
        self.line_cf.set_data(self.t_hist, self.cf_hist)
        self.line_plant_gain.set_data(self.t_hist, self.plant_gain_hist)
        self.line_rms_limit.set_data(self.t_hist, self.rms_limit_hist)

        # Rescale dashboard Y limits
        try:
            if len(self.rms_hist) > 0:
                ymin = min(min(self.rms_hist), min(self.Savg_hist), min(self.level_hist), 0.0)
                ymax = max(max(self.rms_hist), max(self.Savg_hist), max(self.Savg_target_hist),
                          max(self.level_hist), max(self.sat_hist), 
                          max(self.cf_hist) if self.cf_hist else 1.0,
                          max(self.plant_gain_hist) if self.plant_gain_hist else 1.0, 1.0)
                if ymax <= ymin:
                    ymax = ymin + 1.0
                self.ax_dash.set_xlim([max(0, self._tick - self.Ndash), self._tick])
                self.ax_dash.set_ylim([ymin, ymax])
            else:
                # Initial scaling for empty data
                self.ax_dash.set_xlim([0, 50])
                self.ax_dash.set_ylim([0, 1.0])
        except ValueError:
            pass

        # Update equalizer gains display
        eq_info = self.equalizer.get_band_info()
        for bar, gain in zip(self.eq_bars, eq_info['gains']):
            bar.set_height(gain)

        # Draw without blocking
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

# --------------- Helper functions ----------------
def make_bandlimited_noise(n):
    # white -> bandpass -> equalized
    w = np.random.randn(n)
    y = lfilter(bp, [1.0], w)
    
    # Check for NaN after bandpass filter
    if np.any(~np.isfinite(y)):
        print("Debug: NaN in bandpass filter output")
        y = np.random.randn(n) * 0.1
    
    # Apply multi-band equalization
    y_eq = equalizer.apply_equalization(y)
    
    # Check for NaN after equalization
    if np.any(~np.isfinite(y_eq)):
        print("Debug: NaN in equalizer output")
        y_eq = y  # Fall back to non-equalized signal
    
    return y_eq

def inband_psd_estimate(x):
    # Welch PSD in g^2/Hz; x must already be in g
    if len(x) == 0 or np.all(np.abs(x) < 1e-12):
        # Return small but valid values instead of None to avoid NaN
        f_dummy = np.linspace(f1, f2, 100)
        return f_dummy, np.full_like(f_dummy, 1e-8), (1e-8, 1e-8, 1e-4)
    
    f, Pxx = welch(x, fs=fs, nperseg=welch_nperseg, noverlap=welch_noverlap, detrend='constant')
    # Compute metrics over [f1,f2] band
    band = (f >= f1) & (f <= f2)
    if not np.any(band):
        return None, None, None
    
    # Get target PSD for measured frequencies
    target_psd = target_psd_func(f)
    
    # Compute weighted average PSD error (measured vs target)
    # Only consider frequencies where we have target values
    valid_target = ~np.isnan(target_psd) & band
    if not np.any(valid_target):
        return f, Pxx, (0.0, 0.0)
    
    # Average measured PSD in valid band
    S_avg_measured = np.mean(Pxx[valid_target])
    # Average target PSD in valid band  
    S_avg_target = np.mean(target_psd[valid_target])
    
    # Compute RMS via PSD integral over valid band
    df = f[1] - f[0] if len(f) > 1 else 0.0
    a_rms = np.sqrt(np.sum(Pxx[valid_target]) * df) if df > 0 else 0.0
    
    return f, Pxx, (S_avg_measured, S_avg_target, a_rms)

def slew_limit(current, target, max_rate, dt):
    delta = target - current
    max_delta = max_rate * dt
    if delta > max_delta:
        return current + max_delta
    elif delta < -max_delta:
        return current - max_delta
    else:
        return target

# --------------- Safety Limiters ----------------
def crest_factor_limit(signal, max_cf=4.0, soft_knee=0.8):
    """
    Limit crest factor (peak/RMS ratio) of signal using soft compression
    
    Args:
        signal: Input signal array
        max_cf: Maximum allowed crest factor
        soft_knee: Fraction of max_cf where soft limiting begins
    
    Returns:
        tuple: (limited_signal, crest_factor_before, crest_factor_after, limiting_active)
    """
    if len(signal) == 0:
        return signal, 0.0, 0.0, False
    
    # Calculate current RMS and peak
    rms = np.std(signal)
    if rms < 1e-12:  # Avoid division by zero
        return signal, 0.0, 0.0, False
    
    peak = np.max(np.abs(signal))
    current_cf = peak / rms
    
    # If crest factor is acceptable, return unchanged
    if current_cf <= max_cf:
        return signal, current_cf, current_cf, False
    
    # Apply soft compression above the knee
    knee_cf = max_cf * soft_knee
    limited_signal = signal.copy()
    
    if current_cf > knee_cf:
        # Calculate compression ratio
        excess_cf = current_cf - knee_cf
        max_excess = max_cf - knee_cf
        compression_ratio = max_excess / excess_cf if excess_cf > 0 else 1.0
        
        # Apply soft limiting using tanh compression
        scale_factor = knee_cf / current_cf + (max_cf - knee_cf) / current_cf * np.tanh(excess_cf / max_excess)
        limited_signal = signal * scale_factor
        
        # Verify and adjust if needed
        new_peak = np.max(np.abs(limited_signal))
        new_cf = new_peak / rms
        
        if new_cf > max_cf:  # Hard limit if soft limiting wasn't enough
            limited_signal = limited_signal * (max_cf * rms) / new_peak
            new_cf = max_cf
    else:
        new_cf = current_cf
    
    return limited_signal, current_cf, new_cf, True

def rms_limit(signal, max_rms, headroom=0.9):
    """
    Limit RMS level of signal
    
    Args:
        signal: Input signal array
        max_rms: Maximum allowed RMS level
        headroom: Scale factor when limiting is active
    
    Returns:
        tuple: (limited_signal, rms_before, rms_after, limiting_active)
    """
    if len(signal) == 0:
        return signal, 0.0, 0.0, False
    
    current_rms = np.std(signal)
    
    if current_rms <= max_rms:
        return signal, current_rms, current_rms, False
    
    # Scale down to target RMS with headroom
    target_rms = max_rms * headroom
    scale_factor = target_rms / current_rms
    limited_signal = signal * scale_factor
    
    return limited_signal, current_rms, target_rms, True

def apply_safety_limiters(signal, max_cf=4.0, max_rms=1.5, cf_soft_knee=0.8, rms_headroom=0.9):
    """
    Apply both crest factor and RMS limiting to signal
    
    Returns:
        tuple: (limited_signal, limiter_stats)
    """
    # Apply RMS limiting first
    signal_rms_limited, rms_before, rms_after, rms_limiting = rms_limit(signal, max_rms, rms_headroom)
    
    # Then apply crest factor limiting
    signal_final, cf_before, cf_after, cf_limiting = crest_factor_limit(
        signal_rms_limited, max_cf, cf_soft_knee)
    
    limiter_stats = {
        'rms_before': rms_before,
        'rms_after': rms_after if rms_limiting else np.std(signal_final),
        'rms_limiting': rms_limiting,
        'cf_before': cf_before,
        'cf_after': cf_after,
        'cf_limiting': cf_limiting,
        'any_limiting': rms_limiting or cf_limiting
    }
    
    return signal_final, limiter_stats

# --------------- Instantiate plotter -------------
plotter = LivePSDPlotter(f1=f1, f2=f2, target_psd_func=target_psd_func, 
                        target_psd_points=target_psd_points, a_rms_target=a_rms_target, 
                        equalizer=equalizer, smooth_alpha=0.3, update_every=1)

# --------------- DAQ setup ----------------
if SIMULATION_MODE:
    print("=== SIMULATION MODE ===")
    print("Running without DAQ hardware - using plant simulator")
    
    # Use mock objects
    ai_task = MockTask()
    ao_task = MockTask()
    ai_reader = mock_ai_reader
    ao_writer = mock_ao_writer
    
    # No real hardware initialization needed
    print("Simulation setup complete")
    
else:
    print("=== HARDWARE MODE ===")
    ai_task = nidaqmx.Task()
    ao_task = nidaqmx.Task()

try:
    if not SIMULATION_MODE:
        # AI accelerometer channel in g
        ai_ch = ai_task.ai_channels.create_ai_accel_chan(
            physical_channel=device_ai,
            name_to_assign_to_channel="accel",
            sensitivity=accel_mV_per_g,
            sensitivity_units=nidaqmx.constants.AccelSensitivityUnits.MVOLTS_PER_G,
            units=AccelUnits.G
        )
        ai_ch.ai_coupling = Coupling.AC
        ai_ch.ai_excit_src = ExcitationSource.INTERNAL
        ai_ch.ai_excit_val = 0.004  # 4 mA typical IEPE current; check your sensor spec
        ai_ch.ai_term_cfg = TerminalConfiguration.DIFF

        # AI timing
        ai_task.timing.cfg_samp_clk_timing(
            rate=fs,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=buf_samples
        )

        # AO voltage
        ao_task.ao_channels.create_ao_voltage_chan(
            physical_channel=device_ao,
            name_to_assign_to_channel="drive",
            min_val=-ao_volt_limit, max_val=ao_volt_limit, units=VoltageUnits.VOLTS
        )
        ao_task.timing.cfg_samp_clk_timing(
            rate=fs,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=buf_samples
        )

        # Share clocks and start triggers so AI/AO are synchronized
        ao_task.timing.samp_clk_src = ai_task.timing.samp_clk_term
        ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(ai_task.triggers.start_trigger.term)

        # Stream objects
        ai_reader = AnalogSingleChannelReader(ai_task.in_stream)
        ao_writer = AnalogSingleChannelWriter(ao_task.out_stream, auto_start=False)

        # Prime AO buffer with zeros
        ao_writer.write_many_sample(np.zeros(buf_samples))

        # Start tasks (AI first so trigger is armed), then AO
        ai_task.start()
        ao_task.start()
        
        print("Hardware DAQ setup complete")

    # --------------- Control loop ----------------
    level_fraction = initial_level_fraction
    integ = 0.0

    # Keep a rolling input buffer to process blocks
    t_last = time.time()
    t_last_print = time.time()  # Timer for console output
    
    # Initialize plant gain estimate
    if SIMULATION_MODE:
        plant_gain_g_per_V = sim_plant_gain * 0.8  # Start close to actual for faster convergence
        print(f"Starting with plant gain estimate: {plant_gain_g_per_V:.2f} g/V (actual: {sim_plant_gain:.2f} g/V)")
    else:
        plant_gain_g_per_V = 1.0  # initial conservative guess
        print("Starting with conservative plant gain estimate: 1.0 g/V")

    mode_str = "SIMULATION" if SIMULATION_MODE else "HARDWARE"
    print(f"\n=== {mode_str} CONTROL LOOP STARTING ===")
    print("Running. Press Ctrl+C to stop.")
    while True:
        # Generate a block of drive noise and scale
        drive = make_bandlimited_noise(block_samples)
        if np.std(drive) > 1e-12:
            drive = drive / np.std(drive)  # unit RMS voltage profile

        # Target in-band acceleration RMS for this block
        target_rms_block = a_rms_target * level_fraction
        
        # Safety check for NaN in key components
        if not np.isfinite(a_rms_target):
            print("Warning: a_rms_target is NaN, using fallback")
            a_rms_target = 0.1  # fallback value
            target_rms_block = a_rms_target * level_fraction
        
        if not np.isfinite(target_rms_block):
            target_rms_block = 0.1
        
        if np.any(~np.isfinite(drive)):
            drive = np.random.randn(len(drive)) * 0.1

        # Convert target g RMS to volts via plant gain estimate
        volts_block = (target_rms_block / max(plant_gain_g_per_V, 1e-6)) * drive
        
        # Ensure minimum drive level to avoid zero signals
        min_drive_rms = 0.05  # Low minimum drive to allow natural scaling
        current_rms = np.std(volts_block)
        if current_rms < min_drive_rms:
            volts_block = volts_block * (min_drive_rms / max(current_rms, 1e-12))
        
        # Apply safety limiters (crest factor and RMS limiting)
        volts_block, limiter_stats = apply_safety_limiters(
            volts_block, max_crest_factor, max_rms_volts, crest_soft_knee, rms_limit_headroom)
        
        # Final hard clipping as last resort
        volts_block = np.clip(volts_block, -ao_volt_limit, ao_volt_limit)
        
        # Safety: ensure no NaN or inf values in voltage output
        if np.any(~np.isfinite(volts_block)):
            print("Warning: NaN/inf in voltage output, using minimum drive")
            volts_block = np.random.randn(len(volts_block)) * min_drive_rms

        # Write AO and read AI
        ao_writer.write_many_sample(volts_block)
        data = np.empty(block_samples, dtype=np.float64)
        ai_reader.read_many_sample(data, number_of_samples_per_channel=block_samples, timeout=10.0)

        # PSD estimate and metrics
        f, Pxx, metrics = inband_psd_estimate(data)
        if metrics is None:
            print("Welch band empty; check fs/f1/f2.")
            continue
        S_avg_meas, S_avg_target, a_rms_meas = metrics
        
        # Debug: print data statistics occasionally (uncomment for debugging)
        # if np.random.random() < 0.1:  # 10% of the time
        #     print(f"Debug: data range [{np.min(data):.6f}, {np.max(data):.6f}], std={np.std(data):.6f}")

        # Update plant gain estimate g/V using ratio of measured RMS to command RMS
        cmd_rms = np.std(volts_block)  # volts RMS sent
        if cmd_rms > 1e-9:
            new_gain = a_rms_meas / cmd_rms
            plant_gain_g_per_V = 0.8*plant_gain_g_per_V + 0.2*new_gain

        # Update equalizer gains based on measured vs target PSD
        equalizer.update_gains(f, Pxx, target_psd_func)

        # PI control on average PSD level (target vs measured)
        err = S_avg_target - S_avg_meas
        now = time.time()
        dt = max(1e-3, now - t_last)
        t_last = now
        integ += err * dt
        level_fraction_target = level_fraction + Kp*err + Ki*integ

        # Debug PI control occasionally (uncomment for debugging)
        # if np.random.random() < 0.1:  # 10% of the time
        #     print(f"PI Debug: meas={S_avg_meas:.2e}, target={S_avg_target:.2e}, err={err:.2e}, level={level_fraction:.3f}")

        # Constraints
        level_fraction_target = max(0.0, min(2.0, level_fraction_target))
        level_fraction = slew_limit(level_fraction, level_fraction_target, max_level_fraction_rate, dt)

        # Safety: if AO saturates frequently, back off
        sat_frac = np.mean(np.abs(volts_block) >= (0.98*ao_volt_limit))
        
        # Debug: occasionally show voltage levels (uncomment for debugging)
        # if np.random.random() < 0.1:  # 10% of the time
        #     v_rms = np.std(volts_block)
        #     v_peak = np.max(np.abs(volts_block))
        #     print(f"Debug: V_rms={v_rms:.3f}V, V_peak={v_peak:.3f}V, limit={ao_volt_limit}V")
        
        if sat_frac > 0.15:  # Allow higher saturation before backing off
            level_fraction = max(0.0, level_fraction * 0.9)  # Less aggressive backoff
            integ *= 0.8  # Less integral reset
            print("AO near saturation; backing off level.")

        # ---- NEW: live plotting update ----
        plotter.update(f, Pxx, S_avg_meas, S_avg_target, a_rms_meas, level_fraction, sat_frac, plant_gain_g_per_V, limiter_stats)

        # Print status once per second
        now_print = time.time()
        if (now_print - t_last_print) >= 1.0:  # 1 second interval
            t_last_print = now_print
            
            eq_gains_str = f"[{', '.join([f'{g:.2f}' for g in equalizer.gains[:4]])}...]"  # Show first 4 gains
            limiter_str = ""
            if limiter_stats['any_limiting']:
                limiter_parts = []
                if limiter_stats['cf_limiting']:
                    cf_before = limiter_stats.get('cf_before', 0)
                    cf_after = limiter_stats.get('cf_after', 0)
                    if not (np.isnan(cf_before) or np.isnan(cf_after)):
                        limiter_parts.append(f"CF:{cf_before:.1f}→{cf_after:.1f}")
                if limiter_stats['rms_limiting']:
                    rms_before = limiter_stats.get('rms_before', 0)
                    rms_after = limiter_stats.get('rms_after', 0)
                    if not (np.isnan(rms_before) or np.isnan(rms_after)):
                        limiter_parts.append(f"RMS:{rms_before:.2f}→{rms_after:.2f}V")
                if limiter_parts:
                    limiter_str = f", LIMIT=[{', '.join(limiter_parts)}]"
            
            print(f"S_avg: meas={S_avg_meas:.4g}, target={S_avg_target:.4g} g^2/Hz, "
                  f"a_rms={a_rms_meas:.3g} g, level={level_fraction:.3f}, "
                  f"gain={plant_gain_g_per_V:.3g} g/V, sat={sat_frac:.2%}, EQ={eq_gains_str}{limiter_str}")

except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Graceful shutdown
    try:
        ao_task.stop()
        ai_task.stop()
        ao_task.close()
        ai_task.close()
    except Exception:
        pass
    # Keep plot open after stop (optional)
    try:
        plt.ioff()
        plt.show()
    except Exception:
        pass
