"""
Shaker Plant Simulation and Mock DAQ Interface

This module provides:
- Realistic shaker plant simulator with dynamics
- Mock DAQ interface for testing without hardware
- Plant response modeling with resonances and noise
"""

import numpy as np
from scipy.signal import sosfilt, butter, cont2discrete, lfilter, iirpeak
import time


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

        # Resonance filters (frequency-dependent gain shaping)
        self._res_filters = []
        for freq, Q, gain_mult in self.resonances:
            if freq <= 0 or freq >= (self.fs / 2.0):
                continue
            try:
                b, a = iirpeak(freq, Q, fs=self.fs)
                b = b * float(gain_mult)
                zi = np.zeros(max(len(a), len(b)) - 1, dtype=float)
                self._res_filters.append({'b': b, 'a': a, 'zi': zi})
            except Exception:
                continue
        
        print(f"Simulator: gain={base_gain:.1f} g/V, {len(self.resonances)} resonances, "
              f"noise={noise_level:.3f} g RMS")
    
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
        
        # Apply resonance shaping through biquad peaks
        if self._res_filters:
            res_total = np.zeros_like(acceleration)
            for filt in self._res_filters:
                y, filt['zi'] = lfilter(filt['b'], filt['a'], delayed_input, zi=filt['zi'])
                res_total += y
            acceleration += res_total
        
        # Add measurement noise
        if self.noise_level > 0:
            noise = np.random.randn(len(acceleration)) * self.noise_level
            acceleration += noise
        
        # Final safety check: ensure no NaN or inf values
        if np.any(~np.isfinite(acceleration)):
            # Silently replace with noise to avoid spam
            acceleration = np.random.randn(len(acceleration)) * self.noise_level
        
        return acceleration


# --------------- Mock DAQ Interface (for simulation) ---------------
class MockAnalogSingleChannelWriter:
    def __init__(self):
        self.last_data = None
    
    def write_many_sample(self, data):
        self.last_data = data.copy()
        # Simulate write timing
        time.sleep(0.001)


class MockAnalogSingleChannelReader:
    def __init__(self, plant_simulator, num_channels=1):
        self.plant_sim = plant_simulator
        self.num_channels = max(1, int(num_channels))
        self.dt = 1.0 / self.plant_sim.fs
        self._sdof_filters = []
        for idx in range(self.num_channels):
            if idx == 0:
                self._sdof_filters.append(None)
                continue
            damping = 0.05 + 0.03 * idx
            gain = 0.8 + 0.4 * idx
            b_d, a_d = self._design_sdof_filter(freq_hz=200.0, damping=damping)
            state = {
                'b': b_d,
                'a': a_d,
                'u_hist': np.zeros(len(b_d) - 1, dtype=float),
                'y_hist': np.zeros(len(a_d) - 1, dtype=float),
                'gain': gain,
                'noise_scale': (idx + 1) * 0.1 * self.plant_sim.noise_level
            }
            self._sdof_filters.append(state)

    def _design_sdof_filter(self, freq_hz, damping):
        omega_n = 2 * np.pi * freq_hz
        num = [omega_n ** 2]
        den = [1.0, 2.0 * damping * omega_n, omega_n ** 2]
        b_d, a_d, _ = cont2discrete((num, den), self.dt, method='bilinear')
        b_d = b_d.flatten()
        a_d = a_d.flatten()
        if not np.isclose(a_d[0], 1.0):
            b_d = b_d / a_d[0]
            a_d = a_d / a_d[0]
        return b_d, a_d

    def _apply_sdof(self, input_signal, filt_state):
        b = filt_state['b']
        a = filt_state['a']
        u_hist = filt_state['u_hist']
        y_hist = filt_state['y_hist']
        gain = filt_state['gain']

        output = np.empty_like(input_signal)
        for n, u_n in enumerate(input_signal):
            acc = b[0] * u_n
            if len(b) > 1:
                acc += np.dot(b[1:], u_hist)
            if len(a) > 1:
                acc -= np.dot(a[1:], y_hist)

            y_n = acc
            output[n] = y_n * gain

            # Update histories (simple shift register)
            if len(u_hist) > 0:
                u_hist[1:] = u_hist[:-1]
                u_hist[0] = u_n
            if len(y_hist) > 0:
                y_hist[1:] = y_hist[:-1]
                y_hist[0] = y_n

        return output

    def read_many_sample(self, data_array, number_of_samples_per_channel, timeout=10.0):
        # Get the last written AO data and simulate plant response
        if hasattr(mock_ao_writer, 'last_data') and mock_ao_writer.last_data is not None:
            simulated_response = self.plant_sim.process(mock_ao_writer.last_data)
        else:
            simulated_response = np.random.randn(number_of_samples_per_channel) * 0.01

        if simulated_response.size < number_of_samples_per_channel:
            pad = number_of_samples_per_channel - simulated_response.size
            simulated_response = np.pad(simulated_response, (0, pad), mode='edge')

        if np.ndim(data_array) == 1:
            data_array[:] = simulated_response[:len(data_array)]
        else:
            n_channels = data_array.shape[0]
            n_samples = min(number_of_samples_per_channel, data_array.shape[1])
            base = simulated_response[:n_samples]
            data_array[0, :n_samples] = base
            for ch in range(1, n_channels):
                filt_state = self._sdof_filters[ch]
                response = self._apply_sdof(base[:n_samples], filt_state)
                noise = np.random.randn(n_samples) * filt_state['noise_scale']
                data_array[ch, :n_samples] = response + noise

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


# Global mock objects (will be initialized when needed)
mock_ao_writer = None
mock_ai_reader = None


def create_simulation_system(fs, sim_plant_gain, sim_resonances, sim_noise_level, 
                           sim_delay_samples, sim_nonlinearity, num_input_channels=1):
    """Create and initialize the simulation system"""
    global mock_ao_writer, mock_ai_reader
    
    # Create plant simulator
    plant_sim = ShakerPlantSimulator(fs=fs, base_gain=sim_plant_gain, 
                                   resonances=sim_resonances, noise_level=sim_noise_level,
                                   delay_samples=sim_delay_samples, nonlinearity=sim_nonlinearity)
    
    # Create mock objects
    mock_ao_writer = MockAnalogSingleChannelWriter()
    mock_ai_reader = MockAnalogSingleChannelReader(plant_sim, num_channels=num_input_channels)
    
    return mock_ao_writer, mock_ai_reader, MockTask(), MockTask()
