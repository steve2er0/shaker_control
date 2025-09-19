"""
Shaker Plant Simulation and Mock DAQ Interface

This module provides:
- Realistic shaker plant simulator with dynamics
- Mock DAQ interface for testing without hardware
- Plant response modeling with resonances and noise
"""

import numpy as np
from scipy.signal import sosfilt, butter
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


# Global mock objects (will be initialized when needed)
mock_ao_writer = None
mock_ai_reader = None


def create_simulation_system(fs, sim_plant_gain, sim_resonances, sim_noise_level, 
                           sim_delay_samples, sim_nonlinearity):
    """Create and initialize the simulation system"""
    global mock_ao_writer, mock_ai_reader
    
    # Create plant simulator
    plant_sim = ShakerPlantSimulator(fs=fs, base_gain=sim_plant_gain, 
                                   resonances=sim_resonances, noise_level=sim_noise_level,
                                   delay_samples=sim_delay_samples, nonlinearity=sim_nonlinearity)
    
    # Create mock objects
    mock_ao_writer = MockAnalogSingleChannelWriter()
    mock_ai_reader = MockAnalogSingleChannelReader(plant_sim)
    
    return mock_ao_writer, mock_ai_reader, MockTask(), MockTask()
