"""Configuration for the accelerometer monitoring application."""

from typing import Any, Dict, List, Optional, Tuple

# ---------- Acquisition Parameters ----------
SAMPLE_RATE_HZ: float = 2048.0  # Fixed NI-DAQ sample rate (Hz)
BLOCK_SIZE: int = 512            # Samples to read per acquisition block
TIME_WINDOW_SECONDS: float = 2.0  # Duration shown in the time-domain plot
DEFAULT_PSD_WINDOW_SECONDS: float = 0.25  # Welch window length in seconds
PSD_WINDOW_BOUNDS: Tuple[float, float] = (0.1, 10.0)  # Allowed PSD window range in GUI
PSD_OVERLAP: float = 0.5          # Fractional overlap for Welch PSD (0.0-0.95)

# ---------- Channel Configuration ----------
# Each entry describes a single NI-DAQ input channel and its accelerometer scaling.
# Sensitivity is specified in mV/g so voltage inputs can be scaled to g automatically.
CHANNELS: List[Dict[str, Any]] = [
    {
        "physical_channel": "Dev1/ai0",
        "label": "Control Accel",
        "sensitivity_mV_per_g": 100.0,
        "voltage_range": 5.0,             # +/- volts for fallback voltage mode
        "use_accel_channel": True,        # Try NI accel channel with IEPE support first
        "excitation_current_amps": 2.1e-3,
        "terminal_config": "PSEUDO_DIFF"
    }
]

# ---------- Simulation Mode ----------
SIMULATION_MODE: bool = False  # Set True to use synthetic data instead of hardware
SIM_NOISE_LEVEL: float = 0.05  # g RMS noise when running simulation
SIM_SEED: Optional[int] = None  # Optional random seed for repeatable traces

# ---------- Plot Appearance ----------
TIME_PLOT_Y_RANGE: Optional[Tuple[float, float]] = None  # (min, max) for fixed axis or None for auto
PSD_Y_BOUNDS: Optional[Tuple[float, float]] = None       # Optional fixed PSD bounds (linear units)
COLOR_PALETTE: List[str] = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]

"""
Modify CHANNELS and sensitivities to match your hardware wiring. The application reads each
channel continuously at 51.2 kHz, converts voltages to g using the configured sensitivities,
shows the latest time-domain window, and computes PSDs using an adjustable Welch window.
"""
