"""
Configuration file for Random Vibration Shaker Control System

Modify these parameters to customize your system behavior.
"""

# ============= HARDWARE CONFIGURATION =============
DEVICE_AI = "Dev1/ai0"
DEVICE_AO = "Dev1/ao0"
ACCEL_MV_PER_G = 100.0      # Accelerometer sensitivity [mV/g]
ACCEL_EXCITATION_AMPS = 2.1e-3  # IEPE excitation current [A], set 0 for external
AO_VOLT_LIMIT = 1.8         # Absolute max AO voltage magnitude [V]

# Input channel labels for visualization (order matches DAQ wiring)
INPUT_CHANNEL_LABELS = [
    "Control Accel",
    "Response 1",
    "Response 2",
    "Response 3",
]

# ============= SYSTEM PARAMETERS =============
FS = 51200.0                # Sample rate [Hz]
BUF_SECONDS = 4           # Buffer size for AO/AI streaming
BLOCK_SECONDS = 0.5         # Processing block duration
WELCH_NPERSEG = 8192        # Welch segment length (power of 2)

# ============= TARGET PSD PROFILE =============
# List of (frequency [Hz], PSD level [g^2/Hz]) pairs
TARGET_PSD_POINTS = [
    (200.0, 0.001),     # 2.5e-3 g^2/Hz
    (300.0, 0.004),       # 1e-2 g^2/Hz  
    (400.0, 0.004),      # 1e-2 g^2/Hz
    (1000.0, 0.001)    # 2.5e-3 g^2/Hz
]

# === Add

# ============= CONTROL PARAMETERS =============
INITIAL_LEVEL_FRACTION = 0.1    # Start at 100% of target
MAX_LEVEL_FRACTION_RATE = 2   # Max change in level per second
KP = 0.9                        # Proportional gain
KI = .06                        # Integral gain

# ============= EQUALIZER PARAMETERS =============
EQ_NUM_BANDS = 36               # Number of frequency bands
EQ_GAIN_LIMITS = (0.2, 10.0)   # Min/max gain per band
EQ_ADAPT_RATE = 0.2             # Adaptation rate
EQ_SMOOTH_FACTOR = 0.9          # Smoothing factor
EQ_ADAPT_LEVEL_THRESHOLD = 0.4  # Level fraction before full EQ adaptation
EQ_ADAPT_LEVEL_POWER = 1.0      # Exponent applied to level gating (1.0 = linear)
EQ_ADAPT_MIN_WEIGHT = 0.0       # Minimum adaptation weight even below threshold

# ============= SAFETY LIMITERS =============
MAX_CREST_FACTOR = 5.0          # Maximum peak/RMS ratio
MAX_RMS_VOLTS = 1.4             # Maximum RMS voltage output
CREST_SOFT_KNEE = 0.8           # Soft limiting threshold
RMS_LIMIT_HEADROOM = 0.9        # Headroom when limiting

# ============= SIMULATION MODE =============
SIMULATION_MODE = False          # Set to False for real hardware

# Simulation parameters (only used when SIMULATION_MODE = True)
SIM_PLANT_GAIN = 4.0           # Base plant gain [g/V]
SIM_RESONANCES = [             # List of (frequency, Q, gain_multiplier)
    (150.0, 15.0, 3.0),        # 150 Hz resonance, Q=15, 3x gain
    (800.0, 25.0, 2.5),        # 800 Hz resonance, Q=25, 2.5x gain
]
SIM_NOISE_LEVEL = 0.02         # Measurement noise level [g RMS]
SIM_DELAY_SAMPLES = 5          # System delay in samples
SIM_NONLINEARITY = 0.05        # Nonlinearity factor (0 = linear)

# ============= DISPLAY PARAMETERS =============
PLOT_SMOOTH_ALPHA = 0.3        # PSD smoothing factor
PLOT_UPDATE_EVERY = 1         # Update plots every N cycles
CONSOLE_UPDATE_INTERVAL = 5.0  # Console output interval [seconds]

# Real-time viewer tuning
REALTIME_PSD_UPDATE_STRIDE = 5  # Compute PSD for response channels every N blocks


AO_SYNC_WITH_AI = False      # Attempt to share AI sample clock with AO

