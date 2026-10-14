"""
Random Vibration Controller

This module contains:
- Multi-band equalizer for frequency shaping
- PI control system for target tracking
- Safety limiters (crest factor and RMS)
- Signal processing utilities
"""

import numpy as np
from scipy.signal import welch, firwin, lfilter, butter, sosfilt
from scipy.interpolate import interp1d
import time


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
            
            # Create a narrow bandpass filter around the resonance
            bandwidth = high_freq - low_freq
            if bandwidth > 1.0:  # Ensure reasonable bandwidth
                # Design 4th order Butterworth bandpass filter
                sos = butter(2, [low_freq, high_freq], btype='bandpass', fs=self.fs, output='sos')
                self.band_filters.append(sos)
            else:
                # Skip very narrow bands
                self.band_filters.append(None)
    
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
            if sos is None:  # Skip invalid filters
                continue
                
            try:
                # Filter signal through this band
                band_signal = sosfilt(sos, signal)
                
                # Check for NaN in band signal
                if np.any(~np.isfinite(band_signal)):
                    continue  # Skip this band
                
                # Apply gain and add to output
                equalized_signal += gain * band_signal
                
            except Exception:
                continue  # Skip problematic band
        
        # Final check for NaN in output
        if np.any(~np.isfinite(equalized_signal)):
            return signal  # Return original signal if output is bad
            
        return equalized_signal
    
    def update_gains(self, f_measured, psd_measured, target_psd_func, adapt_weight=1.0):
        """Update equalizer gains based on measured vs target PSD."""
        adapt_weight = float(np.clip(adapt_weight, 0.0, 1.0))
        if adapt_weight <= 0.0:
            return

        target_psd = target_psd_func(f_measured)

        # Calculate gain adjustments for each band
        new_gains = self.gains.copy()
        effective_adapt = self.adapt_rate * adapt_weight

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

                # Apply adaptation rate scaled by current weight
                gain_update = 1.0 + effective_adapt * (desired_gain_adjustment - 1.0)
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


class RandomVibrationController:
    def __init__(self, target_psd_points, fs, Kp=0.5, Ki=0.1, 
                 max_level_fraction_rate=0.5, welch_nperseg=2048):
        self.target_psd_points = target_psd_points
        self.fs = fs
        self.Kp = Kp
        self.Ki = Ki
        self.max_level_fraction_rate = max_level_fraction_rate
        self.welch_nperseg = welch_nperseg
        self.welch_noverlap = welch_nperseg // 2
        
        # Frequency band limits
        self.f1 = target_psd_points[0][0]
        self.f2 = target_psd_points[-1][0]
        
        # Create target PSD function
        self.target_psd_func = self._create_target_psd_function(target_psd_points)
        
        # Calculate target RMS
        self.a_rms_target = self._calculate_target_rms(target_psd_points)
        
        # Control state
        self.integ = 0.0
        self.t_last = time.time()
        
        print(f"Controller: target RMS = {self.a_rms_target:.4g} g, band [{self.f1:.0f}, {self.f2:.0f}] Hz")
    
    def _create_target_psd_function(self, target_points):
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
    
    def _calculate_target_rms(self, target_points):
        """Calculate target RMS by integrating the PSD profile"""
        try:
            # Create a fine frequency grid for integration
            f_fine = np.logspace(np.log10(target_points[0][0]), np.log10(target_points[-1][0]), 1000)
            psd_fine = self.target_psd_func(f_fine)
            
            # Check for NaN in PSD values
            if np.any(~np.isfinite(psd_fine)):
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
    
    def estimate_psd(self, x):
        """Estimate PSD and return metrics"""
        # Welch PSD in g^2/Hz; x must already be in g
        if len(x) == 0 or np.all(np.abs(x) < 1e-12):
            # Return small but valid values instead of None to avoid NaN
            f_dummy = np.linspace(self.f1, self.f2, 100)
            return f_dummy, np.full_like(f_dummy, 1e-8), (1e-8, 1e-8, 1e-4)
        
        f, Pxx = welch(x, fs=self.fs, nperseg=self.welch_nperseg, noverlap=self.welch_noverlap, detrend='constant')
        # Compute metrics over [f1,f2] band
        band = (f >= self.f1) & (f <= self.f2)
        if not np.any(band):
            return None, None, None
        
        # Get target PSD for measured frequencies
        target_psd = self.target_psd_func(f)
        
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
    
    def update_control(self, S_avg_meas, S_avg_target, level_fraction):
        """Update PI control and return new level fraction"""
        # PI control on average PSD level (target vs measured)
        err = S_avg_target - S_avg_meas
        now = time.time()
        dt = max(1e-3, now - self.t_last)
        self.t_last = now
        self.integ += err * dt
        level_fraction_target = level_fraction + self.Kp*err + self.Ki*self.integ

        # Constraints
        level_fraction_target = max(0.0, min(2.0, level_fraction_target))
        level_fraction = self.slew_limit(level_fraction, level_fraction_target, self.max_level_fraction_rate, dt)
        
        return level_fraction
    
    @staticmethod
    def slew_limit(current, target, max_rate, dt):
        """Apply slew rate limiting"""
        delta = target - current
        max_delta = max_rate * dt
        if delta > max_delta:
            return current + max_delta
        elif delta < -max_delta:
            return current - max_delta
        else:
            return target


def create_bandpass_filter(f1, f2, fs, numtaps=1025):
    """Create FIR bandpass filter"""
    bp = firwin(numtaps, [f1, f2], pass_zero=False, fs=fs)
    return bp


def make_bandlimited_noise(n, bandpass_filter, equalizer):
    """Generate band-limited noise with equalization"""
    # white -> bandpass -> equalized
    w = np.random.randn(n)
    y = lfilter(bandpass_filter, [1.0], w)
    
    # Check for NaN after bandpass filter
    if np.any(~np.isfinite(y)):
        y = np.random.randn(n) * 0.1
    
    # Apply multi-band equalization
    y_eq = equalizer.apply_equalization(y)
    
    # Check for NaN after equalization
    if np.any(~np.isfinite(y_eq)):
        y_eq = y  # Fall back to non-equalized signal
    
    return y_eq


def _finite_waveform(signal):
    """Validate the single-channel waveform expected by the output limiters."""
    if np.iscomplexobj(signal):
        raise ValueError("Signal must be real-valued")
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or not np.all(np.isfinite(signal)):
        raise ValueError("Signal must be a finite one-dimensional waveform")
    return signal


def _waveform_metrics(signal):
    """Return true RMS (including DC) and crest factor without squaring huge values."""
    peak = np.max(np.abs(signal)) if signal.size else 0.0
    if peak == 0.0:
        return 0.0, 0.0
    scaled_rms = np.sqrt(np.mean((signal / peak) ** 2))
    return float(peak * scaled_rms), float(1.0 / scaled_rms)


def crest_factor_limit(signal, max_cf=4.0, soft_knee=0.8):
    """Compress individual peaks to limit peak / output RMS.

    ``max_cf`` must be finite and at least 1; ``soft_knee`` is in [0, 1]
    (1 selects hard clipping). Acceptable waveforms are unchanged. Compression
    never increases any sample's magnitude and does not modify the input.

    RMS is sqrt(mean(signal**2)), including DC. After soft compression, a
    bounded clipping search accounts for the reduced output RMS. If a sparse
    waveform cannot meet the limit by attenuation, return a zero block rather
    than claim a false reduction. Empty and zero blocks have crest factor 0.

    Returns (limited_signal, measured_cf_before, measured_cf_after, active).
    """
    signal = _finite_waveform(signal)
    if not np.isfinite(max_cf) or max_cf < 1.0:
        raise ValueError("max_cf must be finite and at least 1")
    if not np.isfinite(soft_knee) or not 0.0 <= soft_knee <= 1.0:
        raise ValueError("soft_knee must be finite and in [0, 1]")

    _, current_cf = _waveform_metrics(signal)
    if current_cf <= max_cf:
        return signal, current_cf, current_cf, False

    # Work relative to the peak so very large or small voltages remain stable.
    peak = np.max(np.abs(signal))
    magnitudes = np.abs(signal / peak)
    ceiling = max_cf / current_cf
    knee = ceiling * soft_knee
    compressed = magnitudes.copy()
    if soft_knee == 1.0:
        compressed = np.minimum(compressed, ceiling)
    else:
        above_knee = magnitudes > knee
        width = ceiling - knee
        compressed[above_knee] = knee + width * np.tanh(
            (magnitudes[above_knee] - knee) / width
        )

    if _waveform_metrics(compressed)[1] > max_cf:
        nonzero = compressed[compressed > 0.0]
        # Even equal nonzero magnitudes have CF = sqrt(total / nonzero samples).
        # A single impulse cannot be repaired by making that same impulse smaller.
        if np.sqrt(signal.size / nonzero.size) > max_cf:
            return np.zeros_like(signal), current_cf, 0.0, True

        # The lower bound equalizes all nonzero magnitudes and is feasible.
        # Choose the largest feasible clipping threshold to retain signal energy.
        low, high = float(np.min(nonzero)), float(np.max(nonzero))
        for _ in range(64):
            threshold = low + (high - low) / 2.0
            candidate = np.minimum(compressed, threshold)
            if _waveform_metrics(candidate)[1] <= max_cf:
                low = threshold
            else:
                high = threshold
        compressed = np.minimum(compressed, low)

    limited_signal = np.sign(signal) * compressed * peak
    _, new_cf = _waveform_metrics(limited_signal)
    # Guard rescaling underflow or other floating-point edge cases as well.
    if new_cf > max_cf * (1.0 + 1e-12):
        return np.zeros_like(signal), current_cf, 0.0, True
    return limited_signal, current_cf, new_cf, True


def rms_limit(signal, max_rms, headroom=0.9):
    """
    Limit RMS level of signal
    
    Args:
        signal: Finite one-dimensional input waveform
        max_rms: Finite nonnegative true RMS limit (including DC)
        headroom: Scale factor in [0, 1] when limiting is active
    
    Returns:
        tuple: (limited_signal, rms_before, rms_after, limiting_active)
    """
    signal = _finite_waveform(signal)
    if not np.isfinite(max_rms) or max_rms < 0.0:
        raise ValueError("max_rms must be finite and nonnegative")
    if not np.isfinite(headroom) or not 0.0 <= headroom <= 1.0:
        raise ValueError("headroom must be finite and in [0, 1]")

    current_rms, _ = _waveform_metrics(signal)
    
    if current_rms <= max_rms:
        return signal, current_rms, current_rms, False
    
    # Scale down to target RMS with headroom
    target_rms = max_rms * headroom
    scale_factor = target_rms / current_rms
    limited_signal = signal * scale_factor
    
    rms_after, _ = _waveform_metrics(limited_signal)
    return limited_signal, current_rms, rms_after, True


def apply_safety_limiters(signal, max_cf=4.0, max_rms=1.5, cf_soft_knee=0.8, rms_headroom=0.9):
    """
    Apply both crest factor and RMS limiting to signal
    
    Returns:
        tuple: (limited_signal, limiter_stats). Output metrics describe the final
        waveform after both limiters, before any separate absolute voltage clip.
    """
    # Apply RMS limiting first
    signal_rms_limited, rms_before, _, rms_limiting = rms_limit(signal, max_rms, rms_headroom)
    
    # Then apply crest factor limiting
    signal_final, cf_before, cf_after, cf_limiting = crest_factor_limit(
        signal_rms_limited, max_cf, cf_soft_knee)
    
    limiter_stats = {
        'rms_before': rms_before,
        'rms_after': _waveform_metrics(signal_final)[0],
        'rms_limiting': rms_limiting,
        'cf_before': cf_before,
        'cf_after': cf_after,
        'cf_limiting': cf_limiting,
        'any_limiting': rms_limiting or cf_limiting
    }
    
    return signal_final, limiter_stats

