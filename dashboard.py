"""
Live PSD Dashboard for Random Vibration Control

This module provides real-time visualization of:
- PSD measurements vs target curves
- Control system metrics
- Equalizer gains
- Safety system status
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque


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
        
        # Create bar plot for equalizer gains with narrow bars for many bands
        # Calculate bar width based on number of bands (narrower for more bands)
        num_bands = len(eq_info['centers'])
        if num_bands > 20:
            width_factor = 0.1  # Very narrow for many bands (36+)
        elif num_bands > 12:
            width_factor = 0.2   # Narrow for moderate bands (12-20)
        else:
            width_factor = 0.3   # Normal width for few bands (<12)
        
        bar_widths = eq_info['centers'] * width_factor
        # Use higher alpha for narrow bars to maintain visibility
        alpha_val = 0.9 if num_bands > 20 else 0.7
        self.eq_bars = self.ax_eq.bar(eq_info['centers'], eq_info['gains'], 
                                     width=bar_widths, alpha=alpha_val, color='C4', 
                                     edgecolor='none')  # Remove edges for cleaner look
        
        # Add horizontal line at unity gain
        self.ax_eq.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Unity gain')
        self.ax_eq.legend(loc='best')
        self.ax_eq.set_title('Multi-Band Equalizer Gains')

        self._update_title()

        self.fig.tight_layout()

    def _update_title(self, a_rms_measured=None):
        # Simple title with target and measured g-RMS
        if a_rms_measured is not None:
            self.ax_psd.set_title(
                f"Target g-RMS: {self.a_rms_target:.3g} g  |  "
                f"Measured g-RMS: {a_rms_measured:.3g} g"
            )
        else:
            self.ax_psd.set_title(
                f"Target g-RMS: {self.a_rms_target:.3g} g"
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
        
        # Update title with live measured g-RMS
        self._update_title(a_rms_meas)

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
