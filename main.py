#!/usr/bin/env python3
"""
Random Vibration Shaker Control System

Main execution file that orchestrates:
- Dashboard visualization
- Random vibration controller
- Plant simulation or real hardware
- Safety systems and monitoring

Usage:
    python main.py          # Run in simulation mode
    python main.py --hw     # Run with hardware (requires nidaqmx)
"""

import sys
import numpy as np
import time

# Import our modules
import config
from dashboard import LivePSDPlotter
from rv_controller import (MultiBandEqualizer, RandomVibrationController, 
                          create_bandpass_filter, make_bandlimited_noise, apply_safety_limiters)
from simulation import create_simulation_system

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


def main():
    # ============= LOAD CONFIGURATION =============
    
    # Import all settings from config file
    device_ai = config.DEVICE_AI
    device_ao = config.DEVICE_AO
    fs = config.FS
    buf_seconds = config.BUF_SECONDS
    block_seconds = config.BLOCK_SECONDS
    target_psd_points = config.TARGET_PSD_POINTS
    accel_mV_per_g = config.ACCEL_MV_PER_G
    ao_volt_limit = config.AO_VOLT_LIMIT
    initial_level_fraction = config.INITIAL_LEVEL_FRACTION
    max_level_fraction_rate = config.MAX_LEVEL_FRACTION_RATE
    welch_nperseg = config.WELCH_NPERSEG
    Kp = config.KP
    Ki = config.KI
    eq_num_bands = config.EQ_NUM_BANDS
    eq_gain_limits = config.EQ_GAIN_LIMITS
    eq_adapt_rate = config.EQ_ADAPT_RATE
    eq_smooth_factor = config.EQ_SMOOTH_FACTOR
    max_crest_factor = config.MAX_CREST_FACTOR
    max_rms_volts = config.MAX_RMS_VOLTS
    crest_soft_knee = config.CREST_SOFT_KNEE
    rms_limit_headroom = config.RMS_LIMIT_HEADROOM
    SIMULATION_MODE = config.SIMULATION_MODE
    
    # Check for hardware compatibility
    if not SIMULATION_MODE:
        if not NIDAQMX_AVAILABLE:
            print("ERROR: nidaqmx not available but SIMULATION_MODE=False")
            print("Installing nidaqmx: pip install nidaqmx")
            print("Or set SIMULATION_MODE=True in config.py")
            print("Forcing SIMULATION_MODE=True for now...")
            SIMULATION_MODE = True
        else:
            # nidaqmx is available, try to create a test task to check hardware
            try:
                test_task = nidaqmx.Task()
                test_task.close()
                print("✅ nidaqmx available and working - hardware mode enabled")
            except Exception as e:
                print(f"⚠️  nidaqmx installed but hardware not accessible: {e}")
                print("This is normal if:")
                print("  • You're on macOS (NI-DAQ doesn't support macOS)")
                print("  • No DAQ hardware is connected")
                print("  • DAQ drivers not installed")
                print("Forcing SIMULATION_MODE=True for now...")
                SIMULATION_MODE = True
    
    # Simulation parameters
    sim_plant_gain = config.SIM_PLANT_GAIN
    sim_resonances = config.SIM_RESONANCES
    sim_noise_level = config.SIM_NOISE_LEVEL
    sim_delay_samples = config.SIM_DELAY_SAMPLES
    sim_nonlinearity = config.SIM_NONLINEARITY
    
    # ============= SYSTEM INITIALIZATION =============
    
    # Derived parameters
    buf_samples = int(fs * buf_seconds)
    block_samples = int(fs * block_seconds)
    f1 = target_psd_points[0][0]    # First frequency point
    f2 = target_psd_points[-1][0]   # Last frequency point
    
    # Create FIR bandpass filter
    numtaps = 1025
    bp = create_bandpass_filter(f1, f2, fs, numtaps)
    
    # Create controller
    controller = RandomVibrationController(
        target_psd_points=target_psd_points,
        fs=fs,
        Kp=Kp,
        Ki=Ki,
        max_level_fraction_rate=max_level_fraction_rate,
        welch_nperseg=welch_nperseg
    )
    
    # Create equalizer
    equalizer = MultiBandEqualizer(
        f1=f1, f2=f2, num_bands=eq_num_bands, fs=fs,
        gain_limits=eq_gain_limits, adapt_rate=eq_adapt_rate,
        smooth_factor=eq_smooth_factor
    )
    
    # Create dashboard
    plotter = LivePSDPlotter(
        f1=f1, f2=f2, 
        target_psd_func=controller.target_psd_func, 
        target_psd_points=target_psd_points, 
        a_rms_target=controller.a_rms_target, 
        equalizer=equalizer, 
        smooth_alpha=config.PLOT_SMOOTH_ALPHA, 
        update_every=config.PLOT_UPDATE_EVERY
    )
    
    # ============= DAQ SETUP =============
    
    if SIMULATION_MODE:
        print("=== SIMULATION MODE ===")
        print("Running without DAQ hardware - using plant simulator")
        
        # Create simulation system
        ao_writer, ai_reader, ai_task, ao_task = create_simulation_system(
            fs, sim_plant_gain, sim_resonances, sim_noise_level, 
            sim_delay_samples, sim_nonlinearity,
            num_input_channels=len(getattr(config, "INPUT_CHANNEL_LABELS", ["Control Accel"]))
        )
        
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
                sensitivity=accel_mV_per_g/1000.0, # V/g
                sensitivity_units=nidaqmx.constants.AccelSensitivityUnits.MVOLTS_PER_G,
                units=AccelUnits.G
            )
            ai_ch.ai_coupling = Coupling.AC
            ai_ch.ai_excit_src = ExcitationSource.INTERNAL
            ai_ch.ai_excit_val = 0.004  # 4 mA typical IEPE current
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

        # ============= CONTROL LOOP =============
        
        level_fraction = initial_level_fraction
        
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
        
        t_last_print = time.time()  # Timer for console output
        
        while True:
            # Generate a block of drive noise and scale
            drive = make_bandlimited_noise(block_samples, bp, equalizer)
            if np.std(drive) > 1e-12:
                drive = drive / np.std(drive)  # unit RMS voltage profile

            # Target in-band acceleration RMS for this block
            target_rms_block = controller.a_rms_target * level_fraction
            
            # Safety check for NaN in key components
            if not np.isfinite(controller.a_rms_target):
                print("Warning: a_rms_target is NaN, using fallback")
                controller.a_rms_target = 0.1  # fallback value
                target_rms_block = controller.a_rms_target * level_fraction
            
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
            f, Pxx, metrics = controller.estimate_psd(data)
            if metrics is None:
                print("Welch band empty; check fs/f1/f2.")
                continue
            S_avg_meas, S_avg_target, a_rms_meas = metrics

            # Update plant gain estimate g/V using ratio of measured RMS to command RMS
            cmd_rms = np.std(volts_block)  # volts RMS sent
            if cmd_rms > 1e-9:
                new_gain = a_rms_meas / cmd_rms
                plant_gain_g_per_V = 0.8*plant_gain_g_per_V + 0.2*new_gain

            # Update equalizer gains based on measured vs target PSD
            equalizer.update_gains(f, Pxx, controller.target_psd_func)

            # Update PI control
            level_fraction = controller.update_control(S_avg_meas, S_avg_target, level_fraction)

            # Safety: if AO saturates frequently, back off
            sat_frac = np.mean(np.abs(volts_block) >= (0.98*ao_volt_limit))
            
            if sat_frac > 0.15:  # Allow higher saturation before backing off
                level_fraction = max(0.0, level_fraction * 0.9)  # Less aggressive backoff
                controller.integ *= 0.8  # Less integral reset
                print("AO near saturation; backing off level.")

            # Update live plots
            plotter.update(f, Pxx, S_avg_meas, S_avg_target, a_rms_meas, 
                          level_fraction, sat_frac, plant_gain_g_per_V, limiter_stats)

            # Print status at configured interval
            now_print = time.time()
            if (now_print - t_last_print) >= config.CONSOLE_UPDATE_INTERVAL:
                t_last_print = now_print
                
                eq_gains_str = f"[{', '.join([f'{g:.2f}' for g in equalizer.gains[:4]])}...]"
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
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    # Check for hardware mode flag
    if "--hw" in sys.argv:
        # This would set SIMULATION_MODE = False in a real implementation
        print("Hardware mode requested (not implemented in this demo)")
    
    main()
