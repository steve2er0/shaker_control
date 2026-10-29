from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from run_settings import RunSettings, ConfigurationError, validate_settings, parse_points
from sine_sweep import LogSweepStepper


@pytest.mark.parametrize('changes,field', [
    ({'fs':1000},'target_psd_points'),
    ({'fs':1040},'target_psd_points'),
    ({'fs':np.nan},'fs'),
    ({'fs':np.inf},'fs'),
    ({'fs':'bad'},'fs'),
    ({'buf_seconds':.05},'buf_seconds'),
    ({'welch_nperseg':1024},'welch_nperseg'),
    ({'welch_nperseg':2},'welch_nperseg'),
    ({'target_psd_points':[(100,1e308),(500,1e308)]},'target_psd_points'),
    ({'initial_level_fraction':2.01},'initial_level_fraction'),
    ({'max_rms_volts':2},'max_rms_volts'),
    ({'simulation_mode':'false'},'simulation_mode'),
    ({'test_mode':'sine_sweep','sine_initial_level':1.01},'sine_initial_level'),
    ({'test_mode':'sine_sweep','sine_end_hz':4000},'sine_end_hz'),
    ({'target_psd_points':[(100,.001),(90,.01)]},'target_psd_points'),
])
def test_invalid_combination_is_rejected_before_a_run(settings_data,changes,field):
    settings_data.update(changes)
    assert field in validate_settings(settings_data)
    with pytest.raises(ConfigurationError):
        RunSettings.capture(settings_data)


def test_snapshot_is_detached_and_immutable(settings_data):
    snapshot = RunSettings.capture(settings_data)
    original_id = snapshot.config_id
    settings_data['fs'] = 12000
    settings_data['target_psd_points'][0] = (120,.02)
    assert snapshot.fs == 8000
    assert snapshot.target_psd_points[0] == (100,.001)
    assert snapshot.config_id == original_id
    with pytest.raises(TypeError):
        snapshot.values['fs'] = 123
    with pytest.raises(TypeError):
        snapshot.target_psd_points[0] = (20,.01)
    with pytest.raises(FrozenInstanceError):
        snapshot.values = {}


def test_hardware_channels_and_simulation_channels_are_derived_consistently(settings_data):
    assert RunSettings.capture(settings_data).num_input_channels == 4
    settings_data['simulation_mode'] = False
    snapshot = RunSettings.capture(settings_data)
    assert snapshot.num_input_channels == 4
    assert snapshot.input_channel_labels == ('Control Accel', 'Response 1', 'Response 2', 'Response 3')
    assert [channel['physical_channel'] for channel in snapshot.hardware_input_channels] == [f'Dev1/ai{i}' for i in range(4)]
    settings_data['response_channels'] = []
    assert RunSettings.capture(settings_data).num_input_channels == 1


def test_zero_pi_gains_and_legal_level_boundaries_are_supported(settings_data):
    settings_data.update(kp=0,ki=0,initial_level_fraction=2)
    assert not validate_settings(settings_data)


def test_sine_mode_does_not_require_a_random_band_below_nyquist(settings_data):
    settings_data.update(test_mode='sine_sweep',fs=1000,block_seconds=1,
                         sine_start_hz=20,sine_end_hz=100)
    assert not validate_settings(settings_data)


def test_preset_numbers_are_normalized_and_bad_inactive_numbers_are_rejected(settings_data):
    settings_data['fs'] = '8000'
    assert RunSettings.capture(settings_data).fs == 8000.0
    settings_data['sine_step_dwell'] = 'bad'
    with pytest.raises(ConfigurationError):
        RunSettings.capture(settings_data)


def test_target_integral_uses_power_law_including_inverse_frequency(settings_data):
    settings_data['target_psd_points'] = [(100,.01),(200,.005)]
    assert RunSettings.capture(settings_data).target_rms_g == pytest.approx(np.sqrt(np.log(2)))


def test_point_parser_reports_bad_rows():
    assert parse_points('100, .001\n200 .002 # comment') == [(100,.001),(200,.002)]
    with pytest.raises(ValueError,match='Line 2'):
        parse_points('100 .001\n200 nope')


@pytest.mark.parametrize('start,end',[ (20,2000),(2000,20),(100,100) ])
def test_sweeps_include_exact_endpoints_and_complete(start,end):
    stepper = LogSweepStepper(start,end,3)
    frequencies = [stepper.current_frequency()]
    while stepper.advance():
        frequencies.append(stepper.current_frequency())
    assert frequencies[0] == start
    assert frequencies[-1] == end
    assert stepper.is_finished()


def test_repeating_sweep_restarts_after_endpoint():
    stepper = LogSweepStepper(100,110,1,repeat=True)
    assert stepper.advance() and stepper.current_frequency() == 110
    assert stepper.advance() and stepper.current_frequency() == 100
