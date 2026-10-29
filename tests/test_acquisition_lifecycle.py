"""NI interface doubles verify lifecycle contracts without loading a device."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
import app
from conftest import wait_until


@pytest.mark.parametrize('fail_shared_clock',[False,True])
def test_requested_clock_routing_and_finite_io(window,monkeypatch,fail_shared_clock):
    tasks, writes, reads = [], [], []

    def configure_trigger(*args):
        if fail_shared_clock:
            raise RuntimeError('fixture unsupported shared trigger')

    class Task:
        def __init__(self):
            self.started, self.closed = False, False
            self.ai_channels = SimpleNamespace(add_ai_accel_chan=lambda **kwargs: SimpleNamespace())
            self.ao_channels = SimpleNamespace(add_ao_voltage_chan=lambda **kwargs: None)
            self.timing = SimpleNamespace(cfg_samp_clk_timing=lambda **kwargs: None,samp_clk_term='/Fixture/ai/SampleClock')
            self.in_stream = SimpleNamespace(input_buf_size=0)
            self.out_stream = SimpleNamespace(output_buf_size=0)
            self.triggers = SimpleNamespace(start_trigger=SimpleNamespace(term='/Fixture/ai/StartTrigger',cfg_dig_edge_start_trig=configure_trigger))
            tasks.append(self)

        def start(self):
            self.started = True

        def stop(self):
            assert self.started, 'Unstarted tasks must only be closed'
            self.started = False

        def close(self):
            self.closed = True

    class Writer:
        def __init__(self,*args,**kwargs):
            pass

        def write_many_sample(self,data,timeout=None):
            writes.append(timeout)

    class Reader:
        def __init__(self,*args,**kwargs):
            pass

        def read_many_sample(self,data,number_of_samples_per_channel,timeout):
            reads.append(timeout)
            data[:] = .5*np.sin(2*np.pi*200*np.arange(number_of_samples_per_channel)/8000)

    monkeypatch.setattr(app,'nidaqmx',SimpleNamespace(Task=Task))
    monkeypatch.setattr(app,'NIDAQMX_AVAILABLE',True)
    monkeypatch.setattr(app,'AnalogSingleChannelWriter',Writer,raising=False)
    monkeypatch.setattr(app,'AnalogSingleChannelReader',Reader,raising=False)
    monkeypatch.setattr(app,'AnalogMultiChannelReader',Reader,raising=False)
    for name,values in {
        'AcquisitionType':['CONTINUOUS'], 'TerminalConfiguration':['PSEUDO_DIFF'],
        'ExcitationSource':['INTERNAL','NONE'], 'AccelUnits':['G'], 'Coupling':['AC'],
        'VoltageUnits':['VOLTS'], 'AccelSensitivityUnits':['MILLIVOLTS_PER_G'],
    }.items():
        monkeypatch.setattr(app,name,SimpleNamespace(**{value:value for value in values}),raising=False)
    window.backend_combo.setCurrentIndex(0)
    window.config_tab.fields['sync_ao_clock'].setChecked(True)
    assert window.config_tab.apply_config()
    assert window.start_controller()
    if not fail_shared_clock:
        wait_until(lambda: bool(window.controller_tab.history))
        window.stop_controller()
    wait_until(lambda: window.last_report is not None and not window.busy)
    assert len(tasks) == 2
    assert all(task.closed and not task.started for task in tasks)
    report = window.last_report
    if fail_shared_clock:
        assert report['outcome'] == 'fault'
        assert 'Requested AI/AO clock sharing failed' in report['error']
        assert report['backend'] is None
        assert not reads and not writes
    else:
        assert report['outcome'] == 'stopped'
        assert report['clock_status'] == 'Shared AI clock configured'
        assert writes[0] is None  # Initial zero buffer uses the API's finite default.
        assert writes[1:] and reads
        assert all(math.isfinite(timeout) and timeout > 0 for timeout in writes[1:]+reads)
