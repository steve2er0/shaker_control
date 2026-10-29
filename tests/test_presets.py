import json
import pytest
import workspace_ui


def select_file(monkeypatch,path):
    monkeypatch.setattr(workspace_ui.QFileDialog,'getOpenFileName',lambda *args: (str(path),'JSON'))


def test_preset_roundtrip_preserves_imported_original_and_simulation_labels(window,tmp_path,monkeypatch):
    panel = window.config_tab
    original = tmp_path/'source.json'
    monkeypatch.setattr(workspace_ui.QFileDialog,'getSaveFileName',lambda *args: (str(original),'JSON'))
    panel.save_preset()
    source_bytes = original.read_bytes()
    window.backend_combo.setCurrentIndex(0)
    assert panel.apply_config()
    assert window.settings.num_input_channels == 4
    assert len(window.shared_config.input_channel_labels) == 4
    select_file(monkeypatch,original)
    panel.load_preset()
    assert window.settings.fs == 8000
    assert window.settings.num_input_channels == 4
    assert window.backend_combo.currentData() is True
    assert window.settings.preset_source == str(original.resolve())
    assert not panel.dirty
    assert original.read_bytes() == source_bytes


@pytest.mark.parametrize('payload',[
    [], {'schema_version':9,'settings':{}}, {'settings':[]},
    {'settings':{'fs':1000}}, {'settings':{'simulation_mode':'False'}},
    {'settings':{'input_channel_labels':'one'}},
    {'settings':{'sim_resonances':[[100,'bad',1]]}},
    {'settings':{'eq_gain_limits':[-1,2]}},
    {'settings':{'test_mode':'sine_sweep','eq_gain_limits':[1,float('nan')]}},
])
def test_invalid_preset_is_rejected_atomically(window,tmp_path,monkeypatch,payload):
    panel = window.config_tab
    before = window.shared_config.as_dict()
    source = tmp_path/'source.json'
    source.write_text(json.dumps(payload))
    original = source.read_bytes()
    select_file(monkeypatch,source)
    panel.load_preset()
    assert 'Preset not loaded' in panel.validation_label.text()
    assert window.shared_config.as_dict() == before
    assert source.read_bytes() == original


def test_valid_imported_values_are_shown_without_clamping(window,tmp_path,monkeypatch):
    source = tmp_path/'high-rate.json'
    source.write_text(json.dumps({'settings':{'fs':200000.123456,'welch_nperseg':16000}}))
    select_file(monkeypatch,source)
    window.config_tab.load_preset()
    assert window.settings.fs == window.config_tab.fs_spin.value() == 200000.123456
    assert window.settings.welch_nperseg == window.config_tab.welch_nperseg_spin.value() == 16000
    assert not window.config_tab.dirty
