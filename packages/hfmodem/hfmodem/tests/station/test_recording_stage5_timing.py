# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Selected timing uses paired source-clock edges and retained predecessor state."""
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from hfmodem.core.tx_events import TxEvents
from hfmodem.tests.station.recording_stage5_support import (
    RATE, START, entry, reader, save_selection,
)


@pytest.mark.parametrize('module,protocol', [
    ('tx_timing', 'ARDOP'), ('vara_timing', 'VARA'),
    ('pactor_timing', 'PACTOR'), ('ardop_view', 'ARDOP'),
])
@pytest.mark.parametrize('kind', ['rig rx', 'rig tap'])
def test_timing_matches_identity_clips_absolute_edges_and_rebases_once(tmp_path, module, protocol, kind):
    selected = entry(tmp_path, protocol=protocol.lower())
    path = save_selection(tmp_path, selected)
    journal = TxEvents(protocol, session_id='run')
    journal.record('completed', 2, outcome='sent', recording='/old/name/receive.wav',
        sample_rate=RATE, source_id='source-rx', clock_generation=2,
        ptt_start_sample=7996, ptt_end_sample=8008, audio_start_sample=7998, audio_end_sample=8006)
    journal.record('waveform', 2, outcome='sent', label='selected TX', modulation='4PSK')
    journal.record('completed', 99, outcome='sent', recording=str(path),
        sample_rate=RATE, source_id='other-source', clock_generation=1,
        ptt_start_sample=8001, ptt_end_sample=8002)
    module = reader('logbook_view.' + module)
    args = ('\n'.join(journal.lines()), path, kind, 4)
    kwargs = {'selection': selected['recording']}
    if module.__name__.endswith('tx_timing'):
        args += (protocol,)
    elif not module.__name__.endswith('ardop_view'):
        kwargs['fs'] = RATE
    result = module.intervals(*args, **kwargs)
    events = result if isinstance(result, list) else result[0]
    assert len(events) == 1
    [event] = events
    assert (event['t0'], event['t1']) == (0, 1)
    assert event['original_bounds']['ptt_start_sample'] == 7996
    assert event['original_bounds']['ptt_end_sample'] == 8008
    assert event['source_offset_sample'] == START
    assert event['role'] == 'called' and event['kind'] == 'selected TX'


def test_ardop_party_evaluates_selected_id_and_absolute_epoch_at_each_start(tmp_path):
    selected = entry(tmp_path)
    besra = reader('tools.rehear.besra')
    log = SimpleNamespace(session=0x11, session_span=(0, 1), epochs=[0],
                          selection=selected['recording'], context=selected['recording']['context'])
    party = besra._Party.of(log, START / RATE)
    for at, expected, epoch in [(0, 0x22, 4), (2.1, 0x22, 5), (3.1, None, 5)]:
        party.at = at
        assert party.expect() == expected
        assert party.epoch() == epoch


def test_ardop_grading_counts_only_available_selected_frames(tmp_path):
    module = reader('tools.rehear.besra')
    sessionlog = reader('hfmodem.station.sessionlog')
    selected = entry(tmp_path, gaps=[[8010, 8016]])
    link = selected['recording']['context']['transcript']
    transcript = tmp_path / link['path']
    data = transcript.read_bytes().replace(b'2026-11-01 12:16:42,000',
        b'2026-11-01 12:16:41,500 INFO RX IDLE sess=0x22 ok=True\n2026-11-01 12:16:42,000')
    transcript.write_bytes(data)
    link['range'][1] = len(data)
    link['sha256'] = hashlib.sha256(data[link['range'][0]:]).hexdigest()
    path = save_selection(tmp_path, selected)
    chosen = sessionlog.select(selected['recording'], path, root=tmp_path)
    assert chosen.complete
    assert [f.at for f in chosen.grading] == [1001., 1002.]
    inbound = [(1., SimpleNamespace(name='IDLE')), (1.5, SimpleNamespace(name='IDLE')),
               (2., SimpleNamespace(name='END'))]
    assert module._against(chosen.log, 1000, inbound) == (2, 2)


@pytest.mark.parametrize('edge', ['none', 'clipped'])
def test_ardop_replay_alignment_comes_from_context_without_a_full_tx_edge(tmp_path, monkeypatch, edge):
    module = reader('logbook_view.ardop_view')
    selected = entry(tmp_path)
    observed = []
    class Decoder:
        def __init__(self, **kwargs):
            observed.append((kwargs['expect_session'](), kwargs['rx_epoch']()))
    class Rolling:
        def __init__(self, *args, **kwargs): pass
        def push(self, *args): pass
        def pump(self): pass
        def flush(self): pass
    monkeypatch.setattr(module, 'Demodulator', Decoder)
    monkeypatch.setattr(module, 'RollingDecoder', Rolling)
    tx = [] if edge == 'none' else [{'log_t0': 999.5, 't0': 0., 't1': 1.}]
    log = SimpleNamespace(session=0x11, session_span=(0, 2000), epochs=[0],
                          context=selected['recording']['context'], selection=selected['recording'])
    module.replay(np.zeros(4800), 500, ['W0QXR', 'N0CALL'], tmp_path, tx=tx, log=log,
                  progress=lambda *a, **k: None)
    assert observed == [(0x22, 4)]


def test_selected_sessionlog_read_uses_bound_transcript_range(tmp_path):
    selected = entry(tmp_path)
    module = reader('hfmodem.station.sessionlog')
    path = tmp_path / selected['recording']['context']['transcript']['path']
    log = module.read(path, selection=selected['recording'])
    assert log.session == 0x22 and log.bw == '500'
    assert len(log.frames) == 3 and all(f.session == 0x22 for f in log.frames)
    assert log.session_span[0] <= START / RATE


def test_vara_bandwidth_and_roles_come_from_selected_context(tmp_path):
    selected = entry(tmp_path, protocol='vara')
    path = save_selection(tmp_path, selected)
    module = reader('tools.rehear.kestrel')
    log = SimpleNamespace(path=tmp_path / 'working/run/arm.log', bw='500',
                          selection=selected['recording'], context=selected['recording']['context'])
    assert module._bw(path, log, None) == '2750'
    assert module._dialled(path, selection=selected['recording']) == ('W0QXR', 'N0CALL')


def test_vara_renamed_cutout_calls_are_context_roles(tmp_path):
    selected = entry(tmp_path, protocol='vara')
    path = save_selection(tmp_path, selected)
    module = reader('tools.rehear.kestrel')
    assert module._dialled(path, selection=selected['recording']) == ('W0QXR', 'N0CALL')


def test_pactor_grid_selection_keeps_predecessor_and_rejects_other_clock_generation(tmp_path):
    selected = entry(tmp_path, protocol='pactor')
    module = reader('hfmodem.station.sessionlog')
    path = tmp_path / 'working/run/arm.log'
    with path.open('a') as out:
        out.write('[grid] CS due @ 8020\n[grid] CS due @ 10\n')
    grid = module.shrike_grid(path, RATE, selection=selected['recording'])
    assert grid.aimed == (8020,)
    assert grid.predecessor['sample'] == 7992
    assert grid.clock_generation == 2


def test_pactor_rehear_uses_context_bounds_even_when_cutout_has_no_window_name(tmp_path, monkeypatch):
    selected = entry(tmp_path, protocol='pactor')
    path = save_selection(tmp_path, selected)
    module = reader('tools.rehear.shrike')
    from hfmodem.tests.station.recording_stage5_support import write_json
    write_json(path.with_suffix('.json'), {'recording': selected['recording'], 'samples': 32,
        'end_stream_sample': 8032, 'lost_samples': 0, 'grid_loss_seen': False,
        'xruns': 0, 'peak': .1, 'rms': .01})
    monkeypatch.setattr(module.rxfront, 'load_wav', lambda *a: np.zeros(32))
    monkeypatch.setattr(module, 'FS', RATE)
    for name in ('_rolling', '_ungated', '_anchored'):
        monkeypatch.setattr(module, name, lambda *a: [])
    def legacy_grid(*a, **k):
        assert k.get('selection') == selected['recording'], 'PACTOR read the whole parent transcript grid'
        return SimpleNamespace(within=lambda *a: ([8020], []))
    monkeypatch.setattr(module.sessionlog, 'shrike_grid', legacy_grid)
    log = SimpleNamespace(path=tmp_path / 'working/run/arm.log', selection=selected['recording'])
    result = module.rehear(path, log)
    assert result.verdicts is not None
