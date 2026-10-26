# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from vara_timing import intervals, attribute

LOG = '''recording receive audio to /recordings/session.wav
timing event=key_return mono_s=100.1 key_seq=1 on=1 ok=1
timing event=play_start mono_s=100.12 key_seq=1 capture_samples=480000
timing event=play_return mono_s=102.0 key_seq=1 capture_samples=570240
timing event=key_return mono_s=102.02 key_seq=1 on=0 ok=1
'''


def test_ptt_uses_recording_sample_clock_not_wall_clock_or_filename():
    tx,note=intervals(LOG,'/moved/session.flac','rig rx',30)
    assert len(tx)==1
    assert abs(tx[0]['t0']-9.98)<1e-8 and abs(tx[0]['t1']-11.9)<1e-8
    assert tx[0]['direction']=='tx' and tx[0]['role']=='caller'
    assert 'Software' in note
    assert intervals(LOG,'other.wav','rig rx',30)[0]==[]
    assert intervals(LOG,'session.wav','rig tap',30)[0]==[]
    assert intervals(LOG.replace('on=1 ok=1','on=1 ok=0'),'session.wav','rig rx',30)[0]==[]


def test_roles_keyings_and_boundaries_are_distinct():
    tx,_=intervals(LOG,'session.wav','rig rx',30)
    events=[{'t0':10.2,'t1':11.5}, {'t0':12.3,'t1':14},
            {'t0':11.8,'t1':12.1}, {'t0':1,'t1':2},
            {'t0':1,'t1':1,'role':'caller'}, {'t0':3,'t1':3,'role':'called'}]
    attribute(events,tx)
    assert [e['direction'] for e in events]==['tx','rx','observed','observed','tx','rx']
    assert 'assumed' in events[1]['attribution']


def test_concatenated_logs_do_not_mix_sample_counters():
    other=LOG.replace('session.wav','other.wav').replace('capture_samples=480000','capture_samples=48000')
    tx,_=intervals(other+LOG,'session.wav','rig rx',30)
    assert len(tx)==1 and abs(tx[0]['t0']-9.98)<1e-8
    assert intervals(LOG+LOG,'session.wav','rig rx',30)[0]==[]


def test_receive_frame_with_near_edge_preamble_is_not_lost():
    tx,_=intervals(LOG,'session.wav','rig rx',30)
    events=[{'t0':12.,'t1':16.}, {'t0':11.89,'t1':11.91}]
    attribute(events,tx)
    assert events[0]['direction']=='rx'
    assert events[1]['direction']=='observed'


def test_original_tx_description_stays_with_its_keying():
    tx,_=intervals(LOG+'  tx DATA over #3 at record 5 (131 payload bytes)\n','session.wav','rig rx',30)
    assert tx[0]['kind']=='tx DATA over #3 at record 5'
    assert tx[0]['evidence_type']=='ptt'


def test_host_tx_survives_missing_audio_alignment_and_failed_decode():
    events=[{'t0':1,'t1':2,'evidence_type':'ptt','role':'unknown'},
            {'t0':3,'t1':4,'transmit_label':'SL1 ENTRY','status':'CRC failed','role':'called'}]
    attribute(events,[])
    assert all(e['direction']=='tx' and e['tx_confirmed'] for e in events)


def test_participating_qso_has_two_senders_even_next_to_ptt_edges():
    tx,_=intervals(LOG,'session.wav','rig rx',30)
    events=[{'t0':10.2,'t1':11.5}, {'t0':11.95,'t1':12.07},
            {'t0':11.8,'t1':12.1}, {'t0':1,'t1':2},
            {'t0':11.89,'t1':11.91,'role':'called','kind':'PACTOR-3 SL1 header'}]
    attribute(events,tx,qso=True)
    assert [e['direction'] for e in events]==['tx','rx','rx','observed','rx']
    assert events[3]['in_qso'] is False
    assert all('assumed' not in e['attribution'] for e in events)


def test_connected_replay_assigns_boundary_frame_to_the_peer(tmp_path, monkeypatch):
    import json
    import numpy as np
    import vara_view as view
    from types import SimpleNamespace

    recording = tmp_path / 'receiver.wav'
    recording.write_bytes(b'synthetic recording')
    book = SimpleNamespace(station={'call': 'N0CALL'}, served_path=lambda name: recording)
    entry = {'id': 'synthetic-qso', 'protocol': 'vara', 'call': 'N1TEST',
             'started_utc': '2026-10-10T00:00:00Z', 'bw_hz': 2300,
             'reached': ['connect'], 'audio': [{'kind': 'rig rx', 'url': '/file/receiver.wav'}]}
    tx = [{'id': 'tx-1', 't0': 1., 't1': 2., 'evidence_type': 'ptt',
           'direction': 'tx', 'role': 'caller', 'status': 'logged PTT interval',
           'kind': 'local control', 'symbols': None}]
    monkeypatch.setattr(view, 'load_audio', lambda p: (np.zeros(4 * view.FS), view.FS))
    monkeypatch.setattr(view, 'waterfall', lambda *a: {})
    monkeypatch.setattr(view, 'signal_view', lambda *a: None)
    monkeypatch.setattr(view, 'source_fingerprint', lambda: ('synthetic-source', {}))
    monkeypatch.setattr(view.vara_timing, 'intervals', lambda *a: (tx, 'synthetic clock'))
    monkeypatch.setenv('GIT_DIR', str(tmp_path / 'not-a-checkout'))
    monkeypatch.setattr(view, 'replay', lambda *a, **kw: [
        {'id': 'peer', 't0': 1.95, 't1': 2.07, 'kind': 'peer control',
         'status': 'decoded control', 'symbols': None, 'frame_type': 'observation'}])
    artifact = view.build(book, entry, tmp_path / 'replays')
    events = json.loads((artifact / 'events.json').read_text())
    peer = next(e for e in events if e['id'] == 'peer')
    assert peer['direction'] == 'rx' and peer['role'] == 'called'
    assert 'Peer transmission' in peer['attribution']
    assert json.loads((artifact / 'manifest.json').read_text())['decoder']['commit'] is None
