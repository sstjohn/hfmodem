from vara_packets import consolidate


def event(name, lo, hi, **fields):
    return {'id':name,'kind':name,'t0':lo,'t1':hi,'status':'classified burst',
            'direction':'tx','origin':'test','detail':'test',**fields}


def test_one_packet_keeps_ptt_scanner_and_classifier_as_evidence():
    got=consolidate([event('tx connect-request-retry',10,12,evidence_type='ptt'),
                     event('CR',10.2,10.2),event('connect-response',10.2,11.9)])
    assert len(got)==1 and got[0]['kind']=='tx connect-request-retry'
    assert {o['kind'] for o in got[0]['observations']}=={'CR','connect-response'}


def test_individual_frames_within_an_over_survive():
    got=consolidate([event('tx DATA over #1',10,18,evidence_type='ptt'),
                     event('DATA over',10.1,17.9,frame_type='data',status='CRC pass'),
                     event('frame1',10.2,13.9,frame_type='data',symbols='one',status='CRC pass'),
                     event('frame2',14.1,17.8,frame_type='data',symbols='two',status='CRC pass')])
    assert [e['id'] for e in got]==['frame1','frame2']
    assert all(e['transmit_label']=='tx DATA over #1' for e in got)
    assert len(got[0]['observations'])==2


def test_adjacent_packets_and_opposite_directions_are_not_merged():
    got=consolidate([event('first',1,2,frame_type='data',status='CRC pass'),
                     event('second',2.1,3,frame_type='data',status='CRC pass'),
                     event('rx',1,2,frame_type='data',status='CRC pass',direction='rx')])
    assert len(got)==3


def test_response_at_transmit_boundary_is_not_hidden_inside_our_request():
    got=consolidate([event('tx connect-request-retry',10,12,evidence_type='ptt'),
                     event('CR',10.2,10.2),
                     event('connect-response',11.9,11.9,direction='rx')])
    assert [e['kind'] for e in got]==['tx connect-request-retry','connect-response']
    assert [o['kind'] for o in got[0]['observations']]==['CR']


def test_payload_bytes_are_lossless_and_keep_frame_boundaries():
    from vara_packets import packet_bytes
    blocks=[packet_bytes(bytes(range(256)), 'decoded frame'), packet_bytes(b'', 'decoded frame')]
    assert bytes.fromhex(blocks[0]['hex'])==bytes(range(256))
    assert [b['length'] for b in blocks]==[256,0]


def test_header_detector_timestamp_is_not_attached_as_a_frame_edge():
    packet=event('body',36,36.81,frame_type='data',status='CRC pass',direction='rx')
    header=event('header',35.86,35.86,status='decoded control',direction='rx',timestamp_only=True)
    got=consolidate([packet,header])
    assert [e['id'] for e in got]==['header','body']
    assert 'observations' not in packet


def test_crc_classifier_cannot_hide_decoded_base_payload():
    from vara_packets import packet_bytes
    got = consolidate([
        event('classification', 1, 5, frame_type='data', status='CRC pass'),
        event('base', 1, 5, frame_type='data', status='CRC pass',
              payloads=[packet_bytes(b'\x00\xffhello', 'VARA decoded body')]),
    ])
    assert len(got) == 1 and got[0]['id'] == 'base'
    assert bytes.fromhex(got[0]['payloads'][0]['hex']) == b'\x00\xffhello'
    assert got[0]['observations'][0]['id'] == 'classification'


import pytest


@pytest.mark.parametrize('overs', [1, 2])
def test_blind_replay_retains_crc_passing_base_bytes(tmp_path, monkeypatch, overs):
    import numpy as np
    from types import SimpleNamespace
    import vara_view as view

    samples = np.zeros(4 * view.FS)
    class Gate:
        def push(self, chunk): return []
        def flush(self): return [(0, samples)]
        def reference(self): return None
    monkeypatch.setattr(view.monitor, 'MonitorGate', Gate)
    monkeypatch.setattr(view.monitor, 'HandshakeScanner',
                        lambda calls: SimpleNamespace(push=lambda chunk: []))
    miss = SimpleNamespace(ok=False, frames=(), level=4)
    monkeypatch.setattr(view.varaofdm, 'decode_burst', lambda *a, **kw: miss)
    monkeypatch.setattr(view.varatone, 'decode_burst', lambda *a, **kw: miss)
    monkeypatch.setattr(view, '_tone_burst', lambda *a: False)
    monkeypatch.setattr(view.monitor, 'classify',
                        lambda *a, **kw: view.monitor.Result('DATA over', '7 bytes', 'CRC ok'))
    monkeypatch.setattr(view.monitor.RX, 'detect_overs',
                        lambda x: [(i * len(x) // overs, (i + 1) * len(x) // overs) for i in range(overs)])
    raw = b'\x00\xffhello'
    monkeypatch.setattr(view.phy, 'decode_level',
                        lambda *a, **kw: SimpleNamespace(level=4, frames=(raw,)))
    events = view.replay(samples, 2300, ['N0CALL', 'N1TEST'], tmp_path,
                         progress=lambda *a, **kw: None)
    assert len(events) == overs and all(e['kind'] == 'Base data L4' for e in events)
    assert sum(len(e['payloads']) for e in events) == overs
    assert events[0]['status'] == 'CRC pass'
    assert bytes.fromhex(events[0]['payloads'][0]['hex']) == raw
    assert 'energy-gate window' in events[0]['timing_precision']
