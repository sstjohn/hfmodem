# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from hfmodem.core.tx_events import TxEvents
from tx_timing import intervals, merge


def test_same_contract_for_vara_pactor_and_unaligned_source():
    for protocol in ('VARA','PACTOR'):
        log=TxEvents(protocol)
        log.record('completed',1,outcome='sent',label='data',packet_kind='data',
                   recording='/capture/stream.wav',sample_rate=48000,
                   ptt_start_sample=48000,audio_end_sample=96000,
                   speed_level=6,modulation='DQPSK')
        text='\n'.join(log.lines())
        tx=intervals(text,'/capture/stream.flac','rig rx',10,protocol)
        assert len(tx)==1 and tx[0]['level']==6 and tx[0]['tx_confirmed']
        assert intervals(text,'/capture/tap.wav','rig tap',10,protocol)==[]
        assert intervals(text.replace('"sent"','"refused"'),'/capture/stream.wav','rig rx',10,protocol)==[]


def test_explicit_emission_replaces_early_prose_grid_by_identity():
    explicit = [{'id': 'ptt-session-10', 'tx_id': 10, 't0': 11.87, 't1': 12.11}]
    legacy = [{'id': 'ptt-10', 't0': 11.27, 't1': 11.51,
               'origin': 'original PACTOR session grid'},
              {'id': 'ptt-11', 't0': 12.52, 't1': 12.76,
               'origin': 'original PACTOR session grid'}]
    assert merge(explicit, legacy) == [explicit[0], legacy[1]]


def test_vara_display_preserves_frame_and_strips_only_payload_trailer():
    from hfmodem.kestrel.arq.phy import vara_body
    from vara_packets import vara_payloads, packet_bytes
    # Binary mail can contain 0x14; only the complete caller-specific trailer
    # terminates the DATA payload. Tone frames also retain their two CRC bytes.
    payload = b'FF\r\x14\x01binary'
    raw = vara_body(payload, 'W9SSJ', body_len=426) + b'\x12\x34'
    event = {'kind': 'Tone L9', 'status': 'CRC pass',
             'payloads': [packet_bytes(raw, 'frame')]}
    setup = {'kind': 'Base data L4', 'status': 'CRC pass',
             'transmit_label': 'tx link-setup', 'payloads': [packet_bytes(b'x' * 90, 'body')]}
    vara_payloads([event, setup], 'W9SSJ')
    block = event['payloads'][0]
    assert block['hex'] == raw.hex()
    assert block['payload_hex'] == payload.hex()
    assert block['payload_length'] == len(payload)
    assert setup['payloads'][0]['payload_hex'] is None
