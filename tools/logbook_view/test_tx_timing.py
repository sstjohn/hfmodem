# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import pytest

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


def test_late_and_interrupted_bursts_keep_their_interval_and_say_so():
    log = TxEvents('PACTOR')
    for tx_id, outcome in enumerate(('late', 'interrupted'), 1):
        log.record('completed', tx_id, outcome=outcome, label='data', recording='/capture/stream.wav',
                   sample_rate=48000, ptt_start_sample=48000 * tx_id, audio_end_sample=48000 * tx_id + 24000)
    tx = intervals('\n'.join(log.lines()), '/capture/stream.wav', 'rig rx', 10, 'PACTOR')
    assert [(t['t0'], t['t1']) for t in tx] == [(1.0, 1.5), (2.0, 2.5)]
    assert [t['status'] for t in tx] == ['logged PTT interval (late)', 'logged PTT interval (interrupted)']


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


@pytest.mark.parametrize('host', ['Windows WASAPI', 'Core Audio', 'ALSA'])
def test_transmit_timeline_retains_timestamp_limitation(host):
    from hfmodem.core.audio import timestamp_limitation
    limitation = timestamp_limitation(host)
    log = TxEvents('PACTOR')
    fields = dict(timestamp_limitation=limitation, driver_offset_provenance='nominal',
                  driver_offset_samples=1771) if limitation else dict(driver_offset_samples=756)
    log.record('completed', 1, outcome='sent', recording='/capture/stream.wav',
               sample_rate=48000, ptt_start_sample=48000, audio_end_sample=96000,
               **fields)
    row, = intervals('\n'.join(log.lines()), '/capture/stream.wav', 'rig rx', 10, 'PACTOR')
    assert (row['t0'], row['t1']) == (1, 2)
    if limitation:
        assert row['timestamp_limitation'] == limitation
        assert row['driver_offset_provenance'] == 'nominal'
        assert row['driver_offset_samples'] == 1771
        assert 'estimated' in row['timing_precision'] and limitation in row['timing_precision']
    else:
        assert 'timestamp_limitation' not in row
        assert 'driver_offset_samples' not in row
        assert row['timing_precision'] == 'host sample positions; not measured RF'
