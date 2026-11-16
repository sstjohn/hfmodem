# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Probe records preserve resolved settings and keyed evidence without devices."""
import json
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.core.rigs import resolve_tx_timing
from hfmodem.shrike import onair, p4breakintx, p4probe
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx


@pytest.fixture(autouse=True)
def revision_commands(monkeypatch):
    monkeypatch.setattr(p4probe.subprocess, 'check_output',
        lambda argv, **kwargs: b'abc1234' if argv[1] == 'rev-parse' else b' M code.py')


def options():
    args = onair.build_parser().parse_args(['--p4-link', '--p4-arm', 'chain-a'])
    args.tx_timing_plan = resolve_tx_timing(0, environ={})
    onair._connect_defaults(args)
    onair._p4_entry_defaults(args)
    return args


def test_schema_and_resolved_config():
    args = options()
    args.tx_timing_plan = replace(args.tx_timing_plan, tx_latency_ms=12., latency_samples=576,
        p4_answer_residual_ms=3., p4_answer_residual_samples=144, p4_answer_residual_evidence='bench fixture')
    probe = p4probe.EntryProbe(config=p4probe.probe_config(args))
    result = json.loads(json.dumps(probe.result()))
    assert result['schema'] == 'p4-mail/1'
    assert isinstance(result['config'], dict) and isinstance(result['id'], list)
    config = result['config']
    assert {key for key in vars(args) if key.startswith('p4_')} <= config.keys()
    assert config['p4_arm'] == 'chain-a' and config['p4_opening'] == [7, 38]
    assert config['hold'] == 12
    assert config['tx_latency_ms'] == 12. and config['p4_answer_residual_ms'] == 3.
    assert config['p4_peer_ack']['13'] == 'X27'
    assert config['p4_chain_reset'] == 'turn,form'
    timing = config['timing_plan']
    assert timing['scheduled_offset_s'] == pytest.approx(.905)
    assert timing['long_scheduled_offset_s'] == pytest.approx(3.385)
    assert timing['p4_answer_residual_evidence'] == 'bench fixture'
    revision = config['revision']
    assert isinstance(revision['sha'], str) and isinstance(revision['dirty'], bool)


def test_revision_outside_checkout_and_failure(tmp_path, monkeypatch):
    def not_checkout(*args, **kwargs):
        raise p4probe.subprocess.CalledProcessError(128, args[0])
    monkeypatch.setattr(p4probe.subprocess, 'check_output', not_checkout)
    assert p4probe.code_revision(tmp_path) is None
    def unavailable(*args, **kwargs):
        raise OSError('git unavailable')
    monkeypatch.setattr(p4probe.subprocess, 'check_output', unavailable)
    assert p4probe.code_revision() is None
    assert p4probe.probe_config(options())['revision'] is None


def test_existing_link_fixture_through_probe_writer():
    audio = np.zeros(5*FS)
    onset = round(.923*FS)
    wave = p4breakintx.changeover_packet(b'hello', 0)
    audio[onset:onset+len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options.p4_normal_ack = 'toggle'
    tx.p4_probe.config = p4probe.probe_config(options())
    stream.tx_latency_n = 480
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    result = json.loads(json.dumps(tx.p4_probe.result()))
    assert result['config']['p4_arm'] == 'chain-a'
    packet = next(row for row in result['heard'] if 'data' in row)
    assert packet['form'] == 'body' and packet['counter'] == 0
    assert packet['crc_ok'] is True and packet['copy_index'] == 1
    keyed = next(row for row in result['link_events'] if row['event'] == 'Keyed')
    assert keyed['ack_policy'] == 'toggle'
    assert keyed['chain_epoch'] == 1 and keyed['chain_reason'] == 'opening'
    assert keyed['answers_counter'] == 0
    assert keyed['nominal_s'] == pytest.approx(packet['onset_sample']/FS+.920)
    assert keyed['scheduled_s'] == pytest.approx(keyed['nominal_s']-.010)
    assert {row['chain_epoch'] for row in result['link_events']
            if row['event'] == 'Keyed'} == {1}


def test_copy_index_enrichment_wrap_and_turn():
    probe = p4probe.EntryProbe()
    for epoch, onset, counter in [(1, 0, 0), (1, 0, 0), (1, 1, 0),
            (1, 2, 1), (1, 3, 0), (2, 4, 0)]:
        probe.heard.append(dict(chain_epoch=epoch, onset_sample=onset, counter=counter))
    assert [row['copy_index'] for row in probe.result()['heard']] == [1, 1, 2, 1, 1, 1]


def test_callsign_transmissions_only():
    probe = p4probe.EntryProbe()
    probe.mycall = 'W9SSJ'
    probe.note_id(b'1W9SSJ\r', 'p1', 100, 200)
    probe.note_id(b';FW: w9ssj\r', 'body', 300, 400)
    probe.note_id(b'other station', 'normal', 500, 600)
    probe.note_id(b'hello W9SSJ', 'normal', 700, 800)
    assert [row['kind'] for row in probe.result()['id']] == ['p1', 'body', 'normal']
    assert [row['start_sample'] for row in probe.id] == [100, 300, 700]


def test_receiver_forms_on_normal_short_and_robust_long():
    from hfmodem.shrike import p4shorttx, p4robustlongtx
    from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect
    for waveform, expected in [
            (p4shorttx.short_packet(b'W9SSJ', 0x61), 'normal-short/SL5'),
            (p4robustlongtx.robust_long(b'W9SSJ', 0x61, 3), 'robust-long/SL3')]:
        at = FS
        rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=0.)
        rx.enable_hold()
        rx.clock = Clock(Expect.PACKET, at, .330, 1.25, 0.)
        audio = np.r_[np.zeros(at), waveform, np.zeros(FS//10)]
        obs = (rx._normal(audio, 0) if expected.startswith('normal')
               else rx._long(audio, 0)[0])
        assert obs is not None and obs.frame is not None
        assert obs.form == expected


def test_cw_id_records_completed_sample_interval(tmp_path):
    from hfmodem.tests.shrike.test_identify import transmitter, CALL
    tx, _, _ = transmitter(tmp_path)
    tx.p4_probe = p4probe.EntryProbe()
    tx.identify(CALL)
    assert tx.p4_probe.result()['id'] == [dict(kind='cw', call=CALL,
        start_sample=tx.tx_audio_start, end_sample=tx.tx_end)]


@pytest.mark.parametrize('refused', [False, True])
def test_p1_callsign_id_requires_completed_key(tmp_path, monkeypatch, refused):
    from hfmodem.tests.shrike.test_identify import transmitter, CALL
    tx, _, _ = transmitter(tmp_path)
    tx.p4_probe = p4probe.EntryProbe()
    tx.p4_probe.mycall = CALL
    def key(*args, **kwargs):
        tx.refused = refused
        tx.tx_audio_start, tx.tx_end = 100, 200
    monkeypatch.setattr(tx, '_tx', key)
    tx.send_p1_packet(b'1W9SSJ\r', 100, 1)
    assert tx.p4_probe.id == ([] if refused else [dict(kind='p1', call=CALL,
        start_sample=100, end_sample=200)])


@pytest.mark.parametrize('boundary', ['turn', 'form', 'ladder'])
def test_ack_chain_provenance_at_boundaries(boundary):
    from hfmodem.tests.shrike.test_p4_ack_toggle_1110 import key_packet, short
    from hfmodem.tests.shrike.test_p4turn import machine, propose_ack
    from hfmodem.shrike.p4receiver import Kind
    turn, _ = machine(role='irs', chain_reset='turn,form,ladder')
    key_packet(turn, FS, 0)
    assert turn.keyed_ack['chain_reason'] == 'opening'
    if boundary == 'turn':
        turn._close_receive_epoch('peer turn')
        turn._open_receive_epoch(300000)
    elif boundary == 'ladder':
        assert propose_ack(turn, short(200000)).word == 'X27'
        turn.keyed()
    key_packet(turn, 400000, 1, Kind.LONG if boundary == 'form' else Kind.NORMAL)
    assert turn.keyed_ack['chain_epoch'] == 2
    assert turn.keyed_ack['chain_reason'] == boundary
    assert turn.keyed_ack['answers_counter'] == 1


@pytest.mark.parametrize('protocol', ['p2', 'p3'])
@pytest.mark.parametrize('refused', [False, True])
def test_other_payload_ids_respect_admission(tmp_path, monkeypatch, protocol, refused):
    from hfmodem.tests.shrike.test_identify import transmitter, CALL
    tx, _, _ = transmitter(tmp_path)
    tx.p4_probe = p4probe.EntryProbe()
    tx.p4_probe.mycall = CALL
    tx.tx_audio_start, tx.tx_end = 100, 200
    tx.refused = refused
    if protocol == 'p2':
        monkeypatch.setattr(tx, '_tx_p2', lambda *args, **kwargs: None)
        tx.send_p2_packet(1, b'W9SSJ', 0)
    else:
        tx._log_p3_packet(1, b'W9SSJ', 0)
    assert tx.p4_probe.id == ([] if refused else [dict(kind=protocol, call=CALL,
        start_sample=100, end_sample=200)])


def test_copy_index_resets_when_same_counter_starts_a_new_chain():
    probe = p4probe.EntryProbe()
    probe.heard = [dict(receive_epoch=1, onset_sample=at, counter=0) for at in (100, 200)]
    probe.link_events = [dict(event='Keyed', answers_counter=0,
        accepted_target=at, chain_epoch=epoch) for at, epoch in ((100, 1), (200, 2))]
    assert [row['copy_index'] for row in probe.result()['heard']] == [1, 1]


# The robust long is read by the LONG engine, whose Python fallback answers on
# time only when the machine keeps up.
@pytest.mark.parametrize('form', ['normal-short/SL5',
                                  pytest.param('robust-long/SL3', marks=pytest.mark.realtime)])
def test_heard_forms_through_link_writer(form):
    from hfmodem.shrike import p4shorttx, p4robustlongtx
    audio = np.zeros(8*FS)
    greeting = p4breakintx.changeover_packet(b'hello', 0)
    packet = (p4shorttx.short_packet(b'next', 1) if form.startswith('normal')
              else p4robustlongtx.robust_long(b'next', 1, 3))
    for onset, wave in ((round(.923*FS), greeting), (round(2.173*FS), packet)):
        audio[onset:onset+len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    heard = tx.p4_probe.result()['heard']
    assert any(row.get('form') == 'body' and row.get('crc_ok') for row in heard)
    assert any(row.get('form') == form and row.get('counter') == 1
               and row.get('crc_ok') and row.get('copy_index') == 1 for row in heard)


@pytest.mark.parametrize('residual_ms', [0., 31.4, -31.4])
@pytest.mark.parametrize('geometry', [0, 816])
def test_config_schedule_matches_sample_quantized_scheduler(residual_ms, geometry):
    args = options()
    args.tx_timing_plan = replace(args.tx_timing_plan, tx_latency_ms=7., latency_samples=336, geometry_samples=geometry,
        p4_answer_residual_ms=residual_ms, p4_answer_residual_samples=round(residual_ms*48))
    timing = p4probe.probe_config(args)['timing_plan']
    correction = (336+round(residual_ms*48)-geometry)/FS
    assert timing['scheduled_offset_s'] == pytest.approx(.920-correction)
    assert timing['long_scheduled_offset_s'] == pytest.approx(3.400-correction)
