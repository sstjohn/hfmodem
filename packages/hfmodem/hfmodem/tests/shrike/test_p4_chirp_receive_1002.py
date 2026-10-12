# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recorded X27/v5 to SL1 chirp receive transition."""

import json
from contextlib import redirect_stdout
from os import devnull

import pytest

from hfmodem.shrike import onair, p4chirprx, p4receiver, session
from hfmodem.shrike.p4turn import P4Turn
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx


ARM = CAPTURES / 'onair-1002-2045'
FIELD = bytes.fromhex('524d53205472696d6f646520312e342e342e30205765')
STARTS = (1008626, 1188605, 1368728, 1548763, 1728553)


def setup_link(ladder='chirp', data='chirp'):
    audio = session.load_wav(str(require_input(ARM / 'stream.flac')), FS)
    stream = Stream(audio)
    stream.pos = 759164
    tx = Tx(stream)
    tx.settle = 2352 / FS
    rx = tx.sessrx.p4
    assert not rx.chirp
    probe = json.loads((ARM / 'p4-probe.json').read_text())
    tx.p4_probe.emissions = probe['emissions'][:2]
    args = tx.p4_link_options
    args.p4_ladder = ladder
    args.p4_data = data
    args.p4_hold_cycles = 16
    args.p4_normal_ack = 'parity'
    args.p4_breakin = 'body'
    args.p4_answer_offset = .92
    args.p4_chirp_ack = 'alternate'
    args.p4_irs_silent_cycles = 0
    args.mail_wait_greeting = True
    return stream, tx, rx


def grid():
    return onair._MasterGrid(2797, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800)


def test_x27_v5_path_receives_and_answers_all_five_chirps(monkeypatch):
    stream, tx, rx = setup_link()
    delivered = []
    tx.p4_link_host.deliver = delivered.append
    keyed = tx._tx

    def emit(audio, what):
        keyed(audio, what)
        stream.pos = max(stream.pos, tx.tx_end)

    tx._tx = emit
    seed = P4Turn.seed_opening

    def seed_waiting(machine, *args, **kwargs):
        result = seed(machine, *args, **kwargs)
        machine.allow_breakin = False
        return result

    headers = []
    acquire = p4chirprx.acquire_header

    def placed(audio, **kwargs):
        before = None if rx.clock is None else rx.clock.cycle_s
        header = acquire(audio, **kwargs)
        if header is not None:
            headers.append((header.start_sample, before))
        return header

    with monkeypatch.context() as patch, open(devnull, 'w') as sink, redirect_stdout(sink):
        patch.setattr(P4Turn, 'seed_opening', seed_waiting)
        patch.setattr(p4chirprx, 'acquire_header', placed)
        onair._run_p4_link(tx, stream, grid())

    assert rx.chirp and headers and headers[0][1] == 1.25
    complete = [row for row in tx.p4_probe.heard
                if row['kind'] == 'chirp' and not row['provisional']]
    assert [row['onset_sample'] for row in complete] == list(STARTS)
    assert all(row['status'] == 0x61 and bytes.fromhex(row['data']) == FIELD
               for row in complete)
    assert rx.clock.cycle_s == 3.75
    emissions = tx.p4_probe.emissions[2:]
    # R2 §4 revokes the opening response lease on BODY takeover. The old
    # Y27 extension no longer authorizes the 812590 answer; the greeting
    # repeat and the subsequent short controls still receive their answers.
    assert len(emissions) == 8 and [row['word'] for row in emissions] == ['X27'] * 8
    # B: the greeting repeat matches the deferred answer, which keys on the
    # first copy's raster (768430+.920 s+1.25 s), not the repeat's 5 samples earlier.
    assert [row['audio_start'] for row in emissions[:3]] == [872590, 932585, 992585]
    assert [row['audio_start'] for row in emissions[3:]] == [
        at + round(3.400 * FS) for at in STARTS
    ]
    assert delivered == [b'RMS Trimode 1.4.4.0 Welcome to ', FIELD]


@pytest.mark.parametrize('ladder,data,reused,prearmed,armed', [
    ('chirp', 'none', True, False, True),
    ('long', 'chirp', True, False, True),
    ('chirp', 'none', False, False, True),
    ('long', 'chirp', False, False, True),
    ('long', 'body', True, False, True),
    ('none', 'body', True, False, True),
    ('none', 'body', False, False, True),
    ('long', 'body', True, True, True),
])
def test_link_options_arm_reused_and_fallback_receiver(monkeypatch, ladder, data, reused,
                                                        prearmed, armed):
    stream, tx, rx = setup_link(ladder, data)
    rx.chirp = prearmed
    if not reused:
        tx.sessrx = None
        created = []
        original = p4receiver.P4Receiver

        def make_receiver(*args, **kwargs):
            result = original(*args, **kwargs)
            created.append(result)
            return result

        monkeypatch.setattr(p4receiver, 'P4Receiver', make_receiver)

    class ReachedTurn(Exception):
        pass

    def stop(*args, **kwargs):
        raise ReachedTurn

    monkeypatch.setattr('hfmodem.shrike.p4turn.P4Turn', stop)
    with pytest.raises(ReachedTurn):
        onair._run_p4_link(tx, stream, grid())
    selected = rx if reused else created[0]
    assert selected.chirp is armed
    assert selected.admitted and selected.breakin


def test_recorded_chirp_headers_and_crc_fields():
    audio = session.load_wav(str(require_input(ARM / 'stream.flac')), FS)
    found = []
    for index in range(16):
        nominal = 1008425 + index * 60000
        header = p4chirprx.acquire_header(audio, nominal_sample=nominal,
                                           frequency_offset=-4.5)
        if header is None:
            continue
        found.append(header)
        if index == 15:
            assert abs(header.start_sample - 1908746) <= 96
            assert p4chirprx.decode_packet(audio, start_sample=header.start_sample,
                                            frequency_offset=-4.5) is None
            continue
        result = p4chirprx.decode_packet(audio, start_sample=header.start_sample,
                                         frequency_offset=-4.5)
        assert result is not None and result.frame.status.byte == 0x61
        assert result.frame.data == FIELD
        assert result.marker == header.marker
        assert result.rank0_upper is (index in (0, 6, 12))
    assert [h.marker for h in found] == [8, 9, 9, 9, 9, 9]
    assert all(abs(h.start_sample - at) <= 96 for h, at in zip(found, (*STARTS, 1908746)))


@pytest.mark.parametrize('arm', ['onair-1002-1013', 'onair-1001-1045'])
def test_short_only_recordings_have_no_chirp_headers(arm):
    path = CAPTURES / arm
    audio = session.load_wav(str(require_input(path / 'stream.flac')), FS)
    probe = json.loads((path / 'p4-probe.json').read_text())
    slots = {(row['onset_sample'], round(row['cfo_hz'], 1)) for row in probe['heard']
             if row.get('onset_sample') is not None and row.get('cfo_hz') is not None
             and row['kind'] in ('word', 'short', 'normal', 'body')}
    assert slots
    for onset, cfo in slots:
        assert p4chirprx.acquire_header(audio, nominal_sample=onset,
                                        frequency_offset=cfo) is None
