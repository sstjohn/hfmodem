# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Scripted gateway audio through the link driver and a real B2F client."""
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from hfmodem.shrike import arq, compress, onair, p4breakintx, p4robusttx, p4shorttx
from hfmodem.shrike.p4turn import P4Turn
from hfmodem.shrike.ptc import PtcHost, SimPeer
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx
from hfmodem.winlink import B2FSession, MailClient, compose
from hfmodem.tests.shrike import p4_stock_peer as stock

# Continue at the recorded greeting's body carrier.
peer_body = partial(stock.body, cfo=-2.75)
peer_word = partial(stock.word, cfo=-2.75)


def options(monkeypatch, *extra):
    got = []
    monkeypatch.setattr(sys, 'argv', ['onair', '--after-connect', 'p1-robust',
                                     '--p4-link', '--p4-data', 'body', *extra])
    monkeypatch.setattr(onair, 'run', lambda args: got.append(args) or 0)
    assert onair.main() == 0
    return got[0]


@pytest.mark.parametrize('lead', [0, .074])
@pytest.mark.parametrize('ending', ['sent', 'empty', 'unsupported'])
def test_mail_exchange(monkeypatch, capsys, ending, lead):
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: round(lead * FS))
    args = options(monkeypatch, '--mail-fetch', '--mail-wait-greeting',
                   '--p4-link-timeout', '120', '--p4-entry-timeout', '120')
    stream = Stream(np.zeros(120 * FS))
    tx = Tx(stream)
    receiver = tx.sessrx.p4
    tx.p4_probe.timeout = 120
    tx.p4_link_options = args
    host = PtcHost(SimPeer(), mycall='W9SSJ')
    host.arq.state = arq.State.CONNECTED
    host.arq.role = arq.IRS
    message = compose('W9SSJ', 'SMTP:offline@example.net', 'P4 test', b'hello\r\n',
                      mid='P4TEST0927', date=datetime(2026, 9, 27, tzinfo=timezone.utc))
    session = B2FSession('W9SSJ', target='VE3KPG', comment=False,
                         outbox=[message] if ending != 'empty' else [])
    mail = MailClient(session, host.arq.on_host_data)
    host.app = mail
    mail.link_up()
    tx.p4_link_host, tx.p4_link_mail = host, mail
    closed = []
    tx.p4_link_budget = type('Budget', (), {'close': lambda self, n: closed.append(n)})()
    emitted, counts, waves = [], [], []
    transmit = tx._tx

    def transmit_audio(audio, label):
        waves.append(audio.copy())
        transmit(audio, label)

    tx._tx = transmit_audio
    settled = []
    settle = host.arq.settle_bytes

    def check():
        assert host._txbuf == host.arq._buffer_raw
        assert host._txbuf == len(host.arq._outbuf) + len(host.arq._p4_lent)
        counts.append(host._txbuf)

    def ack(wire):
        nonlocal phase, reply
        settled.append(wire)
        settle(wire)
        check()
        if host._txbuf == 0 and not session.done:
            reply = b'FS Y\r' if phase == 'sending' and ending == 'sent' else b'FF\r'
            phase = 'message' if reply.startswith(b'FS') else 'closing'
            place(receiver.clock.next_due + round(1.25 * FS),
                  peer_body(reply, 0))

    monkeypatch.setattr(host.arq, 'settle_bytes', ack)
    fixture = Path(__file__).parent / 'fixtures/p4-breakin/ve3kpg-audio.wav'
    rate, pcm = wavfile.read(fixture)
    first = json.loads(fixture.with_suffix('.json').read_text())['first_chip']
    greeting = resample_poly(pcm[first:].astype(float) / 32768, FS // rate, 1)
    incoming = [
        peer_body(b'no messages\r', 0),
        peer_body(b'[RMS-1.0-B2FHM$]\r', 0),
        peer_body(b'VE3KPG>\r', 0),
    ]
    phase = 'greeting'
    reply = b''

    def place(at, wave):
        stream.audio[at:at + len(wave)] = wave

    place(round(.923 * FS), greeting)
    keyed = P4Turn.keyed

    def peer(machine, at=None):
        nonlocal phase, reply
        proposal = keyed(machine, at)
        emitted.append(proposal)
        check()
        at = proposal.stream_sample
        if proposal.kind == 'word':
            if phase == 'greeting' and incoming:
                assert not session.sent_text or not incoming
                wave = incoming.pop(0)
            elif session.sent_text:
                wave = peer_body(b'', 0)
            else:
                wave = p4robusttx.robust_short(13)
            place(at + round(.330 * FS), wave)
        elif proposal.kind == 'normal' and not proposal.payload and not session.done:
            place(at + round(.920 * FS), peer_body(reply, 0))
        elif proposal.kind in ('body', 'normal'):
            if proposal.status == 0x40:
                assert session.sent_text
                place(at + round(.920 * FS), peer_body(
                    reply if proposal.state == 'ISS_GRANT' else b'',
                    0 if proposal.state == 'ISS_GRANT' else 0x40))
                phase = 'sending' if phase == 'greeting' else phase
            elif proposal.status == 0x20:
                pass
            else:
                np.testing.assert_array_equal(waves[-1], (p4breakintx.changeover_packet if proposal.kind == 'body' else p4shorttx.short_packet)(
                    proposal.payload, proposal.status))
                if ending == 'unsupported':
                    wave = peer_body(b'cannot decode source', 4)
                else:
                    wave = peer_word('V27' if proposal.status & 1 else 'X27')
                place(at + round(.920 * FS), wave)
        elif proposal.kind == 'short':
            if proposal.value == 37:
                wave = peer_word('V27')
            elif host._txbuf == 0 and not session.done:
                if phase == 'sending' and ending == 'sent':
                    reply = b'FS Y\r'
                    wave = peer_body(reply, 0)
                    phase = 'message'
                else:
                    reply = b'FF\r'
                    wave = peer_body(reply, 0)
                    phase = 'closing'
            else:
                wave = peer_word('X27' if proposal.value == 13 else 'U27')
            place(at + round(.920 * FS), wave)
        return proposal

    monkeypatch.setattr(P4Turn, 'keyed', peer)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    check()
    log = capsys.readouterr().out
    assert closed and len(closed) == 1, (log, emitted, session.exchange, tx.p4_probe.link_events)
    assert '[mail]' in log
    assert not any(e['event'] == 'Refused' for e in tx.p4_probe.link_events)
    assert len(emitted) >= 4 and all(p.kind == 'word' for p in emitted[:3])
    assert session.remote_sid == '[RMS-1.0-B2FHM$]'
    assert session.remote_text.count('RMS Trimode 1.4.4.0') == 1
    assert not host.arq._p4_lent
    if ending == 'unsupported':
        assert session.failure == 'unsupported P4 source status 0x04'
        assert not session.done
        assert 'failed' in tx.p4_probe.stopped
        assert host._txbuf > 0 and not settled
    else:
        wire = b''.join(settled)
        plain = compress.Supervisor().feed(wire)
        assert session.done and not session.failure
        assert host._txbuf == 0 and host.sent_total == len(wire)
        assert plain.count(session.sid.encode()) == 1
        assert all(field in [p.payload for p in emitted if p.kind in ('body', 'normal')]
                   for field in settled)
        assert b'FC EM ' in plain if ending == 'sent' else b'FF\r' in plain
        assert b'FQ\r' in plain
        assert 'exchange complete' in log
        if ending == 'sent':
            assert session.sent_mids == [message.mid]


@pytest.mark.parametrize('resolution', ['confirmed', 'contradicted', 'expired'])
@pytest.mark.parametrize('step', ['ladder', 'data'])
def test_partial_ack_waits_for_confirmation_then_uses_the_deferred_slot(step, resolution):
    from hfmodem.tests.shrike.test_p4turn import machine, word
    from hfmodem.shrike.p4observation import Registered
    from hfmodem.shrike.p4turn import State

    turn, queue = machine(data='body')
    queue.on_host_data(b'A' * 31 + b'B' * 31)
    if step == 'ladder':
        turn._short(0, 1)
        name = 'U27'
    else:
        turn._data(0)
        name = 'X27'
    turn.keyed()
    early = word(name, round(.920 * FS), partial=True)
    turn.observe(early, now=round(1.13 * FS))
    turn.silence(round(1.176 * FS))
    assert turn.decide(round(1.176 * FS)) is None
    turn.observe(word(name if resolution == 'confirmed' else 'Z27',
                      early.observation.stream_sample,
                      outcome=Registered(resolution), replaced=early.observation,
                      answered=None), now=round(1.20 * FS))
    if resolution != 'confirmed':
        assert queue._buffer_raw == 62
        assert bytes(queue._outbuf) + turn.outbox.inflight in (b'A' * 31 + b'B' * 31,
                                                              b'B' * 31 + b'A' * 31)
        assert turn._waiting_ack is None
        return
    proposal = turn.decide(round(2.426 * FS))
    assert proposal.kind == ('body' if step == 'ladder' else 'normal') and proposal.stream_sample == round(2.5 * FS)
    assert proposal.payload == (b'A' if step == 'ladder' else b'B') * 31
    assert queue._buffer_raw == (62 if step == 'ladder' else 31)
    if step == 'data':
        stale = word('V27', round(2.170 * FS), partial=True)
        turn.observe(stale, now=round(2.385 * FS))
        turn.keyed()
        turn.observe(word('V27', stale.observation.stream_sample,
                          outcome=Registered.CONFIRMED, replaced=stale.observation,
                          answered=proposal.stream_sample), now=round(2.51 * FS))
        assert turn.state == State.ISS_DATA
        assert queue._buffer_raw == 31 and turn.outbox.inflight == b'B' * 31


def test_text_is_queued_once_behind_one_sid(monkeypatch):
    from hfmodem.winlink.session import sid_line

    args = options(monkeypatch, '--p4-text', 'FC_TEST', '--p4-link-timeout', '8')
    audio = np.zeros(10 * FS)
    for onset, wave in (
        (.923, stock.body(b'hello', 0)),
        (2.173, p4robusttx.robust_short(13)),
        (4.013, stock.body(b'', 0x40)),
    ):
        at = round(onset * FS)
        audio[at:at + len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options = args
    host = PtcHost(SimPeer(), mycall='W9SSJ')
    host.arq.state = arq.State.CONNECTED
    tx.p4_link_host = host
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    expected = compress.transparent((sid_line() + 'FC_TEST').encode('latin-1'))
    assert bytes(host.arq._outbuf) == expected
    assert host._txbuf == host.arq._buffer_raw == len(expected)
    assert not host.arq._p4_lent and host.sent_total == 0
    packets = tx.p4_probe.emissions[3:]
    assert len(packets) == 3 and all(p['status'] == 0 for p in packets)
    assert len({p['waveform_sha256'] for p in packets}) == 1


@pytest.mark.parametrize('step', ['ladder', 'data', 'peer-close', 'close-silence'])
def test_unresolved_ack_cannot_suspend_retry_or_close_forever(step):
    from dataclasses import replace
    from hfmodem.tests.shrike.test_p4turn import machine, word
    from hfmodem.shrike.p4turn import State

    turn, queue = machine(data='body')
    queue.on_host_data(b'A' * 31)
    if step == 'ladder':
        turn._short(0, 1)
        name = 'U27'
    else:
        turn._data(0)
        name = 'X27'
    turn.keyed()
    onset = round(.920 * FS)
    turn.observe(word(name, onset, partial=True), now=onset + round(.210 * FS))
    limit = onset + round((.275 + 1.25) * FS)
    turn.silence(limit - 1)
    assert turn.decide(limit - 1) is None
    if step == 'peer-close':
        turn.state = State.IRS_CLOSE
    elif step == 'close-silence':
        turn.state = State.ISS_CLOSE
        turn.last = replace(turn.last, kind='short', value=37)
        turn.repeats = 2
    turn.silence(limit)
    assert turn._waiting_ack is None
    if step in ('peer-close', 'close-silence'):
        assert turn.state == State.DOWN
        assert turn.reason == ('peer closed' if step == 'peer-close' else 'close silence')
    else:
        closing = []
        for cycle in range(12):
            if turn.state == State.DOWN:
                break
            proposal = turn.decide(limit + cycle * round(1.25 * FS))
            assert proposal is not None
            if proposal.state == State.ISS_CLOSE:
                closing.append(proposal)
            else:
                assert proposal.repeat
            turn.keyed()
            turn.silence(limit + (cycle + 1) * round(1.25 * FS))
        assert turn.state == State.DOWN
        assert turn.reason == ('retry budget exhausted' if step == 'ladder' else 'close silence')
        if step == 'data':
            assert closing[0].kind == 'body' and closing[0].status == 0x20
            assert [p.value for p in closing[1:]] == [37, 37, 37]
    assert queue._buffer_raw == len(queue._outbuf) == 31
    assert not queue._p4_lent
