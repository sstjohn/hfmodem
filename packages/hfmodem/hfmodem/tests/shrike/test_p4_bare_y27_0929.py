# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The IRS asks for the turn with a bare Y27 and sends data only on the grant."""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import hilbert, resample_poly

from hfmodem.shrike import arq, onair, p4breakintx, p4shorttx, session
from hfmodem.shrike.p4hold import BreakinHold
from hfmodem.shrike.p4receiver import P4Receiver, Emission, Role
from hfmodem.shrike.p4observation import Kind, RESPONSE_S, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.shrike.ptc import PtcHost, SimPeer
from hfmodem.tests.shrike.test_p4turn import body, machine, word
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx
from hfmodem.tests.shrike.test_p4_normal_short_0928 import decide, normal
from hfmodem.winlink import B2FSession, MailClient, compose

ROOT = Path(__file__).resolve().parents[5]
K0003 = ROOT/'working/p4-oracle/kiwi/20260924/7101500/seg-0003.wav'
K0003_RATE, K0003_SHIFT_HZ = 11998.90779, 910.
SID = b'[Pat-1.0.0-B2FHM$]\r;FW: W9SSJ\r;PR: 12345678\r'
IDLE = b''
T = 44160


def queued(**kwargs):
    m, a = machine(role='irs', data='body', breakin='y27', **kwargs)
    a.on_host_data(SID)
    return m, a


def request(m, at=48000):
    p = decide(m, normal(at, status=0, data=IDLE))
    assert (p.kind, p.word, p.state, p.stream_sample) == (Kind.WORD, 'Y27', State.IRS_BREAKIN, at+T)
    m.keyed(p.stream_sample)
    return p.stream_sample


def k0003(t0, t1):
    if not K0003.exists():
        pytest.skip('local k0003 Kiwi recording absent')
    _, pcm = wavfile.read(K0003, mmap=True)
    x = resample_poly(np.asarray(pcm[round(t0*K0003_RATE):round(t1*K0003_RATE)], np.float64)/32768, 4, 1)
    n = np.arange(x.size)
    return (hilbert(x)*np.exp(-2j*np.pi*K0003_SHIFT_HZ*n/FS)).real


@pytest.mark.parametrize('ours,grant', [(1533.339, 1534.261), (1535.195, 1536.110)])
def test_k0003_bare_y27_draws_the_grant_and_the_data_body_follows(ours, grant):
    t0 = ours-1.5
    audio = k0003(t0, grant+1.)
    at = lambda t: round((t-t0)*FS)
    # Our capture is muted while we key; the caller's own word sits there.
    audio[at(ours):at(ours)+round(.27*FS)] = 0
    m, _ = queued()
    assert request(m, at(ours)-T) == at(ours)
    # N9VC's carrier on this recording: fixtures/p4-breakin/n9vc-*.json, +2.5...+2.9 Hz.
    r = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=3.)
    r.enable_hold()
    r.note_emission(Emission(Kind.WORD, Role.ISS, at(ours), .242))
    due = r.clock.next_due
    assert abs(due-at(grant)) <= round(.035*FS)
    early, end = due+round(.215*FS), at(grant)+round(.820*FS)
    regs = r.read_partial(audio[:early], 0, deadline=early)
    for reg in regs:
        m.observe(reg, now=early)
    regs = r.read_window(audio[:end], 0, whole=True)
    for reg in regs:
        m.observe(reg, now=end)
    got = [x.observation for x in regs if x.observation.kind == Kind.BODY]
    assert len(got) == 1 and got[0].frame.status.byte == 0x40
    assert abs(got[0].stream_sample-at(grant)) <= 48
    p = m.decide(got[0].stream_sample+T-m.lead)
    assert (p.kind, p.state, p.status) == (Kind.BODY, State.ISS_DATA_ACK, 0)
    assert p.payload == SID[:31] and p.stream_sample == got[0].stream_sample+T


def test_checked_idle_packet_with_bytes_queued_draws_a_bare_y27():
    m, _ = queued()
    request(m)
    assert m.state == State.IRS_BREAKIN and m.breakins == 1


def test_body_policy_requests_the_idle_turn_with_a_body():
    m, a = machine(role='irs', data='body')
    a.on_host_data(SID)
    m.first_body = False
    p = decide(m, normal(status=0, data=IDLE))
    assert p.kind == Kind.BODY and p.status == 0x40 and p.state == State.IRS_BREAKIN


def test_idle_packet_without_bytes_is_acked():
    m, _ = machine(role='irs', data='body', breakin='y27')
    m.first_body = False
    assert decide(m, normal(status=0, data=IDLE)).word == 'V27'


def test_data_packet_is_acked_not_broken_in_on():
    m, _ = queued()
    m.first_body = False
    assert decide(m, normal(status=0)).word == 'V27'


def test_unread_packet_cannot_draw_the_bare_y27():
    m, a = queued()
    before = bytes(a._outbuf), a._buffer_raw, m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack
    obs = replace(normal(status=0, data=IDLE), frame=None)
    p = decide(m, obs)
    # An unread packet earns only the approved seed repeat, never a turn.
    assert (p.kind, p.word, p.state, p.purpose) == (
        Kind.WORD, 'V27', State.IRS_ANSWER, None)
    assert not p.current_slot_accepted and p.accepted_target is None
    assert p.ack_reason is p.ack_identity is None and not p.payload
    m.keyed()
    assert m.last.word == 'V27' and not m.outbox.inflight
    assert (bytes(a._outbuf), a._buffer_raw, m.ack_word, m.acked_counter, m._last_ack, m._last_body_ack) == before
    assert not [e for e in m.link_events if e['event'] in ('turn', 'data-reply-ack')]


def test_unread_grant_draws_the_request_again():
    m, _ = queued()
    ours = request(m)
    grant = ours+T
    m.observe(word('Y27', grant, extent=.8, responds_to=ours), now=grant+round(.81*FS))
    p = m.decide(grant+T-m.lead)
    # R2 §6 keeps the accepted peer phase; the intervening slot is too early
    # after this unread .8 s grant, so the request uses the next peer slot.
    assert (p.kind, p.word, p.stream_sample) == (Kind.WORD, 'Y27', ours+2*60000)


def test_unanswered_requests_repeat_on_our_raster_and_stop_at_four():
    m, _ = queued()
    at = request(m)
    for k in range(3):
        m.silence(at+round(1.25*FS)-m.lead)
        p = m.decide(at+round(1.25*FS)-m.lead)
        assert (p.word, p.stream_sample) == ('Y27', at+round(1.25*FS))
        at = m.keyed(p.stream_sample).stream_sample
    m.silence(at+round(1.25*FS)-m.lead)
    assert m.state == State.DOWN and m.reason == 'break-in unanswered'


def test_masked_cycle_is_not_peer_silence():
    # 0923: peer raster 104.744 + 1.25 k; our 0.798 s body at 105.664 covers 105.994.
    for duration, silent in ((.798, 0), (.242, 1)):
        r = P4Receiver(FS)
        r.peer_onset = round(104.744*FS)
        r.note_emission(Emission(Kind.BODY if duration > .5 else Kind.WORD, Role.ISS,
                                 round(105.664*FS), duration))
        assert r.finish_cycle(r.clock.next_due) == silent


def driver(audio, *, breakin, timeout):
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_breakin = breakin
    tx.p4_link_options.p4_link_timeout = timeout
    tx.p4_link_host.arq.on_host_data(SID)
    onair._run_p4_link(tx, stream, onair._MasterGrid(0, 60000, 0, packet_n=36000,
                                                     cs_n=12000, d_max_n=4800))
    return [(e['word'], e['status'], e['audio_start']) for e in tx.p4_probe.emissions[1:]]


def place(audio, at, wave):
    audio[at:at+len(wave)] += wave


def test_driver_requests_then_sends_data_on_the_peer_grant():
    audio = np.zeros(7*FS)
    y = round(.923*FS)
    place(audio, y, p4breakintx.changeover_packet(b'RMS', 0))
    ack = y+T
    idle = ack+round(.330*FS)
    place(audio, idle+round(308*FS/28800), p4shorttx.short_packet(IDLE, 1))
    ours = idle+T
    grant = ours+round(.923*FS)
    place(audio, grant, p4breakintx.changeover_packet(b'', 0x40))
    keyed = driver(audio, breakin='y27', timeout=6.)
    assert keyed[0][0] == 'X27' and abs(keyed[0][2]-ack) <= 48
    assert keyed[1][0] == 'Y27' and abs(keyed[1][2]-ours) <= 96
    assert keyed[2][:2] == (None, 0) and abs(keyed[2][2]-(grant+T)) <= 96


class Replay:
    """A recorded WS8EOC arm through the link driver and a B2F client."""

    def __init__(self, name, monkeypatch, *, breakin='y27'):
        path = ROOT/f'captures/onair-0929-{name}'
        if not (path/'stream.flac').exists():
            pytest.skip(f'local September 29 {name} capture absent')
        recorded = json.loads((path/'p4-probe.json').read_text())['emissions']
        self.recorded = recorded
        opening = recorded[:next(i for i, e in enumerate(recorded) if e.get('word'))]
        # L0923 normal-decode rows: 55-69 ms a decode; at that cost the replay
        # keys the arm's own answers through the prompt.
        normal_decode = BreakinHold._normal_decode

        def costed(hold, deadline=None):
            allowance = None if deadline is None else deadline-hold.now()
            clock = [0.]
            decode = hold._decode

            def decoding(scores):
                result = decode(scores)
                clock[0] += .065
                return result

            with monkeypatch.context() as timing:
                timing.setattr(hold, 'now', lambda: clock[0])
                timing.setattr(hold, '_decode', decoding)
                return normal_decode(hold, allowance)

        monkeypatch.setattr(BreakinHold, '_normal_decode', costed)
        got = []
        monkeypatch.setattr(sys, 'argv', ['onair', '--after-connect', 'p1-robust', '--p4-link',
                                         '--p4-data', 'body', '--mail-fetch', '--mail-wait-greeting',
                                         '--no-qrm-guard', '--p4-breakin', breakin])
        monkeypatch.setattr(onair, 'run', lambda args: got.append(args) or 0)
        assert onair.main() == 0
        stream = Stream(session.load_wav(str(path/'stream.flac'), FS))
        tx = Tx(stream)
        tx.p4_link_options = got[0]
        tx.p4_probe.emissions = [{k: e[k] for k in ('audio_start', 'audio_end', 'value', 'header')}
                                 for e in opening]
        tx.p4_probe.values = tuple(e['value'] for e in opening)
        stream.pos = opening[-1]['audio_end']
        tx.settle = .044
        key = tx._tx

        def flushing(audio, what):
            key(audio, what)
            stream.pos = tx.tx_end

        tx._tx = flushing
        host = PtcHost(SimPeer(), mycall='W9SSJ')
        host.arq.state = arq.State.CONNECTED
        host.arq.role = arq.IRS
        message = compose('W9SSJ', 'SMTP:offline@example.net', 'P4 test', b'hello\r\n',
                          mid='P4Y270929', date=datetime(2026, 9, 29, tzinfo=timezone.utc))
        self.mail = MailClient(B2FSession('W9SSJ', role='calling', target='WS8EOC',
                                          password='offline-test-password', outbox=[message]),
                               host.arq.on_host_data)
        host.app = self.mail
        self.mail.link_up()
        tx.p4_link_host, tx.p4_link_mail = host, self.mail
        onair._run_p4_link(tx, stream, onair._MasterGrid(0, 60000, 0, packet_n=36000,
                                                         cs_n=12000, d_max_n=4800))
        self.probe = tx.p4_probe
        self.keyed = tx.p4_probe.emissions[len(opening):]


@pytest.mark.parametrize('name,idle,crc', [('0923', 100.994, False), ('0936', 74.746, True)])
def test_ws8eoc_idle_requires_a_checked_packet_for_a_bare_y27(
        monkeypatch, name, idle, crc):
    run = Replay(name, monkeypatch)
    assert run.mail.session.sent_text
    queued = run.mail._send.__self__
    assert queued._buffer_raw == len(queued._outbuf) > 0
    assert not any(e.get('payload') for e in run.keyed)
    heard = [h for h in run.probe.heard if abs(h['onset_sample']-idle*FS) < FS//10
             and h['kind'] == Kind.NORMAL]
    assert heard and any('status' in h for h in heard) == crc
    repeats = {e["at"] for e in run.probe.link_events
               if e["event"] == "Keyed" and not e.get('current_slot_accepted', True)}
    after = [e for e in run.keyed if e['audio_start'] > idle*FS
             and e['audio_start'] not in repeats]
    assert [e['word'] for e in after] == (['Y27']*3 if crc else ['V27'])
    if crc:
        assert abs(after[0]['audio_start']-(idle*FS+T)) <= 48
    grants = [h for h in run.probe.heard if h.get('status') == 0x40 and h['kind'] == Kind.BODY]
    assert not grants and all(e['word'] for e in run.keyed)
    assert run.probe.reason == ('break-in unanswered' if crc else 'capture ended')


def test_no_ladder_link_sends_body_then_normal_after_the_granted_y27():
    m, _ = queued(ladder='none')
    ours = request(m)
    grant = ours+T
    m.observe(word('Y27', grant, extent=.8, responds_to=ours))
    m.observe(body(grant, IDLE, 0x40))
    p = m.decide(grant+T-m.lead)
    assert (p.kind, p.state, p.status, p.payload) == (Kind.BODY, State.ISS_DATA_ACK, 0, SID[:31])
    m.keyed()
    ack = p.stream_sample+round(RESPONSE_S*FS)
    m.observe(word('X27', ack, extent=.245, responds_to=p.stream_sample), now=ack+round(.28*FS))
    q = m.decide(ack+round(.30*FS))
    assert (q.kind, q.status, q.payload) == (Kind.NORMAL, 1, SID[31:])
    assert (q.stream_sample-p.stream_sample) % round(1.25*FS) == 0


def test_refused_unread_grant_retry_keeps_the_four_key_budget():
    m, _ = queued(ladder='none')
    at = request(m)
    grant = at+round(.935*FS)
    m.observe(word('Y27', grant, extent=.8, responds_to=at))
    due = at+2*60000
    p = m.decide(grant+T-m.lead)
    assert (p.kind, p.word, p.stream_sample) == (Kind.WORD, 'Y27', due)
    m.refused('deadline')
    assert m.breakins == 1
    # R2 §6 advances on the peer phase; Refused spends no request key.
    for _ in range(3):
        due += 60000
        m.silence(due-m.lead)
        p = m.decide(due-m.lead)
        assert p and p.word == 'Y27' and p.stream_sample == due
        m.keyed()
    m.silence(due+60000-m.lead)
    assert m.state == State.DOWN and m.breakins == 4


def test_unanswered_request_after_long_keeps_peer_long_raster():
    from hfmodem.tests.shrike.test_p4_turn_recognition_0929 import idle_long
    m, _ = queued()
    obs = idle_long(48000).observation
    obs = replace(obs, frame=replace(obs.frame, status=replace(obs.frame.status, protocol=0)))
    m.observe(RegisteredObservation(obs, Registered.NEW), now=48000+158400)
    due = 48000+163200
    assert m.decide(due-m.lead).word == 'Y27'
    m.keyed()
    m.silence(due+60000-m.lead)
    assert m.decide(due+60000-m.lead).stream_sample == due+180000


def test_terminal_receive_failure_uses_selected_breakin_policy():
    m, _ = queued()
    m.adapter.terminal = True
    m.observe(body(0, b'bad', 0x04))
    assert m.state == State.IRS_CLOSE
    at = 60000
    m.observe(body(at, b'', 0), now=at+38000)
    p = m.decide(at+T-m.lead)
    assert p.kind == Kind.WORD and p.word == 'Y27' and p.state == State.IRS_BREAKIN


@pytest.mark.parametrize('answer', [True, False])
def test_k0003_long_request_grant_sequence_keeps_one_turn_machine(answer):
    from hfmodem.shrike.p4receiver import Clock, Expect
    base = 1529.406
    audio = k0003(base, 1537.2)
    r = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=3.)
    r.enable_hold()
    r.clock = Clock(Expect.PACKET, FS//2, .330, 3.75, 3.)
    long = r._packet(audio[:round((1533.23-base)*FS)], 0)
    assert long.kind == Kind.LONG and long.frame.data.startswith(b'FF\r')
    assert long.frame.status.byte == 0x42
    m, _ = queued(ladder='none')
    m.observe(RegisteredObservation(long, Registered.NEW), now=long.stream_sample+round(3.31*FS))
    due = long.stream_sample+round(3.4*FS)
    p = m.decide(due-m.lead)
    assert p.word == 'Y27' and p.stream_sample == due
    m.keyed()
    if not answer:
        m.silence(due+60000-m.lead)
        assert m.decide(due+60000-m.lead).stream_sample == due+180000
        return
    r.note_emission(Emission(Kind.WORD, Role.ISS, due, .240))
    origin = round((1533.65-base)*FS)
    end = round((1535.08-base)*FS)
    regs = r.read_window(audio[origin:end], origin, whole=True)
    for reg in regs:
        m.observe(reg, now=end)
    grant = next(reg.observation for reg in regs if reg.observation.kind == Kind.BODY)
    assert grant.frame.status.byte == 0x40
    p = m.decide(grant.stream_sample+T-m.lead)
    assert p.kind == Kind.BODY and p.payload == SID[:31] and p.status == 0
