# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""VE3KPG 2026-09-29 10:24: checked invitations and bounded requests."""

import hashlib
import json
from pathlib import Path

from hfmodem.tests.shrike import p4_stock_peer as stock

import numpy as np
import pytest

from hfmodem.shrike import onair, p4oracle, p4breakintx, p4frame, p4shorttx, p4turn
from hfmodem.shrike.p4receiver import Clock, Emission, Expect, Kind, Observation, P4Receiver, Role
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx
from hfmodem.tests.shrike.test_p4turn import body, machine, word

ROOT = Path(__file__).resolve().parents[5]
SID = b"[Pat-1.0.0-B2FHM$]\r;PR: 25286779\r; VE3KPG DE W9SSJ (EN63bc)\rFC EM 1TQ2Z9X 608 404 0\rF> 7A\r"
GRID = onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800)


def idle_long(at):
    frame = p4frame.decode_field(p4frame.encode_field(b"\x1e" * 69, p4frame.Status.from_byte(0x40)),
                                 data_bytes=69)
    obs = Observation(Kind.LONG, FS, 0, at, at, 3.28, 0, .9, frame=frame)
    return p4turn.RegisteredObservation(obs, p4turn.Registered.NEW)


def granted(ladder, *, queued=True):
    """1024 at 89.749: the peer's idle protocol-2 long draws our answer."""
    m, a = machine(data="body", ladder=ladder, role="irs")
    if queued:
        a.on_host_data(SID)
    offer = 4307951
    m.observe(idle_long(offer), now=offer + round(3.3 * FS))
    grant = m.decide(offer + round(3.4 * FS) - m.lead)
    assert (grant.kind, grant.word, grant.state, grant.stream_sample) == (
        Kind.WORD, 'Y27' if queued else 'X27',
        State.IRS_BREAKIN if queued else State.IRS_ANSWER, offer+163200)
    assert not grant.payload and not m.outbox.inflight and a.tx_acked_bytes == 0
    epoch = m._receive_epoch
    m.keyed()
    assert m.state == (State.IRS_BREAKIN if queued else State.IRS_ANSWER)
    assert m._receive_epoch == epoch and m.grants == 0
    assert not [e for e in m.link_events if e['event'] == 'turn']
    return m, a, grant.stream_sample


@pytest.mark.parametrize("ladder", ["long", "none"])
def test_peer_v27_to_our_request_starts_the_ladder_or_sid_body(ladder):
    m, a, grant = granted(ladder)
    yielded = grant + round(.950 * FS)
    m.observe(word("V27", yielded, extent=.245, responds_to=grant), now=yielded + round(.28 * FS))
    proposal = m.decide(yielded + round(.30 * FS))
    if ladder == 'none':
        assert proposal.kind == Kind.BODY and proposal.state == State.ISS_DATA
        assert proposal.status == 0x40 and proposal.payload == SID[:31]
    else:
        assert (proposal.kind, proposal.value, proposal.state) == (Kind.SHORT, 13, State.ISS_LADDER)
        assert not m.outbox.inflight and bytes(a._outbuf) == SID
    assert proposal.stream_sample == grant + round(3.75 * FS)
    assert any(e["event"] == "turn" and e["responds_to"] == grant for e in m.link_events)


@pytest.mark.parametrize("ladder", ["long", "none"])
def test_v27_that_answers_nothing_of_ours_does_not_take_the_turn(ladder):
    m, a, grant = granted(ladder)
    stray = grant + round(.600 * FS)
    m.observe(word("V27", stray, extent=.245, responds_to=None), now=stray + round(.28 * FS))
    proposal = m.decide(stray + round(.30 * FS))
    assert proposal is None or (proposal.kind == Kind.WORD and not proposal.payload
                                and proposal.state in (State.IRS_BREAKIN, State.IRS_ANSWER))
    assert not [e for e in m.link_events if e['event'] == 'turn']
    assert a.tx_acked_bytes == 0 and bytes(a._outbuf) == SID
    assert m.outbox.inflight == b""


def test_unanswered_word_request_recovers_after_four_actual_keys():
    m, a, grant = granted("none")
    at = grant
    for _ in range(3):
        m.silence(at + 60000)
        proposal = m.decide(at + 180000 - m.lead)
        assert (proposal.kind, proposal.word, proposal.state) == (
            Kind.WORD, 'Y27', State.IRS_BREAKIN)
        assert proposal.stream_sample == at+180000
        at = proposal.stream_sample
        m.keyed()
    m.silence(at + 60000)
    proposal = m.decide(at + 180000 - m.lead)
    assert (proposal.kind, proposal.word, proposal.state) == (
        Kind.WORD, 'X27', State.IRS_ANSWER)
    assert m.state == State.IRS_ANSWER and m.breakins == 4
    assert m.peer_watch.attempts == 4 and m.next_offer_at == at+960000
    assert m._held_answer.answer.word == 'X27'
    assert not m.outbox.inflight and bytes(a._outbuf) == SID
    assert a.tx_acked_bytes == 0 and not proposal.payload
    other, _, first = granted('none')
    other.silence(first+60000)
    other.decide(first+180000-other.lead)
    other.refused('deadline')
    assert other.peer_watch.attempts == 1 and other.breakins == 1


@pytest.mark.parametrize("queued", [True, False])
def test_peer_empty_grant_after_our_v60_answers_with_data_when_bytes_wait(queued):
    """1024 at 100.349: Y27 + idle 0x40 answered our v60, not a grant of ours."""
    m, a = machine(data="body")
    if queued:
        a.on_host_data(SID)
    v60 = round(99.399 * FS)
    m._short(v60, 1)
    m.keyed()
    grant = v60 + round(.950 * FS)
    m.observe(word("Y27", grant, extent=.80, responds_to=v60))
    m.observe(body(grant, b"", 0x40))
    proposal = m.decide(grant + round(.92 * FS) - m.lead)
    assert proposal.stream_sample == grant + round(.92 * FS)
    if queued:
        assert proposal.kind == Kind.BODY and proposal.payload == SID[:31] and proposal.status == 0
        assert proposal.state == State.ISS_DATA_ACK
    else:
        assert proposal.status == 0x40 and not proposal.payload


def test_bytes_arriving_while_irs_request_and_wait_for_associated_release():
    m, a, answer = granted("none", queued=False)
    assert m.state == State.IRS_ANSWER
    a.on_host_data(b"FF\r")
    m.reschedule_breakin(answer+180000-m.lead)
    request = m.decide(answer+180000-m.lead)
    assert (request.kind, request.word, request.state) == (Kind.WORD, 'Y27', State.IRS_BREAKIN)
    assert request.stream_sample == answer+180000 and not m.outbox.inflight
    assert a.tx_acked_bytes == 0
    m.keyed()
    yielded = request.stream_sample+44880
    m.observe(word('V27', yielded, extent=.245, responds_to=request.stream_sample), now=yielded+12000)
    proposal = m.decide(request.stream_sample+180000-m.lead)
    assert (proposal.kind, proposal.payload, proposal.status) == (Kind.BODY, b'FF\r', 0x40)
    assert proposal.stream_sample == request.stream_sample+180000
    assert a.tx_acked_bytes == 0


def test_ladder_none_requires_bodies():
    with pytest.raises(ValueError):
        machine(data="hold", ladder="none")


def _data_exchange(lost, ladder):
    """A peer invitation, then a peer that acks each counted packet by parity at +0.950 except `lost`
    and answers a ladder short as the tape does."""
    audio = np.zeros(24 * FS)
    grant = round(.935 * FS)
    wave = stock.body(b"", 0x40)
    audio[grant:grant + len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options.p4_ladder = ladder
    tx.p4_link_options.p4_data = "body"
    tx.p4_link_options.p4_link_timeout = 21.
    arq = tx.p4_link_host.arq
    text = bytes(range(0x21, 0x21 + 31 + 40 * 4))
    arq.on_host_data(text)
    words = {name: stock.word(name) for name in ("V27", "X27", "U27")}
    keyed = tx._tx
    bodies = []

    def peer(audio_out, what):
        keyed(audio_out, what)
        answer = None
        if "body" in what and "status=0x40" in what and "payload=b''" in what:
            answer = "V27"
        elif "body" in what or "normal" in what:
            counter = sum(i not in lost for i in range(len(bodies))) % 4
            answer = None if len(bodies) in lost else "V27" if counter & 1 else "X27"
            bodies.append(audio_out)
        elif "short" in what:
            answer = "X27" if what.endswith("13") else "U27"
        if answer:
            at = tx.tx_audio_start + round(.950 * FS)
            stream.audio[at:at + len(words[answer])] = words[answer]
        stream.pos = tx.tx_end

    tx._tx = peer
    onair._run_p4_link(tx, stream, GRID)
    return tx, arq, text, bodies


@pytest.mark.parametrize("ladder", ["none", "long"])
def test_body_then_normals_through_counter_wrap_with_a_lost_ack(ladder):
    tx, arq, text, waves = _data_exchange({2}, ladder)
    sent = [e for e in tx.p4_probe.emissions[1:] if e["payload"]]
    fields = [bytes.fromhex(e["payload"]) for e in sent]
    assert [e["status"] for e in sent] == [0, 1, 2, 2, 3, 0]
    assert fields == [text[0:31], text[31:71], text[71:111], text[71:111], text[111:151], text[151:191]]
    from hfmodem.shrike import p4shorttx
    for i, (e, wave) in enumerate(zip(sent, waves)):
        expected = (p4breakintx.changeover_packet if i == 0 else p4shorttx.short_packet)(bytes.fromhex(e["payload"]), e["status"])
        assert e["waveform_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
        assert np.array_equal(wave, expected)
    gaps = np.diff([e["audio_start"] for e in sent]).tolist()
    assert gaps[2] == 60000 and all(g % 60000 == 0 for g in gaps)
    shorts = [e["value"] for e in tx.p4_probe.emissions[1:] if e.get("value") is not None]
    assert shorts == ([] if ladder == "none" else [13, 60])
    assert not [e for e in tx.p4_probe.link_events if e["event"] == "Refused"]
    keyed = [e for e in tx.p4_probe.link_events if e["event"] == "Keyed" and e["kind"] in (Kind.BODY, Kind.NORMAL)]
    assert [(e["status"], e["payload"]) for e in keyed][:6] == [(e["status"], e["payload"]) for e in sent]
    settled = [n for prev, n in zip(arq.io.counts, arq.io.counts[1:]) if n < prev]
    assert settled == [160, 120, 80, 40, 0]
    assert arq._buffer_raw == 0 and not arq._outbuf


def recorded_1024():
    capture = ROOT / "captures/onair-0929-1024"
    if not (capture / "stream.flac").exists():
        pytest.skip("local VE3KPG 2026-09-29 10:24 capture absent")
    audio = onair.session.load_wav(str(capture / "stream.flac"), FS)
    probe = json.loads((capture / "p4-probe.json").read_text())
    # Our own recorded keyings are not the peer; the replay keys its own.
    for e in probe["emissions"]:
        audio[e["audio_start"]:e["audio_end"]] = 0
    return audio


def replay_1024(monkeypatch):
    audio = recorded_1024()
    offer = 4307951
    stream = Stream(audio)
    stream.pos = offer - round(.6 * FS)
    tx = Tx(stream)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_ladder = "none"
    tx.p4_link_options.p4_data = "body"
    tx.p4_link_options.p4_link_timeout = 13.
    tx.p4_link_host.arq.on_host_data(SID)
    rx = tx.sessrx.p4
    start = round(30.8 * FS)
    head = p4oracle.word_at(audio[start:round(31.4 * FS)], FS, start,
                           30.999-start/FS, cfo_hz=1.50, expected_words={"Y27"})
    assert head is not None and head.word == "Y27"
    assert head.cfo_hz == pytest.approx(3.027, abs=.02)
    rx._emit(head, [])
    rx.clock = Clock(Expect.PACKET, offer, .330, 3.75, rx.cfo_hz)
    tx.p4_probe.handoff_audio = offer - 6 * FS, audio[offer - 6 * FS:stream.pos].copy()
    turn = p4turn.P4Turn
    tx.compat_v27_snapshots = []

    def turn_machine(*args, **kwargs):
        m = turn(*args, **{**kwargs, 'role': 'irs'})
        observe = m.observe

        def watched(reg, **kw):
            obs = reg.observation
            def snapshot():
                return (m.state, m.outbox.inflight, bytes(m.outbox.arq._outbuf),
                        m.outbox.arq._buffer_raw, m.outbox.arq.tx_acked_bytes)
            before = snapshot()
            result = observe(reg, **kw)
            if obs.word == 'V27':
                tx.compat_v27_snapshots.append((reg, before, snapshot()))
            return result

        m.observe = watched
        return m

    monkeypatch.setattr(p4turn, 'P4Turn', turn_machine)
    keyed = tx._tx

    def key(audio_out, what):
        keyed(audio_out, what)
        stream.pos = tx.tx_end

    tx._tx = key
    onair._run_p4_link(tx, stream, GRID)
    return tx


def test_1024_replay_requests_only_on_checked_peer_packets(monkeypatch):
    tx = replay_1024(monkeypatch)
    sent = tx.p4_probe.emissions
    requests = [e["audio_start"] for e in sent if e.get('word') == 'Y27' and not e['payload']]
    assert len(requests) == 2
    assert all(abs(a-b) <= 2 for a, b in zip(requests, (4471151, 4651151)))
    assert not any(e.get("value") is not None for e in sent)
    assert not any(e['payload'] for e in sent)
    assert not [r for r in tx.p4_probe.heard if r['kind'] == Kind.BODY
                and r.get('status') == 0x40 and r.get('responds_to') in requests]
    assert tx.p4_link_host.arq._buffer_raw == len(SID)
    assert bytes(tx.p4_link_host.arq._outbuf) == SID


def test_1024_replay_does_not_infer_a_grant_without_a_reply(monkeypatch):
    tx = replay_1024(monkeypatch)
    v27 = [r for r in tx.p4_probe.heard if r.get("word") == "V27" and not r["provisional"]
           and abs(r["onset_sample"] - round(97.849 * FS)) < FS // 50]
    assert len(v27) == 1
    assert v27[0]['onset_sample'] == 4696755
    partial, = [r for r in tx.p4_probe.heard if r.get('word') == 'V27' and r['provisional']]
    assert partial['onset_sample'] == 4696757
    for row in (partial, v27[0]):
        assert row['responds_to'] == 4651151 and row['extent_s'] is None
        assert not row['weak'] and not row['unexpected']
    assert len(tx.compat_v27_snapshots) == 2
    for reg, before, after in tx.compat_v27_snapshots:
        assert reg.responds_to == 4651151 and reg.observation.extent_s is None
        assert before == after == (State.IRS_BREAKIN, b'', SID, len(SID), 0)
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'turn']
    assert tx.p4_link_host.arq.tx_acked_bytes == 0
    assert tx.p4_link_host.arq._buffer_raw == len(SID)
    assert bytes(tx.p4_link_host.arq._outbuf) == SID
    assert not [e for e in tx.p4_probe.emissions if e['payload']]
    assert tx.p4_probe.reason == 'session ceiling'


@pytest.mark.parametrize("after", [.905, .950, .965])
def test_a_word_answering_our_body_is_placed_in_the_090_097_bracket(after):
    body_at = FS
    audio = np.zeros(3 * FS)
    at = body_at + round(after * FS)
    v27 = stock.word("V27")
    audio[at:at + len(v27)] = v27
    rx = P4Receiver(FS, admitted=True, cfo_hz=0., breakin=True)
    rx.enable_hold()
    rx.note_emission(Emission(Kind.BODY, Role.ISS, body_at, .798))
    regs = rx.read_window(audio, 0, whole=True)
    words = [r for r in regs if r.observation.word == "V27" and not r.observation.provisional]
    assert words and abs(words[0].observation.stream_sample - at) <= 2
    assert words[0].responds_to == body_at


def test_ws8eoc_1151_normal_short_headers_tell_first_copies_from_repeats():
    capture = ROOT / "captures/onair-0929-1151"
    if not (capture / "stream.flac").exists():
        pytest.skip("local WS8EOC 2026-09-29 11:51 capture absent")
    audio = onair.session.load_wav(str(capture / "stream.flac"), FS)
    labels = {}
    for t in (24.745, 27.245, 28.495, 29.745, 40.995, 99.745):
        at = round(t * FS)
        rx = P4Receiver(FS, admitted=True, breakin=True)
        rx.enable_hold()
        rx.peer_onset = at
        rx.clock = Clock(Expect.PACKET, at, .330, 1.25, 1.69)
        lo = at - 2 * FS
        rx._normal(audio[lo:at + 2 * FS], lo)
        heard = [e for e in rx.live_events if e["event"] == "normal-header"]
        labels[t] = heard[0]["header"] if heard else None
    assert labels == {24.745: "A", 27.245: "A", 28.495: "B", 29.745: "B", 40.995: "B", 99.745: "B"}
    assert len(p4shorttx.HEADER_B) == p4shorttx.HEADER_CHIPS


@pytest.mark.parametrize("ladder", ["long", "none"])
def test_partial_grant_answer_waits_for_full_confirmation_in_driver(monkeypatch, ladder):
    audio = np.zeros(4*FS)
    at = round(.950*FS)
    wave = stock.word('V27')
    audio[at:at+len(wave)] = wave
    live = Stream(audio)
    tx = Tx(live)
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_ladder = ladder
    tx.p4_link_options.p4_link_timeout = 3.
    tx.p4_link_host.arq.on_host_data(SID)
    receiver = tx.sessrx.p4
    init = p4turn.P4Turn.__init__
    def begin(turn, *a, **kw):
        init(turn, *a, **kw)
        turn._body(0, 0x40, State.ISS_GRANT)
        turn.keyed()
        receiver.note_emission(Emission(Kind.BODY, Role.ISS, 0, .8))
    monkeypatch.setattr(p4turn.P4Turn, '__init__', begin)
    onair._run_p4_link(tx, live, GRID)
    full = [r for r in tx.p4_probe.heard if r.get('word') == 'V27' and not r['provisional']]
    assert full and full[0]['responds_to'] == 0
    sent = tx.p4_probe.emissions
    assert sent
    if ladder == 'none':
        assert sent[0]['payload'] == SID[:31].hex()
    else:
        assert sent[0]['value'] == 13 and not sent[0]['payload']
        assert tx.p4_link_host.arq._buffer_raw == len(SID)
    assert sent[0]['audio_start'] == 120000
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'Refused']


def test_default_hold_policy_does_not_send_queued_bytes_after_grant_v27():
    m, a = machine()
    a.on_host_data(SID)
    m._body(0, 0x40, State.ISS_GRANT)
    m.keyed()
    m.observe(word('V27', 45600, responds_to=0), now=57600)
    p = m.decide(57600)
    assert p is None or not p.payload
    assert not m.outbox.inflight and a._buffer_raw == len(SID)


@pytest.mark.parametrize('status', [0, 0x20])
def test_hold_wakeup_preserves_pending_peer_data_or_close(status):
    m, a = machine(data='body', ladder='none')
    m.state = State.ISS_HOLD
    a.on_host_data(SID)
    m.observe(body(FS, b'peer', status), now=FS+38000)
    assert m.decide(FS+39000) is None
    p = m.decide(FS+44160-m.lead)
    assert p.kind == Kind.WORD and p.stream_sample == FS+44160
    assert p.state == (State.IRS_CLOSE if status else State.IRS_ANSWER)
    assert not m.outbox.inflight


def test_repeated_empty_grant_rechecks_new_application_bytes():
    m, a = machine(data='body', ladder='none', role='irs')
    m.observe(body(0, b'', 0x40), now=38000)
    m.decide(44160-m.lead)
    m.keyed()
    a.on_host_data(SID)
    m.observe(body(60000, b'', 0x40), now=98000)
    p = m.decide(104160-m.lead)
    assert p.payload == SID[:31] and p.status == 0


@pytest.mark.parametrize('breakin,kind', [('body', Kind.BODY), ('y27', Kind.WORD)])
def test_0936_crc_good_idle_breaks_in_when_bytes_wait(breakin, kind):
    from hfmodem.tests.shrike.test_p4_normal_short_0928 import normal
    m, a = machine(role='irs', data='body', breakin=breakin)
    at = round(74.746*FS)
    a.on_host_data(SID)
    m.observe(p4turn.RegisteredObservation(normal(at, status=0, data=b''), p4turn.Registered.NEW),
              now=at+round(.810*FS))
    p = m.decide(at+44160-m.lead)
    assert p.kind == kind and p.state == State.IRS_BREAKIN
    assert p.stream_sample == round(75.666*FS)
