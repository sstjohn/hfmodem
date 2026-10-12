# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Packet acceptance, deadlines, and proposals versus physical emissions."""

from dataclasses import replace
import pytest
from hfmodem.shrike import arq, p4frame, compress
from hfmodem.shrike.p4host import P4Host, P4Outbox, P4StreamFailure
from hfmodem.shrike.p4receiver import (
    Kind,
    Observation,
    Registered,
    RegisteredObservation,
)
from hfmodem.shrike.p4turn import P4Turn, State

FS = 48000


class IO(arq.ArqIO):
    def __init__(self):
        self.counts = []

    def buffer(self, n):
        self.counts.append(n)


def machine(**kwargs):
    a = arq.PactorArq(IO())
    a.state = arq.State.CONNECTED
    return P4Turn(P4Outbox(a), P4Host(), **kwargs), a


def word(
    name,
    at,
    partial=False,
    outcome=Registered.NEW,
    replaced=None,
    answered=None,
    extent=None,
    responds_to="short",
):
    o = Observation(
        Kind.WORD,
        FS,
        0,
        at,
        at,
        0.24,
        0,
        0.9,
        word=name,
        score=.99,
        provisional=partial,
        extent_s=extent,
    )
    return RegisteredObservation(o, outcome, replaced, answered,
                                 at-round(.92*FS) if responds_to == "short" else responds_to)


def body(at, data=b"hello", status=0):
    frame = p4frame.decode_field(
        p4frame.encode_field(data.ljust(31, b"\x1e"), p4frame.Status.from_byte(status)),
        data_bytes=31,
    )
    o = Observation(Kind.BODY, FS, 0, at, at, 0.796, 0, 0.9, frame=frame, after=at)
    return RegisteredObservation(o, Registered.ENRICHED)


def test_byte_accounting_and_escape_cut():
    m, a = machine()
    a.on_host_data(b"A" * 30 + b"\x1eB")
    assert m.outbox.transfer() == b"A" * 30
    assert a._buffer_raw == 33 and len(a._outbuf) == 3
    a.on_host_data(b"C")
    m.outbox.commit()
    assert a._buffer_raw == len(a._outbuf) == 4
    assert m.outbox.transfer() == compress.transparent(b"\x1eBC")
    m.outbox.requeue()
    assert a._buffer_raw == len(a._outbuf) == 4 and a._inflight is None


def test_unsupported_stream_is_terminal_and_duplicates_keep_supervisor_state():
    h = P4Host()
    f = body(0, b"hi").observation.frame
    assert h.accept(f) == b"hi" and h.accept(f) is None
    assert h.accept(replace(f, status=p4frame.Status(1))) == b"hi"
    with pytest.raises(P4StreamFailure):
        h.accept(replace(f, status=p4frame.Status(2, source_mode=1)))
    with pytest.raises(P4StreamFailure):
        h.accept(body(0, b"after").observation.frame)


@pytest.mark.parametrize(
    "extent,accepted,grant",
    [
        (0.245, False, True),
        (0.80, True, True),
        (0.80, False, False),
        (None, False, False),
    ],
)
def test_y27_decides_once(extent, accepted, grant):
    m, a = machine()
    a.on_host_data(b"ours")
    m.outbox.transfer()
    y = FS
    early = word("Y27", y)
    m.observe(early)
    if extent is not None:
        m.observe(
            word(
                "Y27",
                y,
                outcome=Registered.EXTENDED,
                replaced=early.observation,
                extent=extent,
            )
        )
    if accepted:
        m.observe(body(y, b"", 0x40))
    due = y + round(0.92 * FS)
    decision = due - m.lead
    assert m.decide(decision - 1) is None
    p = m.decide(decision)
    assert bool(p) == grant
    if p:
        assert p.stream_sample == due and p.status == 0x40
        m.keyed()
        assert m.state == State.ISS_GRANT
    else:
        assert m.unread == 1
    m.observe(word("Y27", y, outcome=Registered.EXTENDED, extent=0.245), now=due)
    assert m.decide(due) is None
    assert m.outbox.inflight == b"ours"


@pytest.mark.parametrize(
    "resolution", [Registered.CONFIRMED, Registered.CONTRADICTED, Registered.EXPIRED]
)
def test_data_ack_admits_short_but_only_full_ack_commits(resolution):
    m, a = machine()
    a.on_host_data(b"payload")
    m._data(0, State.ISS_DATA_ACK, Kind.BODY)
    m.keyed()
    w = round(0.92 * FS)
    partial = word("X27", w, True)
    m.observe(partial, now=w + round(0.18 * FS))
    p = m.decide(w + round(0.256 * FS))
    assert p.value == 13 and m.outbox.inflight == b"payload"
    short = m.keyed()
    m.observe(
        word(
            "X27" if resolution == Registered.CONFIRMED else "Z27",
            w,
            outcome=resolution,
            replaced=partial.observation,
            answered=short.stream_sample,
        ),
        now=short.stream_sample + round(0.1 * FS),
    )
    if resolution == Registered.CONFIRMED:
        assert m.outbox.inflight == b"" and a._buffer_raw == 0
    else:
        assert m.outbox.inflight == b"payload"
        assert m.decide(
            short.stream_sample
        ).stream_sample == short.stream_sample + round(1.25 * FS)
        assert m.decide(short.stream_sample).kind == Kind.BODY


def test_late_decode_delivers_and_answers_once_over_an_unread_next_copy():
    m, a = machine(role="irs")
    y = FS
    m.observe(word("Y27", y, extent=0.8))
    assert m.decide(y + round(0.846 * FS)) is None
    m.observe(body(y), now=y + FS)
    assert m.decide(y + FS) is None
    y2 = y + round(1.25 * FS)
    m.observe(word("Y27", y2, extent=0.8))
    assert m.decide(y2 + round(0.846 * FS)).stream_sample == y2 + round(0.92 * FS)
    assert m.unread == 2


def test_refused_short_does_not_advance_ladder():
    m, a = machine()
    m.start(0)
    m.keyed()
    m.observe(word("X27", round(0.92 * FS)), now=round(1.1 * FS))
    assert m.decide(round(1.176 * FS)).value == 60
    m.refused("late")
    assert m.index == 0
    assert m.decide(0).stream_sample == round(2.5 * FS)


def test_refused_speculative_short_retains_body_until_resolution():
    m, a = machine()
    a.on_host_data(b"payload")
    m._data(0, State.ISS_DATA_ACK, Kind.BODY)
    m.keyed()
    w = round(0.92 * FS)
    early = word("X27", w, True)
    m.observe(early)
    m.refused("late")
    assert m.state == State.ISS_DATA_ACK and m.outbox.inflight == b"payload"
    m.observe(
        word("X27", w, outcome=Registered.CONFIRMED, replaced=early.observation),
        now=round(1.4 * FS),
    )
    p = m.decide(round(1.4 * FS))
    assert p.value == 13 and p.stream_sample == round(2.5 * FS)
    assert m.outbox.inflight == b""


def test_unread_y27_cancels_retransmission_and_expired_word_does_not():
    m, a = machine()
    m.start(0)
    m.keyed()
    y = round(0.92 * FS)
    early = word("Y27", y, True)
    m.observe(early)
    m.decide(y + round(0.846 * FS))
    m.silence(y + FS)
    assert m.decide(y + FS) is None
    m.observe(
        word("Y27", y, True, outcome=Registered.EXPIRED, replaced=early.observation)
    )
    m.silence(y + FS)
    assert m.decide(y + FS).value == 13


def test_v49_does_not_accept_unknown_word_and_long_requires_renderer():
    m, a = machine(ladder="chirp")
    m.index = 1
    m._short(0, 1)
    m.keyed()
    m.observe(word("UNKNOWN", round(0.92 * FS)))
    assert m.decide(0).value == 49
    n, a = machine()
    n.index = 1
    n._short(0, 1)
    n.keyed()
    n.observe(word("U27", round(0.92 * FS)))
    assert n.decide(0).kind == Kind.LONG
    n.refused("no renderer")
    assert n.state == State.ISS_HOLD


def test_no_close_body_carries_inflight_bytes():
    m, a = machine(app_done=True)
    a.on_host_data(b"last")
    m._data(0)
    sent = m.keyed()
    assert sent.payload == b"last"
    m.observe(word("X27", round(0.92 * FS)), now=round(1.10 * FS))
    p = m.decide(round(1.176 * FS))
    assert p.status == 0x20 and p.payload == b"" and a._buffer_raw == 0


def packet(kind, at, value=None, partial=False):
    return RegisteredObservation(
        Observation(
            kind,
            FS,
            0,
            at,
            at,
            0.758 if kind == Kind.SHORT else 0.72,
            0,
            0.9,
            value=value,
            provisional=partial,
        ),
        Registered.NEW,
    )


def answer(m, name, at):
    m.observe(word(name, at, responds_to=m.last.stream_sample), now=at + round(0.24 * FS))
    p = m.decide(at + round(0.256 * FS))
    assert p is not None
    return m.keyed()


@pytest.mark.parametrize(
    "ladder,data", [("long", "hold"), ("long", "body"), ("chirp", "chirp")]
)
def test_take_turn_ladder_data_and_close_trace(ladder, data):
    m, a = machine(role="irs", ladder=ladder, data=data, app_done=True)
    a.on_host_data(b"A" * 31 + b"B" * 22)
    y = 0
    m.observe(word("Y27", y, extent=0.8))
    m.observe(body(y, b"peer SID"))
    assert m.decide(round(0.846 * FS)).word == "X27"
    m.keyed()
    normal = round(1.25 * FS)
    m.observe(packet(Kind.NORMAL, normal))
    p = m.decide(normal + round(0.846 * FS))
    assert p.status == 0x40
    request = m.keyed()
    assert m.state == State.IRS_BREAKIN
    y = request.stream_sample + round(0.92 * FS)
    m.observe(word("Y27", y, extent=0.8))
    m.observe(body(y, b"", 0x40))
    p = m.decide(y + round(0.846 * FS))
    assert p.payload == b"A" * 31
    sent = m.keyed()
    assert m.state == State.ISS_DATA_ACK
    short = answer(m, "X27", sent.stream_sample + round(0.92 * FS))
    assert short.value == 13
    short = answer(m, "X27", short.stream_sample + round(0.92 * FS))
    assert short.value == (60 if ladder == "long" else 49)
    if ladder == 'chirp':
        short = answer(m, 'X27', short.stream_sample + round(.92*FS))
        assert short.value == 53
    m.observe(
        word(
            "U27", short.stream_sample + round(0.92 * FS)
        )
    )
    p = m.decide(short.stream_sample + FS)
    if data == "hold":
        assert p.kind == Kind.LONG
        m.refused("no renderer")
        assert m.state == State.ISS_HOLD
        assert m.decide(p.stream_sample + round(5 * 1.25 * FS)) is None
        # Peer retakes the turn during the hold; the remaining queue survives.
        y = p.stream_sample + round(4 * 1.25 * FS)
        m.observe(word("Y27", y, extent=0.8))
        m.observe(body(y, b"", 0x40))
        assert m.decide(y + round(0.846 * FS)).status == 0x40
        sent = m.keyed()
        assert m.state == State.ISS_GRANT
        y = sent.stream_sample + round(0.92 * FS)
        m.observe(word("Y27", y, extent=0.8))
        m.observe(replace(body(y, b"FS YY\r"), responds_to=sent.stream_sample))
        assert m.decide(y + round(0.846 * FS)).word == "V27"
        m.keyed()
        assert m.state == State.IRS_ANSWER
        assert bytes(a._outbuf) == b"B" * 22
        return
    assert p.kind == (Kind.CHIRP if data == "chirp" else Kind.NORMAL)
    assert p.payload == b"B" * 22
    sent = m.keyed()
    w = sent.stream_sample + round((3.4 if data == "chirp" else 0.92) * FS)
    close = answer(m, "X27" if data == 'chirp' else "V27", w)
    assert close.status == 0x20 and close.payload == b""
    assert m.counter == 2 and a._buffer_raw == 0
    m.observe(word("X27", close.stream_sample + round(0.92 * FS)))
    assert m.decide(close.stream_sample + FS) is None
    m.silence(close.stream_sample + FS)
    qrt = m.keyed()
    assert qrt.value == 37 and qrt.header == "A"
    m.observe(word("V27", qrt.stream_sample + round(0.92 * FS)))
    assert m.state == State.DOWN and m.reason == "close acknowledged"


def test_data_rollback_ignores_the_valueless_short_answer():
    m, a = machine()
    a.on_host_data(b"payload")
    m._data(0, State.ISS_DATA_ACK)
    m.keyed()
    w = round(0.92 * FS)
    early = word("X27", w, True)
    m.observe(early)
    short = m.keyed()
    m.observe(
        word(
            "Z27",
            w,
            outcome=Registered.CONTRADICTED,
            replaced=early.observation,
            answered=short.stream_sample,
        )
    )
    m.observe(word("X27", short.stream_sample + round(0.92 * FS)))
    p = m.decide(short.stream_sample + FS)
    assert p.kind == Kind.BODY and p.payload == b"payload"
    assert p.stream_sample == short.stream_sample + round(1.25 * FS)
    assert a._buffer_raw == 7


def test_grant_exhaustion_requeues_before_a_fresh_ladder():
    m, a = machine()
    a.on_host_data(b"payload")
    m.outbox.transfer()
    m._body(0, 0x40, State.ISS_GRANT)
    m.keyed()
    for _ in range(3):
        m.silence(0)
        m.keyed()
    m.silence(0)
    p = m.decide(0)
    assert p.value == 13 and p.state == State.ISS_LADDER
    assert not m.outbox.inflight and bytes(a._outbuf) == b"payload"


def test_body_and_chirp_retransmissions_preserve_the_waveform_key():
    for kind in (Kind.BODY, Kind.CHIRP):
        m, a = machine()
        a.on_host_data(b"payload")
        m._data(0, kind=kind)
        first = m.keyed()
        m.silence(FS)
        repeat = m.decide(FS)
        assert repeat.chirp_slot == (1 if first.kind == Kind.CHIRP else 0)
        assert replace(repeat, stream_sample=0, repeat=False, chirp_slot=0) == first
        assert m.outbox.inflight == b"payload"


def test_close_retries_are_a37_and_irs_silence_requests_repeat():
    m, a = machine(app_done=True)
    m._data(0)
    m.keyed()
    for _ in range(3):
        m.silence(FS)
        p = m.keyed()
        assert p.value == 37 and p.header == "A"
    m.silence(FS)
    assert m.state == State.DOWN
    n, a = machine(role="irs")
    a.on_host_data(b"payload")
    n._body(0, 0x40, State.IRS_BREAKIN)
    n.keyed()
    n.observe(word("Z27", round(0.92 * FS)))
    assert n.state == State.IRS_ANSWER
    n.silence(FS)
    seed = n.decide(FS)
    assert seed.word == 'V27' and seed.purpose is None
    assert seed.accepted_target is None and not seed.current_slot_accepted


def test_terminal_source_failure_never_delivers_a_later_supported_packet():
    delivered = []
    from types import SimpleNamespace

    m, a = machine(role="irs")
    m.adapter = P4Host(
        SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    )
    a.on_host_data(b"not exchanged")
    m.observe(body(0, b"first"))
    bad = body(FS, b"bad", 4)
    m.observe(bad)
    assert m.state == State.IRS_CLOSE and m.adapter.failure and delivered == [b"first"]
    m.observe(body(2 * FS, b"later"))
    p = m.decide(2 * FS + round(0.846 * FS))
    assert p.status == 0x40
    m.keyed()
    assert m.state == State.IRS_BREAKIN
    y = p.stream_sample + round(0.92 * FS)
    m.observe(word("Y27", y, extent=0.8))
    m.observe(body(y, b"", 0x40))
    close = m.decide(y + round(0.846 * FS))
    assert close.status == 0x20 and close.payload == b""
    assert delivered == [b"first"] and a._buffer_raw == len(b"not exchanged")


@pytest.mark.parametrize(
    "name,outcome",
    [
        ("X27", Registered.CONFIRMED),
        ("Z27", Registered.CONTRADICTED),
        ("X27", Registered.EXPIRED),
    ],
)
def test_refused_speculation_resolves_on_the_refused_raster(name, outcome):
    m, a = machine()
    a.on_host_data(b"data")
    m._data(0, State.ISS_DATA_ACK)
    m.keyed()
    w = round(0.92 * FS)
    early = word("X27", w, True)
    m.observe(early)
    m.refused("deadline")
    m.observe(
        word(name, w, outcome=outcome, replaced=early.observation), now=round(1.4 * FS)
    )
    p = m.decide(round(1.4 * FS))
    assert p.stream_sample == round(2.5 * FS)
    assert m.counter == (1 if outcome == Registered.CONFIRMED else 0)
    assert p.kind == (Kind.SHORT if outcome == Registered.CONFIRMED else Kind.BODY)


def test_refused_ladder_speculation_expires_to_the_old_step():
    m, a = machine()
    m.start(0)
    m.keyed()
    w = round(0.92 * FS)
    early = word("X27", w, True)
    m.observe(early)
    assert m.decide(w).value == 60
    m.refused("deadline")
    m.observe(
        word("X27", w, True, outcome=Registered.EXPIRED, replaced=early.observation),
        now=round(1.4 * FS),
    )
    p = m.decide(round(1.4 * FS))
    assert p.value == 13 and p.header == "B" and p.stream_sample == round(2.5 * FS)


def test_speculative_shorts_do_not_reset_the_data_retry_budget():
    m, a = machine()
    a.on_host_data(b"data")
    m._data(0, State.ISS_DATA_ACK)
    m.keyed()
    for _ in range(7):
        w = m.last.stream_sample + round(0.92 * FS)
        early = word("X27", w, True)
        m.observe(early)
        short = m.keyed()
        m.observe(
            word(
                "T27",
                w,
                outcome=Registered.CONTRADICTED,
                replaced=early.observation,
                answered=short.stream_sample,
            )
        )
        m.keyed()
    assert m.data_repeats == 7
    m.silence(m.last.stream_sample + FS)
    p = m.decide(m.last.stream_sample + FS)
    assert p.kind == Kind.SHORT and p.value == 13
    assert not m.outbox.inflight and bytes(a._outbuf) == b"data"


# First ten decoded frames in captures/onair-0929-{1151,2200}/p4-probe.json.
# Header A/B labels are from the offline reads cited in plan §3.2 and REPORT.
ACK_GREETINGS = {
    '1151': ((887736, 947733, 1007733, 1067737, 1127734, 1187753,
              1247756, 1307756, 1367756, 1427756),
             b'95 daily minutes remaining with WS8EOC (',
             b'EN72QQ)\r[WL2K-5.0-B2FWIHJM$]\r;PQ: 100167'),
    '2200': ((947828, 1007827, 1067827, 1127835, 1187842, 1247859,
              1307852, 1367852, 1427852, 1487849),
             b'112 daily minutes remaining with WS8EOC ',
             b'(EN72QQ)\r[WL2K-5.0-B2FWIHJM$]\r;PQ: 38772'),
}


def ack_frame(at, data, status, kind=Kind.NORMAL, header='A'):
    n = 31 if kind == Kind.BODY else 40
    frame = p4frame.decode_field(p4frame.encode_field(
        data.ljust(n, b'\x1e'), p4frame.Status.from_byte(status)), data_bytes=n)
    assert frame is not None
    obs = replace(body(at).observation, kind=kind, frame=frame, header=header)
    return RegisteredObservation(obs, Registered.NEW)


def ack_greeting(arm):
    times, c1, c2 = ACK_GREETINGS[arm]
    return [ack_frame(at, b'RMS Trimode 1.4.4.0\r\nW9SSJ has ', 0, Kind.BODY)
            for at in times[:5]] + [
        ack_frame(at, c1 if i < 2 else c2, 1 if i < 2 else 2,
                  header='A' if i in (0, 2) else 'B')
        for i, at in enumerate(times[5:])]


def propose_ack(m, frame):
    at = frame.observation.stream_sample
    m.observe(frame, now=at)
    return m.decide(at + round(m.t[frame.observation.kind]*FS) - m.lead)


@pytest.mark.parametrize('arm', ACK_GREETINGS)
@pytest.mark.parametrize('policy,words,reasons', [
    ('v27', ['V27']*5, ['fixed']*5),
    ('x27', ['X27']*5, ['fixed']*5),
    ('parity', ['V27']*2 + ['X27']*3, ['counter 1']*2 + ['counter 2']*3),
    ('alternate-packet', ['V27']*2 + ['X27']*3,
     ['new packet after X27', 'repeat keeps V27', 'new packet after V27',
      'repeat keeps X27', 'repeat keeps X27']),
])
def test_normal_ack_recorded_repeated_packet(arm, policy, words, reasons):
    m, _ = machine(role='irs', normal_ack=policy)
    expected_words = ['X27']*5 + words
    expected_reasons = ['opening body']*5 + reasons
    for i, frame in enumerate(ack_greeting(arm)):
        previous = m.ack_word
        p = propose_ack(m, frame)
        assert p.word == expected_words[i]
        assert p.ack_reason == expected_reasons[i]
        assert m.ack_word == previous
        row = [e for e in m.link_events if e['event'] == 'normal-answer'][-1]
        assert row['ack_policy'] == policy and row['reason'] == p.ack_reason
        assert row['word'] == p.word and row['previous'] == previous
        assert row['status'] == row['counter'] == frame.observation.frame.status.byte
        assert row['repeat'] == (i not in (0, 5, 7))
        if row['repeat']:
            repeat = [e for e in m.link_events if e['event'] == 'acked-repeat'][-1]
            assert repeat['word'] == p.word and repeat['reason'] == p.ack_reason
            assert repeat['ack_policy'] == policy
            assert p.word == m._last_ack[1].word
        m.keyed()
        assert m.ack_word == p.word


@pytest.mark.parametrize('arm', ACK_GREETINGS)
def test_normal_ack_default_keeps_recorded_words(arm):
    m, _ = machine(role='irs')
    for i, frame in enumerate(ack_greeting(arm)):
        assert propose_ack(m, frame).word == ('X27' if i < 5 else 'V27')
        m.keyed()


@pytest.mark.parametrize('arm', ACK_GREETINGS)
def test_normal_ack_refused_key_keeps_history(arm):
    m, _ = machine(role='irs', normal_ack='alternate-packet')
    frames = ack_greeting(arm)
    for frame in frames[:7]:
        propose_ack(m, frame)
        m.keyed()
    last = m._last_ack
    p = propose_ack(m, frames[7])
    assert p.word == 'X27' and p.ack_reason == 'new packet after V27'
    assert m.ack_word == 'V27'
    m.refused('deadline')
    assert m.ack_word == 'V27' and m._last_ack == last
    p = propose_ack(m, frames[8])
    assert p.word == 'X27' and p.ack_reason == 'new packet after V27'
    m.keyed()
    p = propose_ack(m, frames[9])
    assert p.word == 'X27' and p.ack_reason == 'repeat keeps X27'
    m.keyed()


def test_normal_ack_packet_change_ignores_unrelated_words_and_turn_change():
    m, _ = machine(role='irs', normal_ack='alternate-packet')
    for frame in ack_greeting('2200'):
        propose_ack(m, frame)
        m.keyed()
    # Set history to V27 so an unrelated X27 would visibly corrupt it.
    p = propose_ack(m, ack_frame(1600000, b'007\rCMS via WS8EOC >\r', 3))
    assert (p.word, p.ack_reason) == ('V27', 'new packet after X27')
    m.keyed()
    short = replace(body(1700000).observation, kind=Kind.SHORT, frame=None, value=13)
    p = propose_ack(m, RegisteredObservation(short, Registered.NEW))
    assert p.word == 'X27' and p.ack_reason is None
    m.keyed()
    assert m.ack_word == 'V27'
    p = propose_ack(m, ack_frame(1800000, b'long', 0, Kind.LONG))
    assert p.word == 'X27' and p.ack_reason is None
    m.keyed()
    assert m.ack_word == 'V27'
    p = propose_ack(m, ack_frame(2000000, b'new packet', 3))
    assert (p.word, p.ack_reason) == ('X27', 'new packet after V27')
    m.refused('deadline')
    p = propose_ack(m, ack_frame(2100000, b'\x01\x01\x01wrapped', 0x80))
    assert (p.word, p.ack_reason) == ('X27', 'new packet after V27')
    m.keyed()
    # Granting the turn and returning to IRS does not reset the tracked ack.
    p = propose_ack(m, body(2200000, b'', 0x40))
    assert p.state == State.ISS_GRANT and p.ack_reason is None
    m.keyed()
    assert m.ack_word == 'X27'
    p = propose_ack(m, ack_frame(2300000, b'next turn', 1))
    assert (p.word, p.ack_reason) == ('V27', 'new packet after X27')
    m.keyed()
    assert m.state == State.IRS_ANSWER


@pytest.mark.parametrize('status', [0, 1, 2, 3, 0x80, 0x81, 0x82, 0x83])
def test_normal_ack_parity_covers_idle_and_extended_statuses(status):
    m, _ = machine(role='irs', normal_ack='parity')
    data = b'\x01\x01\x01text' if status & 0x80 else b''
    p = propose_ack(m, ack_frame(48000, data, status))
    assert p.word == ('V27' if status & 1 else 'X27')
    assert p.ack_reason == f'counter {status & 3}'


def test_normal_ack_without_previous_key_and_validation():
    m, _ = machine(role='irs', normal_ack='alternate-packet')
    p = propose_ack(m, ack_frame(48000, b'first normal', 2))
    assert (p.word, p.ack_reason) == ('V27', 'no previous ack')
    m.refused('deadline')
    assert m.ack_word is None
    p = propose_ack(m, ack_frame(108000, b'first normal', 2, header='B'))
    assert (p.word, p.ack_reason) == ('V27', 'no previous ack')
    m.keyed()
    assert machine()[0].normal_ack == 'v27'
    for policy in ('unknown', 'alternate-attempt'):
        with pytest.raises(ValueError):
            machine(normal_ack=policy)


def test_normal_ack_refused_c3_then_extended_packet_reads_last_key():
    m, _ = machine(role='irs', normal_ack='alternate-packet')
    for frame in ack_greeting('2200'):
        propose_ack(m, frame)
        m.keyed()
    p = propose_ack(m, ack_frame(1600000, b'007\rCMS via WS8EOC >\r', 3))
    assert (p.word, p.ack_reason) == ('V27', 'new packet after X27')
    m.refused('deadline')
    p = propose_ack(m, ack_frame(1660000, b'\x01\x01\x01wrapped', 0x80))
    assert (p.word, p.ack_reason) == ('V27', 'new packet after X27')
    m.keyed()
    assert m.ack_word == 'V27'


@pytest.mark.parametrize('kind,status,expected,seed', [
    (Kind.NORMAL, 0x21, 'V27', 'X27'),
    (Kind.LONG, 0x63, 'X27', 'V27'),
    (Kind.BODY, 0, 'V27', 'X27'),
    (Kind.CHIRP, 0x63, 'X27', 'V27'),
    (Kind.SHORT, 13, 'X27', 'V27'),
    (Kind.SHORT, 53, 'U27', 'V27'),
])
def test_normal_ack_unrelated_control_words_do_not_advance(kind, status, expected, seed):
    m, _ = machine(role='irs', normal_ack='alternate-packet')
    for frame in ack_greeting('1151')[:8 if seed == 'X27' else 7]:
        propose_ack(m, frame)
        m.keyed()
    if kind == Kind.SHORT:
        obs = replace(body(1600000).observation, kind=kind, frame=None, value=status)
        incoming = RegisteredObservation(obs, Registered.NEW)
    else:
        incoming = ack_frame(1600000, b'unrelated', status, kind)
    p = propose_ack(m, incoming)
    assert (p.word, p.ack_reason) == (expected, None)
    m.keyed()
    assert m.ack_word == seed


@pytest.mark.parametrize('ack_53', ['U27', 'X27'])
def test_the_word_answering_v53_is_chosen_by_the_operator(ack_53):
    m, _ = machine(role='irs', ack_53=ack_53)
    for frame in ack_greeting('1151')[:7]:
        propose_ack(m, frame)
        m.keyed()
    obs = replace(body(1600000).observation, kind=Kind.SHORT, frame=None, value=53)
    p = propose_ack(m, RegisteredObservation(obs, Registered.NEW))
    assert (p.word, p.peer_control) == (ack_53, 53)


@pytest.mark.parametrize('policy,expected', [
    ('x27', ['X27', 'X27', 'X27', 'X27']),
    ('alternate', ['X27', 'X27', 'V27', 'X27']),
])
def test_chirp_acks_follow_the_operators_policy(policy, expected):
    # K4MSU 2206: X27 moved 0x61 to 0x62, then 0x62 drew X27 on every copy and never moved.
    m, _ = machine(role='irs', chirp_ack=policy)
    words = []
    for i, status in enumerate((0x61, 0x61, 0x62, 0x63)):
        p = propose_ack(m, ack_frame(1600000 + i*180000, b'RMS Trimode', status, Kind.CHIRP))
        words.append(p.word)
        m.keyed()
    assert words == expected


def test_the_operator_can_choose_the_word_for_any_ladder_value():
    m, _ = machine(role='irs', peer_ack={51: 'U27'})
    for frame in ack_greeting('1151')[:7]:
        propose_ack(m, frame)
        m.keyed()
    obs = replace(body(1600000).observation, kind=Kind.SHORT, frame=None, value=51)
    assert propose_ack(m, RegisteredObservation(obs, Registered.NEW)).word == 'U27'
