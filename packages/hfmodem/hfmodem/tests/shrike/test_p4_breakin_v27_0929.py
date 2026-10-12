# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""WS8EOC-80 2026-09-29 23:27 and 23:49: the turn after our empty break-in body."""

import json
from pathlib import Path

import pytest

from hfmodem.shrike import onair, p4turn
from hfmodem.shrike.p4receiver import Emission, Kind, Role
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx
from hfmodem.tests.shrike.test_p4turn import body, machine, word

ROOT = Path(__file__).resolve().parents[5]
GRID = onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800)
B2F = (b";FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;PR: 12345678\r; WS8EOC DE W9SSJ (EN63bc)\r"
       b"FC EM K0BVVV7209VS 654 486 0\rF> 8D\r")
FIELDS = [B2F[i:i + 31] for i in range(0, len(B2F), 31)]

# 2349: our break-in bodies and WS8EOC's V27 to each (onset, provisional read, full read).
BREAKIN_2349 = 1532074
V27_2349 = 1576693, 1588065, 1630455
LEAD_2349 = 2286
DECISION_2349 = 1592074 - LEAD_2349


def breaking_in(key, ladder="none"):
    m, a = machine(role="irs", data="body", ladder=ladder, prekey_lead=LEAD_2349 / FS)
    a.on_host_data(B2F)
    m._body(key, 0x40, State.IRS_BREAKIN)
    m.keyed()
    return m, a


@pytest.mark.parametrize("ladder", ["none", "long"])
def test_2349_partial_v27_to_our_breakin_body_holds_the_repeat_and_the_full_word_starts_data(ladder):
    m, a = breaking_in(BREAKIN_2349, ladder)
    onset, partial, full = V27_2349
    early = word("V27", onset, partial=True, responds_to=BREAKIN_2349)
    m.observe(early, now=partial)
    assert m.decide(DECISION_2349) is None
    m.silence(DECISION_2349)
    assert m.decide(DECISION_2349) is None
    m.observe(word("V27", onset, outcome=p4turn.Registered.CONFIRMED, replaced=early.observation,
                   responds_to=BREAKIN_2349), now=full)
    p = m.decide(full)
    if ladder == 'none':
        assert p.kind == Kind.BODY and p.state == State.ISS_DATA
        assert p.status == 0x40 and p.payload == FIELDS[0]
    else:
        assert (p.kind, p.value, p.state) == (Kind.SHORT, 13, State.ISS_LADDER)
        assert not m.outbox.inflight and bytes(a._outbuf) == B2F
    assert p.stream_sample == BREAKIN_2349 + 2 * 60000
    assert any(e["event"] == "turn" and e["responds_to"] == BREAKIN_2349 for e in m.link_events)


def test_2349_full_v27_read_before_the_decision_keys_data_on_the_next_raster():
    m, a = breaking_in(BREAKIN_2349)
    onset, partial, _ = V27_2349
    early = word("V27", onset, partial=True, responds_to=BREAKIN_2349)
    m.observe(early, now=partial)
    m.observe(word("V27", onset, outcome=p4turn.Registered.CONFIRMED, replaced=early.observation,
                   responds_to=BREAKIN_2349), now=partial + 400)
    p = m.decide(DECISION_2349)
    assert p.stream_sample == BREAKIN_2349 + 60000
    assert p.status == 0x40 and p.payload == FIELDS[0]


@pytest.mark.parametrize("responds_to", [None, BREAKIN_2349 - 60000])
def test_a_v27_that_does_not_answer_our_breakin_body_does_not_take_the_turn(responds_to):
    m, a = breaking_in(BREAKIN_2349)
    onset = BREAKIN_2349 + round(.600 * FS)
    m.observe(word("V27", onset, responds_to=responds_to), now=onset + round(.28 * FS))
    m.silence(DECISION_2349)
    p = m.decide(DECISION_2349)
    assert p is not None and p.state == State.IRS_BREAKIN and p.status == 0x40 and not p.payload
    assert m.outbox.inflight == b""
    assert not any(e["event"] == "turn" for e in m.link_events)


def replay_2349(monkeypatch):
    capture = ROOT / "captures/onair-0929-2349"
    if not (capture / "stream.flac").exists():
        pytest.skip("local WS8EOC 2026-09-29 23:49 capture absent")
    audio = onair.session.load_wav(str(capture / "stream.flac"), FS)
    probe = json.loads((capture / "p4-probe.json").read_text())
    for e in probe["emissions"]:
        audio[e["audio_start"]:e["audio_end"]] = 0
    end = BREAKIN_2349 + 38286
    stream = Stream(audio)
    stream.pos = end
    tx = Tx(stream)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_ladder = "none"
    tx.p4_link_options.p4_data = "body"
    tx.p4_link_options.p4_link_timeout = (1652074 - end) / FS
    tx.p4_link_host.arq.on_host_data(B2F)
    rx = tx.sessrx.p4
    rx.cfo_hz = 17.3
    tx.p4_probe.handoff_audio = end - 4 * FS, audio[end - 4 * FS:end].copy()
    init = p4turn.P4Turn.__init__

    def begin(turn, *a, **kw):
        init(turn, *a, **{**kw, "role": "irs"})
        turn._body(BREAKIN_2349, 0x40, State.IRS_BREAKIN)
        turn.keyed()
        rx.note_emission(Emission(Kind.BODY, Role.ISS, BREAKIN_2349, 38286 / FS))

    monkeypatch.setattr(p4turn.P4Turn, "__init__", begin)
    keyed = tx._tx

    def key(audio_out, what):
        keyed(audio_out, what)
        stream.pos = tx.tx_end

    tx._tx = key
    onair._run_p4_link(tx, stream, GRID)
    return tx


def test_2349_replay_keys_the_sid_body_on_the_raster_after_the_peer_v27(monkeypatch):
    tx = replay_2349(monkeypatch)
    v27 = [r for r in tx.p4_probe.heard if r.get("word") == "V27" and not r["provisional"]
           and abs(r["onset_sample"] - V27_2349[0]) < FS // 50]
    assert v27 and v27[0]["responds_to"] == BREAKIN_2349
    sent = tx.p4_probe.emissions
    assert sent and sent[0]["audio_start"] == BREAKIN_2349 + 60000
    assert sent[0]["status"] == 0x40 and bytes.fromhex(sent[0]["payload"]) == FIELDS[0]


# 2327: our break-in body, WS8EOC's Y27 + idle 0x40 grant read late and repeated,
# our first data body, and the peer's X27 to it.
BREAKIN_2327 = 5672320
GRANTS_2327 = 5716972, 5776950
DATA_2327 = 5821132
X27_2327 = 5865753


def test_2327_grant_then_x27_streams_every_b2f_body_with_counted_status_under_ladder_none():
    m, a = breaking_in(BREAKIN_2327)
    first, repeat = GRANTS_2327
    m.observe(word("Y27", first, extent=.8, responds_to=BREAKIN_2327), now=first + 40000)
    m.observe(body(first, b"", 0x40), now=first + 42000)
    # B: decoded after the lead but before its due, the grant opens the data
    # turn in its own slot; a deadline refusal rolls it to the repeat's slot.
    early = m.decide(first + 42000)
    assert (early.stream_sample, early.state, early.payload) == (
        first + round(.920 * FS), State.ISS_DATA_ACK, FIELDS[0])
    m.refused("deadline")
    m.observe(word("Y27", repeat, extent=.8, responds_to=None), now=repeat + 30000)
    m.observe(body(repeat, b"", 0x40), now=repeat + 30000)
    # The repeat grant now times the data from its own onset, 22 samples
    # before the flight's deferred key at DATA_2327.
    p = m.decide(DATA_2327 - m.lead)
    assert p.stream_sample == repeat + round(.920 * FS) == DATA_2327 - 22
    assert p.state == State.ISS_DATA_ACK
    assert (p.status, p.payload) == (0, FIELDS[0])
    sent = [m.keyed()]
    ack = X27_2327
    while True:
        m.observe(word("V27" if sent[-1].status & 1 else "X27", ack, responds_to=sent[-1].stream_sample), now=ack + 12000)
        p = m.decide(ack + 12000)
        if p is None or not p.payload:
            break
        assert p.kind == Kind.NORMAL and p.stream_sample == sent[-1].stream_sample + 60000
        sent.append(m.keyed())
        ack = sent[-1].stream_sample + round(.930 * FS)
    assert b"".join(s.payload for s in sent) == B2F
    assert [s.status for s in sent] == [k % 4 for k in range(len(sent))]
    assert m.outbox.inflight == b"" and not m.outbox.queued
    assert not any(e.value is not None for e in m._emissions.values() if e.kind == Kind.SHORT)
