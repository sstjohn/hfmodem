# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the session says about the peer, held to what the peer actually sent.

WS8EOC, 7101.5 kHz, 2026-08-26. The gateway sent `0x59A` -- the PACTOR-3 upgrade
grant -- at ZERO bit errors on thirty-six consecutive cycles of its own 1.24999 s
raster, twenty-one of them after this station fell back to PACTOR-1 and sixteen
after the log printed `the peer never took the entry packet`. Then it IDed in CW
and left. It never keyed a data packet at any point: it held the receiving role
for the whole session.

The session's own account said the opposite at three levels and reversed the link
on the strength of it:

    [host] the peer never took the entry packet -> link falls back to PACTOR-1
    [grid] GRID REVERSED -> IRS: transmit anchor +840 ms
    [host] max retries -> yield the link and listen
    verdict: the peer never alternated its acknowledgement (3 CS1/CS2, 0 other)

`0 other` is the one to start from, because it is not a judgement -- it is a
count, and the words were never collected to be counted. `_SessionRx._on` kept
only `cs` events, and PACTOR-1's two unassigned words arrive as `unassigned`, so
ten zero-error `0x59A` reached the state machine and none of them reached the
tally. The verdict then read a gateway commanding a waveform once a second as an
empty channel, and the changeover asked `note_peer_heard` -- somebody is keying,
identified as nobody -- which that gateway satisfied by answering us correctly.

THE CORPUS IS SOMEBODY ELSE'S RECEIVER AND OUR OWN, and neither is our encoder.
`witness.wav` is a KiwiSDR at Empire, Michigan, 103 miles off, on its own clock
and its own dial; `hold_NN.wav` are this session's own windows. Every claim below
about what was on the air is read out of one of the two by `p1rx`, and the
assertions are about what the modem then SAYS. A round trip through
`pactor1.control_signal` would pin the wording and nothing else, which is the
failure mode that has cost this project days five times.

Run:  pytest hfmodem/tests/shrike/test_honest_account.py
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from hfmodem.shrike import onair, pactor1, rxfront, spec
from hfmodem.shrike.arq import (ENTRY_GRANT_CYCLES, IRS, ISS,
                                UPGRADE_SILENCE_CYCLES, State)
from hfmodem.shrike.ptc import Protocol, PtcHost
from hfmodem.shrike.session import load_wav
from hfmodem.tests import evidence

FS = rxfront.FS
SESSION = evidence.CAPTURES / "onair-0826-1041"

#: `0x59A` as an index into `pactor1.CS_WORDS`, and its name in the log.
GRANT_WORD = pactor1.CS_59A
GRANT_NAME = spec.P1_CS_NAMES[GRANT_WORD]

#: Windows of that session whose audio holds a zero-error grant, and how many.
#: Established in `test_codeword_reach`, which reads them out of the recordings.
HELD = {"hold_08": 1, "hold_09": 1, "hold_10": 1, "hold_11": 2, "hold_12": 1,
        "hold_13": 3, "hold_14": 1, "hold_23": 1}

#: ...and windows of the same session with no gateway in them at all: WS8EOC had
#: signed off in CW by 92.1 s. The burst detector still offers eight onsets.
EMPTY = ("hold_24", "hold_25", "hold_26", "hold_27", "hold_28")

requires_session = pytest.mark.skipif(
    not SESSION.is_dir(), reason=f"{SESSION} is not on this machine")


class _Seam:
    """A transmit seam that keys nothing. The FSM needs one to reach CONNECTED."""

    def attach(self, host) -> None:
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


def _grant(t: float = 0.2):
    """The event a zero-error `0x59A` reaches the host as. Its shape is
    `_SessionRx._p1_cs`'s: an `unassigned` kind, no `cs`, the word in `spare`."""
    return rxfront.Event(t, "unassigned", f"{GRANT_NAME} at anchor "
                         f"(0 bit errors, PACTOR-1, shift normal)",
                         protocol="PACTOR-1", spare=GRANT_WORD, sense=0)


def _cs(cs: int, t: float = 0.1):
    return rxfront.Event(t, "cs", f"CS{cs + 1}/anchor (0 bit errors, PACTOR-1)",
                         protocol="PACTOR-1", cs=cs, sense=0)


def _granted() -> PtcHost:
    """A linked ISS that has just taken the gateway's grant and keyed an entry
    packet -- where this station stood at TX[16] of that session."""
    host = PtcHost(peer=_Seam(), mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.on_rx_event(_cs(pactor1.CS_SPEED))     # CS4: the link runs 100 Bd
    host.tick()
    host.arq.on_host_data(b"traffic behind the upgrade")
    host.on_rx_event(_grant())
    assert host.protocol is Protocol.PACTOR3, "the arm did not take the grant"
    return host


def _run_out_the_window(host: PtcHost, answers) -> list[str]:
    """Cycle until the upgrade window ends, feeding `answers()` each cycle."""
    budget = (UPGRADE_SILENCE_CYCLES * len(host.arq.cfg.entry_ladder)
              + ENTRY_GRANT_CYCLES + 1)
    for _ in range(budget):
        for ev in answers():
            host.on_rx_event(ev)
        host.tick()
        if host.protocol is Protocol.PACTOR1:
            break
    return host.log_lines


# -- the fallback line: three states, and it says which -----------------------

def test_the_fallback_names_the_peer_that_answered_rather_than_denying_it():
    """DEFECT 1, the sentence the whole investigation was read off.

    Sixteen zero-error grants arrived after this line printed, and it says the
    peer did nothing. What the receiver was actually in is the state below --
    heard, at zero bit errors, and not acted on -- and that is the finding an
    operator can do something with.

    THE PROPERTY, NOT THE SENTENCE. What has to hold is that the line credits the
    peer with having answered and counts the answers; the wording it does that in
    is free to change, and pinning the whole sentence is what makes the next
    rewording look like a regression.
    """
    host = _granted()
    lines = _run_out_the_window(host, lambda: [_grant()])
    fell = [ln for ln in lines if "falls back to PACTOR-1" in ln]
    assert fell, lines[-4:]
    assert "the peer never took the entry packet" not in fell[0]
    assert "it answered" in fell[0], fell[0]
    assert "zero bit errors" in fell[0], fell[0]
    # ...and it must not claim silence, which is the other window's finding.
    assert "NOTHING AT ALL" not in fell[0], fell[0]


def test_a_window_that_carried_nothing_says_so_and_says_only_that():
    host = _granted()
    lines = _run_out_the_window(host, list)
    fell = [ln for ln in lines if "falls back to PACTOR-1" in ln]
    assert fell and "WE HEARD NOTHING AT ALL" in fell[0], fell[-1:] or lines[-3:]


def test_a_burst_under_our_own_carrier_is_reported_as_ours():
    """The third state, and the one no count of the peer's could ever show.

    `note_burst` is fed `_MasterGrid.nearest_gap_n` -- samples from OUR
    data ending to the burst -- and a negative one is a burst that began while we
    were still keyed. Five of the twenty codewords WS8EOC lost to this station's
    carrier that session are on the far side of the changeover; the other fifteen
    are cycles of exactly this shape.
    """
    host = _granted()

    def answer():
        # The -38 ms the session printed as 1212, eleven cycles running.
        host.arq.note_burst(-38.0)
        return []

    lines = _run_out_the_window(host, answer)
    fell = [ln for ln in lines if "falls back to PACTOR-1" in ln]
    assert fell, lines[-4:]
    assert "WE WERE TRANSMITTING OVER THE ANSWER" in fell[0], fell[0]
    assert "about this station, not about the peer" in fell[0], fell[0]


def test_the_signed_turnaround_is_what_makes_the_three_separable():
    """`nearest_gap_n` around the cycle rather than about it.

    The session printed `nearest at 1212 ms` for eleven consecutive cycles on an
    answer that was 38 ms EARLY and clean -- a near miss on the near edge, given
    the far edge's number. Everything above rests on the sign.
    """
    slot = round(spec.CYCLE_SHORT_S * FS)
    r = onair._MasterGrid(0, slot, round(0.185 * FS),
                          packet_n=round(spec.P1_PACKET_S * FS),
                          cs_n=round(spec.P1_CS_S * FS),
                          d_max_n=onair._d_max_n(spec.CYCLE_SHORT_S, 0.04))
    early = r.anchor + r.rx_ref_n - round(0.038 * FS)
    r.update([early])
    assert r.nearest_gap_n is not None
    assert r.nearest_gap_n / FS == pytest.approx(-0.038, abs=0.002)


# -- the changeover: it follows something the peer sent ------------------------

def test_a_peer_that_only_answers_never_takes_the_channel_from_us():
    """THE BIG ONE. Every reversal of 2026-08-26 fired on our own retry counter.

    The gateway sent a bare control signal every cycle and no data packet at all,
    which is an IRS doing exactly what an IRS owes. Reversing into a role it had
    not asked for put this station's 120 ms codeword 69 ms in front of its answer
    instant for five cycles -- so the trigger is now a direct contributor to
    keying over another station, not only to a wrong verdict.
    """
    host = _granted()
    _run_out_the_window(host, lambda: [_grant()])
    for _ in range(host.arq.cfg.max_retries + 6):
        host.on_rx_event(_grant())
        host.tick()
        if host.arq.state not in (State.CONNECTED, State.DISCONNECTING):
            break
    assert host.arq.role is ISS, "the link reversed onto a peer that was answering"
    assert not any("yield the link and listen" in ln for ln in host.log_lines)
    told = [ln for ln in host.log_lines if "NOT reversing" in ln]
    assert told, "the declined changeover was silent: " + str(host.log_lines[-3:])


def test_a_peer_transmitting_data_while_we_send_still_takes_the_channel():
    """NEGATIVE CONTROL, and the case the yield was built for.

    A CRC-valid data packet arriving while this end also holds the sending role
    is the stranded ISS -- both ends believing they transmit, each deaf to the
    other -- and nothing but an ISS produces one. That yield must still fire, or
    the fix above has traded one dead link for another.
    """
    from hfmodem.shrike.arq import ArqConfig, ArqIO, PactorArq

    class _Air(ArqIO):
        def __init__(self):
            self.lines: list[str] = []

        def log(self, msg):
            self.lines.append(msg)

        def __getattr__(self, name):
            return lambda *a, **k: None

    air = _Air()
    cfg = ArqConfig(max_retries=4)
    a = PactorArq(air, cfg)
    a.on_host_connect("W9SSJ", "WS8EOC")
    a.on_rx_cs(0)                                # CS_ACK: the link is up
    a.on_host_data(b"hello")
    for _ in range(cfg.max_retries + 4):
        if a.role == IRS:
            break
        a.on_rx_packet(1, b"", 0, True)          # its packet, dropped on our role
        a.on_cycle()
    assert a.role == IRS, f"the deafness yield never fired: {air.lines[-3:]}"
    assert any("sent a data packet of its own" in m for m in air.lines), air.lines


# -- the verdict: the words are collected, so the count is a count -------------

def test_the_summary_counts_the_words_the_peer_actually_sent():
    """DEFECT 2. `(3 CS1/CS2, 0 other)` over ten zero-error grants."""
    log = [_cs(pactor1.CS_ACK_A, t) for t in (29.86, 31.11, 32.36)]
    log += [_grant(t) for t in (33.61, 34.86, 36.11, 37.36, 38.61,
                                41.12, 43.59, 46.08, 52.30, 59.73)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        onair._summary(log, [1, 2, 3], 61, [(1, 1.0)], spec.CYCLE_SHORT_S,
                       [1, 2], onair._ConnectEvidence(), ended="under test")
    out = buf.getvalue()
    verdict = next(ln for ln in out.splitlines() if ln.startswith("verdict:"))
    assert ", 0 other" not in verdict, verdict
    assert f"10 other: {GRANT_NAME} x10" in verdict, verdict
    assert "control signals decoded 13 (13 PACTOR-1" in out, out


def test_the_held_answer_line_does_not_call_a_grant_a_repeat_request():
    """`_held_answers`, which is what `** LINK DOWN **` prints.

    In PACTOR-1 an unbroken CS1 is a peer asking for the same packet again. A
    `0x59A` is not in that alternation at all, and calling a run of it a repeat
    request reads a gateway commanding a waveform as a stalled link.
    """
    said = onair._held_answers([_cs(pactor1.CS_ACK_A), _grant(), _grant()])
    assert "every one a repeat request" not in said, said
    assert "the alternation cannot count" in said, said
    assert GRANT_NAME in said, said


# -- and the same, off the recordings the gateway is in -----------------------

def _sessrx() -> onair._SessionRx:
    host = PtcHost(peer=_Seam(), mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.role, host.arq.dxcall = IRS, "WS8EOC"
    host.arq._enter_connected()
    host.on_rx_event = lambda ev: None
    return onair._SessionRx(host, tag="TEST")


def _held_cycle(audio: np.ndarray) -> onair._SessionRx:
    """One receiving cycle as the hold loop runs it, and the receiver after it."""
    rx = _sessrx()
    for i in range(0, audio.size, int(0.25 * FS)):
        rx.bridge(audio[i:i + int(0.25 * FS)])
    rx.skip(0.96)
    rx.flush()
    onair._read_codeword_at_bursts(
        rx, audio, 0, [at for at, _ in onair._peer_bursts(audio, 0)])
    return rx


def _window(name: str) -> np.ndarray:
    return load_wav(SESSION / f"{name}.wav").astype(np.float32)


@requires_session
def test_every_grant_our_own_recordings_hold_reaches_the_tally():
    """The count the verdict is taken over, off the session's own windows.

    Not a rendered codeword: this is the gateway's own audio, through the
    receiver as the cycle runs it, into the list `_summary` reads.
    """
    reached = {name: len([ev for ev in _held_cycle(_window(name)).cs_log
                          if ev.spare == GRANT_WORD])
               for name in HELD}
    assert reached == HELD, reached


@requires_session
def test_the_windows_with_no_gateway_in_them_reach_the_tally_with_nothing():
    for name in EMPTY:
        assert not _held_cycle(_window(name)).cs_log, name


# -- the forecast: it rests on a codeword or it says nothing -------------------

class _Tx:
    """The two instants the forecast reads, and the measured lead between them."""

    def __init__(self, key_up: int, lead_s: float = 0.028):
        self.tx_key_up = key_up
        self.boundary = key_up + round(lead_s * FS)
        self.tx_end = self.boundary + round(spec.P1_PACKET_S * FS)
        self.settle = 0.040


def _grid(onset: int) -> onair._MasterGrid:
    slot = round(spec.CYCLE_SHORT_S * FS)
    r = onair._MasterGrid(0, slot, round(0.185 * FS),
                          packet_n=round(spec.P1_PACKET_S * FS),
                          cs_n=round(spec.P1_CS_S * FS),
                          d_max_n=onair._d_max_n(spec.CYCLE_SHORT_S, 0.04))
    r.update([onset], hushed=True, since_tx=slot)
    return r


def _forecast(rx, onset: int) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        onair._forecast_next_key(rx, _Tx(onset + 2 * FS), _grid(onset), 0)
    return buf.getvalue().strip()


@requires_session
def test_the_forecast_speaks_only_where_a_codeword_was_read():
    """DEFECT 3, against the two kinds of cycle the session had.

    `[predict]` alarmed on one of that session's 22 forecasts and was wrong by
    124 ms: it named TX[19], a `template` cycle the KiwiSDR puts clear by
    113 ms in front and 184 ms behind, and said nothing about the twenty
    codewords the same recording shows we transmitted over. On the cycle it spoke about, the grid
    had already logged `no control signal where it is due` -- so its own
    conditions failed to corroborate it, and nothing checked.
    """
    with_word = _forecast(_held_cycle(_window("hold_23")), 10 * FS)
    assert with_word.startswith("[predict]"), with_word
    assert GRANT_NAME in with_word, with_word
    for name in EMPTY:
        assert not _forecast(_held_cycle(_window(name)), 10 * FS), name


def test_a_cycle_with_bursts_and_no_codeword_gets_no_forecast():
    """The rule with no recording in front of it, which is the one that holds.

    `_report_collision` still times every burst it can see; what it may no longer
    do is extrapolate one it never identified onto a cycle that has not happened.
    """
    onset = 10 * FS
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        onair._report_collision([(onset, round(spec.P1_CS_S * FS))],
                                _Tx(onset + 2 * FS))
    seen = buf.getvalue()
    assert "[clear]" in seen or "[collide]" in seen, seen
    assert "predict" not in seen, seen
