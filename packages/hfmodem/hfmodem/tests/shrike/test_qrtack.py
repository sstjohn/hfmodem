# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The goodbye's acknowledgement, and what a codeword during teardown does.

A PACTOR disconnect is an exchange, not a state change: the QRT bit rides a data
packet and the peer acknowledges it, and `arq.PactorArq._on_ack` reaching
`_finish_disconnected` on `STATUS_QRT` is the ONLY path a link closes cleanly by.
So a receiver that stops admitting codewords when the goodbye goes out cannot
close a link at all -- every session must run its budget out and report the peer
as silent, whatever the peer actually sent.

That is what shipped. `_SessionRx.control_signal` and `_SessionRx._scan` both
gated on `State.CONNECTED`, while the transmit side gates on `LINKED` -- which
names DISCONNECTING outright, "a station saying goodbye is transmitting". This
station granted itself the right to transmit during teardown and denied the peer
the right to be heard.

MEASURED, K4MSU on 3595 kHz, 2026-08-19 22:07. Fifteen packets went out, the last
four carrying QRT; the session reported `HOLD 14 RX (quiet)` and ended `the peer
never acknowledged the goodbye`. The capture that cycle wrote to disk carries a
CS2 at ZERO bit errors, read alike by a free sweep over the window (0.048-0.121 s)
and by the grid-anchored point read (0.087-0.099 s) -- and the cycle before it
carries a CS1, so that CS2 is the alternation, which in PACTOR-1 is what an
acknowledgement IS. The goodbye was answered and the answer was not admitted.

The reversion is the reason this was measured before it was widened. A codeword
arriving during DISCONNECTING can pull the state back to CONNECTED, and the
failure mode to rule out is a link that will not close. The table below is run
rather than argued: only CS3 reverts, it is the peer breaking in, and the same
cycle's tick takes the link back and puts the goodbye on the air again. The
FSM caps the whole teardown at `arq.GOODBYE_CYCLES`, and the `--hold` loop at
its own `onair.QRT_CYCLES`.

Run: python -m hfmodem.tests.shrike.test_qrtack
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, pactor1, rxfront, spec
from hfmodem.shrike.arq import GOODBYE_CYCLES, ISS, State
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests import evidence

FS = rxfront.FS
SLOT_N = round(spec.CYCLE_SHORT_S * FS)
PACKET_N = round(spec.P1_PACKET_S * FS)
SETTLE_N = round(0.04 * FS)
#: K4MSU's turnaround, measured across two sessions ninety minutes apart:
#: 95/95/96 ms at 20:48, and 92/91/95 with the grid acquiring at 93.8 and 95.2.
D_N = round(0.0938 * FS)

CS1, CS2, CS3, CS4 = (pactor1.CS_ACK_A, pactor1.CS_ACK_B,
                      pactor1.CS_CHANGEOVER, pactor1.CS_SPEED)
NAME = {CS1: "CS1", CS2: "CS2", CS3: "CS3", CS4: "CS4"}

#: The 22:07 session's codeword train, one entry per hold cycle, None for a cycle
#: that decoded nothing. Six CS4 and three CS1 reached the state machine; the CS2
#: at hold 14 is the one on disk that did not.
TRAIN = (CS4, CS4, CS4, CS4, None, None, None, None, CS4, None, CS1, CS1,
         CS1, CS2, None, None)
HOLD = 12

#: The windows that session wrote, on this station's disk and no other clone's.
CAPTURES = evidence.CAPTURES / "onair-0819-2207"
RECORDED_ANCHOR = 0.093

#: The 22:10 session, whose sixteen hold windows are the ones a station saying
#: goodbye actually listens in: fourteen of them 0.237-0.245 s.
TEARDOWN = evidence.CAPTURES / "onair-0819-2210"

PACKET_S = spec.P1_PACKET_S

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


class _Peer:
    def __getattr__(self, k):
        return lambda *a, **kw: None


def cs_audio(cs: int | None) -> np.ndarray:
    """One cycle's listening window with the peer's codeword at the turnaround."""
    seg = np.zeros(SLOT_N - PACKET_N - SETTLE_N, np.float32)
    if cs is not None:
        rf = onair._trim_silence(np.asarray(pactor1.control_signal(cs), np.float32))
        seg[D_N:D_N + rf.size] += rf
    return seg


def cs_event(cs: int):
    return rxfront.Event(D_N / FS, "cs", f"CS{cs + 1}", protocol="PACTOR-1",
                         cs=cs, sense=0)


def calling_station(answer: int = CS4) -> PtcHost:
    """A linked ISS, brought up the way the setup loop brings one up.

    The connect answer reaches the FSM through the acquisition search, which
    hands its event straight to `_SessionRx._on` -- so it crosses no state gate,
    and the alternation reference every later codeword is read against is seeded
    by it (`ptc.PtcHost._logical_cs`).
    """
    host = PtcHost(peer=_Peer(), mycall="W9SSJ")
    # The session this file is measured against ran PACTOR-1 at 100 Bd end to
    # end: fifteen `P1 pkt 100Bd` transmissions and nine PACTOR-1 codewords, no
    # PACTOR-3 anything. An upgrade offered on every acknowledgement would put a
    # protocol on the air that the recording does not carry.
    host.stay_in_pactor1 = True
    host.arq.on_host_connect("W9SSJ", "K4MSU")
    host.on_rx_event(cs_event(answer))
    host.tick()
    return host


def saying_goodbye(before=(CS4, CS4, CS1)) -> PtcHost:
    """...with the QRT packet in flight and the alternation reference at CS1."""
    host = calling_station()
    for cs in before:
        host.on_rx_event(cs_event(cs))
        host.tick()
    host.arq.on_host_disconnect()
    host.tick()
    return host


def qrt_in_flight(host: PtcHost) -> bool:
    fl = host.arq._inflight
    return fl is not None and bool(fl.status & spec.STATUS_QRT)


def replay(train, *, gate) -> dict:
    """The train through the real receive path, with `LINKED` set to `gate`."""
    host = calling_station()
    counters: list[int] = []
    real_send = host.send_packet

    def spy(sl, payload, status, breakin=False):
        counters.append(status & 3)
        return real_send(sl, payload, status, breakin=breakin)

    host.send_packet = spy
    rx = onair._SessionRx(host)
    delivered: list[int | None] = []
    states: list[str] = []
    saved, onair.LINKED = onair.LINKED, gate
    try:
        for h, cs in enumerate(train, 1):
            if h == HOLD + 1:
                host.arq.on_host_disconnect()
            rx.new_cycle()
            states.append(host.arq.state)
            delivered.append(rx.control_signal(cs_audio(cs), 0, D_N))
            host.tick()
    finally:
        onair.LINKED = saved
    return {"host": host, "delivered": delivered, "states": states,
            "counters": counters}


# --------------------------------------------------------------------------
def the_goodbye_the_peer_answered() -> None:
    """The 22:07 train, replayed through `_SessionRx` at both gate widths."""
    print("\nThe K4MSU train of 2026-08-19 22:07, through the receive gate")
    wide = replay(TRAIN, gate=(State.CONNECTED, State.DISCONNECTING))
    narrow = replay(TRAIN, gate=(State.CONNECTED,))

    for run, tag in ((narrow, "gate at CONNECTED"), (wide, "gate at LINKED")):
        seen = [NAME[c] for c in run["delivered"] if c is not None]
        print(f"    {tag}: delivered {' '.join(seen)}; "
              f"final state {run['host'].arq.state}")

    check("the goodbye is BUILT in the cycle the host asks for it, so the cycle "
          "after is the first whose receive window DISCONNECTING gates -- which "
          "is the one the peer answers in",
          (wide["states"][HOLD], wide["states"][HOLD + 1])
          == (State.CONNECTED, State.DISCONNECTING),
          f"{wide['states'][HOLD]} then {wide['states'][HOLD + 1]}")
    check("...and the counter the session put on the air advanced once, #1 to #2",
          sorted(set(wide["counters"])) == [1, 2],
          f"#{' #'.join(str(c) for c in wide['counters'])}")
    check("the CS1 of the cycle BEFORE the goodbye reaches the FSM either way -- "
          "that cycle is still CONNECTED, which is why the defect hid",
          narrow["delivered"][HOLD - 1] == CS1 == wide["delivered"][HOLD - 1])

    check("THE FIX: the peer's alternation to CS2 reaches the state machine",
          wide["delivered"][HOLD + 1] == CS2)
    check("...and it acknowledges the goodbye, so the link closes",
          wide["host"].arq.state in (State.DISCONNECTED, State.LISTENING),
          wide["host"].arq.state)
    check("...with nothing of the teardown left standing",
          not wide["host"].arq._qrt_pending and not qrt_in_flight(wide["host"]))

    check("NEGATIVE CONTROL, the behaviour that shipped: the same CS2, off the "
          "same audio, is refused for arriving during DISCONNECTING",
          narrow["delivered"][HOLD + 1] is None)
    check("...so the station spends its remaining cycles repeating a goodbye "
          "that was already answered, and stops in DISCONNECTING",
          narrow["host"].arq.state == State.DISCONNECTING
          and qrt_in_flight(narrow["host"]),
          narrow["host"].arq.state)


def what_a_codeword_does_during_the_goodbye() -> None:
    """Each of the four codewords, one apiece, into a station saying goodbye.

    Established BEFORE the gate was widened, because widening it is what puts
    these paths on the air: the question the table answers is whether any of
    them can leave a link unable to close.
    """
    print("\nOne codeword during DISCONNECTING, alternation reference CS1")
    outcome = {}
    for cs in (CS1, CS2, CS3, CS4):
        host = saying_goodbye()
        assert host.arq.state == State.DISCONNECTING
        host.on_rx_event(cs_event(cs))
        at_rx = host.arq.state
        host.tick()
        outcome[cs] = (at_rx, host.arq.state)
        print(f"    {NAME[cs]}: at the decode {at_rx}, at the end of the cycle "
              f"{host.arq.state}")

    check("CS2 -- the alternation -- acknowledges the goodbye and closes the link",
          outcome[CS2][1] in (State.DISCONNECTED, State.LISTENING))
    check("CS4 on a 100 Bd link acknowledges too, and closes it",
          outcome[CS4][1] in (State.DISCONNECTED, State.LISTENING))
    check("CS1 -- a repeat of the reference -- asks for the goodbye again and "
          "does not close it", outcome[CS1] == (State.DISCONNECTING,
                                                State.DISCONNECTING))
    check("CS3 -- the peer breaking in -- yields the role while preserving "
          "DISCONNECTING; a role change cannot reopen the closing link",
          outcome[CS3][0] == State.DISCONNECTING)
    check("...and the pending goodbye survives: the same tick takes "
          "the link back and puts the goodbye on the air again",
          outcome[CS3][1] == State.DISCONNECTING)


def the_reversion_cannot_lose_the_goodbye() -> None:
    """A peer breaking in every single cycle.

    The shape a real teardown took on 3595 kHz on 2026-08-19: three entries into
    DISCONNECTING and two reversions out of it inside one goodbye, the peer
    keying a CS3-headed packet each time.
    """
    print("\nCS3 every cycle, ten cycles of it")
    host = saying_goodbye()
    ends, keyed = [], 0
    real_send = host.send_packet

    def spy(sl, payload, status, breakin=False):
        nonlocal keyed
        keyed += bool(status & spec.STATUS_QRT)
        return real_send(sl, payload, status, breakin=breakin)

    host.send_packet = spy
    for _ in range(10):
        host.on_rx_event(cs_event(CS3))
        host.tick()
        ends.append(host.arq.state)
    print(f"    ends: {' '.join(dict.fromkeys(ends))}; QRT packets keyed {keyed}")

    check("every cycle of the teardown ends back in DISCONNECTING -- the yield "
          "never strands this station as the receiving one",
          set(ends[:GOODBYE_CYCLES]) == {State.DISCONNECTING}, str(ends))
    check("...and the goodbye is re-sent on each of them rather than dropped",
          keyed == GOODBYE_CYCLES, str(keyed))
    check("...and the teardown is the state machine's OWN to end. A peer that "
          f"goes on breaking in is answered {GOODBYE_CYCLES} times and then "
          "left the channel: the exchange used to run for as long as the peer "
          "kept it up, and only the hold loop could stop it",
          set(ends[GOODBYE_CYCLES:]) == {State.DISCONNECTED}, str(ends))

    silent = saying_goodbye()
    for _ in range(10):
        silent.tick()
    check("NEGATIVE CONTROL: a goodbye no codeword ever answers closes on the "
          "same count -- what a deaf receiver used to make of every session was "
          "a link left in DISCONNECTING for as long as anything kept ticking it",
          silent.arq.state in (State.DISCONNECTED, State.LISTENING),
          str(silent.arq.state))


def the_packet_that_answers_the_goodbye() -> None:
    """The other half of the same gate: a peer breaking in while we say goodbye.

    A codeword is not the only answer a goodbye can get. "In contrast to AMTOR,
    CS3 is transmitted as head portion of a special changeover packet" -- so a
    peer with something left to say answers with 840 ms of data behind the head,
    and `arq.on_rx_packet` yields to it and reads its field in the same call --
    so the frame scan owes teardown the same admission the codeword read does.
    """
    print("\nA changeover packet arriving while the goodbye is in flight")
    audio = np.asarray(pactor1.breakin_signal(b"73 SK", 100, packet_count=0),
                       np.float32)
    for gate, tag in (((State.CONNECTED,), "gate at CONNECTED"),
                      ((State.CONNECTED, State.DISCONNECTING), "gate at LINKED")):
        host = saying_goodbye()
        rx = onair._SessionRx(host)
        rx.new_cycle()
        saved, onair.LINKED = onair.LINKED, gate
        try:
            rx.deep_scan(audio)
        finally:
            onair.LINKED = saved
        took = host.arq.role != ISS
        print(f"    {tag}: frames delivered {rx.count}, "
              f"state {host.arq.state}, role {host.arq.role}")
        if State.DISCONNECTING in gate:
            check("the peer's changeover packet reaches the state machine and "
                  "this station yields the channel to it", rx.count == 1 and took,
                  f"{rx.count} frame(s), role {host.arq.role}")
            check("...and the goodbye is not lost -- it is owed again from the "
                  "receiving side", host.arq._qrt_pending and host.arq._breakin_pending)
        else:
            check("NEGATIVE CONTROL: the shipped gate refuses it, so a gateway's "
                  "last transmission is dropped and the channel is never handed "
                  "back", rx.count == 0 and not took)


def the_capture_that_carried_it() -> None:
    """The two windows themselves, read off disk and driven into the FSM."""
    print(f"\nThe windows {CAPTURES.name} wrote, at the grid's own instant")
    at = round(RECORDED_ANCHOR * FS)
    heard = {}
    for name, want in (("hold_13", CS1), ("hold_14", CS2)):
        heard[name] = audio = onair.session.load_wav(
            str(CAPTURES / f"{name}.wav"), FS)
        got = p1rx.cs_anchored(audio, RECORDED_ANCHOR)
        check(f"{name}.wav reads {NAME[want]} at zero bit errors",
              got is not None and got.index == want and got.errors == 0, str(got))

    host = saying_goodbye()
    rx = onair._SessionRx(host)
    rx.new_cycle()
    saved, onair.LINKED = onair.LINKED, (State.CONNECTED, State.DISCONNECTING)
    try:
        got = rx.control_signal(heard["hold_14"], 0, at)
    finally:
        onair.LINKED = saved
    check("...and hold_14's own audio, handed to a station saying goodbye, "
          "closes the link", got == CS2
          and host.arq.state in (State.DISCONNECTED, State.LISTENING),
          f"{got} {host.arq.state}")


def a_short_window_cannot_carry_a_data_packet() -> None:
    """The frame scan's length floor, against the reader's own limit.

    `_SessionRx._scan` refuses a window under `FS // 2`, and a teardown window is
    0.24 s -- so the floor reads like the thing that closes the frame scan
    exactly when the goodbye needs it, the same shape as the state gate above one
    layer down. It is not, and the arithmetic says so before any audio does: a
    PACTOR-1 packet is `spec.P1_PACKET_S` of air, `p1rx._Tones.fits` offers no
    alignment until the window holds `nbits * sps` samples, and the reader has no
    partial-frame path to fall back to. Below one whole packet it is silent by
    bounds check.

    The floor sits half a packet BELOW that limit. Every window it refuses would
    have returned None anyway, and no window it admits decodes because it was
    admitted -- so lowering it, or varying it with the link state, buys the
    teardown nothing. This is measured rather than argued because a floor moved
    on a guess is how the premise got believed in the first place.
    """
    print("\nThe shortest window a PACTOR-1 data packet can be read from")
    whole = np.asarray(pactor1.packet_signal(b"HELLOP1X", baud=100,
                                             packet_count=1), np.float32)
    read = rxfront.decode_expected_p1_packet
    check("the render decodes whole", read(whole) is not None,
          f"{whole.size} samples, {whole.size / FS:.3f} s")
    lo, hi = 1, whole.size
    while lo < hi:
        mid = (lo + hi) // 2
        if read(whole[:mid]) is not None:
            hi = mid
        else:
            lo = mid + 1
    print(f"    cliff at {lo} samples ({lo / FS:.4f} s); a packet is "
          f"{PACKET_S:.2f} s and the scan's floor is {FS // 2} ({0.5:.2f} s)")
    check("nothing shorter than one whole packet decodes",
          lo >= round(PACKET_S * FS), f"{lo} samples, {lo / FS:.4f} s")
    check("...and the scan's own floor is below that, so it refuses only "
          "windows the reader was going to refuse", FS // 2 < lo,
          f"floor {FS // 2}, cliff {lo}")
    check("a teardown window is shorter than both", 0.245 * FS < FS // 2)


def the_windows_a_goodbye_listens_in() -> None:
    """`onair-0819-2210`, window by window: what the frame scan was ever offered.

    Sixteen hold cycles. Fourteen wrote 0.237-0.245 s, which cannot hold a packet
    and never could. The two that ran long are the cycles the gateway took the
    channel in, and both carry the same `RMS Tri` break-in field under packet
    counter 0 -- an ISS repeating what its IRS never acknowledged. Those are the
    windows the DISCONNECTING branch above exists for, and the floor passes them.
    """
    print(f"\nThe sixteen windows {TEARDOWN.name} wrote")
    short, long_ = [], []
    for wav in sorted(TEARDOWN.glob("hold_*.wav")):
        audio = onair.session.load_wav(str(wav), FS)
        got = rxfront.decode_expected_p1_packet(audio)
        (long_ if audio.size >= round(PACKET_S * FS) else short).append(
            (wav.stem, audio.size / FS, got))
    for stem, dur, got in short + long_:
        print(f"    {stem}  {dur:5.3f} s  "
              f"{'decoded: ' + str(got.text)[:44] if got else 'no packet'}")
    check("most of a teardown is spent in windows shorter than a packet",
          len(short) >= 12, f"{len(short)} of {len(short) + len(long_)}")
    check("...and not one of them decodes, floor or no floor",
          all(got is None for _, _, got in short))
    check("the windows that ran long carry the gateway's break-in packet, and "
          "the floor admits them", bool(long_)
          and all(got is not None and "CRC-VALID" in str(got.text)
                  for _, _, got in long_),
          f"{[(s, bool(g)) for s, _, g in long_]}")


SYNTHETIC = (the_goodbye_the_peer_answered, what_a_codeword_does_during_the_goodbye,
             the_packet_that_answers_the_goodbye,
             the_reversion_cannot_lose_the_goodbye,
             a_short_window_cannot_carry_a_data_packet)
RECORDED = (the_capture_that_carried_it,)
TEARDOWN_RECORDED = (the_windows_a_goodbye_listens_in,)


def _run(stages) -> int:
    global ok
    ok = True
    for stage in stages:
        stage()
    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def main() -> int:
    return _run(SYNTHETIC + (RECORDED if CAPTURES.is_dir() else ())
                + (TEARDOWN_RECORDED if TEARDOWN.is_dir() else ()))


def test_main() -> None:
    assert _run(SYNTHETIC) == 0


@pytest.mark.skipif(
    not CAPTURES.is_dir(),
    reason=f"{CAPTURES} is not on this machine -- on-air captures are gitignored "
           "and live only in the checkout that recorded them")
def test_the_capture_that_carried_it() -> None:
    assert _run(RECORDED) == 0


@pytest.mark.skipif(
    not TEARDOWN.is_dir(),
    reason=f"{TEARDOWN} is not on this machine -- on-air captures are gitignored "
           "and live only in the checkout that recorded them")
def test_the_windows_a_goodbye_listens_in() -> None:
    assert _run(TEARDOWN_RECORDED) == 0


if __name__ == "__main__":
    sys.exit(main())
