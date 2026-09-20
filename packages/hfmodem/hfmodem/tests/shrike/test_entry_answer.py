# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What a read entry packet brings back, through the receiver that has to take it.

The reference peer answers an entry packet it could read with a CS3-headed
changeover packet in the next turnaround, then its greeting a cycle or two later,
then it climbs and asks for the long cycle. Every one of those is a decode this
station had never been asked to make on a live link, and the two that had no
coverage at all through the production entry points are here:

  * the CHANGEOVER, which is `rxfront._cs_event`'s second half -- the packet a
    codeword heads -- reached through `onair._SessionRx.control_signal`;
  * the LONG-CYCLE data packet, which is `deep_scan`'s alone: `_SessionRx.flush`
    keeps `onair.FLUSH_CONTEXT_S` of the buffer and a long packet is four and a
    half times that, so no second reader can cover for it.

AND THE SPLIT COMB, which is what made the changeover worth measuring again.
`spec.SUBBAND_LEAD` puts channel 5 half a symbol ahead of channel 12 on the two
carriers a changeover rides, `placement.CASE0_STAGGER` keys it that way, and
`p3rx.decode_changeover` read both on one clock -- absorbing the split as a joint
shift of the whole frame, which the CRC survives on the bench and pays for on a
channel. The arrangement the sweep already tries IS the clock, because the lead
belongs to the virtual carrier and the swap moves the two together, so reading
each carrier on its own costs no extra sweep and no extra trial.

REAL MATERIAL where there is any: `rf-corpus/PIII_Complete_1` changes hands three
times, twice on the home arrangement and once on the swapped one, and all three
are read here through the session receiver rather than through `p3rx`.

Run:  python -m pytest hfmodem/tests/shrike/test_entry_answer.py
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.shrike import (modem, onair, p3frame, p3rx, placement, rx, rxfront,
                            session, spec)
from hfmodem.shrike.arq import CS_ACK, CS_BREAKIN, IRS, ISS
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.shrike.test_p3level_session import _Seam, _cycle_audio

FS, SPS = onair.FS, rxfront.SPS
RECORDING = corpora.RF_CORPUS / "PIII_Complete_1.wav"

CHANGEOVERS = ((5.5637, bytes.fromhex("0d5054202761"), False),
               (7.7681, bytes.fromhex("0f8f87180155"), False),
               (64.9160, bytes.fromhex("0f8f87180155"), True))
"""Where the recording's link changes hands, the field the new sender put behind
the codeword, and whether that keying is on the swapped arrangement.

The first two are `placement.CHANGEOVER`'s own docstring, measured there; the
third is this session's IRS taking the link back at the end, and it is the one
that shows a real station keying a changeover on the swapped comb."""

GREETING = b"\rPT"
STATUS = 0x20
FIELD = bytes.fromhex("0d5054202761")
"""The IRS's own three bytes and status byte, which `placement.changeover_packet`
reproduces exactly -- so a rendered arm and the recording are the same field."""


class _Session:
    """A `_SessionRx` on a linked PACTOR-3 host that keeps what it delivered."""

    def __init__(self, *, role=ISS, entry_pending: bool = False) -> None:
        host = PtcHost(peer=_Seam(), mycall="W9SSJ")
        host.arq.on_host_listen(True)
        host.arq.role, host.arq.dxcall = role, "DL6MAA"
        host.arq._enter_connected()
        host.protocol = Protocol.PACTOR3
        host.arq.entry_pending = entry_pending
        host.peer.sent.clear()
        self.host = host
        self.rx = onair._SessionRx(host, tag="TEST")
        self.events: list = []
        deliver = host.on_rx_event

        def record(ev):
            self.events.append(ev)
            return deliver(ev)

        host.on_rx_event = record

    def changeover(self, audio: np.ndarray, head_at: int):
        """The cycle a CS3 is due in, read where the grid says it is."""
        self.rx.new_cycle()
        return self.rx.control_signal(audio, 0, head_at)

    @property
    def packets(self) -> list:
        return [ev for ev in self.events if ev.kind == "packet"]


def _head_at(audio: np.ndarray, search_from: int):
    """The codeword `rxfront` reads at the front of a changeover packet.

    The same `_best_cs` the tracked receiver uses, so the frame search starts
    where the live path starts it and not at an alignment chosen here.
    """
    pulse = rx._pulse(SPS)
    delay = (pulse.size - 1) // 2
    Z = {cn: rx._baseband(audio, cn, FS, pulse) for cn in rxfront.HDR_TONES}
    return rxfront._best_cs(Z, delay, search_from)


def _rendered(swapped: bool, snr_db: float, seed: int = 0) -> np.ndarray:
    return _cycle_audio(placement.changeover_packet(GREETING, STATUS,
                                                    swapped=swapped),
                        snr_db=snr_db, seed=seed)


# --------------------------------------------------------------------------- #
# The changeover, through the session receiver
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("at,field,swapped", CHANGEOVERS)
def test_a_real_changeover_reaches_the_host(at: float, field: bytes,
                                            swapped: bool) -> None:
    """DL6MAA's link changing hands, read by the receiver a session runs.

    `p3rx.decode_changeover` is already held to these fields off this recording
    (`test_p3_upgrade`); what is new is the path -- `_SessionRx.control_signal`
    on a CONNECTED PACTOR-3 host, which is where a live station meets one.
    An ISS that reads it must both take the three bytes and hand back the link.
    """
    if not RECORDING.exists():
        pytest.skip(f"recording absent: {RECORDING}")
    audio = session.load_wav(str(RECORDING), FS)
    seg = audio[int((at - 0.2) * FS):int((at + 1.2) * FS)].astype(np.float32)
    ci, be, head = _head_at(seg, int(0.25 * FS))
    assert (ci, be) == (CS_BREAKIN, 0), f"CS{ci + 1} at {be} bit errors"

    sess = _Session()
    assert sess.changeover(seg, head) == CS_BREAKIN
    assert [ev.packet[1:3] for ev in sess.packets] == \
        [(field[-3], spec.field_payload(field[:-3]))], \
        [ev.text for ev in sess.events]
    assert all(ev.breakin for ev in sess.packets)
    assert sess.host.arq.role == IRS, "the ISS did not hand back the link"


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("snr_db", [30.0, 6.0])
def test_our_own_changeover_reads_at_both_arrangements(swapped: bool,
                                                       snr_db: float) -> None:
    """What we key, read back through the live path, on both ARQ cycles.

    A changeover packet carries no header block to name the swap, so the two
    arrangements are the whole of what the reader has to try -- and the peer
    chooses which one by the cycle it answers in.
    """
    audio = _rendered(swapped, snr_db)
    ci, be, head = _head_at(audio, int(0.45 * FS))
    assert (ci, be) == (CS_BREAKIN, 0), f"CS{ci + 1} at {be} bit errors"

    sess = _Session()
    assert sess.changeover(audio, head) == CS_BREAKIN
    assert [ev.packet[2] for ev in sess.packets] == [GREETING], \
        [ev.text for ev in sess.events]
    assert sess.host.arq.role == IRS


@pytest.mark.parametrize("swapped", [False, True])
def test_the_changeover_answers_the_entry_packet(swapped: bool) -> None:
    """The answer path the entry packet is keyed FOR.

    `arq.PactorArq.on_rx_packet` yields on a break-in, and `ptc.PtcHost`
    spends `entry_pending` on the same frame (`_entry_answered`) -- so a peer
    that reads our entry packet and takes the channel with it moves this
    station off speed level 1 and onto the level the entry asked for, in one
    event and not two.
    """
    audio = _rendered(swapped, 20.0)
    _, _, head = _head_at(audio, int(0.45 * FS))
    sess = _Session(entry_pending=True)
    sess.host.arq.speed_level = 1
    assert sess.changeover(audio, head) == CS_BREAKIN
    assert not sess.host.arq.entry_pending
    assert sess.host.arq.speed_level == sess.host.arq.entry_level
    assert sess.host.arq.role == IRS


# --------------------------------------------------------------------------- #
# ...and the clock it is read on
# --------------------------------------------------------------------------- #

def _sweep(audio: np.ndarray, head_at: int, aware: bool, confirm: bool) -> list:
    """`p3rx.decode_changeover`'s own loop, with the clock and the early return
    switchable, so the two readings are compared over the same alignments and
    the same two carrier arrangements.
    """
    pulse = rx._pulse(SPS)
    delay = (pulse.size - 1) // 2
    path = placement.CHANGEOVER
    Z = {cn: rx._baseband(audio, cn, FS, pulse) for cn in spec.VH_CHANNELS}
    base = head_at + placement.CHANGEOVER_HEAD_SYMBOLS * SPS
    out = []
    for swapped in (False, True):
        order = p3frame.VH_ORDER[::-1] if swapped else p3frame.VH_ORDER
        tones = (tuple(spec.CARRIER_SWAP[cn] for cn in path.tones) if swapped
                 else path.tones)
        lead = dict(zip(tones, path.clock_offsets(SPS))) if aware else None

        def at(pos: int, order=order, lead=lead) -> bytes | None:
            softs = rx.case0_softs(Z, pos, fs=FS, delay=delay, order=order,
                                   path=path, lead=lead)
            if softs is None:
                return None
            field, ok = rx.decode_case0_softs(softs, path)
            return field if ok else None

        for k in range(-p3rx.CHANGEOVER_SEARCH, p3rx.CHANGEOVER_SEARCH + 1):
            pos = base + k * p3rx.CONFIRM_STEP
            field = at(pos)
            if field is not None and (not confirm or p3rx.confirmed(at, pos, field)):
                out.append((swapped, k, field))
    return out


@pytest.mark.parametrize("at,field,swapped", CHANGEOVERS)
def test_each_carrier_on_its_own_clock_reads_deeper(at: float, field: bytes,
                                                    swapped: bool) -> None:
    """The margin the one-clock read was giving away, on the real thing.

    Both readings decode these frames -- that is why the defect was invisible.
    What separates them is how much of the sweep validates: read at the
    midpoint of the two carriers the frame accepts at five alignments and at
    four on the swapped keying, and read on each carrier's own clock at seven
    or eight. Same field either way, which is what says the extra alignments
    are the frame and not slack.
    """
    if not RECORDING.exists():
        pytest.skip(f"recording absent: {RECORDING}")
    audio = session.load_wav(str(RECORDING), FS)
    seg = audio[int((at - 0.2) * FS):int((at + 1.2) * FS)].astype(np.float32)
    _, _, head = _head_at(seg, int(0.25 * FS))

    one = _sweep(seg, head, aware=False, confirm=False)
    two = _sweep(seg, head, aware=True, confirm=False)
    assert {f for _, _, f in one} == {f for _, _, f in two} == {field}
    assert {s for s, _, _ in two} == {swapped}, \
        "the accepting arrangement moved with the clock"
    assert len(two) > len(one), f"{len(two)} alignments against {len(one)}"


def test_the_split_comb_is_worth_two_decibels() -> None:
    """...and what those alignments are worth where it matters.

    The rendered changeover in white noise at -19 dB, which is where the
    one-clock read starts to lose it, twenty seeds on each carrier arrangement
    and `p3rx.confirmed` behind every accept -- so what is counted is a
    delivered field and not a soft magnitude. Read at the midpoint of the two
    carriers 12 of 40 arrive; read on each carrier's own clock, 34.

    A CHANGEOVER IS NOT A PACKET THAT CAN BE REPEATED CHEAPLY. It is the peer
    taking the channel, and a station that misses one goes on transmitting into
    it, so two decibels here are not the two decibels of a data frame.
    """
    kept = {}
    for aware in (False, True):
        got = 0
        for swapped in (False, True):
            body = placement.changeover_packet(GREETING, STATUS,
                                               swapped=swapped)
            clean = np.concatenate([np.zeros(SPS), body,
                                    np.zeros(4 * SPS)]).astype(np.float32)
            _, _, head = _head_at(clean, 6 * SPS)
            sigma = float(np.sqrt(np.mean(np.asarray(body, float) ** 2))) \
                * 10 ** (19.0 / 20)
            for seed in range(20):
                noisy = (clean + np.random.default_rng(seed).normal(
                    0, sigma, clean.size)).astype(np.float32)
                got += any(f == FIELD for _, _, f in
                           _sweep(noisy, head, aware, confirm=True))
        kept[aware] = got
    assert kept[True] >= 2 * kept[False], kept
    assert kept[True] >= 30, kept


# --------------------------------------------------------------------------- #
# ...and that it is the codeword the peer sent
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ci", range(len(spec.CS_NAMES)))
def test_a_codeword_reads_back_as_itself_out_of_silence(ci: int) -> None:
    """Every answer the peer can give, rendered into quiet and read blind.

    A codeword has no CRC and the alphabet is six words wide, so a sweep that
    reads an alignment lying half outside the burst gets a word back and cannot
    tell. It lands on a real one: a bare CYCLE-TOG read BREAK-IN at zero bit
    errors 100 ms in front of its own first sample, which armed the 0.6 s
    debounce and hid the burst it came from -- on the QSO bench an ISS handing
    the link back to a peer that had asked for a longer cycle, and 958 bytes of
    a kilobyte never went out.

    Six arms, one per word, because a break-in is the codeword whose cost is the
    channel and the alternatives are not interchangeable.
    """
    audio = np.concatenate([np.zeros(FS), placement.control_signal(ci),
                            np.zeros(FS)]).astype(np.float32)
    got = [ev for ev in rxfront.decode_events(
        audio, cs_max_errors=rxfront.CS_EXPECTED_MAX_ERRORS)
        if ev.kind in ("cs", "unassigned")]
    assert [ev.cs for ev in got] == [ci], \
        [(round(ev.t, 3), ev.text) for ev in got]


def test_a_straddle_reads_a_codeword_and_is_refused_for_where_it_lies() -> None:
    """The alignment that caused this, and the rule that turns it down.

    Ten of the twenty symbols of this read sit in the pad in front of the burst,
    and it is STILL a codeword at zero bit errors -- a different one, BREAK-IN
    out of a CYCLE-TOG. So neither the bit errors nor a second opinion on the
    bits can separate it from the true read; what separates them is that one
    alignment lies inside the burst and the other does not, which is
    `rxfront.CS_EXTENT_FLOOR`'s whole claim.
    """
    # Freeze the original failing stimulus: the old unstaggered, 21-symbol
    # CS6. Current production controls include stagger/runout and cannot stand
    # in for this historical zero-error false BREAK-IN. Keep every gate and
    # exact alignment assertion below unchanged.
    bits = np.array([(0x4B4AD >> i) & 1 for i in range(20)], np.uint8)
    syms = modem.differential_encode(bits, 1)
    burst = modem.modulate_tones({5: syms, 12: syms},
                                modem.ModConfig(sample_rate=FS, matched_pulse=True))
    audio = np.concatenate([np.zeros(FS), burst,
                            np.zeros(FS)]).astype(np.float32)
    pulse = rx._pulse(SPS)
    delay = (pulse.size - 1) // 2
    Z = {cn: rx._baseband(audio, cn, FS, pulse) for cn in rxfront.HDR_TONES}
    # 48780, not the 48660 this pinned: `_best_cs` chooses inside its zero-error
    # plateau by a coherent metric now, and over this render the matched filter's
    # own amplitude is at its MINIMUM at 48660 and peaks at 48870 -- a half
    # symbol apart, so the old first-minimum answer was the worst sampling phase
    # in the burst.
    true_at, straddle = 48780, FS - 10 * SPS

    assert sum(1 for k in range(21) if straddle + k * SPS < FS) == 10, \
        "the premise: ten of the twenty-one symbols lie in the pad"
    assert rx.nearest_control_signal(rx.cs_bits(Z, straddle, delay)) \
        == (CS_BREAKIN, 0), "...and the twenty bits they form are still a word"

    env = rxfront._tone_envelope(Z, delay, straddle, true_at + 42 * SPS)
    assert rxfront._inside_burst(env, straddle, true_at)
    assert not rxfront._inside_burst(env, straddle, straddle)

    ci, be, at = rxfront._best_cs(Z, delay, 52920)
    assert (ci, be, at) == (spec.CS_NAMES.index("CYCLE-TOG"), 0, true_at)


# --------------------------------------------------------------------------- #
# The long cycle's own window
# --------------------------------------------------------------------------- #

def test_a_long_packet_arrives_inside_a_long_cycle() -> None:
    """`deep_scan` is the only reader a 3.37 s packet has, and it is enough.

    The hold loop sizes the receive window off the grid's cycle length, and
    what lands in it on the long cycle is one packet 3.29 s wide. The flush
    cannot cover for a miss there -- it keeps `onair.FLUSH_CONTEXT_S` of the
    buffer, which is a fifth of the packet -- so the claim is not that some
    reader eventually finds it but that this one does, in the cycle it
    arrived in, at the level the reference session runs.
    """
    payload = bytes((3 * i + 3) & 0x5F | 0x20
                    for i in range(spec.SPEED_LEVELS[3].payload_long))
    body = placement.link_packet(3, payload, 0x21, long_cycle=True)
    win = np.zeros(round(spec.CYCLE_LONG_S * FS))
    lead = round(0.2 * FS)
    assert lead + body.size <= win.size, \
        f"the packet does not fit its own cycle: {body.size / FS:.3f} s"
    win[lead:lead + body.size] = np.asarray(body, np.float64)
    sigma = float(np.sqrt(np.mean(np.asarray(body, np.float64) ** 2))) / 10.0
    audio = (win + np.random.default_rng(0).normal(0, sigma, win.size)
             ).astype(np.float32)

    assert body.size > onair.FLUSH_CONTEXT_S * FS, \
        "the premise: a long packet outruns what the flush keeps"

    sess = _Session(role=IRS)
    sess.rx.new_cycle()
    sess.rx.deep_scan(audio)
    assert [ev.packet[0] for ev in sess.packets] == [3], \
        [ev.text for ev in sess.events]
    assert bytes(sess.host.channel(sess.host.ptchn).rx) == payload


# --------------------------------------------------------------------------- #
# The answer a link that never needed one has still never measured
# --------------------------------------------------------------------------- #

ANSWER_LEAD_S = 0.100
"""How far in front of the caller's anchor the peer's PACTOR-3 answer sits.

The retained PACTOR-1 geometry puts the answer a 960 ms packet plus the
turnaround past our boundary and PACTOR-3's own packet is 810 ms, so the
codeword arrives in front of that anchor by the difference -- 150 ms on
`captures/onair-0913-1837`'s comb, where none of them was read. The figure
here is `_p3_cs`'s own "~100 ms", which is over three times `SyncedRx`'s
anchored bracket and well inside `_acquire_answer`'s.
"""

ANSWER_OFFSET_HZ = 15.0
"""...and on a carrier off ours, which the anchored read is not told about."""


def _answer(cs: int, offset_hz: float = ANSWER_OFFSET_HZ) -> np.ndarray:
    """One control signal, on a peer carrier `offset_hz` from the tone plan."""
    burst = np.asarray(placement.control_signal(cs), np.float64)
    t = np.arange(burst.size) / FS
    return np.real(hilbert(burst)
                   * np.exp(2j * np.pi * offset_hz * t)).astype(np.float32)


def _answered_cycle(burst: np.ndarray, at: int, n: int) -> np.ndarray:
    seg = np.zeros(n, np.float32)
    seg[at:at + burst.size] += burst
    return seg


def test_an_iss_that_never_measured_an_answer_acquires_the_peers_codeword():
    """The read `captures/onair-0913-1837` spent 377 seconds without.

    That link came up, took the whole greeting as IRS -- which needs no peer
    codeword at all -- and then reversed. `_p3_cs`'s acquiring search was gated
    on `entry_pending`, spent the moment the entry was confirmed, so the first
    acknowledgement the new ISS needed was read at the retained PACTOR-1 anchor
    and nowhere else: 0 PACTOR-3 control signals decoded against 57 PACTOR-3
    frames.

    Two cycles, because a body-less word is a candidate until a second physical
    cycle puts it on the same clock at the same offset -- and the acquisition is
    spent once: the third cycle reads at the instant this one measured.
    """
    n = round(spec.CYCLE_SHORT_S * FS)
    anchor = round(0.60 * FS)
    at = anchor - round(ANSWER_LEAD_S * FS)
    burst = _answer(CS_ACK)
    seg = _answered_cycle(burst, at, n)

    assert rxfront.SyncedRx().control_signal_at(seg, anchor) is None, \
        "the premise: the anchored read cannot reach this answer"

    sess = _Session(role=ISS, entry_pending=False)
    sess.rx.new_cycle()
    assert sess.rx._p3_cs(seg, anchor, seg_start=0) is None
    assert sess.rx._p3_answer_at is None, "a first coherent head is a candidate"

    sess.rx.new_cycle()
    ev = sess.rx._p3_cs(_answered_cycle(burst, at, n), anchor, seg_start=n)
    assert ev is not None and ev.cs == CS_ACK, ev
    assert sess.rx._p3_answer_at == n + ev.start
    assert sess.rx.p3_receive_offset_hz == ANSWER_OFFSET_HZ

    measured = sess.rx._p3_answer_at
    sess.rx.new_cycle()
    again = sess.rx._p3_cs(_answered_cycle(burst, at, n), anchor, seg_start=2 * n)
    assert again is not None and again.cs == CS_ACK, again
    assert abs(sess.rx._p3_answer_at - (measured + n)) <= SPS // 2


def test_a_link_holding_a_measured_answer_still_reads_only_at_it():
    """NEGATIVE CONTROL: the search is what a link with no instant is owed.

    `_p3_answer_at` is the whole condition. A link that has measured the peer's
    answer clock reads at that clock and at the caller's anchor, and a cycle
    where neither holds a codeword is an unanswered cycle -- not an invitation
    to sweep 31 frequencies over a 460 ms bracket in front of the key, every
    cycle, for the rest of the session.
    """
    n = round(spec.CYCLE_SHORT_S * FS)
    anchor = round(0.60 * FS)
    at = anchor - round(ANSWER_LEAD_S * FS)
    burst = _answer(CS_ACK)

    sess = _Session(role=ISS, entry_pending=False)
    held = anchor - n
    for k in range(2):
        sess.rx.new_cycle()
        sess.rx._p3_answer_at = held
        assert sess.rx._p3_cs(_answered_cycle(burst, at, n), anchor,
                              seg_start=k * n) is None
        assert sess.rx._p3_answer_at == held
        assert sess.rx._p3_head_candidate is None
        assert sess.rx.p3_receive_offset_hz == 0.0
