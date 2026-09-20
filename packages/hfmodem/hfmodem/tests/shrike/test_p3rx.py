# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The PACTOR-III data field, audio in and payload out, on all six speed levels.

What this gate can and cannot say. It renders a packet through shrike's own
transmitter and reads it back, so a pass means the decoder inverts the encoder --
no more. What settles the other question is external, and it is
`test_p3_oracle.py`: an independent PACTOR monitor offered a link setup and four
ARQ cycles of these packets.

Since the packet header block was found in real traffic, both directions run
through it. The transmitter emits the block and the receiver matches it, off the
same tables in `p3frame`, and a packet is now the 81 symbols a real one measures
-- one phase reference, eight header symbols, all 72 rows -- with the carrier
swap alternating cycle by cycle. `test_a_conformant_packet_is_what_a_real_one_is`
is where that shape is pinned.

The negatives below are what keep this gate from being circular:

  * noise decodes to nothing, so the CRC is not accepting whatever it is shown;
  * a packet decodes at its OWN speed level and no other, so the tone-set gate
    and the geometry are doing work the CRC alone would not do;
  * the payload sizes come out of the geometry and land on the published table,
    twelve for twelve across both cycle lengths, which no wrong row count does.

Run:  .venv/bin/python -m pytest packages/hfmodem/hfmodem/tests/shrike/test_p3rx.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import modem, p3frame, p3rx, placement, rx, rxfront, session, spec
from hfmodem.tests.kestrel import corpora

FS = 48000


def _packet(sl: int, payload: bytes) -> np.ndarray:
    """One short-cycle data packet at `sl`, preceded by a second of quiet."""
    path = placement.SPEED_PATHS[sl]
    info = payload + bytes([spec.status_byte(1)])
    pkt = (placement.case0_packet(info) if path.case == 0 else
           placement.data_packet(info, path))
    return np.concatenate([np.zeros(FS), pkt, np.zeros(FS)])


def _payload(sl: int) -> bytes:
    return bytes((0x41 + (i % 26))
                 for i in range(spec.SPEED_LEVELS[sl].payload_short))


def _info(sl: int) -> bytes:
    return _payload(sl) + bytes([spec.status_byte(1)])


@pytest.mark.parametrize("sl", sorted(placement.SPEED_PATHS))
def test_speed_level_round_trips_byte_exact(sl: int) -> None:
    """Speed level 6 is in this list now, and it is what the head rows cost.

    It used to be an expected failure: at rate 8/9 the frame does not survive ONE
    erased row of 72, measured on noiseless softs, and the acquisition burst
    shrike used to send in front of every packet cost five. A conformant packet
    has no burst inside the cycle and gives up nothing, so the two punctured
    levels decode for the first time.
    """
    want = _payload(sl)
    scan = p3rx.decode_p3_packets(_packet(sl, want))
    assert scan.packets, f"SL{sl} did not decode ({scan.trials} trials)"
    got = scan.packets[0]
    assert got.sl == sl
    assert got.payload == want


@pytest.mark.parametrize("sl", sorted(placement.SPEED_PATHS))
def test_speed_level_is_identified_not_guessed(sl: int) -> None:
    """No packet decodes at a speed level it was not sent at.

    Levels 3 and 4 share a tone set and are separated only by their modulation, so
    this is the check that the geometry rather than the spectrum is deciding.
    """
    audio = _packet(sl, _payload(sl))
    for other in placement.SPEED_PATHS:
        if other == sl:
            continue
        scan = p3rx.decode_p3_packets(audio, levels=(other,))
        assert not scan.packets, f"SL{sl} packet decoded as SL{other}"


def test_noise_decodes_to_nothing() -> None:
    """A CRC that accepts noise is not a gate, and this one is offered plenty.

    The blind scan behind the old header path accepts about one position in 4400
    on noise, which is where the corpus's phantom PACTOR-3 headers came from.
    """
    rng = np.random.default_rng(20260731)
    audio = rng.normal(0, 0.1, 8 * FS)
    scan = p3rx.decode_p3_packets(audio)
    assert not scan.packets, [p.report() for p in scan.packets]


def test_trial_count_is_small_enough_to_mean_something() -> None:
    """A decode is evidence only if the scan that found it was short."""
    scan = p3rx.decode_p3_packets(_packet(4, _payload(4)))
    assert scan.packets
    assert scan.expected_false < 0.1, scan.trials


@pytest.mark.parametrize("sl", sorted(spec.SPEED_LEVELS))
def test_geometry_reproduces_the_published_payload_sizes(sl: int) -> None:
    """72 rows short, 320 long -- and every payload size follows from that alone.

    Compare all twelve payload sizes against M.1798.
    """
    assert placement.SPEED_PATHS[sl].crc_bytes - 3 == spec.SPEED_LEVELS[sl].payload_short
    assert placement.LONG_PATHS[sl].crc_bytes - 3 == spec.SPEED_LEVELS[sl].payload_long
    assert placement.LONG_PATHS[sl].n_symbols == placement.LONG_ROWS


@pytest.mark.parametrize("sl", sorted(spec.SPEED_LEVELS))
def test_the_frame_a_level_transmits_is_the_published_one(sl: int) -> None:
    """The comb, the modulation and the code rate, against M.1798 §3.

    Compare the implemented paths with the published speed-level parameters.
    Round trips alone would not detect a shared transmit/receive error.
    """
    s = spec.SPEED_LEVELS[sl]
    for path in (placement.SPEED_PATHS[sl], placement.LONG_PATHS[sl]):
        assert path.speed_level == sl
        assert path.tones == s.channels
        assert path.bits_per_cell == s.bits_per_symbol
        assert path.puncture.rate == s.code_rate
        # `placement.encode_frame` picks the K=9 trellis on case 0 and the K=7 one
        # everywhere else, so the constraint length is a property of the case.
        assert (9 if path.case == 0 else 7) == s.constraint_length


# ---------------------------------------------------------------------------
# The conformant packet: what a station on an established link transmits
# ---------------------------------------------------------------------------

SPS = FS // 100


def _anchors(audio: np.ndarray) -> list:
    pulse = rx._pulse(SPS)
    Z = {cn: rx._baseband(audio, cn, FS, pulse) for cn in range(spec.N_CHANNELS)}
    return p3rx.header_anchors(Z, len(audio), fs=FS)


def test_a_conformant_packet_is_what_a_real_one_is() -> None:
    """Eighty-two symbols, nothing in front of the phase reference.

    The measurement this reproduces is on the air, not here: every packet on the
    reference tape carries energy from the phase reference through symbol 81 and
    stops at 82, at speed level 1 and at speed level 3, and a second station's
    long-cycle level 6 does the same (`placement.ENTRY_TRAILER`). A packet that
    led with a twenty-symbol acquisition burst, or gave five rows up to one, is
    neither.

    Speed levels 1 and 2 run half a symbol longer and only that: their combs are
    staggered (`spec.SUBBAND_LEAD`), so the carriers that start the packet finish
    half a symbol before the ones that do not. Eighty-two symbols each, still
    nothing in front of the phase reference.
    """
    symbols = p3frame.DATA_OFFSET + placement.FRAME_SYMBOLS + 1
    for sl in placement.SPEED_PATHS:
        path = placement.SPEED_PATHS[sl]
        pkt = (placement.case0_packet(_info(sl)) if path.case == 0 else
               placement.data_packet(_info(sl), path))
        # ...plus the transmit pulse ringing out, which is its span and not a
        # symbol (`placement.PROTOCOL_RISE`, which every level now keys).
        cfg = placement.protocol_config()
        want = symbols * SPS + cfg.pulse().size - 1
        assert len(pkt) == want + max(path.clock_offsets(SPS)), (sl, len(pkt) / SPS)


@pytest.mark.parametrize("swapped", (False, True))
@pytest.mark.parametrize("long_cycle", (False, True))
def test_the_staggered_comb_reads_back_on_its_own_clocks(long_cycle: bool,
                                                         swapped: bool,
                                                         monkeypatch) -> None:
    """Speed level 2 out and back, on both cycle lengths and both arrangements.

    Level 2 is the one level whose carriers do not share a symbol clock: three of
    the six run half a symbol ahead of the other three (`spec.SUBBAND_LEAD`). The
    offset belongs to the VIRTUAL carrier, so a swapped cycle carries it onto the
    other physical cluster, and both cycle lengths carry it because a level's
    tone tuple is the same on each. Both of those go through `p3rx.path_for`,
    which rewrites a path's tones into the channels the cycle put them on, so
    this is the check that the stagger survives that rewrite.

    What it is NOT is evidence for the stagger. Our transmitter and our receiver
    read one table, so of course they agree; on clean audio the CRC even survives
    reading the whole comb on one clock, which is measured below and is why the
    external gate in `test_p3_oracle.py` is where the claim lives.
    """
    path = (placement.LONG_PATHS if long_cycle else placement.SPEED_PATHS)[2]
    info = bytes((0x41 + i % 26) for i in range(path.crc_bytes - 2))
    audio = np.concatenate([np.zeros(FS),
                            placement.data_packet(info, path, swapped=swapped),
                            np.zeros(FS)])
    # Grid row 0 as the receiver sees it: the packet's own start, the transmit
    # pulse's group delay, and the phase reference plus header block. Level 2 has
    # no anchor of its own -- four of the sixteen constant headers are too few for
    # `header_anchors` -- so a real receiver reaches it by envelope and CRC.
    row0 = (FS + (placement.protocol_config().pulse().size - 1) // 2
            + p3frame.DATA_OFFSET * SPS)
    head = p3frame.PacketHeader(FS, p3frame.variable_header(
        2, swapped=swapped, long_cycle=long_cycle), 1.0)
    got = p3rx.decode_at(audio, row0, 2, fs=FS, header=head)
    assert got is not None and got.payload == info[:-1]

    # And the clocks are load-bearing rather than decorative. The comparison is
    # the same packet read twice -- on its own two clocks and on the leading one
    # alone -- because how much half a symbol of mis-sampling costs is a property
    # of the transmit/receive pulse PAIR and not a constant: on the generic
    # raised cosine the mis-sampled half fell to cos(45 deg) = 0.70, and now that
    # the transmitter keys the receiver's own matched pulse it holds 0.85. What
    # does not move is the direction and the size of the loss.
    on_air = p3rx.path_for(2, head)

    def _soft() -> np.ndarray:
        return np.abs(rx.demod_cells(audio, row0, on_air, fs=FS,
                                     n_rows=path.n_symbols)
                      ).reshape(path.n_symbols, len(path.tones))

    own = _soft()
    monkeypatch.setitem(spec.SUBBAND_LEAD, 2, (0.0,) * len(path.tones))
    one = _soft()
    assert own[:, :3].mean() > 0.95 and own[:, 3:].mean() > 0.95, own.mean(0)
    # MEASURED at 0.134 (short cycle) and 0.146 (long) with this pulse; the floor
    # keeps the spread between the two cycle lengths as headroom.
    assert own[:, 3:].mean() - one[:, 3:].mean() > 0.10, (own.mean(0),
                                                          one.mean(0))
    # The half that starts the packet is on the clock either way, so nothing of
    # it may move: this is what says the loss above is the mis-sampling.
    assert abs(own[:, :3].mean() - one[:, :3].mean()) < 1e-3, (own.mean(0),
                                                               one.mean(0))


@pytest.mark.parametrize("swapped", (False, True))
@pytest.mark.parametrize("sl", (3, 4, 5, 6))
def test_the_receiver_reads_the_header_the_transmitter_sent(sl: int,
                                                            swapped: bool) -> None:
    """The transmit and receive sides of the header block, against each other.

    Not a tautology, because the two are not the same code: `header_steps` writes
    the published words out through `spec.header_tx_form` and `read_header` scores
    an unassigned match against all sixteen of them. What comes back has to be the
    speed level, the cycle length and the swap that went in -- and the swap is the
    one a receiver has no other way of learning.

    Speed levels 1 and 2 are absent and cannot be here: sixteen published constant
    headers are 192 known bits only where the comb is lit, and those two carry
    none and four of them. `HEADER_FIT` is measured against sixteen channels, so
    they are found by the envelope instead.
    """
    path = placement.SPEED_PATHS[sl]
    audio = np.concatenate([np.zeros(FS),
                            placement.data_packet(_info(sl), path, swapped=swapped),
                            np.zeros(FS)])
    heads = _anchors(audio)
    assert len(heads) == 1, [(h.at, h.vh, round(h.fit, 3)) for h in heads]
    h = heads[0]
    # Cross-shape agreement, not a matched pair: `data_packet` shapes with the
    # default raised cosine while the anchor search matched-filters with
    # `tablegen.symbol_pulse`, so this floor moves with either shape and says
    # nothing about the air. What the pulse answers to is the real capture in
    # `test_a_real_packet_header_places_the_packet_and_names_the_level`.
    #
    # 0.86 rather than 0.87 because `placement.START_PHASE` costs up to 0.023 of
    # it, and only here: the raised cosine spills into the neighbouring 120 Hz
    # slot, so what a channel's matched filter picks up off its neighbour turns
    # on the angle between them. Shaped with the pulse the receiver matches, the
    # same eight readings move by at most 0.006 and half of them improve. The
    # gate a packet has to clear is `p3rx.HEADER_FIT`, 0.80.
    assert h.fit >= 0.86, h.fit
    assert h.vh == p3frame.variable_header(sl, request_status=bool(_info(sl)[-1] & 1))
    assert h.swapped == swapped and not h.long_cycle and sl in h.levels

    scan, _ = p3rx.decode_headed(audio, 0, len(audio), fs=FS)
    assert [(p.sl, p.payload) for p in scan.packets] == [(sl, _payload(sl))]

    # ...and the swap is load-bearing: the same audio read on the home tone order
    # gathers the right energy in the wrong order, which nothing but the CRC sees.
    row0 = h.at + p3frame.DATA_OFFSET * SPS
    assert (p3rx.decode_at(audio, row0, sl, fs=FS) is None) == swapped


@pytest.mark.parametrize("swapped", (False, True))
@pytest.mark.parametrize("sl", (3, 4, 5, 6))
def test_the_start_phase_is_invisible_to_the_receiver(sl: int, swapped: bool,
                                                      monkeypatch) -> None:
    """`placement.START_PHASE` moves no anchor and changes no payload.

    The argument that it is free covers demodulation on its own -- every carrier is
    read differentially, so a constant angle on one cancels in every difference --
    and acquisition is the part that argument does not reach: `header_anchors`
    matched-filters 192 known bits across the whole comb to place a packet to the
    sample, and a transmitter that broke FINDING a packet while leaving READING it
    intact would look exactly like a receiver that had not regressed at all.

    So the packet is built twice, once with the shipped angles and once with the
    all-zero set they replaced, and everything acquisition returns is compared:
    how many anchors, where, which variable header, and the field behind it.
    """
    path = placement.SPEED_PATHS[sl]

    def render() -> np.ndarray:
        return np.concatenate([np.zeros(FS),
                               placement.data_packet(_info(sl), path, swapped=swapped),
                               np.zeros(FS)])

    spread = render()
    monkeypatch.setattr(placement, "START_PHASE", np.zeros(spec.N_CHANNELS))
    aligned = render()

    a, b = _anchors(aligned), _anchors(spread)
    assert [h.vh for h in b] == [h.vh for h in a] and len(a) == 1
    # Within one search step rather than on the sample, and the cost is the
    # raised-cosine transmit pulse rather than the phases: what a channel's matched
    # filter picks up off its neighbour turns on the angle between them, which
    # tilts a fit landscape already flat enough to be tied. Measured over 48
    # renders across the four wide levels and both swaps, the anchor lands on the
    # same sample in 44 and one quarter-symbol step away in 4, never further, never
    # missing, and never on a different variable header; the worst fit of the 48
    # is 0.863 against a `HEADER_FIT` gate of 0.80.
    assert abs(b[0].at - a[0].at) <= SPS // 4
    assert [(p.sl, p.payload) for p in p3rx.decode_p3_packets(spread).packets] \
        == [(p.sl, p.payload) for p in p3rx.decode_p3_packets(aligned).packets] \
        == [(sl, _payload(sl))]
    # `modulate_tones` leaves both at the same peak, so the whole of the gain shows
    # up as RMS: the aligned set spent that peak on the one symbol carrying no data.
    assert np.sqrt(np.mean(spread ** 2)) > 1.15 * np.sqrt(np.mean(aligned ** 2))


def test_the_carrier_swap_alternates_across_cycles() -> None:
    """Two cycles of a link, and the second is not the first.

    Within one packet a fixed tone order is merely wrong half the time; across an
    ARQ raster it is what makes a receiver that never learned the swap read the
    whole session in the wrong order. So the claim is about the sequence: both
    cycles decode, and they disagree about where the carriers are.
    """
    audio = session.build_session("W1AW", [_payload(3)] * 2, sl=3)
    heads = _anchors(audio)
    assert [h.swapped for h in heads] == [False, True], \
        [(round(h.at / FS, 3), h.vh) for h in heads]
    assert np.isclose((heads[1].at - heads[0].at) / FS, spec.CYCLE_SHORT_S,
                      atol=1.01 / (4 * spec.SYMBOL_RATE_BD))
    assert heads[0].tones(placement.DATA3.tones) != heads[1].tones(placement.DATA3.tones)
    scan = p3rx.decode_p3_packets(audio)
    assert [(p.sl, p.payload) for p in scan.packets] == [(3, _payload(3))] * 2


@pytest.mark.parametrize("swapped", (False, True))
@pytest.mark.parametrize("long_cycle", (False, True))
def test_a_tracked_decode_reads_the_cycle_it_is_handed(long_cycle: bool,
                                                       swapped: bool,
                                                       monkeypatch) -> None:
    """The in-session path, on both arrangements and both cycle lengths.

    `rxfront.SyncedRx` is what a connected station uses every cycle, and it is
    where the swap bites hardest: the carriers move to their partner tones on
    EVERY ARQ cycle, so a receiver pinned to the home order can read no more than
    half a link and hands the rest back to a scan thirteen times the price.

    The lock is deliberately half a symbol out. It is inherited from whatever
    alignment last passed a CRC rather than from the packet's own clock, and the
    CRC tolerates a whole symbol where the header fit does not -- which is why the
    header is read finer than the alignments the CRC is offered.

    The second half is what this cost before the header was read at all: with the
    reading turned off the tracked path falls back to the home short cycle, which
    is three of these four cases it cannot read.
    """
    path = (placement.LONG_PATHS if long_cycle else placement.SPEED_PATHS)[2]
    body = b"TRACKED"
    info = (body.ljust(path.crc_bytes - 3, bytes([spec.IDLE]))
            + bytes([spec.status_byte(1)]))
    audio = np.concatenate([np.zeros(FS),
                            placement.data_packet(info, path, swapped=swapped),
                            np.zeros(FS)])
    row0 = (FS + (placement.protocol_config().pulse().size - 1) // 2
            + p3frame.DATA_OFFSET * SPS)

    def tracked(lock: int):
        sync = rxfront.SyncedRx()
        sync.packet_at = lock
        return sync.packet(audio)

    for lock in (row0 - SPS // 2, row0, row0 + SPS // 2 - 1):
        ev = tracked(lock)
        assert ev is not None and ev.packet[2] == body, (lock - row0, ev)

    monkeypatch.setattr(p3rx, "header_of", lambda *a, **k: None)
    assert (tracked(row0) is not None) == (not swapped and not long_cycle)


def test_an_idle_field_delivers_nothing_on_the_tracked_path() -> None:
    """A field of `spec.TEMPLATE` and nothing else is a station with nothing to
    say, and both the scanning decoder and the changeover path have read it as
    empty since the pattern was measured off PIII_Complete_1.

    The tracked path -- the one a connected station uses every cycle -- dropped
    trailing IDLE and nothing else, so the same packet reached it as the pattern
    itself under whatever data type the fill declares. The ARQ hands a field to
    `compress.Decoder` by that declaration, so an idle cycle put a decompressor's
    reading of walking bits on the host data port.
    """
    path = placement.SPEED_PATHS[2]
    status = spec.status_byte(1, data_type=spec.DataType.PMC_ENGLISH)
    info = placement.field_info(b"", path.crc_bytes - 3, status)
    assert info[:-1] == spec.field_fill(len(info) - 1)
    audio = np.concatenate([np.zeros(FS), placement.data_packet(info, path),
                            np.zeros(FS)])
    sync = rxfront.SyncedRx()
    sync.packet_at = FS + (placement.protocol_config().pulse().size - 1) // 2 \
        + p3frame.DATA_OFFSET * SPS
    ev = sync.packet(audio)
    assert ev is not None and ev.packet[2] == b"", ev


@pytest.mark.parametrize("sl", sorted(spec.SPEED_LEVELS))
def test_a_variable_header_says_back_what_it_was_built_from(sl: int) -> None:
    """`variable_header` and `PacketHeader` are inverses, over the whole table."""
    for swapped in (False, True):
        for long_cycle in (False, True):
            vh = p3frame.variable_header(sl, swapped=swapped, long_cycle=long_cycle)
            h = p3frame.PacketHeader(0, vh, 1.0)
            assert (h.swapped, h.long_cycle) == (swapped, long_cycle)
            assert sl in h.levels


# ---------------------------------------------------------------------------
# The confirmation rule: an accept that cannot survive moving the clock
# ---------------------------------------------------------------------------

ARCHIVE = Path(__file__).resolve().parents[5] / "working" / "pactor"
KE5YTA = ARCHIVE / "captures/offair_KE5YTA_p3.wav"
"""16 s of a live PACTOR-3 QSO off 7101.5 kHz, and the only real case-0 header we
hold. Scanned at eighth-symbol resolution its 12,208 positions yield two CRC
accepts, both of them the header, reading `1e1e1e1e1e02` at consecutive
alignments."""


@pytest.mark.parametrize("sl", (2, 3, 4))
def test_a_frame_survives_moving_the_clock(sl: int) -> None:
    """The rule costs a real frame nothing, which is why it can be a rule.

    It is a statement about the matched filter rather than about the signal: the
    filter runs over eight symbols, so its output varies smoothly across one, and
    a frame that decodes at all decodes an eighth of a symbol either side. Held
    here at 3 dB, well below any level these captures were made at.
    """
    rng = np.random.default_rng(3)
    pkt = _packet(sl, _payload(sl))
    audio = pkt + rng.normal(0, np.sqrt(np.mean(pkt ** 2) / 10 ** 0.3), pkt.size)
    row0 = FS + p3rx.GRID_OFFSET * (FS // 100)

    def at(pos: int):
        p = p3rx.decode_at(audio, pos, sl)
        return None if p is None else (p.status, p.payload)

    assert p3rx.confirmed(at, row0, at(row0))


def test_a_real_header_is_confirmed() -> None:
    """Off-air ground truth for the half of the rule that must cost nothing.

    The header is not merely CRC-valid, which by itself is worth one position in
    36,000 and the scan offers 12,208: it is five bytes of 0x1e under status
    0x02, and it holds across the alignments either side.
    """
    if not KE5YTA.exists():
        pytest.skip(f"off-air capture absent: {KE5YTA}")
    audio = rxfront.load_wav(str(KE5YTA))
    hits = list(rx.case0_accepts(audio))
    kept = [(s, f) for s, f in hits
            if p3rx.confirmed(lambda pos: _case0(audio, pos), s, f)]
    assert kept, [(round(s / FS, 3), f.hex()) for s, f in hits]
    assert {f[:6].hex() for _, f in kept} == {"1e1e1e1e1e02"}, \
        [(round(s / FS, 3), f.hex()) for s, f in kept]


def test_an_accident_in_noise_is_not() -> None:
    """...and the half that has to reject something, on a real accident.

    The CRC is 16 bits, so a long enough scan of anything accepts eventually:
    15,408 case-0 positions of white noise produced exactly one, and this is it.
    The window is narrow on purpose -- how RARE an accident is belongs to
    `test_noise_decodes_to_nothing`, and what this one is for is that the rule
    throws away an accept it has already been handed. Which sample it falls on
    is a property of the matched filter, so a pulse change moves it and the
    replacement is found by scanning, never by relaxing the assertion.
    """
    audio = np.random.default_rng(1).normal(0, 0.1, 20 * FS)
    sps = FS // 100
    at = 619980
    hits = list(rx.case0_accepts(
        audio, search=range(at - 4 * sps, at + 4 * sps, sps // 8)))

    assert [s for s, _ in hits] == [at], [(s, f.hex()) for s, f in hits]
    assert not p3rx.confirmed(lambda pos: _case0(audio, pos), *hits[0])


def _case0(audio: np.ndarray, pos: int) -> bytes | None:
    for _, field in rx.case0_accepts(audio, search=range(pos, pos + 1)):
        return field
    return None


def test_a_scan_too_long_to_believe_is_refused() -> None:
    """The budget is arithmetic, and it is what retired the burst-wide SL1 scan.

    That scan spent 19,484 positions on one 88 s FT8 capture and returned a
    five-byte payload. No threshold on the audio was needed to reject it: at
    thirty-two times the budget the trial count alone says it cannot be evidence,
    while the anchored window that replaced it costs a fifteenth of the budget.
    """
    assert p3rx.Scan(trials=19484).expected_false > 30 * p3rx.TRIAL_BUDGET
    assert p3rx.Scan(trials=len(range(-8, 9)) * 4).expected_false < p3rx.TRIAL_BUDGET


def test_an_accept_needs_energy_on_both_carriers() -> None:
    """The presence gate behind every case-0 accept, held at its own boundary.

    A whole packet in noise clears it at the alignment the CRC accepts. The same
    audio with channel 5 replaced by noise -- the construction of the confirmed
    accident held as `rf-corpus/regress/neg_p3_phantom_sl1`, one carrier's signal
    plus the other carrier's floor -- is refused at that same alignment, whatever
    a CRC scan might one day assemble there.
    """
    field = bytes.fromhex("814800008000c8e1")
    pkt = placement.case0_packet(field[:placement.DETECT.crc_bytes - 2],
                                 cfg=modem.ModConfig(sample_rate=FS))
    rng = np.random.default_rng(5)
    pad = np.zeros(FS // 2)
    sigma = float(np.sqrt(np.mean(pkt ** 2))) / 10 ** 0.15         # 3 dB SNR
    audio = np.concatenate([pad, pkt, pad]) + rng.normal(0, sigma, pkt.size + FS)

    pulse = rx._pulse(FS // 100)
    Z = {cn: rx._baseband(audio, cn, FS, pulse) for cn in rx.HEADER_TONES}
    hits = [s for s, f in rx.case0_accepts(audio, fs=FS, Z=Z) if f == field]
    assert hits, "the packet itself no longer decodes"
    assert all(p3rx.sl1_carriers_present(Z, s, fs=FS) for s in hits)

    F = np.fft.rfft(audio)
    F[np.abs(np.fft.rfftfreq(audio.size, 1 / FS) - 1080.0) < 60.0] = 0.0
    notched = np.fft.irfft(F, audio.size) + rng.normal(0, sigma, audio.size)
    Zn = {cn: rx._baseband(notched, cn, FS, pulse) for cn in rx.HEADER_TONES}
    assert not any(p3rx.sl1_carriers_present(Zn, s, fs=FS) for s in hits)


# ---------------------------------------------------------------------------
# A data field off the air, from a station that is not us
# ---------------------------------------------------------------------------

DL6MAA = corpora.REGRESS_FIXTURES / "oracle_pactor3_dl6maa.wav"
"""The whole 79 s of that link, as the corpus holds it.

`occ15` is the first fifteen seconds of the same recording; this is the fixture
the corpus declares, so the one test that says shrike can read somebody else's
speed level 1 does not rest on a file that lives only on one machine."""

OCC15 = ARCHIVE / "captures/occ15.wav"
"""15 s of a real PACTOR-III link, and the recording this receiver could not read
at all until the packet header block was found in it. Six packets on the 1.25 s
cycle, speed level 3 -- which is what an independent decoder reports for it, and
what the variable headers in the recording say for themselves."""


def test_a_real_packet_header_places_the_packet_and_names_the_level() -> None:
    """Acquisition by published codeword, and what it reads out.

    The six anchors fall on an exact 1.25 s raster, so the timing is not fitted;
    every one of them carries a variable header whose speed level is 3; and the
    constant headers alternate between each channel's own word and its
    `spec.CARRIER_SWAP` partner's, cycle by cycle. That last is the frequency
    diversity the specification describes, measured here rather than assumed --
    and it is why a receiver that gathers the tones in one fixed order reads half
    the traffic in the wrong order and none of it at all.
    """
    if not OCC15.exists():
        pytest.skip(f"capture absent: {OCC15}")
    audio = rxfront.load_wav(str(OCC15))
    pulse = rx._pulse(FS // 100)
    Z = {cn: rx._baseband(audio, cn, FS, pulse) for cn in range(spec.N_CHANNELS)}
    heads = p3rx.header_anchors(Z, len(audio), fs=FS)

    assert len(heads) == 6, [(h.at / FS, h.vh, round(h.fit, 3)) for h in heads]
    assert min(h.fit for h in heads) > 0.85
    # To within the quarter symbol the anchor search steps in -- which is the
    # resolution of the answer, not slack allowed for a poor one.
    gaps = np.diff([h.at for h in heads[1:]]) / FS
    assert np.allclose(gaps, spec.CYCLE_SHORT_S, atol=1.01 / (4 * spec.SYMBOL_RATE_BD)), gaps
    assert {h.levels for h in heads[1:]} == {(3,)}
    assert not any(h.long_cycle for h in heads)
    # The swap alternates every cycle, so consecutive packets never agree on it.
    swaps = [h.swapped for h in heads[1:]]
    assert all(a != b for a, b in zip(swaps, swaps[1:])), swaps


def test_a_real_data_field_decodes_at_the_level_the_header_names() -> None:
    """The gate: a CRC-valid PACTOR-III data field from a station that is not us.

    Everything shrike decoded before this was its own transmitter's output, which
    is exactly why the frame conventions could be wrong without any test saying
    so. The link opens on a speed-level-1 entry packet, then one 212-byte level 5
    field, and then settles at level 3, 59 bytes under a status byte reading PMC
    English compression -- each field the level the variable header beside it
    names, and the same progression an independent decoder reports.

    The FIVE level 3 fields are what no accident reaches: they sit on the 1.25 s
    raster to the sample and their sequence numbers run 1, 2, 3, 0, 1 in the order
    the link sent them.

    THE LEVEL 3 FIELDS CARRY NOTHING, and that is a reading rather than a gap:
    every one of them is `spec.TEMPLATE` from byte 0 and nothing else, which is
    what a station with an empty buffer writes and what an independent monitor
    reports as `LEN: 0`. So `info` is the level's whole 59 bytes and `payload` is
    none, and the same is true of the entry packet in front of them. The one
    field here that carries information is the 212-byte level 5 greeting, whose
    own tail resumes the template at byte 4 of it -- a phase no rule here
    reproduces, so those bytes stay in the payload and are the compressed
    stream's business.
    """
    if not OCC15.exists():
        pytest.skip(f"capture absent: {OCC15}")
    scan = p3rx.decode_p3_packets(rxfront.load_wav(str(OCC15)))
    assert [p.sl for p in scan.packets] == [1, 5, 3, 3, 3, 3, 3], \
        [p.report() for p in scan.packets]
    assert not any(p.status & spec.STATUS_LONG_CYCLE for p in scan.packets)
    assert all(0 < len(p.info) <= spec.SPEED_LEVELS[p.sl].payload_short
               for p in scan.packets), [(p.sl, len(p.info)) for p in scan.packets]
    assert [len(p.info) for p in scan.packets] == [5, 212, 59, 59, 59, 59, 59], \
        [len(p.info) for p in scan.packets]
    assert [len(p.payload) for p in scan.packets] == [0, 212, 0, 0, 0, 0, 0], \
        [len(p.payload) for p in scan.packets]
    assert [p.info for p in scan.packets if not p.payload] \
        == [spec.field_fill(len(p.info)) for p in scan.packets if not p.payload]

    body = scan.packets[2:]
    assert [p.seq for p in body] == [1, 2, 3, 0, 1]
    cycles = np.diff([p.start for p in body]) / (FS * spec.CYCLE_SHORT_S)
    assert np.allclose(cycles, 1, atol=0.01), cycles
    # And it is not a scan long enough to have manufactured itself.
    assert scan.expected_false < p3rx.TRIAL_BUDGET, scan.trials


def test_a_real_stations_speed_level_1_field_decodes() -> None:
    """The case-0 convention, read off a station that is not us.

    Case 0 is the one path that had only ever been checked against itself. Its
    cell order, its sixteen-by-nine permutation, its K=9 trellis and its inverted
    sign rule were settled against shrike's own transmitter, and a self-consistent
    error in any of them looks exactly like a working modem: CRC-valid here,
    noise to every station we key at. DL6MAA's entry packet is the check that was
    missing, and it passes.

    THREE readings agree on this packet and only one of them is the CRC:

      * its own variable header is the published word for speed level 1,
        unswapped, short cycle, and fits at 0.98 over the two carriers -- 32 bits
        no accident matches, and the same reading places grid row 0;
      * the accept holds across the alignments either side (`confirmed`) and both
        carriers carry it (`sl1_carriers_present`), inside one trial of the
        anchor rather than a sweep;
      * its five field bytes are the first five of the repeating pattern the
        SAME station's level 3 packets carry four seconds later -- and those come
        off a different code, a different interleave, a different tone set and a
        different anchor. Two unrelated decode chains, one field.

    THAT PATTERN IS THE POINT, not a curiosity of the payload. It is
    `spec.TEMPLATE`, the walking-bit fill a PACTOR-III station writes where it
    has nothing to say, so this packet carries no user data at all -- an empty
    field, a status byte at counter 2 and data type 6, and the CRC:
    `0f 8f 87 c7 c3 1a 66 89`. Every entry packet shrike keys is now that field
    at whatever counter the link is on.
    """
    if not DL6MAA.exists():
        pytest.skip(f"corpus fixture absent: {DL6MAA}")
    scan = p3rx.decode_p3_packets(rxfront.load_wav(str(DL6MAA)))
    entry, *rest = scan.packets
    assert entry.sl == 1 and entry.trials == 1, entry.report()
    assert entry.info == bytes.fromhex("0f8f87c7c3"), entry.info
    assert entry.payload == b"", entry.payload
    assert entry.status == 0x1A and entry.seq == 2
    assert entry.data_type == spec.DataType.PMC_ENGLISH
    assert entry.info == spec.field_fill(5)
    idle = [p.info[:5] for p in rest if p.sl == 3 and p.status & 0x1C == 0x18]
    assert len(idle) == 6 and set(idle) == {entry.info}, idle


def test_the_carrier_swap_reaches_the_speed_level_1_field() -> None:
    """A swapped case-0 packet is unreadable without it, and the swap alternates.

    Which is to say the loss is half of every speed-level-1 link, not a corner:
    the specification exchanges the two virtual carriers on every ARQ cycle, and
    a receiver holding the home order gathers each carrier into the other's
    cells. The blind scan has no way to know -- it reads the home order and that
    is honest -- but an anchored decode does, because the swap is a bit of the
    variable header sitting in front of the field.

    Both carriers are also half a symbol apart (`spec.SUBBAND_LEAD`), so what
    reaches the anchor is a staggered two-tone comb read on one clock, and the
    whole scan has to place it from the variable header alone.
    """
    info = b"HELLO"
    row0 = (FS + (placement.protocol_config().pulse().size - 1) // 2
            + p3frame.DATA_OFFSET * SPS)
    for swapped in (False, True):
        audio = np.concatenate([np.zeros(FS),
                                placement.link_packet(1, info, 0x02,
                                                      swapped=swapped),
                                np.zeros(FS)])
        head = p3frame.PacketHeader(
            0, p3frame.variable_header(1, swapped=swapped), 1.0)
        got = p3rx.decode_at(audio, row0, 1, fs=FS, header=head)
        assert got is not None and got.payload == info, (swapped, got)

        scan = [(p.sl, p.payload) for p in p3rx.decode_p3_packets(audio).packets]
        assert scan == [(1, info)], (swapped, scan)
