# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Shared receive front end: audio in, a stream of decoded PACTOR events out.

This is the single decode path behind both `shrike.monitor` (which renders the
events as text) and the live modem receiver in `shrike.ptc` (which dispatches
them into `PactorArq`). Because they share it, the monitor is a faithful dry-run
of what the session receiver does on the air -- with one stated exception, since
an unstated one is how a dry run stops being faithful: a caller feeding a STREAM
turns `p3_envelope` off, so a live session runs one pass fewer than a monitor
reading the same audio off disk. See `decode_events`.

Each event is the most specific thing the signal supports:
  connect  a PACTOR-1 link-setup burst decoded to a callsign + variant
  cs       a control signal (ACK/NAK/break-in), nearest distance-12 codeword
  packet   a CRC-validated frame -- a PACTOR-3 header/status (SL0) or a PACTOR-1
           ARQ data packet. `Event.packet` is (sl, status, payload, crc_ok), where
           payload is what may be delivered to the host: the P3 case-0 header
           carries no user data so it is empty, a P1 data frame carries its field.
           `Event.protocol` says which of the two it was.
  detect   PACTOR-2 is present. Two different pieces of evidence reach this one
           kind and the line says which: an armed frame marker, eight chips of a
           published codeword carried on both carriers, which names the protocol;
           or, where no marker armed, a pair of carriers with PACTOR-2's spacing
           and simultaneity, which names only a geometry that PACTOR-1 at 200 Bd
           and PACTOR-3 fit as well. Neither carries content -- the field a
           marker heads is read by `p2rx`, whole recordings through
           `decode_bursts` and single cycles through `decode_expected_burst`.
  p1reply  a burst with energy at BOTH PACTOR-1 tones and a duty cycle a reply
           could have. The kind name is older than what the test can support and
           the line does not repeat it: nothing here reads a bit, and nothing
           here establishes that the burst was addressed to us or that it is
           PACTOR-1 at all. Measured on the corpus it also fires on VARA, on
           PACTOR-2 and on a 500 Hz-class ARQ station -- so the line reports the
           shape it measured, in dB, and says what it did not establish.
  fsk      a PACTOR-1 FSK burst is present but did not resolve to a callsign
The live receiver acts only on connect/cs/packet. detect, fsk and p1reply are
shape, and no line among them may be read as a station: `ptc` treats p1reply as
presence and nothing more, which is the same rule stated in code.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Iterator, NamedTuple

import numpy as np

from . import p1rx, p2rx, p3frame, p3rx, pactor2, placement, rx, session, spec

FS = rx.FS_DEFAULT
SPS = rx.SPS_DEFAULT
HDR_TONES = rx.HEADER_TONES

# PACTOR-1 is a 2-FSK with a 200 Hz shift -- ~250 Hz occupied. Anything materially
# wider that merely lights the 1400/1600 bins (FT8's full-passband energy, a 500 Hz
# VARA/ARDOP signal) is not PACTOR-1, whatever those two bins hold.
P1_MAX_BW_HZ = 450
# A control signal is a distance-12 code: a genuine decode lands at 0 errors, and a
# nearest-codeword search over a ~380-position scan will find a 1-error coincidence
# on noise. Confirmed must be a real decode, never a within-radius hit. This is the
# right rule for a receiver that is SWEEPING -- it has no idea when a burst is due,
# so it must not accept one it half-recognises.
CS_MAX_ERRORS = 0
# It was ALSO the right rule for a receiver expecting an answer, and relaxing it
# was this project's most expensive mistake. The reasoning was that a station
# waiting on a reply has a prior worth spending the code's error-correcting radius
# on, and the measured evidence was one 16 s capture with no false hits. On the air
# that relaxation reported ACK, BREAK-IN, CYCLE-TOG and SPEED-UP in sequence from a
# gateway while the operator heard nothing, and declared the link up on the
# strength of it. A 30 s control recording with the transmitter idle then produced
# two "control signals" at radius 2 and none at radius 0.
#
# The real defect was never the radius. It was that the PACTOR-1 control signal was
# not being DECODED at all -- the reply detector only reported presence, and the
# bits were then read off `_p1_cs_bursts`'s burst START, which is a 5 ms energy
# threshold and not a bit grid. That is what produced strong real bursts at 3-5
# errors and made a radius look necessary. `p1rx.cs_bits` now positions the
# twelve-bit window on the signal before reading it, and returns exact matches on
# the same bursts: 30 of 69 across two gateways where it used to return one.
#
# Zero errors, everywhere. The four words sit at mutual distance 8, so a genuine
# burst lands exactly on one and anything else lands 2-4 away.
CS_EXPECTED_MAX_ERRORS = CS_MAX_ERRORS


class _Spectrum:
    """One magnitude spectrum for a hop window, shared by every spectral test.

    _fsk_present, _occupied_bw and _p2_present each used to transform the same
    segment again -- three or four FFTs of a 2 s window per hop, which was the
    receiver's largest single cost on the station Pi.
    """

    __slots__ = ("mag", "freq", "seg")

    def __init__(self, seg: np.ndarray):
        self.seg = seg
        self.mag = np.abs(np.fft.rfft(seg * _window(len(seg))))
        self.freq = np.fft.rfftfreq(len(seg), 1 / FS)


@lru_cache(maxsize=8)
def _window(n: int) -> np.ndarray:
    """A Hann window of length `n`. The spectral tests run per hop over a whole
    capture and all use the same few lengths, so building it each time was ~8% of
    the receiver on the station Pi."""
    return np.hanning(n)


def _occupied_bw(sp: "_Spectrum") -> float:
    """Occupied bandwidth in Hz: the span of passband bins within 14 dB of the peak.
    Cheap, and the strongest single discriminator -- a mode claim whose width is
    impossible for that mode is false however its tones read. Peak-relative rather
    than noise-relative so a signal with sloped edges isn't under-measured."""
    if len(sp.seg) < SPS * 2:
        return 0.0
    F, f = sp.mag, sp.freq
    band = (f > 300) & (f < 2700)
    Fb, fb = F[band], f[band]
    sig = Fb > 0.2 * Fb.max()               # -14 dB
    if sig.sum() < 2:
        return 0.0
    return float(fb[sig].max() - fb[sig].min())


def occupied_bw(seg: np.ndarray) -> float:
    """`_occupied_bw` for a caller that has audio rather than a spectrum.

    Width alone grades nothing -- band noise fills the passband and measures as
    wide as any signal in it -- so this is printed beside a level and never
    thresholded on. What it separates, once something IS known to be there, is a
    250 Hz burst this station failed to read from an emission PACTOR-1 cannot
    make.
    """
    return _occupied_bw(_Spectrum(np.asarray(seg, dtype=np.float64)))


P3_ANCHOR_CHANNELS = (5, 12)
"""The two channels -- 1080 and 1920 Hz -- every PACTOR-III speed level lights.

`spec.SPEED_LEVELS` carries the six channel sets and 5 and 12 are in all of them
[M.1798 §2]; they are also the acquisition burst's own two bands and the pair the
variable header and every control signal ride. So they are what a PACTOR-III
emission can be looked for by without first knowing its speed level.
"""

P3_COMB_KNEE_DB = 8.0
"""Where `p3_comb_db` stops reading as band noise. A REPORTING knee: nothing is
gated on it and no decoder consults it -- it exists so a log line can say which
side of the measurement a window fell on instead of leaving an operator to judge
a bare number.

Measured, on the two 2026-09-11 WS8EOC sessions and this station's own windows:

    our own keyed SL1 entry packet, through the capture mute   +22.8 .. +23.5 dB
    WS8EOC PACTOR-III that CRC-decoded (onair-0911-1022)       +17.2 .. +20.2 dB
    every listen window of onair-0911-2332, the two the
      session printed ANSWER SLOT OCCUPIED for included         -0.8 ..  +2.4 dB
    the 127 ms burst onair-0911-2319 took nothing from              -1.1 dB
    our own PACTOR-1, which lights 1400/1600 Hz and not these       -5.4 dB

Fifteen dB of gap, so the value is placed to be obviously inside it rather than
fitted to either edge.

And the population it costs something against: 5,221 listen windows of this
station's own arms, every fourth session under `captures/`. Median 1.5 dB, 99th
percentile 7.1 dB, so the knee is about where a window stops resembling the
channel. 0.84% of them reach it -- and four of the loudest, from four different
sessions, were read back to see what they were: every one carries channels 4-6
and 11-13 lit with the rest dark at 1.0-1.5 kHz occupied, which is the morning
reception's own signature, and one of their sessions logged `P3 changeover
acquired` twenty-six times. The tail is PACTOR-III that was there, not the
measure fraying.

RE-DERIVED AGAINST LABELLED COPIES ON 2026-09-16 AND LEFT WHERE IT WAS. Every
reading above was taken through a probe pointed at the dial rather than at the
peer, so the knee rested on the wrong channels. Repointed (`p3_comb_db`'s
`offset_hz`), the four copies whose readability is known read

    KB5LZK 40 m 2026-09-15, peer +74.6 Hz  (`fixtures/kb5lzk-answer-0915`)
      cycle 35, a full-strength answer the reader took    12.7 -> 17.5 dB
      cycle 23, the same answer with channel 5 faded       3.1 ->  8.1 dB
    WS8EOC 80 m `captures/onair-0914-2252`, peer -85 Hz, the three SL3
    greeting packets at stream heads 1848762/2688762/2748762
      the registered witness copy, which CRC-decodes       0.6/0.0/2.7 ->
                                                           1.1/3.9/0.6 dB
      the main-input copy, which does not                  1.8/-0.4/0.2 ->
                                                           5.7/3.4/4.1 dB

and they do not separate. KB5LZK orders correctly and WS8EOC does not: the copy
that decodes reads below the copy that does not on two of the three packets, the
two populations interleave, and no knee divides them. The reason is this
measure's own reference rather than the offset. Speed level 3 lights fourteen of
the eighteen channels, so on a strong wide copy the comb median IS a lit channel
-- on the first packet the witness's median is 31 against 22.3 at channel 5, its
weakest lit one -- while a fainter copy of the same packet keeps its median near
the floor. The ratio then ranks the two copies by how much of the comb cleared
the noise, which is not what this reports. A threshold fitted to those four would
be fitted to that artefact, so 8.0 stays where the 2026-09-11 populations put
it. That cost is unmoved either way: an arm has an offset to pass only once
it has tracked a PACTOR-III acquisition, and the 5221 listen windows above
were priced at nominal because nominal is where they were read.
`tests.shrike.test_p3_comb_offset` keeps all four readings in the tree.
"""


def p3_comb_db(seg: np.ndarray, fs: int = FS, offset_hz: float = 0.0) -> float:
    """How far `P3_ANCHOR_CHANNELS` stand over the rest of the 120 Hz comb, in dB.

    Width is not a mode test. `occupied_bw` says so in its own docstring and the
    channel says it louder: measured on `captures/onair-0911-2332`, a window the
    session flagged as occupied read 2381 Hz, a window with nothing in it at all
    read 2398 Hz, and the morning's CRC-valid PACTOR-III frame read 1253 Hz --
    the real reception NARROWEST of the three, because noise fills a passband and
    a signal does not. An hour was spent on the strength of the wide figure.

    What separates them is structure. This asks the one question that needs no
    speed-level hypothesis: are 1080 and 1920 Hz BOTH standing over the channel
    comb's own median? The weaker of the two is taken, so one loud carrier cannot
    answer for both -- which is what puts PACTOR-1's 1400/1600 Hz pair, and any
    other two-tone emission, on the noise side of the measurement.

    Scored over `p3rx.ENVELOPE_SYMBOLS` at a time and reported at its best
    position, so a burst that covers part of the window is not averaged away by
    the rest of it. The median is taken across all eighteen channels rather than
    against a noise floor, which makes the figure a RATIO within one window and
    immune to the 0.8-peak scaling `session.write_wav` applies on the way to disk.

    `offset_hz` IS WHERE THE PEER ACTUALLY IS, and without it the probe measures
    the wrong channels. `TONE_HALFWIDTH_HZ` is 45 and the comb is spaced 120, so a
    station 85 Hz low puts no energy in either nominal anchor and its NEIGHBOURING
    tone 35 Hz inside one of them -- the probe then reads a channel the peer never
    lit against a median it did. Every PACTOR-III peer this station works sits
    tens of hertz off nominal (WS8EOC -85, KB5LZK +75 on the same fortnight), so
    nominal is the case that does not occur. Pass the tracked acquisition offset
    -- `_SessionRx.p3_receive_offset_hz` -- and the whole raster moves with it.

    Returns 0 -- flat, no comb -- for a segment too short to score.
    """
    seg = np.asarray(seg, dtype=np.float64)
    if offset_hz:
        # `p3acquire` imports this module; the cycle is real and this is a
        # reporting path, so the import is paid where the offset is.
        from . import p3acquire
        seg = p3acquire.compensate(seg, offset_hz * p3acquire.FS / fs)
    win = p3rx.ENVELOPE_SYMBOLS * (fs // 100)
    lo, hi = P3_ANCHOR_CHANNELS
    best = 0.0
    for i in range(0, max(seg.size - win, 0) + 1, fs // 100):
        energy = p3rx.channel_energy(seg[i:i + win], fs)
        mid = float(np.median(energy))
        if mid > 0:
            best = max(best, min(energy[lo], energy[hi]) / mid)
    return 20 * float(np.log10(best)) if best else 0.0


EVENT_KINDS = ("connect", "cs", "packet", "detect", "fsk", "p1reply", "unassigned",
               "callb")
"""Every value `Event.kind` can take. This is a PUBLISHED INTERFACE, not a set of
convenient strings, and adding to it is a breaking change for anything that maps
kinds to its own vocabulary.

That is not hypothetical. Adding `p1reply` here raised `KeyError` mid-capture in a
downstream consumer that indexed a dict by kind, and it discarded every later
detection in that recording. The consumer has since been made tolerant of unknown
kinds, but the interface is declared here so the next addition is visible from
this side too.

`unassigned` is the newest, and it is deliberately NOT `cs`: a twelve-bit word
PACTOR-1 gives no meaning has been recognised, and a consumer that dispatched it
as a control signal would be reading a meaning off the table it is absent from
(`pactor1.UNASSIGNED_SIGNALS`). It carries no `cs` index for the same reason, and
names the word in `spare` instead, because what the two words are worth is not
the same: 0x59A in the answer slot is measured -- at two gateways and in eight of
this station's own sessions -- as the PACTOR-3 upgrade grant, and 0x6A9 has no
corpus precedent at all.

`callb` is the branch-B frame family -- Robust Call and both Free Signals -- and
it is deliberately NOT `connect`: the three say different things about who is
free and who is calling whom, `connect`'s address is the CALLER and branch B's is
the DESTINATION, and a consumer that read the two off one kind would have the
direction backwards. It carries its `p1rx.Connect` for the variant and the ident,
and unlike `connect` the frame is CRC-gated.

TWO OF THESE NAMES ARE WRONG and stay wrong for that reason: `p1reply` is not a
reply -- nothing in it reads a bit or identifies a sender -- and `detect` is only
ever a PACTOR-2 carrier pair. Renaming them would break every consumer for a
cosmetic gain, so what is fixed instead is the text an operator reads: the event
text itself, and the display tags in `shrike.monitor` and in creance's runner,
which show P1-BURST and P2-PAIR. `live._print_event` is the one that was not
reached -- it still tags `detect` DETECT and carries no `p1reply` key at all, so
the KeyError this map was given to stop is still live there.
"""


@dataclass
class Event:
    t: float
    kind: str                              # one of EVENT_KINDS
    text: str
    protocol: str | None = None
    """Which protocol the decoder actually read, where the KIND does not say.

    A control signal exists in both: PACTOR-1's are four 12-bit codewords in FSK,
    PACTOR-3's six 20-bit codewords in DBPSK on tones 5 and 12. An adapter that
    inferred the protocol from `kind` therefore announced a decoded PACTOR-3 link
    for an exchange that never left PACTOR-1. It could not have done better --
    the only place the truth appeared was inside `text`, in English, where the
    next rewording of a message would have silently broken it.

    None where `kind` already determines it (`connect`, `fsk`, `p1reply` are
    PACTOR-1). `packet` is NOT one of those: a CRC-valid frame arrives in both
    protocols and this said it was always PACTOR-3, which was wrong from the day
    the PACTOR-1 data decoder landed. Neither is `detect`, whose only emitter is
    the PACTOR-2 carrier pair and which every consumer therefore reported as
    PACTOR-3 -- an unset field reading as a confident wrong answer."""
    connect: p1rx.Connect | None = None
    cs: int | None = None
    spare: int | None = None
    """Which word an `unassigned` event read, as an index into `pactor1.CS_WORDS`
    -- `pactor1.CS_59A` or `pactor1.CS_6A9`.

    A SECOND FIELD RATHER THAN `cs`, because the two ask different things of a
    consumer. `cs` is a word whose meaning PACTOR-1 publishes, and everything
    that reads one acts on that meaning. This names a word the protocol assigns
    none, so what may read it is code with its own measured reason for the one
    word it is looking for -- `ptc.PtcHost` and the 0x59A grant, behind a flag."""
    sense: int | None = None
    """Which FSK shift this control signal arrived in -- 0 as tabulated, 1
    complemented. The peer's Shiftlage for the cycle the burst belongs to, and so
    the one it expects our own transmission in that cycle to arrive in
    (docs/protocols/pactor/pactor1-data-packets.md §7). None where nothing read a shift."""
    packet: tuple | None = None            # (sl, status, payload: bytes, crc_ok)
    breakin: bool = False
    """This packet is a CHANGEOVER packet -- CS3 as its head -- so the station that
    decodes it is being told to stop sending and read the rest of it."""
    start: int | None = None
    """Sample index, WITHIN THE BUFFER THIS DECODE WAS GIVEN, of the frame's grid
    row 0 or the control signal's phase-reference symbol -- where the demodulator
    actually locked, not where the trigger fired. `SyncedRx` tracks it to aim the
    next cycle's decode; a caller that slides a rolling buffer must not, because
    the origin moves under it."""
    cycle_long: bool | None = None
    """Decoded P3 frame length; status bit 5 is only a request, not geometry."""
    carrier_swapped: bool | None = None
    """Measured P3 physical carrier order; independent of header request bit 0."""
    information: bytes | None = None
    """Decoded P3 data, fill and status bytes, before the CRC; never re-padded."""


def load_wav(path: str) -> np.ndarray:
    """Left channel of a WAV as float in [-1, 1], resampled to FS."""
    return session.load_wav(path, FS)


def _pattern_a() -> np.ndarray:
    dA = p3frame.PATTERN_A[1:] * np.conj(p3frame.PATTERN_A[:-1])
    return dA / np.abs(dA)


CALL_B_TONE_RATIO = 1.5
"""How far a tone pair has to stand above the sweep's own median band energy
before the branch-B decoder is pointed at it.

WHERE TO LOOK, NEVER WHETHER TO BELIEVE: the CRC decides, and this only says
which of 43 sweep centres are worth the pass. Measured on 100 Bd trains buried in
noise, the ratio at the pair a real train is keying is 3.1 at the weakest signal
`decode_call_b_all` reads at all and 1.7 at one it reads nothing from, while four
seeds of band noise reach 1.04-1.18 -- so the bar sits below every frame that
could have been decoded and above noise, which is the only place a saving is
allowed to sit."""


def call_b_centres(audio: np.ndarray) -> list[float]:
    """Which of `p1rx.CALL_B_SWEEP`'s tone-pair centres this buffer could hold."""
    p = np.abs(np.fft.rfft(audio)) ** 2
    df = FS / audio.size

    def tone(hz: float) -> float:
        return float(p[int((hz - 25) / df):int((hz + 25) / df) + 1].sum())

    lo = np.array([tone(fc - p1rx.CALL_B_SHIFT / 2) for fc in p1rx.CALL_B_SWEEP])
    hi = np.array([tone(fc + p1rx.CALL_B_SHIFT / 2) for fc in p1rx.CALL_B_SWEEP])
    floor = float(np.median(np.concatenate([lo, hi]))) + 1e-30
    keep = np.minimum(lo, hi) > CALL_B_TONE_RATIO * floor
    return [float(fc) for fc in p1rx.CALL_B_SWEEP[keep]]


def call_b_frames(audio: np.ndarray, centre: float | None = None
                  ) -> list[tuple[float, p1rx.Connect]]:
    """Every branch-B frame in `audio`: Robust Call, or either Free Signal.

    `p1rx.decode_call_b_all` with the sweep narrowed first. Each of its 43 centres
    costs its own pass over the buffer -- 87 ms of every 3 s window, spent on a
    quiet channel entirely on pairs that are not there -- against one FFT to name
    the pairs that are. A caller holding a link pins `centre` instead and pays for
    one pass.
    """
    if centre is not None:
        return p1rx.decode_call_b_all(audio, centre=centre)
    out: list[tuple[float, p1rx.Connect]] = []
    for fc in call_b_centres(audio):
        for t, cn in p1rx.decode_call_b_all(audio, centre=fc):
            # Neighbouring centres read the same key-down, and it is one burst.
            if all(abs(t - was) >= p1rx.CALL_B_LEN * 8 / 100 for was, _ in out):
                out.append((t, cn))
    return sorted(out, key=lambda hit: hit[0])


def _fsk_present(sp: "_Spectrum") -> bool:
    """PACTOR-1 link-setup is FSK keyed between 1400 and 1600 Hz; flag when *both*
    tones stand above the off-tone noise. The floor excludes the 1300-1700 Hz band
    so a lone carrier (which would drag a tone-inclusive median up with it) cannot
    masquerade as two-tone keying."""
    if len(sp.seg) < SPS * 4:
        return False
    F, f = sp.mag, sp.freq
    off = (f > 300) & (f < 2700) & ~((f > 1300) & (f < 1700))
    floor = np.median(F[off]) + 1e-9
    mark = F[(f > 1340) & (f < 1460)].max()
    space = F[(f > 1540) & (f < 1660)].max()
    if min(mark, space) <= 6 * floor:
        return False
    return _occupied_bw(sp) <= P1_MAX_BW_HZ


# PACTOR-2 rides two carriers about 200 Hz apart. So does PACTOR-1: its FSK tones
# are 1400/1600, and both measure 199.2 Hz apart here -- spacing alone cannot tell
# them apart, which is how an earlier detector graded every P1 connect as P2. What
# separates them is that P2 keys both carriers at once while FSK alternates between
# them.
#
# Concurrency was written down as that discriminator and then never asked for: it
# was computed nowhere, and the figure the line printed as "concurrency" was the
# amplitude ratio between the two peaks. What stood in for it was a rule that
# discarded any pair within 25 Hz of 1400/1600 -- which is where an ON-FREQUENCY
# PACTOR-2 station sits, so the detector was constitutionally unable to report the
# one case that matters, and did report an off-frequency stranger at 1704/1910 Hz
# during a session where it was read as the gateway answering.
#
# Measured over the corpus, concurrency does the exclusion's job and does it at
# the right frequency: 0.54-0.98 across the two HB9AK PACTOR-2 fixtures against
# 0.01-0.28 over the three real PACTOR-1 ones, with the pair at 1400/1600 in both
# cases. Counted in lines off the live `decode_events` path, exclusion out and
# gate in: the PACTOR-2 fixtures go 3 -> 7 and 0 -> 3, the on-frequency case
# becoming visible for the first time; the PACTOR-1 fixtures stay at zero, where
# the exclusion had them; the wideband, clipped and VARA fixtures go 4 -> 0; and
# 4.8 h of passive band listening goes 36 -> 20. The two-sided PACTOR-1 recording
# goes 24 -> 14 and those 14 are the PACTOR-3 in its passband, which keys many
# carriers at once and so satisfies this outright -- they are why the line names a
# geometry rather than a mode.
#
# None of it used to reach a window where `_fsk_present` fires: that took the
# `fsk` branch and never asked. PACTOR-1 is the narrower mode and the width gate
# was reading the wider one -- a real PACTOR-2 station measures 326-366 Hz here,
# inside `P1_MAX_BW_HZ`, so HB9AK's 40 s QSO came out as 57 `fsk` lines, 4
# `p1reply`, one PACTOR-1 CONNECT naming a callsign out of 8-DPSK, and 7 of these,
# none of them on a burst. The precedence is now settled by `_p2_markers` above
# this test rather than by width, and this gate reports what is left: a window
# with the geometry and no marker in it.
P2_SPACING_HZ = (150.0, 250.0)
P2_MIN_CONCURRENCY = 0.35
# Balance is what is left of a concentration test. The pair's share of in-band
# energy (0.90) and an occupied width (600 Hz) were gates here too, and both were
# calibrated on clean lab clips: a real weak PACTOR-2 reply measures 0.39 and
# 1800 Hz on them, because on a weak signal those figures describe the receiver's
# passband. Balance survives because it catches what they were really for -- a
# lone carrier paired with its own noise sidelobe 200 Hz away, which is perfectly
# "concentrated". PACTOR-2 drives both carriers equally, measured at 0.97, against
# 0.00-0.04 for the single-carrier signals this used to admit.
P2_MIN_BALANCE = 0.30
# How far the pair must stand above adjacent noise-only guard bands.
P2_MIN_EXCESS = 3.0


def _carrier_concurrency(seg: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Fraction of active symbol slots in which BOTH carriers are strong.

    The slots are a fixed SPS grid from the segment's start and nothing aligns
    them to the signal, so a window straddling a tone change sees both tones at
    roughly half strength -- which clears the 0.35 threshold below. Real traffic
    does not key that way and it does not show: shrike's own PACTOR-1 transmitter
    measures 0.0 here and real off-air PACTOR-1 0.01-0.28. A synthetic that
    alternates on every single symbol does, at 0.42-0.96, so do not grade one
    with this.
    """
    n = np.arange(SPS)
    wl = np.exp(-2j * np.pi * f_lo * n / FS)
    wh = np.exp(-2j * np.pi * f_hi * n / FS)
    lo, hi = [], []
    for i in range(0, len(seg) - SPS, SPS):
        s = seg[i:i + SPS]
        lo.append(abs(s @ wl))
        hi.append(abs(s @ wh))
    lo, hi = np.array(lo), np.array(hi)
    if lo.size == 0 or lo.max() <= 0 or hi.max() <= 0:
        return 0.0
    on_l, on_h = lo > 0.35 * lo.max(), hi > 0.35 * hi.max()
    active = (on_l | on_h).sum()
    return float((on_l & on_h).sum() / active) if active else 0.0


# PACTOR-2's two carriers sit at 1400 and 1600 Hz -- 200 Hz apart, CENTRED ON 1500.
# The centre is what the earlier gates were missing: VARA's two strongest peaks are
# also ~200 Hz apart and comparably strong, but they land wherever its passband
# happens to be (measured 605/405 Hz, centre 505), while an on-frequency P2 signal
# measured 1400/1606, centre 1503. The window allows for receiver mistuning -- the
# sigidwiki lab clips sit at centre 1694 -- but assumes the rig is roughly on
# frequency, which is the operating case. captures/PACTOR-II.wav sits at centre
# ~1019 Hz -- about 480 Hz low -- and is deliberately NOT matched: widening far
# enough to catch an archive clip that mistuned would start admitting VARA again.
P2_CENTRE_HZ = (1350.0, 1800.0)


def _peak_pair(sp: "_Spectrum") -> tuple[float, float, float] | None:
    """The two strongest narrow peaks: (lo, hi, amplitude ratio) or None."""
    F, f = sp.mag, sp.freq
    b = (f > 300) & (f < 2900)
    F, f = F[b], f[b]
    floor = np.median(F) + 1e-12
    idx = [i for i in range(2, len(F) - 2)
           if F[i] == max(F[i - 2:i + 3]) and F[i] > 4 * floor]
    idx.sort(key=lambda i: -F[i])
    top = []
    for i in idx:
        if all(abs(f[i] - f[j]) > 60 for j in top):
            top.append(i)
        if len(top) == 2:
            break
    if len(top) < 2:
        return None
    a, c = sorted(top, key=lambda i: f[i])
    return float(f[a]), float(f[c]), float(min(F[a], F[c]) / max(F[a], F[c]))


class _P2Pair(NamedTuple):
    """What the carrier-pair test measured. Every field appears in the line it
    produces, so a reader can judge the claim instead of taking the label."""
    f_lo: float
    f_hi: float
    concurrency: float          # active slots with BOTH carriers up
    balance: float              # weaker peak / stronger peak
    excess_db: float            # pair over the adjacent guard bands


def _p2_present(sp: "_Spectrum") -> _P2Pair | None:
    """A pair of carriers with PACTOR-2's spacing, balance and simultaneity.

    This is where the signal IS and how strong it is, never what it says. It
    cannot tell PACTOR-2 from another two-carrier mode of the same geometry: the
    corpus's two-sided PACTOR-1 recording carries PACTOR-3 in the same passband
    and still yields 14 lines, because PACTOR-3 keys many carriers at once and so
    satisfies a simultaneity test outright. Only a decode separates them, and the
    PACTOR-2 byte decode is not reachable from here.
    """
    seg = sp.seg
    if len(seg) < 4 * SPS:
        return None
    pair = _peak_pair(sp)
    if pair is None:
        return None
    f_lo, f_hi, ratio = pair
    if not P2_SPACING_HZ[0] <= f_hi - f_lo <= P2_SPACING_HZ[1]:
        return None
    if not P2_CENTRE_HZ[0] <= 0.5 * (f_lo + f_hi) <= P2_CENTRE_HZ[1]:
        return None
    if ratio < P2_MIN_BALANCE:
        return None
    # The pair must stand above adjacent guard bands, not merely above the local
    # median: a median-relative peak search finds a "pair" in pure noise, which
    # fired 7 times across 40 empty windows on 30 m. Same absolute reference the
    # PACTOR-1 reply test uses.
    F, f = sp.mag, sp.freq
    half = 60.0
    tone = F[(np.abs(f - f_lo) < half) | (np.abs(f - f_hi) < half)].mean()
    guard = F[((f > 700) & (f < 1000)) | ((f > 2000) & (f < 2400))].mean() + 1e-12
    if tone / guard < P2_MIN_EXCESS:
        return None
    # Last, because it is the only test here that costs a pass over the audio.
    conc = _carrier_concurrency(seg, f_lo, f_hi)
    if conc < P2_MIN_CONCURRENCY:
        return None
    return _P2Pair(f_lo, f_hi, conc, ratio, 20 * float(np.log10(tone / guard)))


class _P2Frame(NamedTuple):
    """An armed PACTOR-2 frame marker and the data field it heads."""
    t: float
    end: float
    path: pactor2.Path
    score: float


def _p2_markers(audio: np.ndarray) -> list[_P2Frame]:
    """Where PACTOR-2 frames are in `audio`, on the marker's own evidence.

    THE ONLY TEST IN THIS FILE THAT CAN NAME PACTOR-2. Everything else offered
    for the job is a geometry -- two carriers 200 Hz apart keyed together, which
    PACTOR-1 at 200 Bd and PACTOR-3 satisfy as readily -- while this correlates
    eight chips of a published complex codeword against the differential phase of
    both carriers and arms at `p2rx.MARKER_ARM` of a noiseless one. The codeword
    index it recovers carries the speed level and the frame length, so the span
    the frame occupies is read rather than assumed.

    Measured across the 31 fixtures of `rf-corpus/regress`, at 1% of real time:
    41 arm on the two PACTOR-2 recordings that are graded against bytes, 8 on the
    archive clip captioned PACTOR-2, and ONE anywhere else -- in 68 s of VARA.
    Zero on all five PACTOR-1 positives, zero on all seven negatives, zero on
    both PACTOR-3 fixtures. That separation is what lets it take precedence over
    `_fsk_present` without the FSK path losing anything.
    """
    out = []
    for t, _bin, k, _sense, score, _swapped in p2rx.find_markers(audio, FS, 0.90):
        if score < p2rx.MARKER_ARM:
            continue
        path = (pactor2.PATHS_LONG if (k >> 3) & 1 else pactor2.PATHS)[(k >> 1) & 3]
        out.append(_P2Frame(t, t + path.n_symbols / p2rx.SYMBOL_RATE, path, score))
    return out


# A PACTOR-1 peer answers a SELCALL with a short FSK burst in its slot of the ARQ
# cycle -- measured off W6IDS at 1395/1600 Hz, ~350 ms, 100 Bd, tones ALTERNATING
# (concurrency 0.13-0.43 against 0.64 for two simultaneous carriers).
#
# WHAT THIS ESTABLISHES, AND ONLY THIS: a burst of roughly that shape was in the
# window. It does not read a bit -- at the SNR a real reply arrives with, the bit
# decisions are not trustworthy (eye 0.79 against the 0.86 of a capture that only
# just decodes) -- and it has no way at all to tell whose burst it was or whether
# it answers anything. The line used to end "a peer answered"; on 2026-08-03,
# listening passively with the transmitter idle, it said so 162 times across
# 4.8 h on the eight Winlink channels `tools/nightwatch.sh` sweeps.
P1_REPLY_MIN_MS = 120
# Wider than the 0.13-0.43 measured above, and it stays wider: raising the floor
# costs real PACTOR-1 faster than it removes anything else. At 0.10 the corpus's
# PACTOR-1 fixtures fall from 64 lines to 44 while the rest fall only from 15 to
# 6, which is not separation. The both-tones gate below is what does that work.
P1_REPLY_CONC = (0.05, 0.55)
# How far the 1400/1600 pair must stand above adjacent noise-only guard bands.
# Deliberately PRECISE rather than sensitive. Measured across five recorded
# replies the burst excess runs 1.09-2.46x while quiet windows run 0.96-1.24x --
# they overlap, so only the clear ones can be claimed. At 2.0 this catches 2 of 5
# replies and produces no false positive on any quiet window or on FT8. A missed
# reply costs a retry; a false link-up would have us transmitting data at a
# station that never answered.
P1_REPLY_MIN_EXCESS = 2.0
# ...and then EACH TONE separately, because that excess is a mean over the two
# tone windows and a mean is met by one of them alone at twice the figure. In most
# of the 162 windows above, one tone stood 3-6x over the guard while the other sat
# at 0.7-1.0x: a lone carrier, which cannot be 2-FSK however long it lasts.
#
# 1.7 is the knee, and it is a knee rather than a margin: measured window by
# window, the real PACTOR-1 fixtures hold 61 of 64 at 1.7 and start going at 1.8
# (59) and 2.0 (54), while every step below 1.7 buys back more noise than signal.
#
# Counted in lines off the live `decode_events` path: the night above goes 162 ->
# 42, and over the corpus fixtures the seven VARA recordings go 51 -> 9 and dense
# local FT8 goes 8 -> 0, while the three real PACTOR-1 recordings go 66 -> 63.
# Those 9 are why the line still refuses to name a peer.
P1_REPLY_MIN_TONE_EXCESS = 1.7
# Fraction of the window a genuine reply may occupy before it looks continuous.
P1_REPLY_MAX_DUTY = 0.55


# The early test decides only whether to STOP LISTENING, and being wrong there is
# cheap -- a false stop costs one window, while a missed one costs the handshake.
# So it is deliberately more sensitive than P1_REPLY_MIN_EXCESS, which gates a
# LINE AN OPERATOR READS and must stay precise. (That comparison used to say the
# link-up decision; nothing here has decided that since `ptc` was made to treat
# `p1reply` as presence, and a threshold justified by a job it no longer does is
# the same defect as a message justified by a decode that did not happen.)
P1_STARTING_EXCESS = 1.4


def p1_reply_starting(seg: np.ndarray, min_ms: float = 40.0) -> bool:
    """Is a peer's burst UNDER WAY? A cheaper, earlier test than _p1_reply.

    The full test needs >=120 ms of burst plus a 0.40 s span to compare against
    its guard bands, which a short slice cannot supply -- so a listener polling in
    slices never breaks early and answers a cycle late. This asks only whether the
    tone pair is up right now, which is enough to stop listening and reply.
    """
    if seg.size < SPS * 2:
        return False
    F = np.abs(np.fft.rfft(seg * _window(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / FS)
    tone = F[((f > 1360) & (f < 1440)) | ((f > 1560) & (f < 1640))].mean()
    guard = F[((f > 900) & (f < 1200)) | ((f > 1800) & (f < 2100))].mean() + 1e-12
    if tone / guard < P1_STARTING_EXCESS:
        return False
    n = np.arange(SPS)
    wm = np.exp(-2j * np.pi * 1400 * n / FS)
    ws = np.exp(-2j * np.pi * 1600 * n / FS)
    hot = 0
    for i in range(0, seg.size - SPS, SPS // 2):
        w = seg[i:i + SPS]
        if abs(w @ wm) + abs(w @ ws) > 0:
            hot += 1
    return hot * (SPS / 2) / FS * 1000 >= min_ms


# A control signal is 12 bits at 100 Bd = 120 ms. The window is generous either
# side because the burst is bounded by an energy threshold, not by a clock.
#
# The two corpus bursts that decode at zero errors now measure 119.0 and 121.0 ms
# (they read 80 and 75 against the old fixed-threshold profile, which is what put
# this floor at 60 rather than 80). The floor stays at 60 anyway, because a burst
# that fades mid-word arrives here in pieces and the shortest of them still
# decodes: 60 is far from the 0 ms that noise, a 200 Hz-shift signal and a
# PACTOR-3 burst all produce, and nothing is bought by tightening it.
CS_BURST_MS = (60, 260)


CS_WIN_S = 0.020        # spectral analysis window
CS_HOP_S = 0.001        # and how far it advances


def _cs_profile(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p1 excess, wideband excess, window CENTRE times) over a listen window.

    Excess is against the MEDIAN of the passband, not against a pair of bands
    picked to look quiet. A fixed guard at 2600-2900 Hz sits in the SSB filter's
    skirt -- 10-15 dB down on an FT-891 in PKTUSB -- so that ratio reported the
    receiver's own shape and read 4-11x on captures holding nothing but band
    noise, which is how a night of empty windows looked like fragmented replies.
    The median is insensitive to that and to a few narrowband signals sharing the
    passband. Calibration, on corpus audio: a real control signal peaks at 8.1-8.6
    and band noise at 2.8.

    Times are the window's CENTRE because that is where its measurement belongs:
    stamping a 20 ms window by its leading sample reports every edge 10 ms early
    and, worse, made the caller's arithmetic look like it was about the signal.
    """
    w, h = int(CS_WIN_S * FS), int(CS_HOP_S * FS)
    if seg.size < w * 4:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    # The window length is fixed, so the three band masks are too. Built once:
    # at this hop they would otherwise be rebuilt a thousand times a second.
    f = np.fft.rfftfreq(w, 1 / FS)
    band = (f > 300) & (f < 2600)
    tone = ((f > 1360) & (f < 1440)) | ((f > 1560) & (f < 1640))
    wideband = (f > 1900) & (f < 2500)
    p1, wide, ts = [], [], []
    for i in range(0, seg.size - w, h):
        F = _Spectrum(seg[i:i + w]).mag
        g = np.median(F[band]) + 1e-12
        p1.append(F[tone].mean() / g)
        wide.append(F[wideband].mean() / g)
        ts.append((i + w / 2) / FS)
    return np.array(p1), np.array(wide), np.array(ts)


# WHETHER ANYTHING WAS THERE, decided on how long the FSK-tone energy LASTED and
# not on how high it reached. The peak is the mean of two spectral bins over the
# passband median across a 20 ms window, and a listen window holds one of those
# per millisecond -- so its maximum is an extreme-value statistic and grows with
# the window. Band noise alone reaches 4.4x in the 0.24 s windows of a keyed cycle
# and 4.7x in the 1.25 s ones of a hushed cycle, on the same channel in the same
# minute, which is a threshold on how long we listened rather than on what was
# there. At 3.5 it fired on 18 of the 21 cycles of the N5UXT 14110.0 run of
# 2026-08-14 and called each one interference; the operator at the rig, asked what
# the channel sounded like, said "quiet -- no real QRM".
#
# Duration separates where amplitude does not. Over 15,397 quiet listen windows --
# the 21 of that run, 15,156 sliced from ten 80 m monitor sessions the receiver
# logged zero detections in, the 6950 kHz out-of-band control and
# `neg_noise_chatham` -- the longest run of FSK-tone energy is 11 ms, and 8 ms
# outside that run. One further 80 m recording is deliberately not in that
# population: it reaches 76 ms, and the monitor log beside it reports a 1400/1600
# burst at the same point, so it is the detector working. A control signal is
# 120 ms: across the eight WS8EOC sessions of 2026-07-30, in which the gateway
# answered every cycle, 146 of 166 receive windows exceed 20 ms and 103 hold a run
# of control-signal length outright.
#
# 20 ms is 1.8x the quiet maximum and a sixth of a control signal. It costs three
# of the twenty answered windows anchored in `test_p1cs` their middle label -- they
# profile at 9, 14 and 16 ms and read as nothing there -- and that is the right
# direction to be wrong in, because the line below claims only what it measured
# and a fragment too weak to measure is not evidence of a peer.
#
# An occupied channel is not thereby called quiet: 7102.0 kHz, twenty carrier-pair
# detections up to +21.5 dB over guard, exceeds 20 ms on 13 of 13,067 windows
# because its traffic is not at the PACTOR-1 tones, while 7101.5 and 7106.5 --
# which are -- exceed it on every window of both recordings, at 74-229 ms.
P1_ENERGY_MIN_MS = 20


def cs_evidence(seg: np.ndarray) -> str:
    """What a listen window actually contained, in one line.

    A cycle that decodes nothing used to report only that it decoded nothing,
    which reads the same whether the band was empty, someone answered in a mode
    that is not this one, or the detector was broken. It was in fact the third,
    for a whole session, and the log could not have said so.

    Three states, because there are three things this can tell apart: nothing at
    the tones, energy that is not a control signal, and a burst of the right
    length. What it CANNOT tell apart is what put the energy there -- a station we
    failed to read, a passing signal, a noise excursion -- so the middle line
    names none of them. It used to name interference, on a channel the operator
    could hear was quiet, and a false cause in the log is how a session acquires
    an explanation it never earned.

    Runs and bursts come off one profile through `_p1_runs`, which is the
    instrument `_p1_cs_bursts` filters, so the line and the verdict cannot drift
    apart. They had drifted: this counted its own runs off the raw threshold mask
    while the burst detector closed short holes and refined to half height, so a
    window holding a burst the detector measured at 111 ms was reported as an
    18 ms run of energy that was not shaped like one.
    """
    prof = _cs_profile(seg)
    p1 = prof[0]
    if p1.size == 0:
        return "no audio"
    runs = _p1_runs(prof)
    run_ms = round(max((d for _, d in runs), default=0.0) * 1000)
    head = f"peak {p1.max():.1f}x median, longest run {run_ms} ms"
    n = len(_cs_length(runs))
    if n:
        return f"{head}, {n} burst(s) of control-signal length -- CLOSE, keep this capture"
    if run_ms < P1_ENERGY_MIN_MS:
        return f"{head} -- nothing at the FSK tones above the band noise"
    return (f"{head} -- energy at the FSK tones, not a control signal's length; "
            f"not attributed (a burst we could not read and interference "
            f"read alike here)")


def _half_height(xs: np.ndarray, lo: int, hi: int) -> tuple[int, int]:
    """Refine a detected run [lo, hi) to its half-height crossings.

    The profile is the burst convolved with the 20 ms analysis window, so a
    rectangular burst arrives as a plateau with symmetric ramps. Its width AT
    HALF PLATEAU is the burst's own duration -- exactly, for any symmetric
    window, because the two ramps are mirror images about the burst's edges. Its
    width at a fixed absolute threshold is not: that reads the ramp wherever it
    happens to cross, so the answer moves with how strong the burst is. A 120 ms
    control signal profiled as 130 that way, and the 10 ms was read as evidence
    that `spec.P1_CS_S` was wrong.

    The plateau is the run's MEDIAN level and the floor the median of the 50 ms
    either side -- neither assumed, and neither a peak. A real burst fades across
    its own 120 ms and dips at every tone transition, so its maximum is an
    outlier and halving it lands above the plateau: on 26 clean bursts that alone
    costs 10 ms and widens the spread from 4 ms to 13.
    """
    guard = int(round(0.05 / CS_HOP_S))
    near = np.concatenate([xs[max(0, lo - guard):lo], xs[hi:hi + guard]])
    floor = np.median(near) if near.size else 0.0
    half = 0.5 * (floor + np.median(xs[lo:hi]))
    a, b = lo, hi - 1
    while a > 0 and xs[a - 1] >= half:
        a -= 1
    while a < b and xs[a] < half:
        a += 1
    while b + 1 < xs.size and xs[b + 1] >= half:
        b += 1
    while b > a and xs[b] < half:
        b -= 1
    return a, b


def _p1_runs(prof: tuple[np.ndarray, np.ndarray, np.ndarray]
             ) -> list[tuple[float, float]]:
    """Every continuous stretch of narrowband FSK energy: (start_s, dur_s).

    The shared body of the three questions asked of this profile -- "is there a
    control signal here", "where did the peer start transmitting" and "what was
    in this window at all" -- which differ only in the lengths they accept. It
    takes the profile rather than the audio so the third can report the same runs
    the first two act on without paying for a second pass over the window.
    """
    p1, wide, ts = prof
    if p1.size == 0:
        return []
    xs = np.where(wide < 2.5, p1, 0.0)
    on = xs > 4.0
    # Close brief holes before segmenting. A control signal is one continuous
    # burst, so a gap of a symbol or two is an artefact of the analysis rather
    # than a boundary -- the wideband veto above flickers on a clean two-tone
    # signal, because leakage into 1900-2500 Hz varies window to window, and it
    # was chopping a single 120 ms burst into fragments too short to qualify.
    hole = int(round(0.020 / CS_HOP_S))
    gap = 0
    for i in range(len(on)):
        if not on[i]:
            gap += 1
        else:
            if 0 < gap <= hole:
                on[i - gap:i] = True
            gap = 0
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            a, b = _half_height(xs, i, j)
            out.append((ts[a], (b - a) * CS_HOP_S))
            i = j
        else:
            i += 1
    return out


# A mark/space balance gate on control-signal candidates, tried and not adopted.
# All four control signals carry exactly six ones and six zeros --
# `[bin(w).count("1") for w in pactor1.CONTROL_SIGNALS]` is `[6, 6, 6, 6]`. So a
# genuine one spends half its length on each tone, and mark/space energy over the
# burst comes out near unity whatever the codeword. That is a property of the code
# itself, not a threshold fitted to a recording.
#
# Measured across the negative corpus: it takes the candidates reaching the decoder
# from 43 to 7 while keeping both known-good anchors. Tightening to 0.80-1.25
# leaves 2 but drops an anchor -- a real burst fades across its 120 ms, so the
# usable tolerance is wider than the arithmetic alone suggests.
#
# NOT USED, and here so the idea is not had again from scratch. It measures well
# on paper and fails on the pipeline:
#
#   * gate 0.70-1.43 cuts foreign candidates 43 -> 7 with both anchors kept, when
#     each burst is handed to it on its own;
#   * inside `_p1_runs`' sliding window the SAME anchor burst at t=8.69 measures
#     1.574 in one bracketing and 1.376 in another, so it is rejected or kept
#     depending on where the window happened to fall.
#
# A 15% swing with window alignment makes it a coin toss on real signals, and the
# cause is imprecise burst boundaries -- the very thing it was meant to work
# around. It also buys nothing measurable: false accepts are already zero, so all
# it can do is cost true positives. Revisit only if burst extents become exact.


def _cs_length(runs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Those runs whose length is a PACTOR-1 control signal's."""
    return [(t, d) for t, d in runs
            if CS_BURST_MS[0] / 1000 <= d <= CS_BURST_MS[1] / 1000]


def _p1_cs_bursts(seg: np.ndarray) -> list[tuple[float, float]]:
    """Runs whose length is a PACTOR-1 control signal's: (start_s, dur_s).

    Its own detector rather than a branch of the reply-presence test, which was
    tuned to notice that SOMEONE answered and floors out at 120 ms -- long enough
    to miss a real 115 ms control signal entirely. Presence and identity want
    different detectors: this one is looking for a specific length of a specific
    thing, and the wideband veto in `_p1_runs` keeps a PACTOR-3 data burst from
    qualifying.
    """
    return _cs_length(_p1_runs(_cs_profile(seg)))


# Below this a run is a fragment -- a transition flicker, a noise excursion -- and
# not a transmission worth referencing a raster to. The same floor `CS_BURST_MS`
# uses, and for the same measured reason: a real control signal profiles at its
# nominal length -- 119.0 and 121.0 ms on the two corpus bursts -- and a burst
# that fades mid-word arrives in pieces, the shortest of which still decodes.
P1_ONSET_MIN_MS = CS_BURST_MS[0]


def p1_bursts(seg: np.ndarray,
              min_ms: float = P1_ONSET_MIN_MS) -> list[tuple[float, float]]:
    """Each PACTOR-1 burst in this window: where it starts and how long it runs.

    THE LENGTH WAS ALWAYS MEASURED AND ALWAYS DROPPED. `p1_burst_onsets` is this
    with the second half thrown away, and every caller that then needed a length
    assumed one: `onair._report_collision` timed every burst as a 120 ms
    codeword, so on 2026-09-04 it reported the 193 ms tail of a packet we had
    transmitted over as ending 1363 ms before our carrier where it ended 1290.
    A run is what it is; the detector knows.
    """
    return [(t, d) for t, d in _p1_runs(_cs_profile(seg))
            if d >= min_ms / 1000 and t > CS_WIN_S / 2]


def p1_burst_onsets(seg: np.ndarray,
                    min_ms: float = P1_ONSET_MIN_MS) -> list[float]:
    """Where each PACTOR-1 burst in this window STARTS, in seconds. Any length.

    `_p1_cs_bursts` answers a narrower version of this, and its 260 ms ceiling is
    exactly what makes it the wrong instrument for a transmit raster: a station
    holding the channel sends 0.96 s data packets, so a receiver anchored on
    control-signal-length runs alone would follow its peer while it was the ISS
    and free-run the moment the link turned around. Same energy threshold, same
    wideband veto; only the ceiling is dropped.

    A run that is already under way at the window's first sample is NOT an onset
    and is dropped: no rising edge was observed, so all it says is that the burst
    began at some unknown time before we started listening. Reporting it as a
    start would hand a raster the moment our own T/R recovery ended, which is a
    property of our receiver and not of the peer's clock. The first profile
    window is centred half a window in, so that is where the test sits.
    """
    return [t for t, _ in p1_bursts(seg, min_ms)]


# WHAT A DETECTOR SHAPED LIKE PACTOR-1 CANNOT SEE, and it is both halves of the
# one above. `_p1_runs` measures two 80 Hz bins against the passband MEDIAN and
# then vetoes any window carrying energy at 1900-2500 Hz, so that a PACTOR-3 data
# burst cannot be read as a control signal. A peer that leaves PACTOR-1 for
# something 1.5-2.6 kHz wide trips the veto and, at the same time, raises the very
# median the tone bins are divided by -- so the excess FALLS as the channel gets
# louder. Measured over the fifteen consecutive listen windows WS8EOC filled in
# captures/onair-0828-1844, the tone excess peaks at 1.8-2.9 against a threshold
# of 4.0, where the six cycles it answered PACTOR-1 in reach 8.0-26.7. The
# session logged `nothing heard` on all fifteen and signed off.
#
# So this asks the other question and only it: HOW LOUD IS THE PASSBAND. No tones,
# no bandwidth, no length -- a burst of any modulation raises it, and so does a
# rise in the band noise. It is therefore not a detection on its own and must not
# be read as one: the caller says WHERE it is asking and against what this session
# has heard that place be quiet, and only that comparison claims anything.
OCCUPANCY_BAND_HZ = (300.0, 2600.0)
#: Short enough that a control signal spans two dozen of them, long enough that a
#: frame's power is a power and not a sample. The profile grid `_cs_profile` runs
#: on, taken from the same argument.
OCCUPANCY_FRAME_S = 0.005
#: Below this a tenth-percentile is the minimum wearing a sleeve.
OCCUPANCY_MIN_FRAMES = 8


def quiet_level_db(seg: np.ndarray, t0: float = 0.0,
                   t1: float | None = None) -> float | None:
    """The QUIETEST TENTH of `seg[t0:t1]`, as mean passband power in dB.

    A TENTH RATHER THAN THE MEDIAN, because the median is the burst. Over the six
    windows of `captures/onair-0828-1844` in which WS8EOC answered with a real
    control signal, the codeword covers more than half the span an answer is
    searched across, so the median measures the codeword: -13.7 and -6.5 dB in
    the first and last of them, against -20.2 and -14.3 for the channel
    underneath. What a later window has to be compared against is the channel,
    or the comparison is between two signals.

    AND A TENTH RATHER THAN THE MINIMUM. The two agree closely on this material --
    the floor they hand the caller differs by 0.6 dB across those six windows --
    and the tenth is the one that does not move with how long the caller listened,
    a single frame's power being an extreme-value statistic like the peak
    `P1_ENERGY_MIN_MS` was written to replace.

    BAND-LIMITED FIRST, and that is not cosmetic. The SSB filter's skirt and the
    codec's own noise live outside 300-2600 Hz and move with neither the band nor
    the far end, so a wideband RMS carries them along and dilutes what did change.
    Between WS8EOC's answered cycles and the fifteen it filled, the band limit
    takes the separation from 3.2 dB to 9.0.

    Returns None where the span is too short or the PCM cannot measure RF:
    non-finite samples, or an exactly constant run spanning two power frames.
    Ten milliseconds contains at least three cycles at the passband's lowest
    frequency. Such a digital hold is not a quiet RF baseline, regardless of
    whether another part of the window contains a short valid burst.
    """
    a = np.asarray(seg, dtype=np.float64)
    a = a[int(round(t0 * FS)):None if t1 is None else int(round(t1 * FS))]
    w = int(round(OCCUPANCY_FRAME_S * FS))
    if a.size < w * OCCUPANCY_MIN_FRAMES:
        return None
    if not np.all(np.isfinite(a)):
        return None
    changes = np.r_[0, np.flatnonzero(np.diff(a) != 0) + 1, a.size]
    if np.max(np.diff(changes)) >= 2 * w:
        return None
    lo, hi = OCCUPANCY_BAND_HZ
    F = np.fft.rfft(a)
    f = np.fft.rfftfreq(a.size, 1 / FS)
    F[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(F, n=a.size)
    n = a.size // w
    e = (x[:n * w].reshape(n, w) ** 2).mean(1)
    return float(10 * np.log10(np.percentile(e, 10) + 1e-20))


#: Short enough that a 120 ms codeword covers a dozen of them and its leading edge
#: lands in one, long enough that a bin's power is a power. The report this came
#: out of measured in these and nothing here re-derives them.
ONSET_BIN_S = 0.010
#: The lead a step is measured against, and the tail it is measured over. Three
#: bins of each: fewer and the median is a sample, more and a turnaround at the
#: near edge of the band has no lead in front of it.
ONSET_LEAD_BINS = 3
ONSET_TAIL_BINS = 3
#: The latest `d` a turnaround may be claimed at. Every control signal WS8EOC sent
#: on 2026-09-04 arrives at 90-100 ms and none of the eight sessions on file
#: carries one past 134 ms (`PEER_TURNAROUND_S`), so a step found after this is a
#: change in the channel rather than somebody answering our packet.
ONSET_STEP_MAX_S = 0.130
#: What "the same band" is measured to. Fine enough to separate the 2180-2400 Hz
#: group of 2026-09-04's first flag from the 1400/1600 an answer sits at, coarse
#: enough that a 220 Hz emission lands in a handful of bins rather than a comb.
ONSET_BAND_HZ = 50.0


class SlotOnset(NamedTuple):
    """Whether the energy in a listen window STARTED inside it, and where it sits.

    `bands` is passband power per `ONSET_BAND_HZ`, linear, for a caller keeping a
    running floor of its own; `step_db` is the whole of the finding.
    """

    step_db: float
    bands: np.ndarray


def answer_onset(seg: np.ndarray, t0: float = 0.0,
                 t1: float | None = None) -> SlotOnset | None:
    """The rise from this window's flat lead to the rest of it, in dB.

    WHAT A LEVEL CANNOT SAY, and it is the half `quiet_level_db` warns about. A
    level compares a window against another window and reports that the channel
    got louder; it cannot tell an answer to our packet from a station that was
    already transmitting when our carrier dropped. An answer has a turnaround, so
    it BEGINS inside the window, at the one instant our own transmission fixes --
    and the beginning is a property of this window alone, which is why nothing
    outside it is consulted.

    MEASURED, `captures/onair-0904-1659`, 28 listen windows of one WS8EOC arm.
    The five cycles the peer answered in step +3.8, +3.8, +4.1, +5.7 and +6.1 dB;
    the two the arm called occupied and hung up on step +0.1 and +0.7, because
    both were up in the first bin the receiver could hear after unkey and flat to
    the end. Nothing else in that session's answer band separates them: the two
    flags are the top two of a level distribution spanning 8.6 dB with nothing in
    it but the channel, and windows recorded after the link was down with nobody
    on the air reach within 1.1 dB of the level threshold.

    AGAINST THE WINDOW'S OWN LEADING BINS AND NOT A SESSION MEDIAN. Subtracting
    the median of the session's windows per bin is the right way to draw this
    offline -- it takes out the receiver's mute recovery -- but a median a live
    station can compute is drawn from the handful of windows it has collected so
    far, which on a link answering every cycle are the answers themselves. On the
    same 28 windows that reference costs 2 dB of the separation (the five answers
    fall to +1.8 to +2.8 and the flags to -1.3 and -0.9) for a recovery shape
    that is 0.9 dB across the band. So the lead is this window's own, and it
    starts at `t0` -- the caller opens the band past the mute release for its own
    reasons and the first bin scored is the first bin it asked about.

    THE SPLIT IS SEARCHED AND NOT ASSUMED, over `d` up to `ONSET_STEP_MAX_S`,
    because a turnaround is the far end's hardware and moves by tens of
    milliseconds between stations. Medians on both sides: a step is what is being
    asked for, and one bin of key-up transient must not read as one.

    Returns None where the window is too short to have a lead and a tail.
    """
    a = np.asarray(seg, dtype=np.float64)
    a = a[int(round(t0 * FS)):None if t1 is None else int(round(t1 * FS))]
    w = int(round(ONSET_BIN_S * FS))
    n = a.size // w
    if n < ONSET_LEAD_BINS + ONSET_TAIL_BINS + 1:
        return None
    lo, hi = OCCUPANCY_BAND_HZ
    F = np.fft.rfft(a)
    f = np.fft.rfftfreq(a.size, 1 / FS)
    band = (f >= lo) & (f <= hi)
    power = np.abs(F[band]) ** 2
    F[~band] = 0
    x = np.fft.irfft(F, n=a.size)
    e = 10 * np.log10((x[:n * w].reshape(n, w) ** 2).mean(1) + 1e-20)
    last = min(n - ONSET_TAIL_BINS,
               int((ONSET_STEP_MAX_S - t0) / ONSET_BIN_S) + 1)
    if last <= ONSET_LEAD_BINS:
        return None
    step = max(float(np.median(e[k:]) - np.median(e[:k]))
               for k in range(ONSET_LEAD_BINS, last))
    k = ((f[band] - lo) // ONSET_BAND_HZ).astype(int)
    bands = np.bincount(k, weights=power,
                        minlength=int((hi - lo) // ONSET_BAND_HZ) + 1)
    return SlotOnset(step, bands)


class _P1Burst(NamedTuple):
    """What the 1400/1600 burst test measured -- not who sent it, and not that it
    is PACTOR-1. Every field appears in the line it produces."""
    ms: float
    concurrency: float          # active slots with BOTH tones up
    weaker_tone_db: float       # the quieter of the two tones, over the guard bands


def _p1_reply(seg: np.ndarray) -> _P1Burst | None:
    """A burst keying BOTH PACTOR-1 tones, of a length and duty a reply could have.

    Shape only. See `P1_REPLY_MIN_MS` for what that does and does not establish.
    """
    if seg.size < SPS * 4:
        return None
    n = np.arange(SPS)
    wm = np.exp(-2j * np.pi * 1400 * n / FS)
    ws = np.exp(-2j * np.pi * 1600 * n / FS)
    m, sp = [], []
    for i in range(0, seg.size - SPS, SPS // 2):
        w = seg[i:i + SPS]
        m.append(abs(w @ wm)); sp.append(abs(w @ ws))
    m, sp = np.array(m), np.array(sp)
    if m.size == 0 or m.max() <= 0 or sp.max() <= 0:
        return None
    # Absolute reference, not a relative one: a threshold like 0.4*max is met by
    # any fluctuation, so on noise it always finds a "burst". Compare the tone
    # pair against guard bands either side that should carry only noise.
    # Measure on the burst, not the whole window: a 350 ms reply diluted across a
    # 1.5 s slot barely clears the floor even when it is plainly audible.
    tot0 = m + sp
    span = max(1, int(P1_REPLY_MIN_MS / 1000 * FS / (SPS // 2)))
    k = int(np.argmax(np.convolve(tot0, np.ones(span), "valid"))) * (SPS // 2)
    burst = seg[k:k + int(0.40 * FS)]
    if burst.size < SPS * 2:
        burst = seg
    F = np.abs(np.fft.rfft(burst * _window(len(burst))))
    f = np.fft.rfftfreq(len(burst), 1 / FS)
    guard = F[((f > 900) & (f < 1200)) | ((f > 1800) & (f < 2100))].mean() + 1e-12
    mark = F[(f > 1360) & (f < 1440)].mean() / guard
    space = F[(f > 1560) & (f < 1640)].mean() / guard
    if 0.5 * (mark + space) < P1_REPLY_MIN_EXCESS:
        return None
    if min(mark, space) < P1_REPLY_MIN_TONE_EXCESS:
        return None
    tot = m + sp
    hot = tot > 0.4 * tot.max()
    if not hot.any():
        return None
    ms = float(hot.sum() * (SPS / 2) / FS * 1000)
    if ms < P1_REPLY_MIN_MS:
        return None
    # A reply is a BURST inside the listening window; a 500 Hz-class mode
    # (VARA-500, ARDOP-500) transmits continuously and covers both 1400 and 1600
    # from centre 1500, so it clears the guard-band test -- one such station
    # tripped this 18 times in 20 s. Occupied bandwidth cannot separate them
    # (on a weak signal that measure returns the passband width), but duty cycle
    # can: a reply occupies a fraction of the window, a data stream fills it.
    if hot.mean() > P1_REPLY_MAX_DUTY:
        return None
    # KNOWN LIMITATION, and the reason the line reports a shape rather than a
    # station: a busy 500 Hz-class ARQ signal (VARA-500 / ARDOP-500) still passes
    # everything here. Width does not separate them -- measured over a 0.4 s burst
    # that station spans 95-272 Hz at -6 dB against 205-239 Hz for a real PACTOR-1
    # reply, because it is OFDM and a short window catches only a few subcarriers.
    # Duty cycle removes its continuous stretches but not its ARQ turnarounds,
    # which are genuinely bursty, and the both-tones test removes most but not all
    # of what is left: 13 windows across the corpus's VARA, PACTOR-2 and FT8
    # fixtures still arrive here. Separating them needs a DECODE, not another
    # spectral statistic, and at the SNR a reply arrives with the bits are not
    # recoverable. So this can report a burst that belongs to somebody else, and
    # `decode_events` writes that into the line rather than into this comment.
    on_m, on_s = m > 0.35 * m.max(), sp > 0.35 * sp.max()
    act = (on_m | on_s).sum()
    conc = float((on_m & on_s).sum() / act) if act else 0.0
    if not P1_REPLY_CONC[0] <= conc <= P1_REPLY_CONC[1]:
        return None
    return _P1Burst(ms, conc, 20 * float(np.log10(min(mark, space))))


CS_EXTENT_FLOOR = 0.10
"""How far under a codeword read's own peak its quietest sample may run.

The gate on the blind sweep, and it is the BURST'S EXTENT rather than its
strength or its evenness. Both of those were tried and neither can carry it: a
codeword answering a data packet runs 0.17 of the packet beside it, so a level
referred to anything but the read itself throws real answers away, and the edge
alignments the sweep already accepts are uneven down to 0.003 of a uniform read.
What separates a reading from a straddle is where the samples lie, and the
envelope is what says where the burst is.

Measured on `_tone_envelope`, quietest sample of the read against the read's own
peak: 0.570 over the 39 codewords of `PIII_Complete_1`, 0.418 over our own render
from a clean channel down to 3 dB. A read hanging four symbols off the burst
scores 0.006 and the ten-symbol one that caused this 0.000. So 0.10 sits a factor
of 4.2 under the weakest read it admits and 17 over the tightest straddle it
turns down."""

CS_EXTENT_SLACK = 1
"""Symbols at each end of the read allowed to sit on the burst's shoulders.

One at the head because a shaped burst rises over about a symbol -- our own
render's envelope reaches a tenth of its peak 1.09 symbols in and the phase
reference is read at 1.375 -- and one at the tail because the reference peer keys
TWENTY-TWO symbols: a run-out follows the twentieth bit at full amplitude and no
phase step (`test_p3_upgrade` measures it), so a correct read's last sample sits a
symbol inside the keying and an extent taken on the falling edge may fall short of
it by that much."""


def _tone_envelope(Z: dict, delay: int, lo: int, hi: int) -> np.ndarray:
    """One-symbol running mean of the header tones' magnitude over [lo, hi).

    The instrument that says where a burst starts and stops, at the ~300 Hz the
    matched filter leaves. The running mean and not the magnitude itself: a DBPSK
    envelope nulls at every phase reversal, down to 0.19 of the burst's own peak
    mid-burst, so the raw trace has a hole every other symbol and no edges at all.
    """
    n = hi - lo
    e = np.zeros(n + SPS)
    for cn in HDR_TONES:
        seg = np.abs(Z[cn][lo + delay - SPS // 2:lo + delay - SPS // 2 + n + SPS])
        e[:seg.size] += seg
    c = np.concatenate([[0.0], np.cumsum(e)])
    return (c[SPS:SPS + n] - c[:n]) / SPS


def _inside_burst(env: np.ndarray, base: int, st: int) -> bool:
    """Whether all 21 symbols read at `st` lie inside the burst `env` describes."""
    a, b = st - base, st - base + 20 * SPS
    if a < 0 or b >= env.size:
        return False
    body = env[a + CS_EXTENT_SLACK * SPS:b + 1 - CS_EXTENT_SLACK * SPS]
    return bool(body.min() > CS_EXTENT_FLOOR * env[a:b + 1].max())


def _cs_coherence(Z: dict, delay: int, st: int, ci: int) -> float:
    """How well the differential symbols read at `st` match codeword `ci`, 0..1.

    `rx.cs_bits`' own arithmetic with the magnitudes kept instead of thrown away
    at the slicer, normalised by the energy it read -- the same ratio
    `p3acquire._acquire` ranks its frequency hypotheses by. A hard decision cannot
    separate two alignments that both decode; this can, and its peak is the
    burst's phase reference.
    """
    signs = 1.0 - 2.0 * rx._CS_TABLE[ci]
    idx = st + np.arange(spec.CS_BITS_PER_TONE + 1) * SPS + delay
    num = den = 0.0
    for cn in HDR_TONES:
        y = Z[cn][idx]
        d = y[1:] * np.conj(y[:-1])
        num += float((d.real * signs).sum())
        den += float(np.abs(d).sum())
    return num / den if den else 0.0


def _best_cs(Z: dict, delay: int, pos: int,
             span: range | None = None) -> tuple[int, int, int]:
    """Nearest control signal over the burst around `pos`, at its best alignment.

    The pilot metric peaks anywhere within the 21-symbol CS, so the default search
    goes back a full burst length. `span` overrides it with absolute sample
    positions, which is how a tracking receiver aims at the alignment it locked on
    last cycle instead of re-deriving it. Returns (index, least Hamming distance,
    the alignment that scored it).

    THE ALIGNMENT IS THE COHERENT PEAK, not the first alignment that decoded.
    Twenty bits, no CRC, six words at mutual distance twelve: a wide band of
    alignments reads the same word at zero errors, fifteen of them -- 8.75 ms --
    on `fixtures/ws8eoc-0910/first-rms.wav`. Keeping the first strict minimum
    returned the LEADING EDGE of that plateau, 300 samples (6.25 ms) in front of
    the head the acquisition, the crop manifest and the coherent argmax all agree
    on, and every consumer of the instant -- a peer's raster phase, a changeover's
    body offset, an answer's turnaround position -- carried that bias. So the
    plateau the first minimum opens is walked to its end and `_cs_coherence`
    chooses within it: on the five morning crops that lands 0, 0, -30, -12 and
    +30 samples from the recorded onset, against the edge's -300 to -150.

    HALF THE DEFAULT SWEEP READS OFF THE END OF THE BURST, and a twenty-bit word
    has no CRC to notice. The alphabet is six words at mutual distance twelve, and
    a straddle lands on one: a bare CYCLE-TOG rendered into quiet read BREAK-IN at
    zero bit errors 100 ms in front of its own first sample, ten of its twenty
    decision variables in the pad, which armed the 0.6 s debounce and hid the
    burst that was really there -- on the QSO bench an ISS handing the link back
    to a peer that had only asked for a longer cycle, and 958 bytes of a kilobyte
    never sent.

    So an alignment is admissible only if its 21 symbols lie inside the burst, and
    the burst is what the two tones' envelope says it is. The gate is on the
    DEFAULT sweep alone. A caller passing `span` is aiming at a band it chose --
    a turnaround slot, or a few symbols either side of last cycle's lock -- which
    may hold no burst or several, and there is no one burst for its alignments to
    be inside of; `SyncedRx._cs_at_lock` carries its own radius-1 rule instead.
    """
    env = base = None
    if span is None:
        span = range(max(0, pos - 22 * SPS), pos + 2 * SPS, SPS // 16)
        base = span.start
        env = _tone_envelope(Z, delay, base, span[-1] + 21 * SPS)
    best = (0, 99, pos)
    plateau: list[int] = []
    for st in span:
        if st < 0:
            continue
        bits = rx.cs_bits(Z, st, delay)
        if bits is None:
            break
        ci, be = rx.nearest_control_signal(bits)
        extends = bool(plateau) and (ci, be) == best[:2] \
            and st - plateau[-1] == span.step
        if not (be < best[1] or extends):
            continue
        if env is not None and not _inside_burst(env, base, st):
            continue
        if extends:
            plateau.append(st)
        else:
            best, plateau = (ci, be, st), [st]
    if len(plateau) > 1:
        at = max(plateau, key=lambda st: _cs_coherence(Z, delay, st, best[0]))
        best = (best[0], best[1], at)
    return best


def _status_str(b: int) -> str:
    parts = [f"seq={b & spec.STATUS_SEQ}", f"type={(b >> 2) & 0x7}"]
    if b & spec.STATUS_CHANGEOVER:
        parts.append("CHANGEOVER")
    if b & spec.STATUS_QRT:
        parts.append("QRT")
    return " ".join(parts)


def _cs_event(audio: np.ndarray, ci: int, be: int, at: int, t: float,
              tag: str, *, details: bool = True) -> "Event":
    """What a decoded PACTOR-3 codeword is -- which for CS3 is a whole PACKET.

    CS3 is never bare. It is the first twenty symbols of the new sender's own
    changeover packet, and the three bytes behind it are the first three of what
    that station has to say: in `PIII_Complete_1` they are the carriage return
    and the `PT` that the greeting behind them reads `C-II DSP/QUICC System -
    Maildrop QRV` from. A receiver that stops at the codeword yields the channel
    correctly and loses them, which is the defect the PACTOR-1 seam had until
    `p1rx` read the rest of the burst.

    So a CS3 whose frame validates arrives as a `packet` event carrying
    `breakin`, which is the shape `arq.PactorArq.on_rx_packet` already reads --
    it yields the link and reads the packet as one event, no cycle in between.
    A CS3 whose frame does NOT validate still arrives as a `cs`, and the link
    still yields on it; a codeword is evidence enough to stop transmitting.

    `details` is what a caller with a key to make says when the body will not
    fit in front of it. The 0.84 s frame behind the head is demodulated on both
    carrier arrangements, 7.6 ms median and 14.4 at the ninetieth percentile
    over the 55 linked cycles of `captures/onair-0913-2152`, against the 7.6 ms
    a cycle holds between its last read and the admission check -- so the read
    that yields the channel was also the read that spent the slot. Declining it
    costs the three bytes a cycle: the codeword still yields the link, and the
    body is re-read behind the key out of the same buffer.
    """
    if details and ci == placement.BREAKIN_CS:
        # Over a SLICE, because the caller's buffer is not bounded and
        # `rx._baseband` is an 1860-tap convolution over whatever it is handed --
        # the reason `_sampled_baseband` exists. The frame ends one packet after
        # the head, and eight symbols either side covers the filter's own edges.
        lo = max(0, at - 8 * SPS)
        hi = min(len(audio), at + round(placement.PACKET_S * FS) + 8 * SPS)
        field, ok, swapped = p3rx.decode_changeover_details(audio[lo:hi], at - lo)
        if ok:
            info, st = field[:-2], field[-3]
            data = spec.field_payload(bytes(info[:-1]))
            return Event(t, "packet",
                         f"P3 CHANGEOVER {len(data)}B status=0x{st:02x} "
                         f"({_status_str(st)}) {data!r}  [CRC-VALID]",
                         protocol="PACTOR-3", breakin=True, cs=ci, start=at,
                         packet=(placement.CHANGEOVER.speed_level, st, data, True),
                         cycle_long=False, carrier_swapped=swapped,
                         information=bytes(info))
    return Event(t, "cs", f"{spec.CS_NAMES[ci]}  ({be} bit errors{tag})",
                 protocol="PACTOR-3", cs=ci, start=at)


def _packet_event(p: "p3rx.P3Packet", *, t: float | None = None,
                  tag: str = "") -> "Event":
    """The Event for a decoded PACTOR-3 data field.

    Field layout is the spec's: data field, then the status byte, then the CRC
    (M.1798 S1), and `p3rx.packet_of` is what cuts it -- the status carries the
    mod-4 packet counter, so dropping it made every frame read as sequence 0, and
    the fill rule is the one `p3rx` and the changeover path already apply.

    THE SPEED LEVEL IS THE PEER'S, not this receiver's home geometry. It used to
    be `placement.HEADER.speed_level` for every in-session packet whatever the
    peer sent, so `arq._gear_cs` computed its next-level request off a 2 that was
    a default rather than a reading, and a peer at level 5 was asked to move to
    level 3.
    """
    return Event(p.start / FS if t is None else t, "packet",
                 f"SL{p.sl} DATA {len(p.payload)}B status=0x{p.status:02x} "
                 f"({_status_str(p.status)}) {p.payload!r}  "
                 f"[CRC-VALID{tag}]",
                 protocol="PACTOR-3",
                 packet=(p.sl, p.status, p.payload, True), start=p.start,
                 cycle_long=p.long_cycle, carrier_swapped=p.carrier_swapped,
                 information=p.info + bytes([p.status]) if p.info else None)


# PACTOR-1 data frames are gated inside `p1rx.decode_p1_packets` now -- the header
# test that used to live here, plus an eye test the header alone could not replace.
# The numbers that justified this one are in `p1rx.EYE_MIN` and are far worse than
# they looked from here: measured over 200 corpus captures the header test took 412
# accepts to 2, but a 2.4 hour sample was not enough audio to see that the residual
# scales with the trial count, and on the full 12.25 hour corpus the CRC alone
# accepts 6341 frames.
#
# The counter's low bit agrees with the header on every W4DNA frame (0xAA <-> odd
# count), and adding that as a third test took those 2 to 0. It was deliberately not
# applied, on the grounds that a peer whose convention differed would be rejected
# for every cycle of a whole session rather than for one -- and the corpus has since
# produced that peer. The JN36lf 14110 kHz station sends count 3 under header 0x55
# and count 0 under header 0xAA, the opposite parity to W4DNA's, and a parity test
# would have thrown away both of its frames. See docs/protocols/pactor/pactor1-data-packets.md.


def _p1_packet_event(p, start: int | None, *, t: float | None = None,
                     tag: str = "") -> "Event":
    """The Event for a PACTOR-1 data frame whose first bit is at sample `start`.

    `start` is where the FRAME is, not where the search that found it began --
    `_packet_event`'s rule, and PACTOR-1 was the one packet event that did not
    follow it. `decode_events` timestamped these at the hop it was sweeping, up
    to two seconds before the burst, so the time depended on where the caller's
    hop grid happened to fall rather than on the audio. That is invisible to a
    caller reading a whole file at one origin and not to `live.RollingRx`, whose
    grid moves with every window: DL6MAA's 200 Bd announcement, a single burst at
    3.4000 s in `rf-corpus/PIII_Complete_1`, came out as two packets at 2.5 and
    3.0 s -- one frame, byte for byte the same, past a dedupe keyed on the time.

    `t` overrides it for a caller that has its own clock and a window that starts
    at zero on it.
    """
    # The counter and the type are two of the status byte's five fields, and a
    # line that renders only those prints a peer asking for the channel as
    # `0x41 (cnt=1 type=0)` -- which reads as an ordinary packet.
    flags = "".join(f" {name}" for name, on in
                    (("BK", p.changeover_request), ("QRT", p.qrt)) if on)
    return Event(start / FS if t is None else t, "packet",
                 f"P1 {'BREAK-IN' if p.breakin else 'DATA'} {p.baud}Bd  "
                 f"status=0x{p.status:02x} "
                 f"(cnt={p.packet_count} type={p.data_type}{flags})  "
                 f"{len(p.payload)}B {p.payload[:24]!r}  [CRC-VALID{tag}]",
                 protocol="PACTOR-1", packet=(0, p.status, p.payload, True),
                 breakin=p.breakin, start=start)


def decode_expected_p1_packet(audio: np.ndarray,
                              memory: "p1rx.PacketMemory | None" = None
                              ) -> "Event | None":
    """The PACTOR-1 data frame due in this window, gated on the CRC and nothing else.

    The PACTOR-1 half of `decode_expected_packet`, and it exists because the blind
    path cannot serve a station that is holding a link. `decode_events` runs the P1
    data scan only once it has decoded a CONNECT in the SAME buffer -- reasonable
    for a monitor sweeping a band, where the scan is far too costly to run on every
    burst -- but in a QSO the connect is OURS. Nothing the peer sends can ever set
    that flag, so shrike transmitted a correct PACTOR-1 data phase and could not
    read one: a Winlink RMS taking the channel and sending its banner went
    unanswered, cycle after cycle.

    A connected station needs no such gate. It knows a frame is due because it just
    acknowledged the last one, and `p1rx`'s scan is already CRC-gated -- so the
    question is only whether the evidence is strong enough to act on, which is what
    `p1rx`'s header and eye gates answer.

    Unlike the PACTOR-3 case this is CHEAP -- measured on the same 1.25 s window,
    5 ms against `decode_expected_packet`'s 68 -- because `p1rx` already hoists
    everything that does not depend on the alignment: the MARK/SPACE magnitudes are
    two sliding single-bin DFTs over the whole buffer, computed once, after which
    each of the thousands of candidate starts is an array lookup rather than a
    correlation. The same lesson `SyncedRx` and `_sampled_baseband` were built from,
    applied a layer down, which is also why there is no PACTOR-1 counterpart to
    `SyncedRx`: at 5 ms a cycle there is nothing left worth tracking.

    Both frames are looked for. A station holding the channel is expecting data, and
    the one thing that can interrupt it is the peer's CHANGEOVER packet -- CS3 as
    its head and 840 ms of the new sender's first packet behind it. Nothing else in
    the receiver can deliver one: the control-signal detector rejects it on length
    and its bit-grid positioner rejects it on the tail, both by construction. Only
    this pass sees it, so leaving it out is the same as not implementing break-in.

    IT IS NOT A NET, and what it catches depends on where the window opens rather
    than on the frame. `p1rx`'s break-in scan is driven by the envelope's RISING
    EDGES, so a changeover whose carrier is already up when the buffer starts has
    no edge to be found at. Measured over the 23 real changeover packets in the
    corpus, varying nothing but the lead in front of the burst: 20 ms of lead reads
    21 of them, 50 ms reads 20, 200 ms reads 23. The session's hold window opens at
    the sample our own transmission ended on (`onair._LiveInput.flush_to`) and the
    peer keys its changeover 71.3-71.9 ms after that -- measured off WS8EOC's own
    emissions on both arms of 2026-08-30 -- so what this pass runs on in a QSO is
    the 50 ms column, not the 200 ms one.

    `memory` is the session's `p1rx.PacketMemory`, one per link, held by the
    caller for `p2rx.decode_expected_burst`'s reason: a cycle cannot remember
    anything, and the copies a station failed to read singly are exactly what
    ARQ repetition puts back on the air. A single-shot decode clears it; a miss
    feeds it this window's softs at both bauds -- the peer's geometry is not
    readable from a copy that did not decode, and each baud accumulates in its
    own lane -- and a combined decode comes back through the same event shape,
    tagged for what it is.
    """
    for p in p1rx.decode_p1_packets(audio):
        if memory is not None:
            memory.clear()
        return _p1_packet_event(p, p.start, t=0.0, tag=", expected")
    for p in p1rx.decode_p1_packets(audio, breakin=True):
        if memory is not None:
            memory.clear()
        return _p1_packet_event(p, p.start, t=0.0, tag=", expected")
    if memory is not None:
        for baud in (100, 200):
            softs = p1rx.packet_softs(audio, baud=baud)
            if softs is None:
                continue
            p = memory.add(softs, baud)
            if p is not None:
                return _p1_packet_event(p, None, t=0.0, tag=", combined")
    return None


def decode_expected_packet(audio: np.ndarray,
                           memory: "p3rx.FieldMemory | None" = None
                           ) -> "Event | None":
    """Find a data frame in a window where one is DUE, at whatever level it is.

    ANCHORED ON THE PACKET'S OWN HEADER BLOCK, which is what `p3rx.decode_headed`
    is, and this used to be a CRC timing scan on ONE geometry instead:
    `rx.decode_case1_header`, i.e. speed level 2, short cycle, home tone order.
    Measured through `onair._SessionRx.deep_scan` on a CONNECTED PACTOR-3 host,
    one rendered packet per row, that reader delivered SL2 short and nothing
    else -- SL1, SL3-6, every long cycle and a swapped SL2 all returned None,
    while `p3rx.decode_p3_packets` read all eleven off the same audio. Level 3
    is `arq.ArqConfig.entry_sl` and the level the reference session runs, so the
    half of a mail exchange this station receives was closed for the levels it
    was most likely to be offered.

    The block carries the three things a scan cannot recover -- the speed level
    (modulo four; the channel occupancy separates 1 from 5 and 2 from 6), the
    cycle length and the carrier swap -- so it costs the CRC one position per
    admissible level rather than a window of them, and case 0 arrives on the same
    path as every other level rather than only through the monitor's sweep.

    What it costs: measured on a 1.30 s window, 74-84 ms against the pinned
    scan's 48, of which 37 ms is the eighteen-channel baseband and 36 the
    quarter-symbol anchor sweep. It is not what the rolling receiver runs every
    slide -- it is for a caller that knows a frame is due in the window it is
    holding, and `SyncedRx` is what turns one of these into a lock, after which
    the read is nine alignments on one geometry.

    `memory` is the session's `p3rx.FieldMemory`, held across cycles for
    `decode_expected_p1_packet`'s reason: ARQ repeats arrive one per cycle, and
    a memory built inside one scan can never see two of them.
    """
    try:
        scan, _ = p3rx.decode_headed(audio, 0, len(audio), memory=memory)
    except Exception:
        return None
    p = max(scan.packets, key=lambda p: p.start, default=None)
    return None if p is None else _packet_event(p, tag=", expected")


def _case1_persists(seg: np.ndarray, at: int, field: bytes) -> bool:
    """The SL>=2 header at `at` is still there an eighth of a symbol either side.

    This is the frame a live link reads, so what it costs one was measured before
    it went in front of one: a rendered header validates across eight of the
    seventeen eighth-symbol offsets around row 0 from a clean channel down to
    0 dB SNR, and across six at -3 dB. The rule asks for two."""
    def decode(pos: int) -> bytes | None:
        f, ok, _ = rx.decode_case1_header(seg, fs=FS, search=range(pos, pos + 1))
        return f if ok else None

    return p3rx.confirmed(decode, at, field)


def _confirmed_case0(seg: np.ndarray) -> bytes | None:
    """The first case-0 field in `seg` that survives moving the clock, or None.

    There is no comb test here and there is not going to be one. PACTOR-3 rides a
    120 Hz grid and noise does not, so a comb detector looks like the cheap
    discriminator a CRC accept cannot fake -- and measured, it is not one. On the
    three TI0BCR dwells off 14.113 MHz it reported 13 to 15 tones of 17 with the
    spectral autocorrelation on 120 Hz, and those captures carry no PACTOR at all:
    in-band power across the reported burst stands at +0.15, +0.04 and -0.32 dB
    against the 1.96 s of lead-in recorded before it, the comb reads as strongly in
    that lead-in and in the tail as in the burst, its lines sit at 1133, 1283,
    1373, 1523, 1613 and 1763 Hz -- spacings of 150 and 90 Hz alternating, none of
    them on the 480 + 120k channel grid -- and an independent decoder handed all
    three at full input amplitude reports nothing. On the DL6MAA recording, which
    that decoder reads end to end, the same detector reported 4 of 17. It was
    measuring how many birdies a receiver's site has, and it ranked real PACTOR-3
    below noise.

    What separates them is `p3rx.confirmed`, which asks the accept rather than the
    spectrum.
    """
    pulse = rx._pulse(SPS)
    Z = {cn: rx._baseband(seg, cn, FS, pulse) for cn in rx.HEADER_TONES}

    def at(pos: int) -> bytes | None:
        for _, f in rx.case0_accepts(seg, fs=FS, search=range(pos, pos + 1), Z=Z):
            return f
        return None

    for start, field in rx.case0_accepts(seg, fs=FS, Z=Z):
        if p3rx.confirmed(at, start, field):
            return field
    return None


def _decode_at(audio: np.ndarray, Z: dict, delay: int, pos: int) -> "Event | None":
    """The three things that ride tones 5 and 12, tried in order of specificity.

    A header and a control signal never occur at once, so a CS is only considered
    where neither header validated. Each slice is BOUNDED: the header decoders
    CRC-scan every symbol position they are given, so handing one the rest of the
    capture makes each trigger cost O(file) and the whole scan quadratic.

    Both headers are confirmed before they are reported. The bound above keeps a
    trigger's case-0 scan to 328 positions, which at one accept in 36,000 is not
    small enough on its own: it manufactured four PACTOR-3 headers over the
    corpus, on frequencies where no PACTOR station transmits.
    """
    seg = audio[max(0, pos - int(0.15 * FS)):pos + int(1.0 * FS)]
    try:
        field = _confirmed_case0(seg)
    except Exception:
        field = None
    if field is not None:
        st = field[5]
        # The case-0 header carries no user data of its own, so the deliverable
        # payload is empty -- the frame still drives the sequence/turnaround
        # through its status byte.
        #
        # SPEED LEVEL 1, and it reported 0. The decoder's case numbers are one less
        # than the protocol's levels throughout (`placement.SPEED_PATHS`), and 0 is
        # the number the PACTOR-1 seam uses for "this protocol has no speed level"
        # -- so a PACTOR-3 level 1 frame arrived at the link layer wearing the one
        # value that means PACTOR-1. The rule that a lone level 1 accept must not
        # move a link is written against the level, and could not see this one.
        return Event(pos / FS, "packet",
                     f"HEADER field={field.hex()}  status=0x{st:02x} "
                     f"({_status_str(st)})  [CRC-VALID, confirmed]",
                     protocol="PACTOR-3", packet=(1, st, b"", True))

    # The SL>=2 header, which is the one shrike's own transmitter sends. Only
    # case 0 was tried here, so every data packet shrike put on the air was
    # undecodable by shrike -- the link came up and then carried nothing, because
    # transmit and receive did not meet. It is a different frame geometry, not a
    # fallback: a case-0 packet does not validate as case 1 (`tests/shrike/test_rx.py` checks).
    lo = max(0, pos - int(0.15 * FS))
    wide = audio[lo:pos + int(1.6 * FS)]
    try:
        f1, ok1, at = rx.decode_case1_header(wide, fs=FS)
        ok1 = ok1 and _case1_persists(wide, at, f1)
    except Exception:
        f1, ok1, at = b"", False, -1
    if ok1:
        # Keep the trigger's timestamp -- it is what the monitor and the corpus
        # regression report -- but carry the alignment the demodulator actually
        # locked on, so a tracking receiver can aim the next cycle at it.
        return _packet_event(p3rx.packet_of(f1, placement.HEADER, lo + at),
                             t=pos / FS)

    # No "header present but the CRC failed" line: with the trigger this weak, a
    # failed CRC is the ordinary case on any signal, and asserting a header is
    # there is exactly the unsupported claim the trigger cannot make.
    # The pilot correlation peaks somewhere inside a CS burst rather than on its
    # phase-reference symbol, so scan back a burst length for the alignment of
    # least codeword distance.
    ci, be, at = _best_cs(Z, delay, pos)
    if be <= CS_MAX_ERRORS:
        return _cs_event(audio, ci, be, at, pos / FS, "")
    return None


def p3_packet_events(audio: np.ndarray, *, envelope: bool = True,
                     not_before: float | None = None) -> Iterator[Event]:
    """Ordinary P3 events, optionally bounded by a live epoch's header start.

    Packet.start is data-row zero, not the opening reference/header symbol.
    The transition floor is relative to this audio, and deliberately does not
    require matched-filter lookback to begin after the enabling control.
    """
    for p in p3rx.decode_p3_packets(audio, envelope=envelope).packets:
        if (not_before is not None
                and p.start - p3frame.DATA_OFFSET * SPS < round(not_before * FS)):
            continue
        yield Event(p.start / FS, "packet", p.report(), protocol="PACTOR-3",
                    packet=(p.sl, p.status, p.payload, True), start=p.start,
                    cycle_long=p.long_cycle, carrier_swapped=p.carrier_swapped)


def decode_events(audio: np.ndarray, hop_s: float = 0.5,
                  cs_max_errors: int = CS_MAX_ERRORS,
                  acquiring: bool = False,
                  p3_envelope: bool = True,
                  p3_packets: bool = True,
                  p3_packets_after: float | None = None) -> Iterator[Event]:
    """Decoded PACTOR events in `audio`.

    `acquiring` selects the connect-answer path: a sliding correlator over the
    window, restricted to CS1/CS4 and demanding an exact match, instead of
    reading twelve bits at one envelope-derived point. That is the split a
    working implementation makes -- a running link knows its grid and reads a
    point, an acquiring one does not and has to search. Measured on real gateway
    replies recorded with our transmitter off, the point read finds 10% of
    W6IDS's bursts and the search finds 36%.

    OFF by default, and it must stay off once connected: the search is 224 trials
    per burst and only the two-codeword alphabet and the zero-error requirement
    keep it from finding a word in noise. Pass it only while CONNECTING.

    `p3_envelope` is the one thing here whose cost does not scale with the audio,
    and a caller feeding a stream turns it off. It is the PACTOR-3 data pass's
    fallback for a burst with no header block (`p3rx.decode_p3_packets`), and its
    trial budget is spent per burst rather than per second -- so the cost steps
    the moment the buffer first holds a whole PACTOR-3 body, not gradually as it
    grows. MEASURED on `watch_pactor3_maryland.wav`, one call: 31.8 ms over a
    0.75 s window and 846.7 ms over 0.85 s, 1.86 s over 1.00 s and 7.54 s over
    4.00 s, with no event returned at any length. `live.RollingRx` is where that
    lands on a clock; a caller reading a file is not on one and keeps it.

    `p3_packets` controls only the ordinary PACTOR-3 packet enumeration. A live
    session still connecting in PACTOR-1 may defer that expensive pass without
    disabling P1 answers/grants, P3 controls/changeovers, or explicit acquisition.
    File readers and callers without that protocol context retain the default.
    `p3_packets_after` is the optional opening-header floor of that transition,
    relative to this audio; it does not constrain controls or other readers.
    """
    delay = (len(rx._pulse(SPS)) - 1) // 2
    Z = {cn: rx._baseband(audio, cn, FS, rx._pulse(SPS)) for cn in HDR_TONES}
    dA = _pattern_a()
    starts = np.arange(0, max(1, len(audio) - 9 * SPS), SPS // 4)
    scores = rx.pilot_corr(Z, starts, delay, dA)
    floor = np.median(scores) + 1e-9
    # KNOWN LIMITATION, measured 2026-07-25: a data packet decodes on a clean
    # channel and at NO signal-to-noise ratio below it, 40 dB included, for the
    # same reason the acknowledgement did not -- the ratio below reads 2.49-2.51
    # on a rendered packet against its 2.5 threshold. The control signal could be
    # rescued by looking at the one best position, because _best_cs there is 3.5 ms
    # and the codeword is its own gate. A header cannot: the pilot peak is a good
    # position estimate on a clean channel (a stable -24.8 symbols) and a random
    # one in noise (-111, +44, +39 symbols measured), so only the exhaustive CRC
    # scan finds the frame, and running that once per clip costs 1.1 s on real
    # off-air audio and 2.6 s on noise -- 450% to 1036% of a core at the session's
    # 0.25 s slide. Tried, measured, reverted. The fix is not a cheaper gate but a
    # scan the ARQ layer asks for once per cycle, in the slot where a frame is
    # actually due, instead of the receiver guessing every slide.
    hop = int(hop_s * FS)
    # Separate debounce per family: a PACTOR-1 connect leaks FSK energy into the
    # header tones and can raise a spurious P3 pilot, so a shared timer would let
    # the (weaker) false detect shadow the real connect. Keep them independent.
    last_p3 = last_fsk = last_p2 = last_cs = last_spare = last_callb = -1e9
    # Separate from `last_fsk` on purpose: that one debounces the PRESENCE reports
    # as well, and a report that cannot read the burst it saw must not be able to
    # hide one that can. See the PACTOR-1 data scan below.
    last_dec = -1e9
    # Set by a decoded connect, and it gates the PACTOR-1 data scan. Correct for
    # what this is -- a receiver sweeping a band it knows nothing about -- and
    # useless to a station holding a link, because in a QSO the connect is OURS
    # and nothing the peer sends will ever set it. That case has its own entry
    # point: `decode_expected_p1_packet`.
    in_session = False

    # The PACTOR-3 DATA field, which until the packet header block was found in
    # real traffic was the one thing in the protocol this receiver could see and
    # not read. It runs as its own pass rather than inside the hop loop because
    # its acquisition is the packet's own header and its gate is the tone set,
    # neither of which has anything to do with where a 0.5 s hop happens to fall
    # -- and because it must not be able to arm the debounce that the header and
    # control-signal decodes share.
    if p3_packets:
        yield from p3_packet_events(audio, envelope=p3_envelope,
                                   not_before=p3_packets_after)

    # PACTOR-2 acquisition, its own pass for the reason the PACTOR-3 one is: the
    # marker is where the frame is, and that has nothing to do with where a hop
    # falls. It is ALSO the precedence, and the reason for both is that PACTOR-1
    # is the narrower mode: a PACTOR-2 station measures 326-366 Hz on
    # `_occupied_bw`, inside `P1_MAX_BW_HZ`, so every burst of it satisfied
    # `_fsk_present` and left through the `fsk` branch before this pass existed.
    # The `detect` line said so of a real PACTOR-2 station exactly never.
    p2_frames = _p2_markers(audio)
    for fr in p2_frames:
        yield Event(fr.t, "detect",
                    f"PACTOR-2 frame marker, {fr.path.name}, correlation "
                    f"{fr.score:.2f} -- arms at {p2rx.MARKER_ARM} of a "
                    f"noiseless 1.00, a published codeword on both carriers, so "
                    f"this one DOES name the protocol; the field behind it is "
                    f"read by p2rx, not here",
                    protocol="PACTOR-2")

    for hi in range(0, len(audio), hop):
        seg0, seg1 = hi, min(hi + hop, len(audio))
        sel = (starts >= seg0) & (starts < seg1)
        if sel.any():
            k = np.argmax(scores[sel])
            local, pos = scores[sel][k], int(starts[sel][k])
            # The pilot correlation is a TRIGGER, not evidence, and it is not
            # reported as any. Measured against its own clip-relative floor it does
            # not separate PACTOR-3 from anything: real off-air P3 scores 2.50 and
            # plain FT8 scores 2.79, so "pilot detected (corr 2.7x noise)" was a
            # claim the number cannot support. It survives only to say where the
            # CRC-gated header decode -- which is discriminative -- is worth trying.
            # The ratio against the clip's median is not enough on its own. In a
            # mostly-quiet clip the median collapses toward zero and every
            # position clears 2.5x -- including one in the silence, which then
            # armed the 0.6 s debounce and suppressed the real burst arriving
            # after it. Requiring a share of the clip's PEAK as well rejects a
            # position that is only "above" a floor that is not there.
            peak = float(scores.max()) + 1e-12
            # A caller EXPECTING an answer gets a control-signal check that does
            # not wait on the pilot ratio, because that ratio is exactly what
            # fails there: measured on a rendered ACK in noise it sits at
            # 2.49-2.51 against a 2.5 threshold -- standing on the signal, so
            # whether it fires is a coin toss -- while _best_cs at that same
            # position names the right codeword every time.
            #
            # It looks in ONE place -- the clip's best-scoring position -- because
            # that is what expecting an answer means. Allowed to range over every
            # hop it found a radius-1 coincidence in the noise ahead of the real
            # burst and reported the wrong control signal first, which is worse
            # than missing one. The codeword is the evidence: 20 bits at
            # distance 12, so a hit at the one candidate position is a decode.
            answered = False
            if cs_max_errors > CS_MAX_ERRORS and local >= 0.9 * peak \
                    and pos / FS - last_cs > 0.6:
                ci, be, at = _best_cs(Z, delay, pos)
                if be <= cs_max_errors:
                    last_cs = pos / FS
                    answered = True
                    yield _cs_event(audio, ci, be, at, pos / FS, ", expected")
            # A header and a control signal ride the same two tones but never at
            # once, so having taken one there is nothing else to look for here.
            if not answered and local > 2.5 * floor and local >= 0.3 * peak \
                    and pos / FS - last_p3 > 0.6:
                ev = _decode_at(audio, Z, delay, pos)
                # Arm the debounce only on a DECODE. Arming it on the attempt let
                # a failure suppress a success: a burst framed by quiet triggers
                # once on its leading edge, where the alignment is wrong, and
                # again 0.15 s later where it is right -- and the second was
                # inside the first's 0.6 s shadow, so a control signal that
                # decodes cleanly on its own went missing whenever it arrived
                # surrounded by silence, which on the air is always.
                if ev is not None:
                    last_p3 = pos / FS
                    yield ev
        seg = audio[seg0:min(len(audio), seg0 + 2 * FS)]
        sp = _Spectrum(seg)
        for b_t0, b_dur in _p1_cs_bursts(seg):
            cs = p1rx.decode_control_signal(seg, b_t0, b_dur)
            # ZERO errors or nothing. The four words sit at mutual distance 8, so a
            # burst that is one of them lands exactly on it; 1-4 errors means it is
            # something else, and every acknowledgement shrike ever invented came
            # from accepting those.
            #
            # A word PACTOR-1 assigns no meaning is REPORTED AS ITSELF. It
            # reached here as a six-error CS1 and was discarded; it is now named
            # in `spare`, and it is still not a `cs` -- what a consumer does
            # with a word by name is its own affair, and only the 0x59A grant
            # has a reader (`ptc.PtcHost`). The acquire search below gets the
            # burst it has always got, and the debounce is this branch's own:
            # sharing `last_cs` would let a word that means nothing suppress the
            # acknowledgement arriving behind it.
            spare = cs is not None and cs[1] == 0 and cs.unassigned
            if acquiring and (cs is None or cs[1] != 0 or spare):
                got = p1rx.acquire_control_signal(seg, b_t0, b_dur)
                if got is not None and (hi + b_t0 * FS) / FS - last_cs > 0.3:
                    last_cs = (hi + b_t0 * FS) / FS
                    yield Event((hi + b_t0 * FS) / FS, "cs",
                                f"{spec.P1_CS_NAMES[got[0]]}  (acquired, PACTOR-1)",
                                protocol="PACTOR-1", cs=got[0], sense=got[2])
                    continue
            if spare:
                if (hi + b_t0 * FS) / FS - last_spare > 0.3:
                    last_spare = (hi + b_t0 * FS) / FS
                    yield Event((hi + b_t0 * FS) / FS, "unassigned",
                                f"{spec.P1_CS_NAMES[cs[0]]}  (0 bit errors, "
                                f"PACTOR-1, shift "
                                f"{'inverted' if cs.sense else 'normal'})",
                                protocol="PACTOR-1", spare=cs[0], sense=cs.sense)
                continue
            if cs is not None and cs[1] == 0 and \
                    (hi + b_t0 * FS) / FS - last_cs > 0.3:
                last_cs = (hi + b_t0 * FS) / FS
                yield Event((hi + b_t0 * FS) / FS, "cs",
                            f"{spec.P1_CS_NAMES[cs[0]]}  (0 bit errors, PACTOR-1)",
                            protocol="PACTOR-1", cs=cs[0], sense=cs.sense)

        # A window an armed marker accounts for is PACTOR-2, and the four tests
        # below it are PACTOR-1 shape and a geometry three modes fit -- none of
        # which has anything to say about it. Only the CRC-gated PACTOR-1 data
        # scan is left running, because a validated frame is not a claim a
        # correlation elsewhere in the window may overrule.
        p2mark = next((fr for fr in p2_frames if fr.t < (seg0 + len(seg)) / FS
                       and fr.end > seg0 / FS), None)
        present = p2mark is None and _fsk_present(sp)
        # Both PACTOR-1 decoders read the same 100 Bd MARK/SPACE magnitudes off
        # this hop's audio, and each used to build its own -- a duplicated O(N)
        # pass on every FSK hop of a session. Built at most once per hop, by
        # whichever question gets there first.
        p1_tones = None
        if present and hi / FS - last_fsk > 0.5:
            p1_tones = p1rx.tone_series(seg)
            conn = p1rx.decode_connect(seg, tones=p1_tones)
            if conn is not None:
                last_fsk = last_dec = hi / FS
                in_session = True
                yield Event(hi / FS, "connect", conn.report(), connect=conn)
                continue
        # The other link-setup frame: 11 bytes at 100 Bd throughout, no sync byte
        # and a CRC, which the sync-and-character lock above cannot see at all. A
        # Robust Call train read as nothing but energy, so a monitor transcript
        # beside one says "quiet" over a channel somebody is calling on.
        #
        # The tone pair is the station's own only while a link is up; a sweeping
        # receiver has no idea where the pair is, and every branch-B train on file
        # arrived on somebody else's -- 607/807 through 1906/2107 Hz. So the pin
        # is `in_session` and the default is `call_b_frames`' narrowed sweep.
        #
        # NOT BEHIND `_fsk_present`, for that same reason, and this was measured:
        # gated on it, the monitor over K6SDR's Robust Call train on 10.145 MHz
        # still printed "no PACTOR signal detected" -- the gate asks for energy at
        # 1400 and 1600 Hz and that train is at 607/807. The frame carries a
        # CRC-16 and needs no energy gate in front of it; what bounds the cost is
        # `call_b_centres`, which is the same question asked at the pair that is
        # actually there.
        #
        # Its own debounce, on the frame length: the hops overlap four deep and
        # the same burst decodes in each of them, while a real train repeats every
        # 1.25 s and every burst of it is worth a line.
        if p2mark is None:
            centre = (p1rx.MARK + p1rx.SPACE) / 2 if in_session else None
            fresh = [(seg0 / FS + t, cn)
                     for t, cn in call_b_frames(seg, centre)
                     if seg0 / FS + t - last_callb > p1rx.CALL_B_LEN * 8 / 100]
            if fresh:
                last_fsk = last_callb = fresh[-1][0]
                for at, cn in fresh:
                    yield Event(at, "callb", cn.report(), protocol="PACTOR-1",
                                connect=cn)
                continue
        # The PACTOR-1 ARQ data phase, asked as its own question rather than as
        # the branch of the connect test it used to be. Two guards came off it and
        # each was costing real traffic on the one third-party session in the
        # corpus that carries any.
        #
        # `_fsk_present`, because the two decoders want different evidence.
        # `decode_connect` is a structural lock -- sync byte, character run,
        # terminator -- with nothing unforgeable in it, so it needs a gate in front
        # of it or it names a callsign out of noise; taking the gate away is worth
        # `Normal Call: E` and `Normal Call: .` across the corpus, and 1,080
        # accepts in all -- swept ungated over the 902 corpus captures in 1.5 s
        # windows a quarter-second apart, 37,309.5 s of real off-air HF, 144,203
        # windows and 9,372,437 sync alignments. A data frame is
        # CRC-gated, and a CRC is not a threshold that can be argued with. The gate
        # meanwhile judges the wrong thing: `_occupied_bw` measures the whole 2 s
        # hop rather than the signal at the FSK tones, so any co-channel
        # transmission in the passband vetoes the window. KE5YTA's packet at
        # t=11.00 arrives under two tones 20 and 30 times the noise floor and is
        # thrown away because a 1000 Hz signal starts up beside it.
        #
        # `last_fsk`, because a PRESENCE REPORT MUST NOT SUPPRESS A DECODE. The
        # `p1reply` line at t=10.50 -- which says only that a burst was there and
        # explicitly does not claim to read it -- armed the same timer the decoder
        # is gated on, and shadowed the packet 0.5 s behind it. That is the failure
        # `last_p3` carries a comment about, in the other family: a report that
        # knows nothing hiding one that knows something. Decodes get their own
        # timer, and presence is not allowed to touch it.
        #
        # What stays is `in_session`, which is the guard that means anything here:
        # data frames cannot precede a connect. And the cost that put the scan
        # behind the gate does not survive measurement -- `p1rx` hoists everything
        # alignment-independent, so this is 5 ms on a 1.25 s window against
        # `decode_expected_packet`'s 68. Swept over all 896 corpus captures and all
        # 27 regression fixtures, it recovers KE5YTA's two lost packets and finds
        # nothing anywhere else.
        if in_session and hi / FS - last_dec > 0.5:
            if p1_tones is None:
                p1_tones = p1rx.tone_series(seg)
            pkts = p1rx.decode_p1_packets(seg, tones100=p1_tones)
            if pkts:
                last_fsk = last_dec = hi / FS
                for p in pkts:
                    yield _p1_packet_event(p, seg0 + p.start)
                continue
        if present and hi / FS - last_fsk > 0.5:
            # ENERGY AT BOTH TONES, WHICH IS NOT THE SAME AS PACTOR-1. This said
            # "PACTOR-1 FSK present" until 2026-08-14, when a WWV recording taken
            # to check the timebase became the control nobody had run: 103 of
            # these in 159 s of a standards time broadcast -- a steady carrier
            # and one tick a second, no ARQ mode at all. Dial 9998.5 puts WWV's
            # carrier at 1510 Hz, between the two tones, and the AF chain's odd
            # harmonics do the rest. A quiet band at 6800 kHz gave zero in 211 s,
            # so the detector is not free-running; it needs a strong carrier near
            # 1500 Hz, and a watched channel is exactly where one lives.
            yield Event(hi / FS, "fsk",
                        "energy at both FSK tones (1400/1600 Hz) -- shape only, "
                        "no callsign, and a strong carrier near 1500 Hz reads "
                        "the same")
            continue
        pr = _p1_reply(seg) if p2mark is None else None
        if pr is not None and hi / FS - last_fsk > 0.5:
            last_fsk = hi / FS
            # THE CODEWORD READ ALREADY FAILED HERE. `_p1_cs_bursts` above tried to
            # decode a control signal out of this same window and got nothing at
            # zero errors -- which is what the distance-8 code is for, since a burst
            # that is one of the four words lands exactly on it and a burst that is
            # not lands 2-4 away. So everything below the decode is shape, and the
            # line is written to be unmistakable about that at three in the morning:
            # this used to end "a peer answered" and it was quoted back to the
            # operator as a peer answering, from a night on which nothing did.
            yield Event(hi / FS, "p1reply",
                        f"1400/1600 Hz burst {pr.ms:.0f} ms, weaker tone "
                        f"+{pr.weaker_tone_db:.1f} dB over guard, concurrency "
                        f"{pr.concurrency:.2f} -- shape only, not decoded, not "
                        f"attributed (VARA and 500 Hz ARQ look the same)")
            continue
        p2 = _p2_present(sp) if p2mark is None else None
        if p2 is not None and hi / FS - last_p2 > 1.0:
            last_p2 = hi / FS
            # The frequencies, because the omission cost a night: this line said
            # only "spacing 150 Hz" for a pair at 1704/1910 Hz, and an off-frequency
            # stranger was read as the gateway answering us. And `protocol`,
            # because the kind alone said PACTOR-3 to every downstream consumer.
            #
            # THE TEXT NAMES NO PROTOCOL, for the reason `_p2_present` gives in
            # its own docstring: the four gates above are a geometry, and three
            # different modes satisfy them. This line used to end "PACTOR-2
            # geometry" and the phrase was read back as an identification --
            # a 2026-08-18 session opened on a live link abandoned to it. What
            # was underneath it that night is measured: the captures of
            # captures/onair-0818-2235 carry a burst of about a second apiece at
            # 1400/1600 Hz keyed 200 Bd on the idle byte, which alternates every
            # 20 ms and so passes the concurrency test on any window that
            # straddles a transition -- PACTOR-1, at the speed level we ourselves
            # transmit, and hold_08, hold_12 and hold_17 decode CRC-valid.
            yield Event(hi / FS, "detect",
                        f"carrier pair {p2.f_lo:.0f}/{p2.f_hi:.0f} Hz "
                        f"+{p2.excess_db:.1f} dB over guard, concurrency "
                        f"{p2.concurrency:.2f}, balance {p2.balance:.2f} -- "
                        f"shape only, no protocol named (PACTOR-1 at 200 Bd, "
                        f"PACTOR-2 and PACTOR-3 all read like this)",
                        protocol="PACTOR-2")


# ---------------------------------------------------------------------------
# In-session receive: demodulate what is due, where it is due
# ---------------------------------------------------------------------------

def _frame_span(path) -> int:
    """Samples `path`'s grid needs past row 0 -- `rx.decode_frame`'s own margin.

    Two rows of it are that margin; the rest is the row count, which the cycle
    length moves by more than a factor of four, and the stagger the last carrier
    starts its clock on.
    """
    return (path.n_symbols + 2) * SPS + max(path.clock_offsets(SPS))


def _packet_span(span: int) -> int:
    """The PACKET inside a `_frame_span`: its last two rows are the margin.

    What a receiver has to WAIT for and what a window has to HOLD are different
    lengths, and one number was serving both. The margin carries no signal, and
    `_sampled_baseband` reads past the end of a buffer as silence, so a frame
    whose packet is inside the window decodes whether or not its margin is. Held
    to the longer figure, a window ending a few hundred samples inside that
    margin excluded the packet's own alignment from `_candidates` and the read
    fell back a whole symbol -- 125 of the 181 IRS cycles of
    `captures/onair-0913-2320`, on a packet that ended 16.8 ms inside every one.
    """
    return span - 2 * SPS


MATCHED_DELAY_N = (rx._pulse(SPS).size - 1) // 2
"""Group delay of the matched filter, in samples.

The filter's output at one grid instant is the window of audio ENDING this far
past it, so a reader's last row needs this much air behind it or it is
demodulated out of whatever the buffer was padded with."""

UNREAD_TAIL_N = 3 * SPS - MATCHED_DELAY_N
"""Samples at the end of a `_frame_span` that no grid instant reads.

The tracked reader's last row sits one symbol inside the packet's own span and
the matched filter reaches its group delay past that, so the two trailing rows
`_frame_span` carries are the filter's reach and 511 samples over. A window that
stops anywhere inside this tail has lost nothing the decode would have used --
which is why the 2026-09-13 IRS windows, closed on the capture clock and so a
holdback short of their nominal span, still decoded every frame their anchor
admitted. What the shortfall did cost was the ANCHOR: `_p3_packet` floored its
projection against the same nominal span.
"""


_CS_SPAN = (spec.CS_BITS_PER_TONE + 1) * SPS
"""Samples a control signal occupies: 20 DBPSK symbols plus the phase reference."""


@lru_cache(maxsize=1)
def _matched_filter() -> np.ndarray:
    """`rx._pulse` reversed, ready to dot against a window of baseband."""
    return rx._pulse(SPS)[::-1].copy()


def _sampled_baseband(audio: np.ndarray, tones, idx: np.ndarray) -> dict:
    """`rx._baseband` for each tone, evaluated ONLY where `idx` reads it.

    The matched filter is 1860 taps and the convolution produces one output per
    input sample -- some 43,000 for the window one data frame lives in, of which
    the grid reads 81. A sweeping receiver has no choice, because it does not know
    which 81. A receiver that has locked does know, and a filter evaluated only at
    its sample instants is one small matrix product instead of six transforms of
    the window. Measured on the station Pi, that convolution was 75% of a tracked
    decode.

    Returns full-length arrays with only `idx` populated, so `rx.cells_from_baseband`
    and `rx.cs_bits` index them exactly as they index the real thing: the demod
    downstream is untouched and cannot drift away from the blind path's. Agrees
    with `rx._baseband` at those positions to 4e-16.
    """
    taps = _matched_filter()
    m = taps.size
    # bbp[j:j+m] is the history rx._baseband's 'full' convolution reads for output
    # j, so only the samples some window touches need mixing down at all.
    lo, hi = int(idx.min()), int(idx.max())
    b0, b1 = max(0, lo - m + 1), min(len(audio), hi + 1)
    pad = np.zeros(m - 1, complex)
    rows = idx - b0
    out = {}
    for cn in tones:
        bb = audio[b0:b1] * rx.carrier(cn, FS, b0, b1 - b0)
        win = np.lib.stride_tricks.sliding_window_view(
            np.concatenate([pad, bb, pad]), m)[rows]
        z = np.zeros(len(audio) + m - 1, complex)
        z[idx] = win @ taps
        out[cn] = z
    return out


class SyncedRx:
    """In-session receiver: demodulate the frame that is DUE, where it is due.

    `decode_events` is the band monitor's job. It does not know whether anything
    is present, at what offset, or of what kind, so it correlates the pilot at
    every hop across the buffer and re-runs the header and control-signal scans
    wherever one fires. A connected station is not in that position: it agreed the
    cycle, it knows from its own role whether the peer owes it a data packet or a
    control signal, and it has already decoded one of them. PACTOR is
    cycle-synchronous, so the next frame's row 0 lands one cycle later, within the
    grid's jitter -- a few symbols, not a few seconds.

    So this searches a handful of symbol positions either side of the alignment it
    last decoded at. Two things get cheaper together: the CRC scan shrinks from
    every symbol in the window to nine candidates, and the six per-tone matched
    filters stop convolving the window at all -- `_sampled_baseband` evaluates them
    at the instants the grid reads and nowhere else. Measured on a rendered level 2
    packet, 9.8 ms against the blind scan's 132.5 ms.

    WHERE A FRAME IS AND WHAT SHAPE IT IS ARE DIFFERENT QUESTIONS. The carrier swap
    moves every virtual carrier to its partner tone on each ARQ cycle, the cycle
    length moves the row count and the speed level moves the whole comb, and a
    station holding a lock knows none of the three from its own timing -- so
    `_packet_at_lock` reads the packet's header block and the lit channels before
    its grid. That is 2.3 ms of the 9.8 and not one extra CRC trial.

    THE BLIND PATH STAYS. This is a mode, not a replacement -- the monitor has no
    session to sync to and must keep sweeping. A miss here returns None, which
    means "I cannot answer cheaply, run the blind scan"; the caller feeds whatever
    that produces back through `observe` and the lock is restored. Two consecutive
    misses drop the lock outright, so a stale alignment is not scanned forever.

    A lock is only meaningful for a caller that hands whole, cycle-aligned windows
    -- `Event.start` is measured from the start of the buffer, so a rolling buffer
    moves the origin under it.
    """

    TRACK_SYMBOLS = 4
    """Symbols either side of the tracked alignment. Wide enough for cycle jitter
    and a resampled peer, narrow enough that the scan is nine Viterbi runs."""
    MAX_MISSES = 2

    def __init__(self, track_symbols: int = TRACK_SYMBOLS,
                 memory: "p3rx.FieldMemory | None" = None):
        self.track_symbols = track_symbols
        self.packet_at: int | None = None
        self.packet_level: int | None = None
        self.cs_at: int | None = None
        self.misses = 0
        self.tracked = 0                   # decodes made from a lock
        self.locks = 0                     # locks taken from a blind decode
        self.rotation: float | None = None
        """The angle the last CRC-valid tracked read found its field at.

        One number for the session: it is where the peer's transmitter, this
        station's clock and the residual carrier offset between them put the
        constellation, and the arm of 2026-09-13 holds it within a degree and a
        half across both of its speed levels. `p3rx.field_rotation` measures it
        far better than a header block can but only modulo one state's turn,
        and this is what says which turn."""
        self.memory = p3rx.FieldMemory() if memory is None else memory
        """Soft combining for the tracked read.

        THE SESSION'S OWN, where the caller hands one in. A locked link's frame
        is read here and the repeats the peer sends arrive at both readers --
        this one on the cycles the lock answers and the blind sweep on the ones
        it misses -- so two memories hold halves of one run of copies and
        neither has the sum. Sharing costs nothing: `FieldMemory` keys on the
        geometry either reader read, so a copy at a level the link is not on
        resets it there as it always did.

        What it exposes is a cycle both readers feed -- this one at the lock and
        the sweep at the block behind it -- which puts two demodulations of one
        transmission in the sum. Neither arm of 2026-09-13 does it at its own
        carrier correction, because the level 2 block anchors for the sweep on
        none of their cycles."""

    @property
    def locked(self) -> bool:
        return self.packet_at is not None or self.cs_at is not None

    def observe(self, ev: Event) -> None:
        """Take a lock from whatever the blind path decoded."""
        if ev.kind == "packet":
            level = ev.packet[0] if ev.packet is not None else None
            if ev.protocol not in (None, "PACTOR-3") or level not in placement.SPEED_PATHS:
                # A P1 packet's level 0 and FSK bit anchor are not a P3 grid.
                # The same session observer sees both protocols at upgrade.
                self.packet_at = self.packet_level = None
                return
        if ev.start is None:
            return
        if ev.kind == "packet":
            self.packet_at, self.misses = ev.start, 0
            self.packet_level = level
            self.locks += 1
            self.memory.clear()            # the field is delivered
        elif ev.kind == "cs":
            self.cs_at, self.misses = ev.start, 0
            self.locks += 1

    def packet(self, audio: np.ndarray, *,
               preferred_only: bool = False) -> "Event | None":
        """The data frame due this cycle, or None -- meaning fall back to blind.

        `preferred_only` keeps the read on the last CRC-validated level and
        stops there; see `_packet_at_lock` for what the ladder behind it costs.
        """
        return self._try(
            lambda a, at: self._packet_at_lock(a, at,
                                               preferred_only=preferred_only),
            self.packet_at, audio)

    def control_signal(self, audio: np.ndarray,
                       max_errors: int = CS_EXPECTED_MAX_ERRORS) -> "Event | None":
        """The peer's answer due this cycle, or None -- fall back to blind."""
        return self._try(lambda a, at: self._cs_at_lock(a, at, max_errors),
                         self.cs_at, audio)

    WIDEBAND_EARLY_N = round(.025 * FS)
    """Maximum missing short-packet tail admitted by the bounded CRC trial."""

    def sl1_packet_at(self, audio: np.ndarray, at: int) -> "Event | None":
        """One header-selected SL1 CRC at the current retained packet clock.

        The first ordinary packet after CHANGEOVER has no learned speed yet.
        A speculative SL3 read must not suppress this two-carrier packet or
        consume its reply deadline. Fit only nine quarter-symbol header
        positions, then decode once at the strongest header's row zero. A
        faded header uses the retained clock and the existing two short SL1
        carrier-order hypotheses. No alignment, speed, frequency, or combining
        ladder is opened; a failed CRC changes no receiver state.
        """
        home = placement.SPEED_PATHS[1]
        lead, delay = p3frame.DATA_OFFSET * SPS, MATCHED_DELAY_N
        span = _packet_span(_frame_span(home))
        if at < lead + SPS or len(audio) < at + span:
            return None
        starts = range(at - SPS, at + SPS + 1, SPS // 4)
        idx = np.arange(starts.start - lead, starts[-1] + 1, SPS // 4) + delay
        # Both carrier arrangements share the tone set, but not their clocks.
        idx = np.unique(np.add.outer(idx, home.clock_offsets(SPS)).ravel())
        z = _sampled_baseband(audio, home.tones, idx)
        h = p3rx.header_of(z, starts, home)
        if h is not None and h.fit >= p3rx.anchor_gate(home):
            if 1 not in h.levels or h.long_cycle:
                return None
            trials = [(h.at + lead, h)]
        else:
            trials = [(at, None), (at, p3frame.PacketHeader(0, 0, 0.0, carrier_swapped=True))]
        for row, header in trials:
            if len(audio) < row + span:
                continue
            path = p3rx.path_for(1, header)
            idx = row + np.arange(-1, path.n_symbols) * SPS + delay
            idx = np.unique(np.add.outer(idx, path.clock_offsets(SPS)).ravel())
            z = _sampled_baseband(audio, path.tones, idx)
            packet = p3rx.decode_at(audio, row, 1, Z=z, header=header)
            if packet is None:
                continue
            self.packet_at, self.packet_level = row, 1
            if header is h and h is not None:
                self.rotation = h.rot
            self.misses = 0
            self.tracked += 1
            self.memory.clear()
            return _packet_event(packet, tag=", SL1 pre-key")
        return None

    def sl2_packet_at(self, audio: np.ndarray, at: int) -> "Event | None":
        """Bounded short-SL2 read after an emitted CS4 or an established SL2.

        The old pre-key path tried the last validated SL1 and speculative SL3,
        never the SL2 we had just requested. A weak SL2 header is common: keep
        the established packet clock and try both physical arrangements, with
        rotation measured from the field. One CRC per arrangement, no scan or
        memory update on failure. A credible different header vetoes the trial.
        """
        home = placement.SPEED_PATHS[2]
        lead, delay = p3frame.DATA_OFFSET * SPS, MATCHED_DELAY_N
        span = _packet_span(_frame_span(home))
        if at < lead + SPS or len(audio) < at + span:
            return None
        starts = range(at - SPS, at + SPS + 1, SPS // 4)
        idx = np.arange(starts.start - lead, starts[-1] + 1, SPS // 4) + delay
        idx = np.unique(np.add.outer(idx, home.clock_offsets(SPS)).ravel())
        z = _sampled_baseband(audio, home.tones, idx)
        header = p3rx.header_of(z, starts, home)
        credible = header is not None and header.fit >= p3rx.anchor_gate(home)
        if credible and (2 not in header.levels or header.long_cycle):
            return None
        row = header.at + lead if credible else at
        if len(audio) < row + span:
            return None
        first = header.swapped if header is not None else False
        for swapped in (first, not first):
            h = p3frame.PacketHeader(row - lead, 2, 0.0,
                                    carrier_swapped=swapped)
            path = p3rx.path_for(2, h)
            idx = row + np.arange(-1, path.n_symbols) * SPS + delay
            idx = np.unique(np.add.outer(idx, path.clock_offsets(SPS)).ravel())
            z = _sampled_baseband(audio, path.tones, idx)
            near = header.rot if header is not None else self.rotation or 0.0
            rot, _ = p3rx.field_rotation(z, row, path, near)
            h = replace(h, rot=rot)
            packet = p3rx.decode_at(audio, row, 2, Z=z, header=h)
            if packet is None:
                continue
            self.packet_at, self.packet_level = row, 2
            self.rotation, self.misses = rot, 0
            self.tracked += 1
            self.memory.clear()
            return _packet_event(packet, tag=", SL2 pre-key")
        return None

    WIDEBAND_HEADER_BODY_N = round(.050 * FS)
    """Body support needed after row zero for the bounded opening-header read."""

    def wideband_header_at(self, audio: np.ndarray,
                           at: int) -> "p3frame.PacketHeader | None":
        """Read only a credible wideband header at a retained peer row zero.

        Three quarter-symbol positions and the existing constant-header gate
        bound this probe. Only the opening header and 50 ms of body are used,
        including when the header advertises a long packet. The returned `at`
        is HEADER phase; add `DATA_OFFSET * SPS` to recover its data row zero.

        This is geometry evidence, not a delivered packet: no CRC, receive
        lock, retry bookkeeping, or soft-combining memory is changed. A caller
        may retain the remaining body, but must not acknowledge this header.
        """
        header = self._wideband_header_candidate_at(audio, at)
        if (header is None
                or header.fit < p3rx.anchor_gate(placement.SPEED_PATHS[3])
                or not any(sl >= 3 for sl in header.levels)):
            return None
        return header

    def _wideband_header_candidate_at(self, audio: np.ndarray,
                                     at: int) -> "p3frame.PacketHeader | None":
        """The same bounded header probe, before its public credibility gate.

        A below-gate result is only a constellation hint for an explicitly
        enabled short-body CRC hypothesis. It cannot establish frame geometry.
        """
        probe = placement.SPEED_PATHS[3]
        lead = p3frame.DATA_OFFSET * SPS
        extent = at + self.WIDEBAND_HEADER_BODY_N
        if at < lead + SPS // 4 or len(audio) < extent:
            return None
        audio = audio[:extent]
        delay = (_matched_filter().size - 1) // 2
        starts = range(at - SPS // 4, at + SPS // 4 + 1, SPS // 4)
        idx = np.arange(starts.start - lead, starts[-1] + SPS + 1, SPS // 4) + delay
        Z = _sampled_baseband(audio, probe.tones, idx)
        return p3rx.header_of(Z, starts, probe)

    def wideband_packet_at(self, audio: np.ndarray, at: int, *,
                           allow_short_fallback: bool = False,
                           can_decode=None) -> tuple["Event | None", bool]:
        """One header-selected short SL3 frame at a retained peer clock.

        The boolean says a wideband header owned this trial, including a CRC
        miss. That miss must not start another level or alignment ladder under
        the same key. Other levels and long headers leave their existing readers
        alone. Early-tail recovery is not established for every wideband level.

        v24's first SL3 frame decodes from audio ending 20 ms before its nominal
        end. Missing filter support is zero, never future audio. The header gate,
        an explicit 25 ms tail limit, and the normal CRC still apply.

        `allow_short_fallback` permits one short SL3 hypothesis when that
        header falls below its gate. Its data row is the caller's retained
        clock, not the weak header's position or cycle bits; the field refines
        rotation, and then the CRC decides -- once on each physical carrier
        order, the block being too weak to name it, and once more on the sum of
        this cycle's softs with the repeats behind them. A credible long or
        other-level header still excludes this fallback. No alignment, level or
        frequency ladder is opened, and the clock does not move. A failed weak
        hypothesis does not claim header ownership or change receiver state.

        `can_decode`, when supplied, is checked after the header and immediately
        before the sparse body work. The caller retains its transmit-admission
        guard: permission to attempt a CRC is not permission to key late.
        """
        probe = placement.SPEED_PATHS[3]
        lead = p3frame.DATA_OFFSET * SPS
        span = _frame_span(probe)
        if at < lead + SPS // 4 or len(audio) < at + _packet_span(span) - self.WIDEBAND_EARLY_N:
            return None, False
        # Bound both the transform and the allocation to one short packet.
        extent = at + span + SPS
        audio = audio[:extent]
        if len(audio) < extent:
            audio = np.pad(audio, (0, extent - len(audio)))
        delay = (_matched_filter().size - 1) // 2
        header = self._wideband_header_candidate_at(audio, at)
        if header is None:
            return None, False
        fallback = header.fit < p3rx.anchor_gate(probe)
        if fallback:
            if not allow_short_fallback:
                return None, False
            # This is a declared short-SL3 hypothesis, not a lowered header
            # gate. Only its body CRC may turn it into a delivered packet.
            header = p3frame.PacketHeader(
                at - lead, 4 | int(header.swapped), header.fit,
                rot=header.rot, carrier_swapped=header.swapped)
        elif header.long_cycle or [sl for sl in header.levels if sl >= 3] != [3]:
            return None, False
        if can_decode is not None and not can_decode():
            return None, not fallback
        sl = 3
        path = p3rx.path_for(sl, header)
        start = header.at + lead
        grid = np.add.outer(
            start + (np.arange(path.n_symbols + 1) - 1) * SPS + delay,
            path.clock_offsets(SPS)).ravel()
        # A weak hypothesis also offers its softs to the memory, which reads
        # them an eighth of a symbol either side of the grid.
        idx = np.unique(np.add.outer(
            grid, (-p3rx.CONFIRM_STEP, 0, p3rx.CONFIRM_STEP) if fallback
            else (0,)).ravel())
        Z = _sampled_baseband(audio, path.tones, idx)
        # SL3's unstaggered field has the same rotation weight under either
        # carrier permutation, so the angle is one measurement for both and
        # the arrangement is not something it can rank.
        near = (self.rotation if fallback and self.rotation is not None
                else header.rot)
        rot, _ = p3rx.field_rotation(Z, start, path, near)
        # BOTH ARRANGEMENTS, one CRC each and no new gate. The block's own
        # reading of the swap is all a below-gate hypothesis has, and at fit
        # 0.65-0.78 it is wrong about half the time -- which is once an ARQ
        # cycle against a gateway that alternates, as K0NTS's did on
        # 2026-09-15 while its field decoded offline on the other order. The
        # tone SET is closed under the swap, so the second trial re-reads no
        # baseband and moves no clock: the same instants in the other cell
        # order, with the CRC still the only acceptor.
        heads = [replace(header, rot=rot)]
        if fallback:
            heads.append(replace(header, rot=rot,
                                 carrier_swapped=not header.swapped))
        order = ""
        for h in heads:
            packet = p3rx.decode_at(audio, start, sl, Z=Z, header=h)
            if packet is not None:
                order = " swapped" if h.swapped else " home"
                break
        if packet is None:
            if not fallback:
                return None, True
            return self._wideband_combined(audio, start, path, rot, Z), False
        self.packet_level, self.rotation = sl, rot
        self.misses = 0
        self.tracked += 1
        self.memory.clear()
        tag = (f", wideband pre-key short fallback SL{sl}{order}" if fallback
               else ", wideband pre-key")
        return _packet_event(packet, tag=tag), True

    def _wideband_combined(self, audio: np.ndarray, start: int,
                           path: "placement.Path", rot: float,
                           Z: dict) -> "Event | None":
        """This cycle's failed softs, summed with the repeats behind them.

        A COPY THAT FAILS IS STILL EVIDENCE, and discarding it was the whole of
        the K0NTS stall: the gateway repeated one 60-byte field for forty
        seconds, every copy reached this CRC and none of them passed it alone,
        while the sum of the same copies -- the arm's own windows, its own
        clock, its own carrier order, nothing else changed -- decodes twice.
        `FieldMemory` keys on the geometry and corroborates its own accept, so
        a hypothesis that was never credible enough to deliver a packet is
        still safe to add to: a wrong arrangement is a wrong cell order, and
        what refuses it is the same CRC that refused the copy.
        """
        cells = {off: p3rx._cells(audio, start + off, path, rot, fs=FS, Z=Z)
                 for off in (-p3rx.CONFIRM_STEP, 0, p3rx.CONFIRM_STEP)}
        if any(c is None for c in cells.values()):
            return None
        field = self.memory.add(cells, path)
        if field is None:
            return None
        self.packet_level, self.rotation = path.speed_level, rot
        self.misses = 0
        self.tracked += 1
        return _packet_event(
            p3rx.packet_of(field, path, start),
            tag=f", wideband pre-key short fallback SL{path.speed_level} combined")

    def control_signal_at(self, audio: np.ndarray, at: int,
                          max_errors: int = CS_EXPECTED_MAX_ERRORS,
                          *, details: bool = True) -> "Event | None":
        """The peer's answer at an instant the CALLER predicts, not one we locked.

        The PACTOR-3 counterpart of `p1rx.cs_anchored`, and it exists for the same
        situation: a station holding a cycle grid knows where the turnaround is
        from the grid, not from having decoded there before. `_MasterGrid.rx_due`
        is that prediction, and it is available in the first cycle after the
        upgrade -- where a lock taken from a previous PACTOR-3 decode is not,
        because there has not been one.

        No miss bookkeeping: the anchor is the caller's, so a miss here says
        nothing about a lock this object holds and must not drop one.

        MEASURED on the two real off-air PACTOR-3 control signals the corpus
        holds -- the CS5 and CS1 an established link sends back in its turnaround
        slots. Both read at ZERO bit errors from anchors 30 ms either side of the
        burst, every step of a 10 ms sweep, which is far wider than the cycle
        grid's own jitter. Against audio that is not PACTOR-3 it accepts nothing:
        0 in 200 PACTOR-1 control signals over both shift senses, 0 in 200
        PACTOR-1 data packets and 0 in 400 windows of white noise.

        `details` reaches `_cs_event`: a caller whose key is close may take the
        codeword without the changeover body behind it.
        """
        return self._cs_at_lock(audio, at, max_errors, details=details)

    def control_signal_tracked(self, audio: np.ndarray, at: int, *,
                               details: bool = True) -> "Event | None":
        """The answer where the PEER'S OWN cadence puts it, not where a grid guesses.

        `control_signal_at` reads at an instant its caller predicts and spends
        `track_symbols` either side of it, because a caller predicting from a
        transmit grid may be a few symbols out. A caller projecting from the last
        answer it decoded is not: it is reading the same station's next word off
        a cadence it has already measured. So the bracket is
        `ANSWER_TRACK_SYMBOLS` and the radius is the tracked one, which is the
        rule `_cs_at_lock` has held a lock to since 2026-08-28 and which this
        path was missing only because `CS_EXPECTED_MAX_ERRORS` is PACTOR-1's
        twelve-bit constant.
        """
        return self._cs_at_lock(audio, at, self._CS_TRACKED_MAX_ERRORS,
                                details=details, track=self.ANSWER_TRACK_SYMBOLS)

    def _try(self, decode, at: int | None, audio: np.ndarray) -> "Event | None":
        if at is None:
            return None
        ev = decode(audio, at)
        if ev is None:
            self.misses += 1
            if self.misses >= self.MAX_MISSES:
                self.packet_at = self.cs_at = None
                self.packet_level = None
            return None
        self.misses = 0
        self.tracked += 1
        return ev

    def _candidates(self, audio: np.ndarray, at: int, span: int, step: int,
                    lead: int = 0, track: int | None = None) -> range | None:
        """Alignments around the lock, on its own grid, that fit inside `audio`.

        `lead` is what the caller reads BEFORE the alignment -- a data packet's
        header block sits ahead of grid row 0 -- so the earliest candidate is that
        much into the buffer rather than at its start. `track` narrows the
        bracket for a caller whose anchor is better than the lock's.
        """
        last = len(audio) - span
        reach = (self.track_symbols if track is None else track) * SPS
        lo, hi = at - reach, at + reach
        while lo < lead:
            lo += step
        while hi > last:
            hi -= step
        return range(lo, hi + 1, step) if lo <= hi else None

    def _index(self, cand: range, path, delay: int,
               offsets: np.ndarray) -> np.ndarray:
        """Sample instants the whole search reads, `offsets` from the first row 0.

        A block that sits ahead of the grid takes negative ones. Each candidate
        reads the same instants one symbol further on, which is why the span is
        widened by the search itself -- and, on a level whose comb is staggered, by
        the clock each carrier is read on. The sparse baseband holds only the
        samples named here, so an instant missing from this set reads as silence
        rather than as an error.
        """
        idx = cand.start + offsets + delay
        return np.unique(np.add.outer(idx, path.clock_offsets(SPS)).ravel())

    def _packet_at_lock(self, audio: np.ndarray, at: int, *,
                        preferred_only: bool = False) -> "Event | None":
        """The frame due at `at`, at whatever speed level the peer sent it.

        THE LEVEL IS THE PEER'S TO CHOOSE and this used to assume it was 2. The
        header block was read for the swap and the cycle length and then handed
        to `p3rx.path_for` with `placement.HEADER.speed_level` -- a literal --
        so `header.levels` was computed, discarded, and the frame demodulated on
        a six-tone grid whatever comb was lit. A level 3 packet is fourteen tones
        and does not validate on that grid at all.

        Try the last CRC-validated level first. Interference can make occupancy
        rank several wide combs ahead of an unchanged SL1 retry, spending the
        response deadline on failed geometries. This preference changes only
        their order: a miss still tries every occupancy-ranked candidate, so the
        peer can change speed. The header and body CRC validate each hypothesis.

        Without that prior success, channel occupancy orders the candidates as
        in `p3rx.decode_headed`: the header carries the level modulo four, so it
        cannot separate levels 1 and 5, or 2 and 6.

        A CALLER WITH A KEY IN FRONT OF IT TAKES THE PREFERRED LEVEL ALONE.
        Walking the occupancy ranking costs up to 80 ms on the 80 m arm of
        2026-09-13, which is ten times the whole pre-key budget, and it bought
        nothing there: the bounded read takes the same 104 frames at 1.48 ms
        median and 7.42 ms at worst, none of them over budget against 35 of 177.
        The sweep behind our own carrier has a cycle to spend and keeps the
        ladder, which is where the peer's speed change is picked up.
        """
        preferred = self.packet_level
        if preferred is not None:
            ev = self._level_at_lock(audio, at, preferred)
            if ev is not None:
                return ev
            if preferred_only:
                return None
        on = audio[at:at + int(p3rx.LEVEL_WINDOW_S * FS)]
        for sl in p3rx.levels_present(p3rx.channel_energy(on, fs=FS)):
            if sl == preferred:
                continue
            ev = self._level_at_lock(audio, at, sl)
            if ev is not None:
                self.packet_level = sl
                return ev
        return None

    def _level_at_lock(self, audio: np.ndarray, at: int,
                       sl: int) -> "Event | None":
        """One speed level's hypothesis of the frame at `at`.

        READ THE HEADER FIRST, because what it names decides which samples the
        grid is even made of. A virtual carrier moves to its partner tone on
        every ARQ cycle, so a tracked decode pinned to the home order gathers the
        right energy in the wrong order on every second cycle and passes a CRC on
        none of them; the cycle length moves the row count the same way. Both
        come out of the eight-symbol block the packet opens with, which is what
        that block is for, and `p3rx` already knows how to turn one into a
        geometry.

        On a quarter symbol, because the lock is not: it is a symbol grid
        inherited from whatever alignment last passed a CRC, so it can stand half
        a symbol off the packet's own clock, which the CRC tolerates and the
        header fit does not. `p3rx.header_of` carries the figures.

        `p3rx.decode_at` rather than the frame chain directly, because speed
        level 1 is case 0 -- its own cell order, its own permutation and a K=9
        trellis -- and a tracked receiver that could not read one had no path to
        the peer's entry answer or to the packet a cycle behind a break-in.

        THE BLOCK NAMES THE LEVEL AND THE FIELD SETTLES THE REST. Above level 1
        the arrangement and the angle come from `p3rx.field_rotation` over the 72
        rows rather than from eight symbols in front of them, and a copy that
        fails every alignment goes to `self.memory` -- which is what the
        arrangement and the angle buy, a copy in the same cell order and on the
        same axes as the one before it.
        """
        lead = p3frame.DATA_OFFSET * SPS
        delay = (_matched_filter().size - 1) // 2
        home = placement.SPEED_PATHS[sl]
        cand = self._candidates(audio, at, _packet_span(_frame_span(home)),
                                SPS, lead)
        if cand is None:
            return None
        step = SPS // 4
        head_at = range(cand.start, cand[-1] + 1, step)
        head = _sampled_baseband(
            audio, home.tones,
            self._index(cand, home, delay,
                        np.arange(-lead, (len(cand) - 1) * SPS, step)))
        header = p3rx.header_of(head, head_at, home, fs=FS)
        # Below the header gate, do not trust its level or cycle-length bits.
        # The established short-frame fallback still uses the same timing
        # bracket and CRC; SL1 also tests the partner carrier arrangement.
        below = header if (header is not None
                           and header.fit < p3rx.anchor_gate(home)) else None
        if below is not None:
            header = None
        if header is not None and sl not in header.levels:
            return None
        # SPEED LEVEL 2 BARELY REACHES ITS GATE, and that is a property of its
        # comb rather than of the signal. Six carriers light four of the sixteen
        # constant words, so `HEADER_FIT` scored over all sixteen -- the other
        # twelve being noise -- is a figure it can never make; read here on the
        # level's own six it can, but each of those carriers holds a sixth of the
        # burst where speed level 1's two hold a half. MEASURED at the lock over
        # WS8EOC's 80 m arm of 2026-09-13, whose gateway repeated one level 2
        # packet for thirty-three cycles: 0.67 to 0.87, twenty-nine of
        # thirty-five under the gate, where the level 1 frames before them read
        # 0.91 to 0.95 against their own.
        #
        # A gate is what stops a SEARCH manufacturing anchors, and a tracked read
        # is not a search: the position is the session's own clock, and the block
        # is only being asked for the swap, the cycle length and the angle. So
        # the reading REPLACES the unread-block hypothesis and the CRC decides.
        # Replaces rather than joins, because it is the same hypothesis better
        # informed -- home order at zero rotation is what `path_for(sl, None)`
        # assumes, and this is that geometry with the swap and the angle
        # measured. Over the 35 level 2 cycles of that arm the substitution
        # takes the tracked reader from 1 read to 3 at 16.0 ms median against
        # 15.9, where keeping both hypotheses cost 22.1 for the same three.
        #
        # AT EVERY LEVEL THE BLOCK CANNOT CARRY, not only at level 2: above it
        # the sixteen constant words are there to be read and a block under the
        # gate is a faded one, which is the same position -- the level bits are
        # not to be trusted, the swap and the angle are still wanted, and the
        # field below now supplies both far better than the block could.
        headers = [below] if below is not None and sl >= 2 else [header]
        if header is None and sl == 1:
            # A faded header does not establish the request bits or geometry.
            # Keep the existing short-frame hypothesis and its timing bracket,
            # but let the CRC choose either physical order. WS8EOC 2026-09-11
            # has one such header among 35 otherwise readable repeated frames.
            headers.append(p3frame.PacketHeader(0, 0, 0.0, carrier_swapped=True))
        for trial_header in headers:
            path = p3rx.path_for(sl, trial_header)
            # A long cycle is four and a half times the grid, so the window that held
            # the short one may not hold it; asking again is how that comes out as a
            # miss rather than as a read off the end of the buffer.
            cand = self._candidates(audio, at, _packet_span(_frame_span(path)),
                                    SPS, lead)
            if cand is None:
                continue
            grid = np.arange(-1, path.n_symbols + len(cand) - 1) * SPS
            Z = _sampled_baseband(audio, path.tones,
                                  self._index(cand, path, delay, grid))
            # THE ANGLE COMES OFF THE FIELD, not off the block in front of it.
            # `p3rx.field_rotation` folds the data out of all 432 cells where
            # the block offers 48 on four constant words out of sixteen: over
            # the 35 level 2 cycles of the 80 m arm of 2026-09-13 the block
            # lands within 20 deg of the session's own angle eight times and the
            # field thirty-two, and the eight cycles whose field passes a CRC at
            # the lock read 41.6 to 45.4 deg on every one.
            #
            # THE ARRANGEMENT IS THE BLOCK'S WHERE THE BLOCK CLEARS ITS GATE,
            # which is what the gate is for. Over it the block has read both
            # staggers on a quarter symbol and its fit separates them; the same
            # sum that measures the angle cannot, because at speed level 2 the
            # swap moves which cluster leads by exactly the half symbol
            # `spec.SUBBAND_LEAD` is -- a lock standing half a symbol off reads
            # as heavily on the wrong arrangement as on the right one, and
            # `test_p3rx.py` holds a rendered packet at both. Under the gate
            # there is no reading to prefer and the heavier sum is taken, which
            # on that arm names the arrangement the packet was sent in on every
            # one of the eight cycles that pass a CRC, by 1.6x to 10x, where
            # the block's own variable-header fit is a coin at 0.42-0.55.
            from_field = sl >= 2 and trial_header is not None
            if from_field and trial_header is below:
                trial_header = self._field_shape(Z, at, sl, trial_header)
                path = p3rx.path_for(sl, trial_header)
            # NEAREST THE LOCK FIRST, which is where a held link's frame is. The
            # candidates were walked in time order, so the alignment the cycle
            # before decoded at -- the one an unmoved grid puts this cycle's row 0
            # on -- sat fifth of nine, and the four in front of it each spent a
            # Viterbi run over the whole frame before the CRC turned them down.
            # That is most of what a tracked read costs at the wide levels:
            # measured through `onair._SessionRx.deep_scan`, speed level 6 goes
            # 50.5 ms to 23.4 on the short cycle and 190.6 to 67.5 on the long one,
            # with the same nine alignments still admissible and the same CRC
            # admitting them. `tests/shrike/test_rxcost.py` holds the table.
            for start in sorted(cand, key=lambda st: (abs(st - at), st)):
                h = trial_header
                if from_field:
                    rot, _ = p3rx.field_rotation(Z, start, path, h.rot, fs=FS)
                    h = replace(h, rot=rot)
                p = p3rx.decode_at(audio, start, sl, fs=FS, Z=Z, header=h)
                if p is not None:
                    if from_field or (header is not None
                                      and trial_header is header):
                        # Only a measurement updates it: the level 1 fallback
                        # header carries a placeholder angle, not a reading.
                        self.rotation = h.rot
                    self.memory.clear()
                    return _packet_event(p, tag=", tracked")
            if from_field and sl == self.packet_level:
                # The level the link is ON, not one the ladder behind a miss is
                # guessing at: `FieldMemory` groups by geometry and resets on a
                # change, so a speculative copy at the wrong level ends the run
                # of repeats the memory exists to gather.
                ev = self._combined_at_lock(audio, at, cand, path, trial_header, Z)
                if ev is not None:
                    return ev
        return None

    def _field_shape(self, Z: dict, at: int, sl: int,
                     header: "p3frame.PacketHeader") -> "p3frame.PacketHeader":
        """`header` with the carrier arrangement and the angle read off the field.

        The two arrangements are the same tones on the two symbol clocks the
        carrier swap chooses between, so the heavier reading is the one the
        packet's own cells agree on -- the measurement the block was too short
        to make. The turn is settled against the session's angle where it has
        one and against the block's own reading where it does not.
        """
        near = header.rot if self.rotation is None else self.rotation
        best = None
        for swapped in (header.swapped, not header.swapped):
            h = replace(header, carrier_swapped=swapped)
            rot, weight = p3rx.field_rotation(Z, at, p3rx.path_for(sl, h),
                                              near, fs=FS)
            if best is None or weight > best[0]:
                best = weight, replace(h, rot=rot)
        return best[1]

    def _combined_at_lock(self, audio: np.ndarray, at: int, cand: range,
                          path: "placement.Path",
                          header: "p3frame.PacketHeader",
                          Z: dict) -> "Event | None":
        """This cycle's copy, summed with the repeats behind it.

        ONE MORE CRC AT THE ONE POSITION the rest of this read is aimed at. The
        tracked path already offers the CRC nine unconfirmed alignments a cycle
        on the strength of the session's clock, so a tenth is a tenth more
        exposure, and what stands behind it is `FieldMemory`'s own key check and
        its corroboration rule.

        It is the field's own angle that makes a copy worth keeping. Combining
        needs every copy in one cell order and on one set of axes, and where
        the block supplied the angle the level 2 repeats of the 80 m arm of
        2026-09-13 disagreed by up to a half turn -- a sum of those carries
        less than one copy does. Read off the field, all 33 of them correlate
        +0.49 to +0.83 against the copy that decoded.
        """
        start = min(cand, key=lambda st: (abs(st - at), st))
        rot, _ = p3rx.field_rotation(Z, start, path, header.rot, fs=FS)
        cells = {off: p3rx._cells(audio, start + off, path, rot, fs=FS, Z=Z)
                 for off in (-p3rx.CONFIRM_STEP, 0, p3rx.CONFIRM_STEP)}
        if any(c is None for c in cells.values()):
            return None
        field = self.memory.add(cells, path)
        if field is None:
            return None
        self.rotation = rot
        return _packet_event(p3rx.packet_of(field, path, start),
                             tag=", tracked combined")

    _CS_TRACKED_MAX_ERRORS = 1

    ANSWER_TRACK_SYMBOLS = 1
    """Symbols either side of the peer's OWN answer clock that are searched.

    A bracket is a prior written down, and the prior behind a projection from
    the peer's last decoded answer is far sharper than the one behind a lock:
    KB5LZK's 40 m arm of 2026-09-15 answered on a cadence whose residual against
    a straight line is 1.5 ms over thirty clean cycles, so `track_symbols`'s four
    symbols is twenty times the jitter it is absorbing, and every alignment in it
    is another twenty-bit word the code can be talked into.

    Spend the difference on the RADIUS instead, which is what the evidence
    wanted. Both words that arm dropped -- 1 bit error in 20 with channel 5
    faded, and 0 errors on channel 12 with channel 5 faded -- sit at Hamming 1
    from a control signal when the two carriers are summed, which is inside the
    distance-12 alphabet and outside `CS_EXPECTED_MAX_ERRORS`. Measured over the
    same 16,045-sample window the arm read: at four symbols radius 1 fabricates
    a codeword in 4 of 400 windows of white noise, at two symbols 2, and at ONE
    symbol none -- 0 of 400, and 0 of 165 off-answer windows of that arm's own
    band noise, the same clean record radius 0 has at four symbols -- while the
    true answers go from 14 of 31 to 16."""

    def _cs_at_lock(self, audio: np.ndarray, at: int,
                    max_errors: int, *, details: bool = True,
                    track: int | None = None) -> "Event | None":
        # A control signal's alignment is sub-symbol, so the candidates are the
        # sixteenths the blind scan uses -- but over a few symbols either side of
        # the lock, not the whole burst length it has to search back from cold.
        cand = self._candidates(audio, at, _CS_SPAN, SPS // 16, track=track)
        if cand is None:
            return None
        delay = (_matched_filter().size - 1) // 2
        idx = np.arange(cand.start,
                        cand[-1] + spec.CS_BITS_PER_TONE * SPS + 1,
                        SPS // 16) + delay
        Z = _sampled_baseband(audio, HDR_TONES, idx)
        ci, be, st = _best_cs(Z, delay, 0, span=cand)
        # A TRACKED control signal is held to a TIGHTER radius than a swept one,
        # and the reason is the prior rather than caution. A control signal has no
        # CRC -- only a Hamming radius over 20 bits -- so unlike a packet it cannot
        # tell a bad alignment from a good one, and searching a narrow window with
        # a radius wide enough for a cold scan buys fabrication: measured, a lock a
        # few symbols outside this window returned a CONFIDENTLY WRONG codeword
        # rather than a miss, a CS3 five symbols late reporting CS5, which is an
        # invented ARQ command. A hit clears the miss counter, so the wrong lock
        # then sustained itself.
        #
        # A receiver that already knows where the burst is does not need that
        # radius to absorb alignment error, which is what it was really paying for.
        # At radius 1 every fabricated codeword disappears -- the map goes clean to
        # misses either side of the true window -- while a true lock still decodes
        # 8/8 at every SNR from 30 dB down to 3 dB. So the tighter rule costs
        # nothing that was being used, and a miss here is free: it hands the burst
        # back to the blind scan, which takes a global minimum over the whole burst
        # and finds the zero-error truth.
        if be > min(max_errors, self._CS_TRACKED_MAX_ERRORS):
            return None
        return _cs_event(audio, ci, be, st, st / FS, ", tracked", details=details)
