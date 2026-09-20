# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ARDOP receiver: samples in, decoded frames out.

Runs the receive chain the spec (``docs/protocols/ardop/11-WAVEFORM.md`` §7) lays out —
``SearchingForLeader → AcquireSymbolSync → AcquireFrameSync → AcquireFrameType →
DecodeFrameType → AcquireFrame → DecodeFrame`` — over a whole capture:

1. find the two-tone leader (:func:`besra.dsp.detect.leader_start`);
2. lock the 4FSK frame-type header onto the symbol grid and minimum-distance
   decode its 10 tones back to ``(type, session)`` via :func:`besra.frame.frame.decode_header`;
3. demodulate the body for that type — 4FSK tone decode, or differential
   4PSK/8PSK/16QAM on 1/2/4/8 carriers — and hand each carrier block to the byte
   layer (RS + the frame-type-bound CRC) for correction.

The DSP lives in :mod:`besra.dsp.detect`; the byte layer (frame catalog, RS, CRC,
callsign unpacking) in :mod:`besra.frame` / :mod:`besra.fec` / :mod:`besra.crc`.
This module is the seam between them on receive, the mirror of
:mod:`besra.phy.modulator` on transmit.

The tones ARDOP transmits sit at their true audio frequencies here (no sideband
mix), so 4FSK tone index maps straight to its dibit and differential PSK phase is
read without the reference's sign flip. Symbol timing is recovered by locking the
per-carrier block's own CRC/RS — a clean capture admits exactly one alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dsp import detect
from ..frame import callsign
from ..frame import frame as F
from ..fec import rs
from .. import crc
from . import memory
from . import quality as Q

SAMPLE_RATE = 12000
_LEADER_SYMS = 12          # 240 ms default leader
_HEADER_SYMS = 10          # frame-type header, 4FSK 50 baud
_SYM_50 = SAMPLE_RATE // 50   # 240 samples
_HEADER_LEN = _HEADER_SYMS * _SYM_50  # 2400 samples

_SHORT_CONTROL = {"BREAK", "IDLE", "DISC", "END", "ConRejBusy", "ConRejBW",
                  "DATAACK", "DATANAK"}
_BITS_PER_SYMBOL = {F.Mod.PSK4: 2, F.Mod.PSK8: 3, F.Mod.QAM16: 4}

# Absolute floor on the 10-symbol header crispness (Σ per-symbol max/total tone
# power, so ∈ [0, 10]). A clean header sits near the ceiling; random noise that
# happens to satisfy the frame-type parity scores far lower. Rejecting below the
# floor is the receiver's byte-quality gate — ardopcf's bytQualThres (spec §3) —
# and stops noise minting phantom control frames. Calibrated to clear the widest
# gap between the 59 clean fixtures and 12 kHz noise (see test_demod_robustness).
# Clean headers score ~9.99 of a 10 ceiling; noise that satisfies the parity peaks
# near 6.5, so 8.0 sits in the void between them.
_HEADER_CRISP_FLOOR = 8.0
# Acceptance needs corroboration proportional to how little the frame carries. A
# data frame's RS+payload CRC, and a ConReq/ID/Ping's Packed6 callsign + CRC, are
# strong second gates behind the header — 8.0 is enough (a false one dies there).
# A bare control/ack frame (DATANAK/DATAACK/ConAck/PingAck/ConRej/BREAK/…) has no
# callsign and ≤3 body bytes, so a 16-bit header CRC is essentially its only gate,
# and that collides across a long acquisition search. Measured: real such headers
# score ~9.99 (all 59 fixtures); off-air VARA/PACTOR murk that CRC-matched them
# scored 8.58–9.15, while genuine off-air ARDOP ConReqs (callsign-corroborated,
# kept) sat at 9.46–9.94. So bare control frames must clear a high floor; callsign
# and data frames keep 8.0 because their content is the real gate.
_CONTROL_CRISP_FLOOR = 9.5
# An ARQ session in progress changes that arithmetic: the session id we are party
# to is known in advance, and a bare control bearing exactly that id is not a
# frame from nobody — the id match is ~8 bits of corroboration on top of the
# header parity, which is what the strict floor exists to substitute for.
# Measured (logs/onair/20260803T151721Z-besra-7102000.wav): KE8LVA answered 6 of
# 11 ConReqs with ConAck2000 carrying the expected session 0x51, headers reading
# crisp 8.3–9.4 at their true +6 Hz CFO — every one rejected by the 9.5 floor,
# and the connect failed with the answers on tape. That capture's noise never
# exceeded 6.91 over ~845k scanned window positions, so 8.0 keeps the calibrated
# margin over noise while admitting a session-corroborated reply.
_EXPECTED_CONTROL_CRISP_FLOOR = 8.0


#: The smallest payload any rung of the data ladder carries, and so the test that
#: separates a data frame from the controls: ConReq/ID/Ping hold 12 bytes, ConAck
#: and PingAck 3, the bare controls none.
_DATA_K = 16


def _crisp_floor(ftype: int, session: int, expected: int | None) -> float:
    fd = F.FRAMES[ftype]
    if fd.k >= _DATA_K:                                  # a real data frame
        return _HEADER_CRISP_FLOOR
    if fd.name.startswith("ConReq") or fd.name in ("IDFrame", "Ping"):
        return _HEADER_CRISP_FLOOR                       # carries a validatable callsign
    if expected is not None and session == expected:
        return _EXPECTED_CONTROL_CRISP_FLOOR             # session-corroborated control
    return _CONTROL_CRISP_FLOOR                          # bare control/ack: header is all there is

# A leader estimated within this many Hz of centre is treated as on-tune, so the
# zero-offset fixtures decode on the untouched real signal (bit-exact) while a
# genuinely mistuned frame is read on shifted tones.
_CFO_DEADBAND = 3.0

# Acquisition works within bounded regions sized off the 12-symbol default leader,
# and the scan steps past a decoded frame by its known length. Every quantity that
# used to be global — burst onset, carrier-offset window, leader threshold, the
# body transform — is measured on one of these local spans instead.
_LEADER_LEN = _LEADER_SYMS * _SYM_50           # 2880 samples
_SILENCE_GATE = 0.02                            # energy below this × capture peak is silence
_CFO_SPAN = _LEADER_LEN + _SYM_50              # leader region handed to the CFO estimate
_LEAD_SPAN = _LEADER_LEN + _HEADER_LEN         # window the leader edge is pinned within
_BODY_BACK = 12000                             # body-slice margins spanning the timing search
_BODY_FWD = 6000


@dataclass(slots=True)
class DecodedFrame:
    """One decoded frame. ``ok`` is the integrity verdict (RS/CRC for data, majority
    for ConAck/PingAck, RS + callsign validity for ID/ConReq/Ping, always true for a
    bare control frame). ``payload`` is the recovered data bytes; the remaining fields
    are populated for the frame classes that carry them.

    ``quality`` is the 0–100 decode quality of the body (:mod:`besra.phy.quality`) —
    the number this end reports back in a DATAACK/DATANAK and the peer gearshifts on.
    A bare control frame has no body to measure and leaves it None, exactly as the
    reference leaves ``intLastRcvdFrameQuality`` at the previous frame's reading.

    ``header_only`` separates the two things ``ok=False`` used to say at once: a
    frame this receiver accepted and whose body the channel destroyed, versus ten
    tones that satisfied the frame-type parity and were reported because nothing
    better decoded. The second is a guess at a type, not a sighting of one, and
    everything that reads these frames back — the RX log, the monitor — has to be
    able to tell them apart. See :meth:`Demodulator._scan_header`."""
    type: int
    session_id: int
    ok: bool
    payload: bytes = b""
    name: str = ""
    offset: int = 0            # sample index of the frame-type header in the capture
    quality: int | None = None
    header_only: bool = False
    #: The body ran off the end of the capture, so ``quality`` is a sentinel rather
    #: than a reading. It has to be 0 on the wire — `intLastRcvdFrameQuality` is a
    #: standing value in the reference as it is here, so leaving it unset sends the
    #: NAK out carrying the last good frame's score — and 0 is also a legal grade,
    #: which is how five q=0 sightings on 2026-08-26 were read as the demodulator
    #: scoring a 16QAM body it could not follow when nothing had arrived to score.
    truncated: bool = False
    conack_timing_ms: int | None = None
    pingack_sn_db: int | None = None
    pingack_quality: int | None = None
    caller: str | None = None
    target: str | None = None
    grid: str | None = None
    #: Memory-ARQ readings of this decode attempt's carriers — the validated
    #: payload of each that decoded, the soft reading of each that did not. They
    #: ride here rather than straight into :class:`besra.phy.memory.BlockMemory`
    #: because acquisition decodes bodies speculatively and only one attempt
    #: becomes the frame; see :meth:`besra.phy.memory.BlockMemory.keep`.
    soft: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def unverified(self) -> bool:
        """True when ``ok`` is not a verdict: a bare control has no body, so
        :meth:`Demodulator._decode_body` reports it ``ok`` by returning, and the
        ten header tones are the whole of the evidence for it.

        The report is all that can be fixed here, because nothing in the audio
        separates a real bare control from ten tones that satisfy the parity.
        Measured over the eleven 2026-08-23 ARDOP arms: the energy where a body
        would be reads 0.0 dB against the header's own for the 39 real bare
        controls and for the 10 phantoms alike, and the in-band-to-out-of-band
        ratio at the same place tracks the rig's IF filter width rather than
        anything the peer sent. The parity has no rejection power either — ten
        symbols of a steady tone decode to `16QAM.500.100.O sess=0x00`, which is
        the phantom this receiver logged 15 times in one session, and every short
        periodic symbol pattern lands on some frame type.

        So a bare control is corroborated by its repeat and by nothing else, which
        is a rule the protocol layer keeps (`ArqSession._corroborated`); a log that
        renders one the way it renders an RS-and-CRC decode is claiming a check
        that was never run."""
        fd = F.FRAMES.get(self.type)
        return fd is not None and fd.name in _SHORT_CONTROL


def _addressed(frame: DecodedFrame, expected: int | None) -> bool:
    """Whether a frame with nothing behind its header is addressed to the session
    this station is party to — `ArqSession.on_receive`'s own address filter, run
    over the frames whose type is a guess rather than a decode.

    A frame that validated is a sighting whatever id it bears, and a third
    station's traffic on a shared channel is a real thing this receiver should
    report: both 2026-08-26 W6IDS recordings hold `4PSK.200.100 sess=0x05 ok=True`
    frames at quality 62-66 from a QSO that was not ours, four across the two.

    A frame that did not is a type read off ten tones, and the ten tones are also
    where the id came from — so the id is the only corroboration on offer, and
    inside a session its value is known in advance. An uncorroborated claim bearing
    another id is a claim about audio nobody addressed to us: the protocol layer
    already discards exactly those as someone else's traffic, and reporting them
    anyway is the log claiming a check that was never run. Measured across the two
    recordings, replayed whole through the live path with the session's own hint
    and mute: 51 reported frames become 50 and 75 become 72, and the four that go
    are `16QAM.500.100.O sess=0x00` against a session of 0x0d — one graded 23, so
    a reading of q is not what separates them.

    ConReq/Ping/ID are exempt for the reason they are exempt there: they ride a
    forced wire id and are addressed by callsign instead, so the session id says
    nothing about them either way. A guessed one names nobody and the protocol
    layer drops it on that; it survives here as a sighting of a leader and ten
    tones, which is what it is.

    Bandwidth is not the test, though the phantom invites it: `16QAM.500.100` on a
    link running 200 Hz rungs looks impossible and is not. Both those sessions
    negotiated BW500, and the rung in use is the ISS's own choice frame by frame,
    so a 500 Hz frame was legal there throughout and a gate on the ladder's width
    would refuse the peer's next climb.

    ``expected`` is None wherever no session is in progress, and then nothing here
    applies: a listening receiver keeps every sighting it had, which is the point
    of listening.

    Necessary, and not sufficient, which took a keying to establish. The id is not
    independent of the type: both come off the same ten symbols, so a header the
    search mints mints an id with it, and one candidate in 256 mints the session's.
    On 2026-08-28 one did, and a DATANAK went out for it. What the audio can be
    asked instead is `_leader_backed`, in `Demodulator._scan_header`.
    """
    if not (frame.header_only or frame.unverified):
        return True
    if F.FRAMES[frame.type].forces_session:
        return True
    return expected is None or frame.session_id == expected


@dataclass
class RxFunnel:
    """What the acquisition search did, in ardopcf's `LogStats` vocabulary
    (`ARQ.c:2612`), for the stages this receive path has.

    ``leader_detects`` is the two-tone leader signature holding where it was not
    holding a symbol earlier, anywhere in the ±200 Hz of tuning offset a receiver
    must tolerate (:func:`_holds_by_offset`) rather than on the nominal
    1475/1525 Hz bins alone. That search is the row's own, not the walk's: the
    walk triggers on the nominal bins (:meth:`Demodulator._next_leader`) because
    an offset that holds is an acquisition candidate and sixteen extra offsets are
    sixteen extra chances for noise to mint one. So this is a report on what
    reached the receiver rather than a count of what acquisition acted on, and it
    is a trigger rather than the precondition it is in the reference — a leader
    can hold with nothing decoding after it, and a frame can decode from a leader
    too clipped or too soft to hold.

    ``leader_offset`` is where those holds sat when none of them sat on the
    nominal bins. Retuning is the one thing the operator can do that the modem
    cannot, so a mistuned channel says so in Hz rather than leaving it to be read
    off a zero.

    The frame-type rows are one per header scan, as
    `intGoodFSKFrameTypes` and `intFailedFSKFrameTypes` count
    `MinimalDistanceFrameType` (`SoundInput.c:2412`); a candidate is scanned at up
    to six tone shifts here and once there, so these rows run well ahead of the
    detects rather than behind them.

    `LeaderSyncs` and `FrameSyncs` have no counterpart and are not invented: they
    are stages of a state machine walking a stream through one leader, where this
    path pins an edge (`detect.leader_start`, measured against its slice's own
    peak, so a position rather than a detection) and searches header positions
    directly. ``frames`` runs the other way — the reference decodes a stream once
    and has nothing to count, while here a frame lives in some twenty-five
    overlapping windows and is reported from one (`RollingDecoder.DEDUP_S`).

    **No row is a frame that arrived and was lost.** Audio with no frame in it at
    all fails 4700 header scans a minute, because that is the acquisition walk
    working; a real 40 m channel fails 8500. Filtering does not rescue the reading
    either: leader holds overlapping neither a reported frame nor our own PTT look
    like unanswered arrivals and are not — 78% of them are the noise floor itself
    on a bare channel, and a session's rate sits *below* the pooled bare-channel
    rate over 7.25 hours of corpus. The funnel is worth reporting for its shape
    across a session and across bands, and for nothing read off a single row."""

    capture_s: float = 0.0
    leader_detects: int = 0
    leader_peaks: np.ndarray = field(
        default_factory=lambda: np.zeros(_LEADER_SHIFTS.size))
    good_frame_types: int = 0
    failed_frame_types: int = 0
    frames: int = 0

    @property
    def leader_offset(self) -> int | None:
        """The carrier offset the leaders sat at, to the 25 Hz
        `detect.leader_presence_grid` resolves — or None when any of them held on
        the nominal bins, because an offset is worth reporting when it is the
        whole reason a leader is not where the receiver is looking and a channel
        carrying on-tune leaders too has nothing for the operator to correct.

        Which offset is by the presence the leader reached there, and not by how
        often or how long it held: a leader reads strongest at its own offset and
        flickers around the floor at the neighbouring ones, so the neighbours
        collect the *most* holds. On `7102k_065457.wav`, leaders at −195/−159/−198
        Hz, −175 holds 24 times to −200's 19 and reads 0.81 where −200 reads 0.96.
        ``leader_peaks`` is zero at an offset nothing ever held at, so one loud
        window of noise cannot name the channel."""
        if not self.leader_peaks.any() or self.leader_peaks[_LEADER_SHIFTS == 0].any():
            return None
        return int(_LEADER_SHIFTS[self.leader_peaks.argmax()])

    def __str__(self) -> str:
        off = self.leader_offset
        tune = f" ({off:+d} Hz off tune)" if off is not None else ""
        return (f"RX funnel over {self.capture_s:.1f} s: "
                f"LeaderDetects={self.leader_detects}{tune}  "
                f"Good Frame Type Decodes={self.good_frame_types}  "
                f"Failed Frame Type Decodes={self.failed_frame_types}  "
                f"Frames reported={self.frames} "
                f"(search bookkeeping, not frames missed)")


def _angle_diff(a: float, b: float) -> float:
    """Differential phase in milliradians, wrapped to ±π (``ComputeAng1_Ang2``)."""
    d = a - b
    if d < -3142:
        return d + 6284
    if d > 3142:
        return d - 6284
    return d


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One header reading: its crispness, where it starts, the type and session id
    its ten tones spell, and — only when the tones say so — the session id this
    station is party to, offered as the reading a single slipped tone would have
    hidden. See :func:`_slipped_session`."""
    crisp: float
    offset: int
    ftype: int
    session: int
    repaired: int | None


def _slipped_session(col: np.ndarray, ftype: int, session: int,
                     expected: int | None) -> int | None:
    """``expected``, when the four session-carrying tones read as this station's own
    session id with exactly one tone slipped to its runner-up; None otherwise.

    Nothing in the header protects the session id. Both parity symbols are computed
    from the raw type (:func:`besra.frame.frame.header_symbols`), so symbols 5-8 are
    returned XORed with the type and unchecked, and one slipped tone silently
    rewrites two of the eight bits. `ArqSession.on_receive` then reads a foreign
    session and discards the frame — payload, RS, CRC and all — as a stranger's.

    Measured on `logs/onair/20260829T015913Z-besra-3590508.wav` at t=103.075 s, arm
    04 of the 2026-08-29 KN4LQN link: a `4PSK.200.100S.E` decoded `ok=True` at q=60
    bearing `sess=0xa4` against the session's `0xac`. Its ten symbols differ from
    the correct header in symbol 7 alone, read 1 where 3 was second-strongest by
    6.6 dB — the softest reading in the header bar the parity, and everything else
    at 15-27 dB. The peer retransmitted the frame 3.0 s later, it decoded at 0xac
    and q=80, and its payload was byte-identical: the same sixteen bytes of the CMS
    greeting, thrown away once and ACKed once.

    The test is a soft-decision re-read against a hypothesis this end already holds,
    not a search: the type symbols and both parities are left exactly as they
    arrived, so no correction here can turn one frame type into another, and the one
    substituted tone must be the *runner-up* at its symbol rather than any of the
    three alternatives. A stranger's session id has to fall inside a single dibit of
    ours (12 of the 255 that are not ours) and then put our tone second at that
    dibit, which an uncorrelated header does about one time in three.

    That residue is why :meth:`Demodulator._scan_header` spends this only on a frame
    whose body carries its own proof. A bare control's session id is eight bits of
    corroboration that `_crisp_floor`, :func:`_addressed` and the `_scan_header`
    fallback all rest on for want of anything else, and manufacturing the match
    would hollow out all three."""
    if expected is None or session == expected:
        return None
    want = F.header_symbols(ftype, expected)[5:9]
    got = col.argmax(0)
    slipped = [j for j in range(4) if want[j] != int(got[j])]
    if len(slipped) != 1:
        return None
    j = slipped[0]
    return expected if int(np.argsort(-col[:, j])[1]) == want[j] else None


def _header_candidates(au: np.ndarray, lo: int, hi: int, shift: float,
                       expected: int | None = None,
                       cap: int = 8) -> list[_Candidate]:
    """Offsets in ``[lo, hi)`` whose ten 4FSK tones decode to a known frame type
    and clear that type's crispness floor, as :class:`_Candidate` records,
    crispest first, at most one per half symbol of proximity, at most ``cap``.

    Only the header's own span is transformed. The scan reads a few thousand
    window starts and the leader-stepping loop above re-runs it on every
    iteration, so a whole-capture ``tone_mag_series`` computes far more columns
    than are ever indexed. ``shift`` carries the leader's frequency offset into the
    tone set instead of de-rotating the capture, for the reason
    :func:`detect.leader_start` gives."""
    if hi <= lo:
        return []
    tones = tuple(f + shift for f in detect.FSK_TONES[50])
    mags = detect.tone_mag_series(au[lo:hi + _HEADER_LEN], _SYM_50, tones)
    cols = (np.arange(hi - lo)[:, None]
            + _SYM_50 * np.arange(_HEADER_SYMS)[None, :])
    block = mags[:, cols]                                # (4 tones, offsets, 10)
    symbols = np.argmax(block, axis=0)                   # tone index == dibit
    crisp = (block.max(0) / np.maximum(block.sum(0), 1)).sum(1)
    viable = np.flatnonzero(crisp >= _HEADER_CRISP_FLOOR)   # the least any type faces
    out: list[_Candidate] = []
    for i in viable[np.argsort(-crisp[viable], kind="stable")]:
        if len(out) >= cap:
            break
        if any(abs(int(i) + lo - c.offset) < _SYM_50 // 2 for c in out):
            continue
        decoded = F.decode_header([int(s) for s in symbols[i]])
        if decoded is None or decoded[0] not in F.FRAMES:
            continue
        if crisp[i] < _crisp_floor(decoded[0], decoded[1], expected):
            continue
        out.append(_Candidate(float(crisp[i]), lo + int(i), decoded[0], decoded[1],
                              _slipped_session(block[:, i, 5:9], decoded[0],
                                               decoded[1], expected)))
    return out


def _leader_backed(au: np.ndarray, hs: int, shift: float) -> bool:
    """True when the symbol immediately before the header start ``hs`` carries the
    two leader tones *together* — the physical signature that distinguishes a
    leader from 4FSK body symbols, which ring one tone at a time. The score is
    |1475|·|1525| against the strongest single FSK tone bin squared: near 1 for a
    balanced simultaneous pair (measured 0.68 on KE8LVA's channel-smeared off-air
    leader), under 0.01 for a lone body tone. Abutment matters as much as the
    tones: a leader merely somewhere earlier in the span also vouches for 10 body
    symbols of its own frame read as a "header".

    This is the price of admission for a frame whose header is essentially its
    whole content (bare controls, ConAck/PingAck) claimed from the widened
    clipped-leader search: without it, any 10 body symbols of an undecoded frame
    that happen to spell a valid type and clear the crispness floor would mint a
    phantom control frame, trivially "validated" because there is no body to
    check."""
    n = _SYM_50
    lo_edge = hs - n - n // 2       # half a symbol of slack on the early side
    if lo_edge < 0:
        return False
    seg = au[lo_edge:hs]
    lo = np.abs(detect.sliding_bin(seg, n, detect.LEADER_TONES[0] + shift))
    hi = np.abs(detect.sliding_bin(seg, n, detect.LEADER_TONES[1] + shift))
    tones = tuple(f + shift for f in detect.FSK_TONES[50])
    peak = detect.tone_mag_series(seg, n, tones).max(axis=0)  # magnitude²
    score = lo * hi / np.maximum(peak, 1e-9)
    return bool(score.max() >= 0.5)


#: A 20 ms window is leader-like at or above this `detect.leader_presence`, and a
#: leader is *held* where at least `_LEADER_HELD_FRAC` of the next 8 symbols are.
#: Eight, not the nominal twelve, because an ARQ reply routinely loses the front of
#: its leader to the receiving station's own post-TX recovery. Measured on the
#: 2026-08-05 KE8LVA session: every real leader in it — the gateway's and our own —
#: holds 0.25 over 62–90% of its 240 ms, against a channel reading 0.034 at the
#: median and 0.179 at the 90th percentile, and the pair admits 74 walk positions in
#: 179 s where a bare energy walk would take 745.
_LEADER_PRESENCE_FLOOR = 0.25
_LEADER_HELD_SYMS = 8
_LEADER_HELD_FRAC = 0.5

#: The offsets the leader row searches: the ±200 Hz of tuning error a receiver is
#: required to tolerate (spec §4.1/§7), on the grid `detect.leader_presence_grid`
#: reads.
_LEADER_SHIFTS = np.arange(-200, 201, detect.LEADER_SHIFT_HZ)

#: How many ranked candidates a window offers on top of its two event detectors.
#: A cost knob, not a detection threshold — every one is another acquisition
#: attempt. Measured on the 2026-08-05 KE8LVA session: 3 is where the late cluster
#: empties, and 8 buys one more frame for 45% more median window time.
_RANKED_ONSETS = 3


def _leader_hold(au: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How much two-tone leader the next 8 symbols carry, per sample offset, and
    where that clears :data:`_LEADER_PRESENCE_FLOOR` often enough to be a leader.

    ON THE NOMINAL BINS, 1475/1525 Hz, and blind to a channel tuned off them. The
    presence measure has no search in it, so a leader shifted out of those bins
    reads as absence rather than as a weak leader: on `rf-corpus/7102k_065457.wav`
    the first three leaders sit at −195, −159 and −198 Hz and read 0.131, 0.122
    and 0.232 here against a 0.25 floor, while at the estimated shift the same
    three read 0.927, 0.450 and 0.958. What recovers them is not this function —
    it is the acquisition walk, which estimates the offset per candidate
    (`detect.estimate_leader_cfo`), and five clean frames still decode off that
    capture. So a low score here is "no leader in the nominal bins", and any rate
    or hold figure counted off it inherits that scope — the in-session and
    bare-channel rates for "packets we neither ACK nor NAK" were counted off it,
    and describe an arrival on frequency.

    The funnel's leader row is not: it searches the offsets
    (:func:`_holds_by_offset`). This one stays on the nominal bins because what it
    feeds is the walk, where an offset that holds becomes an acquisition attempt
    and a searched offset that holds on noise becomes a frame nobody sent.
    """
    span = _LEADER_HELD_SYMS * _SYM_50
    pres = detect.leader_presence(au)
    if pres.size < span:
        empty = np.zeros(pres.size)
        return empty, empty.astype(bool)
    held = (detect.boxcar((pres >= _LEADER_PRESENCE_FLOOR).astype(np.float64), span)
            >= _LEADER_HELD_FRAC * span)
    return detect.boxcar(pres, span), held


def _rises(hot: np.ndarray, before: bool = False) -> np.ndarray:
    """Where ``hot`` turns on. The walk triggers on the edge and never on the level,
    so a detector already satisfied at the search start (``before``) offers nothing
    until it has gone quiet and come back."""
    return np.flatnonzero(hot & ~np.concatenate(([before], hot[:-1])))


def _holds_by_offset(au: np.ndarray) -> tuple[int, np.ndarray]:
    """How many times the leader signature came up held at *any* offset in
    :data:`_LEADER_SHIFTS`, and the presence it reached at each of the offsets it
    held at — the funnel's leader row, and the answer to the question that row's
    name asks rather than to "was there a leader where we are tuned".

    Counted once per place and not once per offset: no leader sits more than half
    a grid step from a trial offset, which costs it a fifth of its presence, so it
    holds at its neighbours as well. Summed down the search the 2026-08-05 KE8LVA
    session reads 13314 where the same holds counted once read 2379.

    The hold is :func:`_leader_hold`'s, offset by offset: at least
    `_LEADER_HELD_FRAC` of the next `_LEADER_HELD_SYMS` symbols above
    `_LEADER_PRESENCE_FLOOR`, counted where it turns on. Read a symbol at a time
    rather than a sample at a time, which is the reference's own search cadence
    and what makes seventeen offsets cheaper than the one the walk reads at every
    sample.

    Deliberately not wired into the walk. Measured on the 2026-08-05 KE8LVA
    session, which is on frequency: fed to :meth:`Demodulator._next_leader` as an
    acquisition trigger it reported four frames nobody sent — two of them
    `DATANAK`s on a session nobody held, the failure `_CONTROL_CRISP_FLOOR`
    exists to refuse — and put 24% more failed header scans through the funnel.
    Requiring the hold at one offset keeps noise out of the count (8 s of it
    scores zero holds at all seventeen), but every offset a walk is handed is an
    acquisition attempt, and the noise only has to win one of them."""
    pres = detect.leader_presence_grid(au, _LEADER_SHIFTS, _SYM_50)
    if pres.shape[-1] < _LEADER_HELD_SYMS:
        return 0, np.zeros(_LEADER_SHIFTS.size)
    held = (detect.boxcar((pres >= _LEADER_PRESENCE_FLOOR).astype(np.float64),
                          _LEADER_HELD_SYMS) >= _LEADER_HELD_FRAC * _LEADER_HELD_SYMS)
    return _rises(held.any(axis=0)).size, np.where(held.any(axis=1), pres.max(axis=1), 0.0)


def _ranked_onsets(score: np.ndarray) -> np.ndarray:
    """The window's most leader-like places, weak ones included, in stream order.

    Both event detectors can go silent for a whole window — see
    :meth:`Demodulator._next_onset` and :meth:`Demodulator._next_leader` — and a
    walk with nothing left to try stops six seconds early. This is what remains
    when neither fires: a *rank* rather than a floor, so it is never empty, and
    bounded by `_RANKED_ONSETS` rather than by how loud the channel is."""
    keep: list[int] = []
    work = np.asarray(score, dtype=np.float64).copy()
    for _ in range(_RANKED_ONSETS):
        if not work.size or work.max() <= 0.0:
            break
        i = int(np.argmax(work))
        keep.append(i)
        work[max(0, i - _LEADER_LEN):i + _LEADER_LEN] = 0.0
    return np.array(sorted(keep), dtype=int)


def _body_span(ftype: int) -> int:
    """Samples from the body start to the end of the frame body, from the frame
    type alone — leader and header excluded. Stepping the scan cursor past a
    decoded frame by this known length skips it cleanly, where chasing trailing
    energy overshoots into a following frame's leader when the gap is short."""
    fd = F.FRAMES[ftype]
    name = fd.name
    if name in _SHORT_CONTROL:
        return 0
    if name.startswith("ConAck") or name == "PingAck":
        return 12 * _SYM_50
    if name == "IDFrame" or name.startswith("ConReq") or name == "Ping":
        return 16 * 4 * _SYM_50
    if fd.mod is F.Mod.FSK4:
        sps = SAMPLE_RATE // fd.baud
        parts = 3 if ftype in (0x7A, 0x7B) else 1
        block_bytes = fd.k // parts + fd.r // parts + 3
        return parts * block_bytes * 4 * sps
    block_bytes = fd.k + fd.r + 3
    nsym = block_bytes * 8 // _BITS_PER_SYMBOL[fd.mod]
    return (nsym + 1) * 120


def frame_span(ftype: int) -> int:
    """Samples from a frame's header to its last body sample — its whole extent
    measured from ``DecodedFrame.offset``, and known the moment the header decodes.
    The receive path's hold-back is built on that: a frame whose span has arrived is
    one the newest window edge cannot have cut."""
    return _HEADER_LEN + _body_span(ftype)


class Demodulator:
    """Decode ARDOP frames from a 12 kHz int16 (or float) sample stream.

    ``expect_session`` is an optional zero-argument callable returning the session
    id the ARQ layer is currently party to (or None). It is read once per
    ``decode`` pass and admits session-matching bare controls at the lower
    ``_EXPECTED_CONTROL_CRISP_FLOOR``; without it every bare control faces the
    strict unsolicited floor, exactly as before.

    ``rx_epoch`` is the other thing only the ARQ layer knows: a counter that moves
    whenever a repeat of a frame type would no longer be a repeat of the same bytes
    — this station entering the IRS role, or the session ending. Memory ARQ
    (:mod:`besra.phy.memory`) is discarded when it moves. Without it the memory
    still resets on every frame-type and session change it can see for itself, and
    still places each arrival in the stream so that re-reading one is not a repeat
    of it."""

    def __init__(self, leader_threshold: float = 0.3, expect_session=None,
                 rx_epoch=None):
        self._leader_threshold = leader_threshold
        self._expect_session = expect_session
        self._rx_epoch = rx_epoch
        self._next_at = 0
        self.memory = memory.BlockMemory()
        self.funnel = RxFunnel()

    # -- top-level scan -----------------------------------------------------

    def decode(self, samples, at: int | None = None) -> list[DecodedFrame]:
        """Every frame found in ``samples``. Frames are located by their leaders, so
        a capture may hold several back to back (or one, as the fixtures do).

        ``at`` is where these samples begin in the received stream, which memory
        ARQ needs to tell a repeat from a re-reading of the same arrival. Left
        unsaid, each pass is taken to follow the last, which is what a caller
        handing over consecutive blocks means; the overlapping windows of
        :class:`besra.radio.RollingDecoder` re-read the same audio and say so."""
        au = np.asarray(samples, dtype=np.float64)
        at = self._next_at if at is None else at
        self._next_at = at + au.size
        energy = detect.boxcar(au * au, _SYM_50)
        peak = float(energy.max()) if energy.size else 0.0
        score, held = _leader_hold(au)
        detects, peaks = _holds_by_offset(au)
        self.funnel.leader_detects += detects
        self.funnel.leader_peaks = np.maximum(self.funnel.leader_peaks, peaks)
        ranked = _ranked_onsets(score)
        expected = self._expect_session() if self._expect_session is not None else None
        self.memory.epoch(self._rx_epoch() if self._rx_epoch is not None else 0)
        self.memory.stream(at)
        out: list[DecodedFrame] = []
        cursor = 0
        while cursor < au.size - _HEADER_LEN:
            frame = self._acquire_header(au, energy, peak, held, ranked, cursor, expected)
            if frame is None:
                break
            # Every frame that is not a short control participates in the
            # memory-ARQ key, exactly as `LastDataFrameType` does in the reference:
            # a ConAck or a rung of the rate ladder arriving between repeats means
            # the next frame of the old type is a new block, not another copy.
            if frame.name not in _SHORT_CONTROL:
                self.memory.keep(frame)
            # Reported, not received: an unaddressed claim (`_addressed`) still
            # spends its span off the cursor and still keys memory ARQ, so
            # acquisition walks the capture exactly as it did before and only the
            # report changes.
            if _addressed(frame, expected):
                out.append(frame)
            cursor = frame.offset + _HEADER_LEN + _body_span(frame.type)
        return out

    # -- leader + frame-type header ----------------------------------------

    def _acquire_header(self, au: np.ndarray, energy: np.ndarray, peak: float,
                        held: np.ndarray, ranked: np.ndarray, cursor: int,
                        expected: int | None) -> DecodedFrame | None:
        """Locate the next frame at or after ``cursor`` — leader, then header, then
        body — returning the decoded frame (``offset`` set) or None when no leader
        remains.

        Every step works on a bounded region around the candidate frame — the burst
        onset, the carrier-offset window, and the leader threshold are all measured
        locally — so an earlier weak frame or a later strong one elsewhere in the
        capture cannot skew them. ``shift`` carries the leader's frequency offset
        (beyond the on-tune deadband) into the downstream tone/carrier detectors,
        the same device :func:`detect.leader_start` uses to read a mistuned leader
        without a capture-length de-rotation.

        The header sits at most 12 symbols after the visible leader edge — *at
        most*, because an ARQ reply's leader routinely loses its front to the
        receiving station's own post-TX recovery (measured: KE8LVA's ConAck2000
        answers arrived ~80 ms clipped, every cycle), and a frame whose leader is
        clipped by N ms carries its header N ms earlier than the full-leader
        geometry says. So the header is searched from one symbol after the leader
        edge out to the full-leader position, and a candidate is accepted on the
        body's own verdict: the crispest header whose body validates (RS/CRC,
        majority, callsign) wins. Bare control frames have no body, so their
        acceptance stays what it always was — the high crispness floor — and when
        nothing validates the crispest candidate at the full-leader position is
        reported unvalidated, exactly the pre-widening behaviour. A leader with no
        acceptable header — a false trigger in a trailer or in noise — is stepped
        over, not treated as the end of the capture.

        The walk has three independent detectors — :meth:`_next_onset`,
        :meth:`_next_leader` and :meth:`_next_ranked` — and tries the earliest of
        them. Asking the leader signature only where energy found nothing was not
        enough: the silence gate
        is a fraction of the *window's* peak, so one loud burst raises it over a
        weaker earlier frame, the energy walk "succeeds" past that frame, and the
        leader walk is never consulted. Measured on the shipped KE8LVA greeting
        fixture with 0.2 s of noise appended after it — at the recording's own
        level the greeting decodes, at ten times the level it decodes nothing, and
        disabling the energy walk brings it back. 203 of the 690 windows in that
        session (29.4%) hold more than one energy rise, so a walk that can pick
        the wrong one is the ordinary case rather than an edge.

        No detector is authoritative, so a candidate that yields nothing steps the
        walk on by a leader length *or* to another detector's candidate, whichever
        is nearer: a spurious leader signature in a frame's tail used to step
        straight over the following frame's real leader.

        The two event detectors can both fall silent for a whole window — a band
        with traffic never re-crosses the silence gate, and a leader softened by
        the channel never reaches the presence floor — and a walk out of candidates
        stops where it stands. :meth:`_next_ranked` is what it falls back on, and
        the reference implementation is why: ardopcf runs a fresh two-tone test
        every 20 ms hop for as long as the audio lasts, with no arming condition at
        all (`SearchFor2ToneLeader3`, one attempt per 240-sample hop, forever), so
        it can never be blinded for a window the way an edge-triggered walk can.

        None of it is free. Measured through the live path on the 2026-08-05 KE8LVA
        session (571 windows): a 6.25 s window costs 52 ms at the median and 208 ms
        at the 90th, against 60 and 247 for the two-detector walk over the same
        audio — cheaper despite the extra attempts, because `_header_candidates` no
        longer sorts the whole window. Both figures are inside the 0.25 s step."""
        search = cursor
        while search < au.size - _HEADER_LEN:
            onsets = sorted({c for c in (self._next_onset(energy, peak, search),
                                         self._next_leader(held, search),
                                         self._next_ranked(ranked, search)) if c >= 0})
            if not onsets:
                return None
            onset = onsets[0]
            cfo = detect.estimate_leader_cfo(au[onset:onset + _CFO_SPAN])
            est = 0.0 if abs(cfo) < _CFO_DEADBAND else cfo

            # The estimate is aliased when the leader's front is clipped (see
            # `detect.estimate_leader_cfo`), so a failed acquisition retries near
            # zero: a reply mangled by our own post-TX recovery comes from a
            # station we are already netted to, and the small grid exists because
            # the crispness floors sit inside the sensitivity band of a ~5 Hz
            # reading error (measured on KE8LVA's off-air ConAck2000: crisp 9.69
            # read at its true +5 Hz, under the 9.5 floor read on-tune).
            fallback = None
            last_ls = -1
            for shift in dict.fromkeys((est, 0.0, 5.0, -5.0, 10.0, -10.0)):
                got = self._frame_at(au, onset, shift, expected,
                                     legacy=shift in (est, 0.0))
                if got is None:
                    continue
                frame, ls = got
                last_ls = max(last_ls, ls)
                if frame is not None and frame.ok:
                    return frame
                if fallback is None and frame is not None:
                    fallback = frame
            if fallback is not None:
                return fallback
            stepped = (last_ls + _LEADER_LEN if last_ls >= 0
                       else onset + _LEADER_LEN)  # step past the false leader
            search = min([stepped, *onsets[1:]])  # …but not past the untried one
        return None

    def _frame_at(self, au: np.ndarray, onset: int, shift: float,
                  expected: int | None,
                  legacy: bool) -> tuple[DecodedFrame | None, int] | None:
        """One acquisition attempt at a given tone shift: pin the leader edge near
        ``onset``, scan the header window, validate candidates by their bodies.
        None when no leader edge is found; otherwise ``(frame, leader_start)``
        where ``frame`` is the body-validated winner, or (``legacy`` only) the
        crispest unvalidated candidate at the full-leader position, or None when
        nothing decodes."""
        lead_lo = max(0, onset - _SYM_50)
        ls = detect.leader_start(au[lead_lo:onset + _LEAD_SPAN],
                                 self._leader_threshold, shift)
        if ls < 0:
            return None
        ls += lead_lo
        frame = self._scan_header(au, ls, shift, expected, legacy)
        if frame is None:
            self.funnel.failed_frame_types += 1
        else:
            self.funnel.good_frame_types += 1
        return frame, ls

    def _scan_header(self, au: np.ndarray, ls: int, shift: float,
                     expected: int | None, legacy: bool) -> DecodedFrame | None:
        """Scan the header window for a leader edge at ``ls``: body-validated
        candidates win; failing those — on the estimate shift only, the surface
        the pre-clipped-leader receiver had — the crispest candidate at the
        full-leader position, or bearing the session id this end is party to, is
        reported as today's possibly-not-ok frame. The retry shifts get no such
        fallback: a bare control is trivially "ok", and letting one through from a
        retry shift's narrow window minted phantom DATAACK/DATANAK frames out of
        an undecoded frame's body.

        The position gate stands in for corroboration, and in standing in for it
        assumes a 240 ms leader. KE8LVA leads its ARQ data with ~168 ms: its
        header sits 2013 samples past the leader edge in the shipped 2026-08-05
        greeting fixture and 1985-2058 across the thirteen bursts of
        ``logs/onair/20260809T231613Z``. So the gate threw away a frame the scan
        had already found — the true 4PSK.500.100.O header was the *crispest*
        candidate in all thirteen, and every one was reported instead as a
        4PSK.200.100.E at the nominal position, carrying a session id that was not
        the session's. A candidate bearing the expected session id is corroborated
        by eight bits that owe nothing to where it sits, which is what the gate is
        substituting for.

        A frame whose header is its whole content must show leader ahead of it here
        for the same reason it must in the validated loop above — `_leader_backed`
        is the only physical corroboration a bare control has, and the fallback path
        is where it is needed most. *Every* frame reported from this loop is one:
        `header_only` is the statement that nothing behind the header agreed with
        it, so the `k < 12` exemption the validated loop grants — RS and payload CRC
        will refuse a false data frame — has nothing left to grant here, and the
        loop ran without the test for the frames most able to walk in. On
        2026-08-28 a `16QAM.500.100.E` was minted 0.52 s into the body of a
        `4FSK.200.50S.E` W6IDS was still transmitting, out of a rolling window that
        opened after that frame's leader, and it carried this station's own session
        id because a header the search mints mints its id too. It drew a DATANAK
        1 ms later, on top of the gateway. The leader is what the audio can be
        asked for and what it does not have: a body's 4FSK symbols ring one tone at
        a time, and a window that opens mid-frame has nothing at all before the
        header it claims.

        An uncorroborated fallback faces the bare-control floor, whatever kind of
        frame it claims to be. `_crisp_floor` lets a data frame in at 8.0 because
        "a false one dies" on its RS and payload CRC — but this loop runs only
        after that second gate has failed or does not exist, so the header is once
        again all there is and `_CONTROL_CRISP_FLOOR` is the floor written for
        exactly that. Measured on the 2026-08-05 KE8LVA session: the three
        `16QAM.500.100.O` phantoms scored 8.64, 8.21 and 8.21 — over the data
        floor, under the control floor — while the real `4PSK.500.100.O` headers
        under them read 9.59 on the on-tune grid and carry the session id besides.

        A frame reported from here is marked `header_only`: it is a type this
        receiver could not confirm by anything except the parity of ten tones, and
        a log that renders it the same as a decode turns one receiver's guess into
        a sighting of a frame nobody sent.

        The validated loop is also the only place a slipped session tone is put
        back (`_slipped_session`), and it is put back only for a data frame — one
        whose RS and frame-type-bound CRC have just vouched for the body behind the
        header. A control frame reaching that line is `ok` by having no body to
        check, so correcting its id would substitute a guess for the eight bits of
        corroboration the id is carrying, and the fallback loop below never
        corrects at all: `header_only` says nothing agreed with the header, which
        is when the raw id is the only evidence there is."""
        span = au.size - _SYM_50 + 1
        nominal = ls + _LEADER_LEN
        lo = max(0, ls + _SYM_50)
        hi = min(span - _HEADER_LEN, nominal + 170)
        cands = _header_candidates(au, lo, hi, shift, expected)
        for c in cands:
            # Frames whose header is essentially their whole content must show
            # leader ahead of the header to be claimed outside the narrow
            # full-leader window — see `_leader_backed`. Data/station frames
            # carry their own proof (RS, CRC, callsign) and need none.
            if F.FRAMES[c.ftype].k < 12 and not _leader_backed(au, c.offset, shift):
                continue
            frame = self._decode_body(au, c.offset, c.ftype, c.session, shift)
            if frame.ok:
                if c.repaired is not None and F.FRAMES[c.ftype].k >= _DATA_K:
                    frame.session_id = c.repaired
                return frame
        if not legacy:
            return None
        for c in cands:
            if c.session != expected and c.crisp < _CONTROL_CRISP_FLOOR:
                continue
            if not _leader_backed(au, c.offset, shift):
                continue
            if c.offset >= nominal - 170 or c.session == expected:
                frame = self._decode_body(au, c.offset, c.ftype, c.session, shift)
                frame.header_only = True
                return frame
        return None

    @staticmethod
    def _next_onset(energy: np.ndarray, peak: float, start: int) -> int:
        """First rising edge through the silence gate at or after ``start``. Energy
        already above the gate at ``start`` is the previous frame's trailer, not a
        new burst, so it is passed over: acquisition triggers only where energy
        climbs out of silence. Gating against the capture peak (not a fraction of a
        local peak that a strong neighbour sets) is what keeps a weaker frame's
        onset from being skipped, and triggering on the rise — not on any hot
        sample — is what stops a decoded frame's trailer from re-acquiring just past
        its body when the next frame sits close behind."""
        gate = _SILENCE_GATE * peak
        if gate <= 0.0 or start >= energy.size:
            return -1
        rises = _rises(energy[start:] >= gate,
                       bool(start > 0 and energy[start - 1] >= gate))
        return start + int(rises[0]) if rises.size else -1

    @staticmethod
    def _next_leader(held: np.ndarray, start: int) -> int:
        """First place at or after ``start`` where the two-tone leader signature is
        held — the acquisition walk's second opinion, of equal standing to
        :meth:`_next_onset`.

        `_next_onset` triggers on energy climbing out of silence, and a band with
        traffic on it never goes back down: its noise floor alone sits above 2% of
        the loudest frame in the window, so there is exactly one rise — the window's
        own first sample — and acquisition that fails there is blind for the
        remaining six seconds. Measured on the 2026-08-05 KE8LVA session: the
        greeting frame at 122.10 s was found only in the last rolling window that
        could still hold it whole, 1.83 s after its final sample, and the answer
        went out 2.24 s after the frame ended, onto a gateway already transmitting.

        The leader's own signature is what remains when energy says nothing, and it
        is scale-free (`detect.leader_presence`), so it needs no quiet reference.

        Two detectors are not two chances on every frame, and the receive lateness
        is not spent. This one has an absolute floor, and a leader that does not
        reach it gets no candidate from here. Measured on the live path over that
        same session — 0.1 s blocks with our own keyed stretches dropped, the audio
        `RadioAudio._capture` actually delivers — the gateway's 20 data frames split
        in two: 9 report within 0.40 s of their last sample, 11 report at 1.80-2.30
        s, and nothing lands between. The leader ahead of each says why. Presence
        peaks 0.42-0.84 (median 0.69) on the nine, and 0.17-0.62 (median 0.28) on
        the eleven, seven of them under this 0.25 floor outright and the rest short
        of the 8-symbol hold `_leader_hold` asks for on top of it. Both readings
        are on the nominal bins, which is a scope and not a bug here — that
        session is on frequency — but it is why these figures do not transfer to
        a mistuned capture; see `_leader_hold`. With no candidate
        from here and only the window's own first sample from `_next_onset`, the two
        of them together leave the walk one attempt per window, so such a frame is
        acquired only once the window's first sample *is* its leader: the last window
        that can still hold it whole, `RollingDecoder.OVERLAP_S + STEP_S` less the
        frame's own span, which for these 4.17 s repeats is the 2.08 s the late
        figures cluster at. :meth:`_next_ranked` is the third detector that keeps
        the walk from ending there.

        The floor is not the thing to move, and that is measured too. At 0.18 it
        mints six `DATANAK`s on a session nobody holds, spaced on the 3.8 s grid of
        our own ConReqs — bare controls read out of undecoded bodies, exactly what
        `_CONTROL_CRISP_FLOOR` refuses. What answers the blindness is not another
        threshold on this measurement but :func:`_ranked_onsets`, which offers the
        window's best remaining candidates whether or not any floor is cleared."""
        if start >= held.size:
            return -1
        rises = _rises(held[start:], bool(start > 0 and held[start - 1]))
        return start + int(rises[0]) if rises.size else -1

    @staticmethod
    def _next_ranked(ranked: np.ndarray, start: int) -> int:
        """The next of :func:`_ranked_onsets`' candidates at or after ``start`` —
        the walk's third detector, and the only one that cannot fall silent."""
        i = int(np.searchsorted(ranked, start))
        return int(ranked[i]) if i < ranked.size else -1

    # -- body dispatch ------------------------------------------------------

    def _decode_body(self, au: np.ndarray, header_start: int, ftype: int,
                     session: int, shift: float):
        fd = F.FRAMES[ftype]
        body_start = header_start + _HEADER_LEN
        base = DecodedFrame(type=ftype, session_id=session, ok=True, name=fd.name,
                            offset=header_start)

        if fd.name in _SHORT_CONTROL:
            return base

        # Read the body from a bounded slice around it (with margin for the timing
        # search), at tones/carriers shifted by the leader offset — never a
        # capture-length transform nor a capture-length de-rotation. Reading a
        # shifted bin of the untouched signal is the same measurement as
        # de-rotating and reading the nominal bin, minus the full-length array.
        lo = max(0, body_start - _BODY_BACK)
        hi = min(au.size, body_start + _body_span(ftype) + _BODY_FWD)
        work = au[lo:hi]
        off = body_start - lo

        if fd.name.startswith("ConAck"):
            return self._decode_conack(work, off, shift, ftype, base)
        if fd.name == "PingAck":
            return self._decode_pingack(work, off, shift, ftype, base)
        if fd.name == "IDFrame":
            return self._decode_station(work, off, shift, ftype, base, is_id=True)
        if fd.name.startswith("ConReq") or fd.name == "Ping":
            return self._decode_station(work, off, shift, ftype, base, is_id=False)

        # Data frames.
        if fd.mod is F.Mod.FSK4:
            return self._decode_fsk_data(work, off, shift, ftype, base)
        return self._decode_psk_data(work, off, shift, ftype, base)

    # -- 4FSK helpers -------------------------------------------------------

    def _fsk_symbol_series(self, work: np.ndarray, baud: int,
                           shift: float) -> np.ndarray:
        sps = SAMPLE_RATE // baud
        tones = tuple(f + shift for f in detect.FSK_TONES[baud])
        return detect.tone_mag_series(work, sps, tones)

    @staticmethod
    def _fsk_tones(mags: np.ndarray, start: int, sps: int, nbytes: int) -> np.ndarray | None:
        """The four tone magnitudes of each of ``4·nbytes`` symbols — the raw material
        both the byte decision and the decode-quality measurement read."""
        idx = start + sps * np.arange(4 * nbytes)
        if idx[-1] >= mags.shape[1]:
            return None
        return mags[:, idx]

    def _decode_conack(self, work, off, shift, ftype, base):
        mags = self._fsk_symbol_series(work, 50, shift)
        sym = self._fsk_tones(mags, off, _SYM_50, 3)
        timing = _majority(_fsk_bytes(sym)) if sym is not None else None
        base.ok = timing is not None
        if sym is not None:
            base.quality = Q.fsk_quality(sym)
        if timing is not None:
            base.conack_timing_ms = 10 * timing
        return base

    def _decode_pingack(self, work, off, shift, ftype, base):
        mags = self._fsk_symbol_series(work, 50, shift)
        sym = self._fsk_tones(mags, off, _SYM_50, 3)
        val = _majority(_fsk_bytes(sym)) if sym is not None else None
        base.ok = val is not None
        if sym is not None:
            base.quality = Q.fsk_quality(sym)
        if val is not None:
            base.pingack_sn_db = ((val & 0xF8) >> 3) - 10
            base.pingack_quality = (val & 7) * 10 + 30
        return base

    def _decode_station(self, work, off, shift, ftype, base, is_id: bool):
        """ID / ConReq / Ping: a 12-data + 4-RS block, no CRC (RS is the check)."""
        mags = self._fsk_symbol_series(work, 50, shift)
        base.ok = False
        nominal = self._fsk_tones(mags, off, _SYM_50, 16)
        for delta in range(-40, 44, 4):
            sym = self._fsk_tones(mags, off + delta, _SYM_50, 16)
            if sym is None:
                continue
            data, ok = rs.rs_correct(_fsk_bytes(sym), 4)
            if not ok:
                continue
            try:
                base.caller = callsign.unpack_callsign(data[:6])
                if is_id:
                    base.grid = callsign.unpack_grid(data[6:12])
                else:
                    base.target = callsign.unpack_callsign(data[6:12])
            except ValueError:
                continue
            base.ok = True
            nominal = sym
            break
        if nominal is not None:
            base.quality = Q.fsk_quality(nominal)
        return base

    def _decode_fsk_data(self, work, off, shift, ftype, base):
        fd = F.FRAMES[ftype]
        sps = SAMPLE_RATE // fd.baud
        mags = self._fsk_symbol_series(work, fd.baud, shift)

        # Only the long 600-baud frame (0x7A/0x7B) splits into three sub-blocks.
        parts = 3 if ftype in (0x7A, 0x7B) else 1
        pk, pr = fd.k // parts, fd.r // parts
        block_bytes = pk + pr + 3
        payload = bytearray()
        ok_all = True
        errors = 0
        recovered = False
        truncated = False
        measured: list[np.ndarray] = []
        for part in range(parts):
            got = None
            soft = None
            base_start = off + part * block_bytes * 4 * sps
            for delta in range(-40, 24, 2):
                sym = self._fsk_tones(mags, base_start + delta * sps, sps, block_bytes)
                if sym is None:
                    continue
                raw = _fsk_bytes(sym)
                data, err = _rs_accept(raw, pk, pr, ftype)
                if data is not None:
                    got, errors = data, errors + err
                    self.memory.keep_good(base, part, got)
                    measured.append(sym)
                    break
                # The alignment memory ARQ keeps: the one whose symbols point
                # hardest at a single tone, the 4FSK reading of the same "tightest
                # on the constellation" the PSK path locks its timing by.
                crispness = float((sym.max(0) / np.maximum(sym.sum(0), 1e-12)).mean())
                if soft is None or crispness > soft[0]:
                    soft = (crispness, sym)
            else:
                # A part that no alignment read still tells the peer something: read
                # its quality off the nominal alignment so the NAK carries a
                # measurement. This one is graded on what arrived now, not on the
                # average — a block memory ARQ lifted was still a weak arrival, and
                # the peer's ladder needs to hear that.
                sym = self._fsk_tones(mags, base_start, sps, block_bytes)
                if sym is None:
                    truncated = True
                else:
                    measured.append(sym)
                got = self.memory.good(base, part)
                recovered = recovered or got is not None
                if got is None and soft is not None:
                    avg = self.memory.fsk(base, part, soft[1])
                    if avg is not None:
                        data, err = _rs_accept(_fsk_bytes(avg), pk, pr, ftype)
                        if data is not None:
                            got, errors, recovered = data, errors + err, True
                            self.memory.keep_good(base, part, got)
            if got is None:
                ok_all = False
                got = b""
            payload += got
        base.ok = ok_all
        base.payload = bytes(payload)
        # A frame the capture cut short reports nothing arrived, whichever of its
        # parts fell off the edge. On the three-part 600-baud frame the earlier
        # parts do fit, so `measured` is non-empty and grading it grades a prefix —
        # which is what the PSK path's own note calls no better: a cut 4FSK.2000.600
        # scored 95 off its first part while the same frame complete but weak reads
        # 23-29 at -3 dB and 13 at -9 dB, so truncation outranked every real channel.
        base.truncated = truncated or not measured
        base.quality = (0 if base.truncated else
                        Q.fsk_quality(np.concatenate(measured, axis=1)))
        # `fd.r` and `fd.carriers`, not the per-part `pr` and the part count: the
        # reference sums RS errors over all three parts of a 600-baud frame and
        # compares against intNumCar=1, intRSLen=150 (ARDOPC.c case 0x7a), so its
        # bar is 36 errors, not the 35 that (parts=3, pr=50) computes. A part
        # memory ARQ supplied was not read off this arrival, so a frame carrying
        # one does not claim the floor at all — see the PSK path.
        if not recovered:
            _apply_rs_floor(base, errors, fd.carriers, fd.r)
        return base

    # -- PSK / QAM ----------------------------------------------------------

    def _decode_psk_data(self, work, off, shift, ftype, base):
        fd = F.FRAMES[ftype]
        cc = fd.carriers
        bps = _BITS_PER_SYMBOL[fd.mod]
        block_bytes = fd.k + fd.r + 3
        nsym = block_bytes * 8 // bps

        # All carriers are sampled at the same instants, so timing is one shared
        # offset. Lock it on carrier 0 by how tightly its differential phases fall on
        # the constellation grid — a cheap search that avoids RS-decoding every offset.
        cbin0 = detect.carrier_bin_series(work, _carrier_freq(0, cc) + shift)
        step = 1571.0 if fd.mod is F.Mod.PSK4 else 785.4  # 90° or 45° in mrad
        centre = self._lock_timing(cbin0, off, nsym, step)
        if centre is None:
            # Timing locks on nothing only when the body ran off the end of the
            # capture. Leaving `quality` unset there hands the ARQ layer no
            # reading, and `intLastRcvdFrameQuality` is a *standing* value in the
            # reference as it is here — so the NAK for a body that never arrived
            # went out carrying the last good frame's score (measured: a 96, then
            # a DATANAK 0x1D, which reads back as 96). It reports nothing, because
            # nothing is what arrived; grading the prefix instead is no better,
            # since the padding past a capture's edge is phase-perfect and scores
            # 100. Truncation is the characteristic failure at the top of the
            # gearshift ladder, which is where an honest number matters most.
            base.ok = False
            base.truncated = True
            base.quality = 0
            return base

        payload = bytearray()
        ok_all = True
        errors = 0
        recovered = False
        symbols = None
        for car in range(cc):
            cbin = (cbin0 if car == 0 else
                    detect.carrier_bin_series(work, _carrier_freq(car, cc) + shift))
            got = None
            soft = None
            for delta in (centre, *range(centre - 3, centre + 4)):
                read = self._psk_block(cbin, off + delta, nsym, fd.mod)
                if read is None:
                    continue
                block, dphase, mag = read
                # Quality comes off the last carrier, as it does in the reference —
                # from the alignment that validated, or the nominal one if none did.
                if car == cc - 1 and symbols is None:
                    symbols = (dphase, mag)
                data, err = _rs_accept(block, fd.k, fd.r, ftype)
                if data is not None:
                    got, errors = data, errors + err
                    self.memory.keep_good(base, car, got)
                    if car == cc - 1:
                        symbols = (dphase, mag)
                    break
                # What memory ARQ remembers of a carrier that would not read: the
                # alignment whose phases sit tightest on the constellation, since a
                # repeat is only worth averaging against this one's best reading.
                fit = _grid_err(dphase, step)
                if soft is None or fit < soft[0]:
                    soft = (fit, dphase, mag)
            if got is None:
                # A carrier a previous repeat already validated is finished, and a
                # frame whose carriers arrive correct on different repeats is
                # assembled out of both (the reference's `CarrierOk`).
                got = self.memory.good(base, car)
                recovered = recovered or got is not None
            if got is None and soft is not None:
                avg = self.memory.psk(base, car, soft[1], soft[2])
                if avg is not None:
                    block = _psk_decide(avg[0], avg[1], fd.mod)
                    data, err = _rs_accept(block, fd.k, fd.r, ftype)
                    if data is not None:
                        got, errors, recovered = data, errors + err, True
                        self.memory.keep_good(base, car, got)
            if got is None:
                ok_all = False
                got = b""
            payload += got
        base.ok = ok_all
        base.payload = bytes(payload)
        if symbols is not None:
            dphase, mag = symbols
            base.quality = (Q.qam_quality(dphase, mag, step) if fd.mod is F.Mod.QAM16
                            else Q.psk_quality(dphase, step))
            # The RS-clean floor speaks for the frame that just arrived, so a frame
            # any of whose carriers came out of memory does not get it: with every
            # carrier answered from the cache, `errors` is zero because nothing was
            # read, and reporting the peer an 80 for a body we could not read is
            # exactly the invitation up the ladder the quality number exists to
            # withhold. What arrived is graded on its own constellation.
            if not recovered:
                _apply_rs_floor(base, errors, cc, fd.r)
        return base

    @staticmethod
    def _lock_timing(cbin, off, nsym, step) -> int | None:
        """Symbol-timing offset that best fits the differential phases to the
        constellation grid (mean distance to the nearest ``step`` multiple)."""
        best, best_err = None, None
        for delta in range(-80, 20):
            idx = off + delta + 120 * np.arange(nsym + 1)
            if idx[-1] >= cbin.size or off + delta < 0:
                continue
            phase = 1000.0 * np.angle(cbin[idx])
            dphase = np.diff(phase)
            dphase -= 6284.0 * np.round(dphase / 6284.0)
            err = _grid_err(dphase, step)
            if best_err is None or err < best_err:
                best_err, best = err, delta
        return best

    def _psk_block(self, cbin, base, nsym, mod):
        """One carrier's raw block: reference symbol then ``nsym`` differential
        symbols, as ``(bytes, differential phases, magnitudes)`` — the phases and
        magnitudes are what :mod:`besra.phy.quality` measures the decode on."""
        idx = base + 120 * np.arange(nsym + 1)
        if idx[-1] >= cbin.size or base < 0:
            return None
        vec = cbin[idx]
        phase = 1000.0 * np.arctan2(vec.imag, vec.real)
        mag = np.abs(vec)
        dphase = [_angle_diff(phase[i + 1], phase[i]) for i in range(nsym)]
        return _psk_decide(dphase, mag, mod), np.asarray(dphase), mag


# --------------------------------------------------------------------------- #
# Per-modulation symbol decisions (spec §4.1).
# --------------------------------------------------------------------------- #

def _fsk_bytes(sym: np.ndarray) -> bytes:
    """The bytes ``sym`` (4 tone magnitudes per symbol) spells, MSB dibit first."""
    tones = np.argmax(sym, axis=0).reshape(-1, 4)
    vals = (tones[:, 0] << 6) | (tones[:, 1] << 4) | (tones[:, 2] << 2) | tones[:, 3]
    return bytes(int(v) for v in vals)


def _rs_accept(block: bytes, k: int, r: int, ftype: int) -> tuple[bytes | None, int]:
    """A carrier block's payload if it RS-corrects to a codeword whose count byte
    fits and whose frame-type-bound CRC holds, with the symbols RS had to fix — or
    ``(None, 0)``. The one gate every block passes through, whether it arrived just
    now or came back off a memory-ARQ average."""
    data, ok = rs.rs_correct(block, r)
    if ok and data[0] <= k and crc.check_crc16_frametype(data, ftype):
        return data[1:1 + data[0]], _rs_errors(block, data, r)
    return None, 0


def _grid_err(dphase, step: float) -> float:
    """Mean distance of differential phases to the nearest constellation point."""
    return float(np.abs(dphase - step * np.round(np.asarray(dphase) / step)).mean())


def _psk_decide(dphase, mag, mod) -> bytes:
    """The bytes a carrier's differential phases spell under ``mod``."""
    if mod is F.Mod.PSK4:
        return _decode_4psk(dphase)
    if mod is F.Mod.PSK8:
        return _decode_8psk(dphase)
    return _decode_16qam(dphase, mag[1:], int(mag[0]) * 3 // 4)


def _rs_errors(raw: bytes, data: bytes, r: int) -> int:
    """Symbols the RS decoder corrected. A successful decode lands on a valid
    codeword, so re-encoding the recovered data reproduces it and the byte
    differences from what was received are exactly the corrections."""
    fixed = data + rs.rs_parity(data, r)
    return sum(a != b for a, b in zip(raw, fixed))


def _apply_rs_floor(base: DecodedFrame, errors: int, blocks: int, r: int) -> None:
    """A frame that RS-corrected with few errors was read well however loose its
    constellation looked, and the reference says so (SoundInput.c, before
    ``returnframe``): it lifts such a frame's quality to `Q.RS_CLEAN_FLOOR`.

    Both quotients are integer, as they are in the C."""
    if base.ok and errors // blocks < r // Q.RS_CLEAN_DIVISOR:
        base.quality = max(base.quality or 0, Q.RS_CLEAN_FLOOR)


def _majority(triple: bytes) -> int | None:
    """The 2-of-3 majority value of a redundantly repeated byte (ConAck/PingAck)."""
    if triple[0] == triple[1] or triple[0] == triple[2]:
        return triple[0]
    if triple[1] == triple[2]:
        return triple[1]
    return None


def _carrier_freq(carrier: int, count: int) -> int:
    """The audio frequency of carrier ``carrier`` of ``count`` (spec §4 table).
    Single carrier sits at 1500 Hz; multi-carrier layouts are 200 Hz-spaced and
    symmetric about 1500, carrier 0 lowest."""
    if count == 1:
        return 1500
    return (1600 - count // 2 * 200) + 200 * carrier


def _decode_4psk(dphase) -> bytes:
    out = bytearray()
    for i in range(0, len(dphase), 4):
        raw = 0
        for k in range(4):
            p = dphase[i + k]
            raw <<= 2
            if -786 < p < 786:
                pass
            elif 786 <= p < 2356:
                raw += 1
            elif p >= 2356 or p <= -2356:
                raw += 2
            else:
                raw += 3
        out.append(raw)
    return bytes(out)


def _psk8_point(p: float) -> int:
    if -393 < p < 393:
        return 0
    if 393 <= p < 1179:
        return 1
    if 1179 <= p < 1965:
        return 2
    if 1965 <= p < 2751:
        return 3
    if p >= 2751 or p < -2751:
        return 4
    if -2751 <= p < -1965:
        return 5
    if -1965 <= p <= -1179:
        return 6
    return 7


def _decode_8psk(dphase) -> bytes:
    out = bytearray()
    for i in range(0, len(dphase), 8):
        bits = 0
        for k in range(8):
            bits = (bits << 3) + _psk8_point(dphase[i + k])
        out += bytes([(bits >> 16) & 0xFF, (bits >> 8) & 0xFF, bits & 0xFF])
    return bytes(out)


def _decode_16qam(dphase, mag, threshold: int) -> bytes:
    """8 phase points plus an absolute half-amplitude bit (0x08) against a threshold
    that tracks the two rings (``Decode1CarQAM``)."""
    out = bytearray()
    thr = threshold
    for i in range(0, len(dphase), 2):
        data = 0
        for k in range(2):
            data = (data << 4) + _psk8_point(dphase[i + k])
            if mag[i + k] < thr:
                data += 8
                thr = (thr * 900 + int(mag[i + k]) * 150) // 1000
            else:
                thr = (thr * 900 + int(mag[i + k]) * 75) // 1000
        out.append(data)
    return bytes(out)


def decode(samples) -> list[DecodedFrame]:
    """Convenience: decode every frame in ``samples`` with a default demodulator."""
    return Demodulator().decode(samples)
