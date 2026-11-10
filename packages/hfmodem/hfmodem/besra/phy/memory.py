# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Memory ARQ: what a carrier block that failed leaves behind for its repeat.

ARDOP resends an unacknowledged data frame under the same frame type until the
peer acks it — the even/odd twin in the catalog is what distinguishes the *next*
block — so two frames of one type arriving in a row are the same bytes sent
twice. The reference keeps two things from each attempt (SoundInput.c), and besra
keeps the same two:

* the carriers that already decoded, by their validated payload (``CarrierOk``,
  ``bytFrameData``). A carrier that passed RS and its CRC is finished; the repeat
  need only supply the ones that did not, and a frame whose carriers arrive
  correct on different repeats is assembled out of both.
* the *soft* reading of every carrier that did not — differential phase per
  symbol for PSK and QAM (``SavePSKSamples``, ``SaveQAMSamples``), per-symbol
  relative tone magnitudes for 4FSK (``SaveFSKSamples``) — averaged across the
  repeats and re-decoded. Two readings of one symbol, each wrong sometimes, agree
  on the truth more often than either does alone.

Never the decided bytes of a block that failed: bytes are already a decision, and
averaging decisions throws away the confidence that makes the averaging worth
anything.

Nothing here decides or corrects. A caller hands in the reading of a block that
failed and gets back the average to re-decode, which still has to pass RS and the
frame-type-bound CRC exactly as a fresh block does — but that gate is not what
keeps a stale memory honest, because a *validated* carrier put back from memory
was already through it. What keeps it honest is only ever the rule below about
which frames the memory belongs to.

Readings ride on the ``DecodedFrame`` they were taken for and are kept only when
that frame is the one the receiver settles on. Acquisition decodes bodies
speculatively — several header candidates, several tone shifts — and all but one
of those attempts is thrown away; without that, the search would average a frame
against six readings of itself.
"""

from __future__ import annotations

import numpy as np

from ..dsp.templates import SAMPLE_RATE

#: How far apart two arrivals may sit and still be the same block, in samples.
#:
#: An ISS repeats an unacknowledged data frame on a timer and gives that block up
#: after a fixed budget, so a run of one type longer than the budget is not one
#: block being repeated: it is a second block that happens to share the type, and
#: a reading from the first must not be put back into it.
#:
#: The budget is the whole derivation, because ARDOP's ARQ has no repeat *count*.
#: ardopcf stamps ``dttTimeoutTrip`` when it composes a new data frame
#: (``ARQ.c:898``) and never restamps it for a repeat; ``GetNextARQFrame`` stops
#: repeating at ``(Now - dttTimeoutTrip) / 1000 > ARQTimeout`` (``ARQ.c:462``) and
#: ``ARQTimeout`` is 120 s (``ARDOPC.c:101``, host-settable 30-240). besra's ISS
#: has the same shape and a shorter budget — repeats bypass ``_touch()``, so
#: ``_timeout_s`` 90 s ends them. The peer is the one deciding how long it
#: repeats, so the bound takes the reference's number:
#:
#:      120 s x 12000 samples/s = 1_440_000 samples
#:
#: measured from the first arrival held under a key, which is the reference's own
#: ``MemarqTime`` — set when the memory is first written, compared against
#: ``1000 * ARQTimeout`` in ``CheckMemarqTime`` (``SoundInput.c:152-181, 305``).
#:
#: For scale rather than for the derivation: the repeat grid is the frame plus
#: ``ComputeInterFrameInterval`` (``ARQ.c:879-895``), 1.5-2.1 s of it plus the
#: measured remote leader, so 4PSK.500.100's 2.4 s frames come round about every
#: 4.4 s and the span admits some 28 of them. Nothing on the air has come close:
#: KE8LVA's four greetings spanned ~20 s, and the gain audit's three arrivals
#: ~10 s.
#:
#: Samples rather than seconds because a reading is already placed in the stream
#: by sample position and nothing here reads a clock — the whole-capture pass
#: that found the frame this bound is for has no wall time to read. Samples
#: rather than a count of transmissions because that count is not one number: it
#: is this span divided by a repeat grid that varies with frame duration and with
#: the peer's measured leader, so it is derived *from* the span rather than
#: standing in for it.
_REPEAT_SPAN = 120 * SAMPLE_RATE


def _angle_avg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The reference's ``WeightedAngleAvg``: unit vectors summed, angle taken.

    Not an average of the complex samples, and the difference is the whole point.
    Magnitude is absent despite the name, so a repeat that arrived under a static
    crash casts the same one vote as a quiet one — where averaging the samples
    themselves would let the crash decide the symbol. The cost falls where two
    readings disagree outright: near-opposite angles sum to nearly nothing and the
    angle that comes back is arbitrary. That symbol is then a guess, which is what
    it already was, and RS still has to carry the block."""
    a, b = a / 1000.0, b / 1000.0
    return 1000.0 * np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b))


def _span(ftype: int) -> int:
    """How far apart two readings must sit to be two arrivals: the frame's own
    extent, since a repeat cannot begin before the first copy's last sample. Taken
    from the demodulator, and imported here rather than at module scope because
    that module imports this one."""
    from .demodulator import frame_span
    return frame_span(ftype)


def _relative(tones: np.ndarray) -> np.ndarray:
    """Each 4FSK symbol's tone magnitudes scaled to sum to one — the reference's
    own device in ``SaveFSKSamples``. What carries a symbol is which tone leads its
    neighbours, so a repeat that happened to arrive louder must not outvote a
    quieter one that read the tones more clearly."""
    tones = np.asarray(tones, dtype=np.float64)
    return tones / np.maximum(tones.sum(axis=0, keepdims=True), 1e-12)


def _fold_psk(readings):
    """Arrivals oldest first into the reference's running pair of averages: phases
    through ``WeightedAngleAvg``, magnitudes through the plain mean it gives them,
    amplitude being the signal for 16QAM rather than a weight."""
    _, phase, mag = readings[0]
    for n, (_, dphase, m) in enumerate(readings[1:], start=1):
        phase = _angle_avg(phase, dphase)
        mag = (mag * n + m) / (n + 1)
    return phase, mag


def _fold_fsk(readings):
    acc = readings[0][1]
    for n, reading in enumerate(readings[1:], start=1):
        acc = (acc * n + reading[1]) / (n + 1)
    return acc


class BlockMemory:
    """Per-carrier memory of one frame, held for that frame's repeats.

    The memory belongs to one ``(epoch, frame type, session)``. Anything else
    arriving clears it, because only a repeat of the same frame under the same
    session is the same bytes — this is the reference's ``LastDataFrameType``
    comparison, which resets on every frame type it sees that is not a short
    control, so a rate shift down the ladder or a connect handshake wipes the
    slate. ``epoch`` carries the resets a receiver cannot see for itself: the
    reference calls ``ResetMemoryARQ`` from ARQ.c wherever this station takes the
    IRS role, which is where a link turnover would otherwise hand a fresh block
    the memory of the one before the turn.

    Those two are the reference's whole receiver-side rule, and on their own they
    are not enough here. ardopcf reads a stream once, so its ``LastDataFrameType``
    walks the transmissions in the order they were sent; besra's live path decodes
    overlapping windows and re-reads each frame two dozen times, so the type it
    last saw wanders backwards as an old frame is re-acquired behind a newer one.
    On `20260819T024752Z` that resurrected the handshake's second frame after its
    third had already been read, and the fourth — cut by the window edge, so all
    its carriers failed — was answered whole from the second's validated payload
    and reported as a clean decode of bytes nobody had transmitted. So a reading
    is placed in the stream, and a frame that sits behind the newest transmission
    the memory has accounted for neither adds to it nor draws on it.

    Neither rule bounds how long a reading may live, and the third one does:
    nothing is remembered further back than an ISS can go on repeating one block
    (``_REPEAT_SPAN``). That is the reference's ``MemarqTime`` guard, which besra
    had none of.

    What is left after all three is two same-type blocks meeting under one key
    inside the span, where a carrier validated on the older is put back into the
    newer whole and reaches the application as clean data — a payload out of
    ``CarrierOk`` having already been through RS and the CRC on the frame it was
    read from. Nothing on the stream separates them: the two arrivals sit one
    repeat interval apart and a repeat lost to a fade leaves the same gap. ardopcf
    is spared it by reading the stream once, so the alternating type between two
    blocks always reaches ``LastDataFrameType`` before the second is decoded.
    besra reads overlapping windows and reaches it late — on `20260829T185417Z`
    the `4PSK.200.100S.E` between two `.O` blocks was acquired after the second
    had already been reported, and W6IDS's proposal reached the mail client with a
    block of the address duplicated over the block that carried the subject, its
    `F>` never delivered and 524 bytes of mail left at the CMS.

    So the memory a frame leaves when it comes out *whole* is dropped, and only
    what a frame that came out short leaves is kept. That is the whole of what
    ``CarrierOk`` is for — a frame whose carriers arrive correct on different
    repeats — and the only thing it can no longer do is re-emit a block already
    delivered out of a repeat too weak to read alone. On the 2026-08-05 KE8LVA
    session that is three of the four greeting arrivals the whole-capture pass
    reaches, and seven of the eight a rolling decoder reads over it ungated —
    every one of them the bytes the session had already delivered, which
    `ArqSession._receive_data` drops on the type alternation in any case. A repeat
    that fails there now draws the ACK for what was delivered, or, where it
    degrades to a header-only sighting, a NAK — one asked-for retransmit, against
    a block invented whole.
    """

    def __init__(self) -> None:
        self._epoch = 0
        self._at = 0
        self._key: tuple[int, int, int] | None = None
        self._newest = -1
        self._since = 0
        self.reset()

    def reset(self) -> None:
        self._good: dict[int, bytes] = {}
        self._reads: dict[int, dict[int, tuple]] = {}

    def epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def stream(self, at: int) -> None:
        """Where in the received stream the samples of this decode pass begin."""
        self._at = at

    def where(self, frame) -> int:
        """Where this frame's header sits in the received stream."""
        return self._at + frame.offset

    def live(self, frame) -> bool:
        """True when what is remembered belongs to this frame."""
        at = self.where(frame)
        return (self._key == (self._epoch, frame.type, frame.session_id)
                and self._newest <= at <= self._since + _REPEAT_SPAN)

    def repeats(self, index: int) -> int:
        """Arrivals whose soft reading is held for this carrier (or 600-baud
        sub-packet)."""
        return len(self._reads.get(index, ()))

    # -- one decode attempt, recorded on the frame it produced ---------------

    def good(self, frame, carrier: int) -> bytes | None:
        """The payload a previous repeat already validated for this carrier."""
        return self._good.get(carrier) if self.live(frame) else None

    def keep_good(self, frame, carrier: int, payload: bytes) -> None:
        frame.soft[carrier] = ("good", payload)

    def psk(self, frame, carrier: int, dphase: np.ndarray,
            mag: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Record a failed carrier's differential phases and magnitudes, and return
        them averaged with what previous repeats read — or None when this is the
        first reading of the carrier and there is nothing to average against."""
        reading = ("psk", np.asarray(dphase, dtype=np.float64),
                   np.asarray(mag, dtype=np.float64))
        frame.soft[carrier] = reading
        earlier = self._earlier(frame, carrier)
        return _fold_psk([*earlier, reading]) if earlier else None

    def fsk(self, frame, part: int, tones: np.ndarray) -> np.ndarray | None:
        """Record a failed 4FSK block's tone magnitudes and return them averaged
        with what previous repeats read, or None on the first reading."""
        reading = ("fsk", _relative(tones))
        frame.soft[part] = reading
        earlier = self._earlier(frame, part)
        return _fold_fsk([*earlier, reading]) if earlier else None

    def keep(self, frame) -> None:
        """Keep the readings of the attempt that became ``frame``, under its key."""
        at = self.where(frame)
        if at < self._newest:
            return
        key = (self._epoch, frame.type, frame.session_id)
        if key != self._key or at - self._since > _REPEAT_SPAN:
            self._key = key
            self._since = at
            self.reset()
        self._newest = at
        if frame.ok:
            self.reset()                     # a finished block, and nothing to finish
            return
        span = _span(frame.type)
        for index, reading in frame.soft.items():
            if reading[0] == "good":
                self._good[index] = reading[1]
            else:
                held = {p: r for p, r in self._reads.get(index, {}).items()
                        if abs(p - at) >= span}
                self._reads[index] = {**held, at: reading}

    def _earlier(self, frame, index: int) -> list[tuple]:
        """The held readings of this carrier that came from earlier arrivals.

        Readings are held by stream position and read back by distance, which is
        not the same test. Holding them by position was meant to say that
        re-reading one arrival replaces its own reading rather than voting with
        it, and that only holds where acquisition pins the header on the same
        sample twice. The overlapping windows of the live path re-read each
        arrival around two dozen times and pin its leader edge a little
        differently as they slide over it, so one transmission comes back under
        positions a few samples apart and every one of them becomes an arrival in
        its own right. Measured on `20260826T133925Z`, replayed as the session
        heard it: 52 of the 81 averages that session produced were an arrival
        folded against a second copy of its own reading, and one arrival on its
        own through the live path produced 32.

        A repeat cannot begin before the first copy's last sample, so a reading
        nearer than the frame's own length is this arrival and not another one.
        """
        if not self.live(frame):
            return []
        at, span = self.where(frame), _span(frame.type)
        return [r for p, r in sorted(self._reads.get(index, {}).items())
                if abs(p - at) >= span]
