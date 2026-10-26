# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""OFDM link-setup — TX and RX, riding each bandwidth's validated DATA chain.

The link-setup is the burst that carries the CALLER callsign: step 4 of the
connect handshake  [spec 05 §5.3, §5.3.2]. It is not a waveform of its own — the
caller-ID frame uses the DATA waveform at the negotiated setup speed  [spec 04 §4.2A],
so TX delegates to the codec for the offered speed. At the default level 4,
the established TX/RX codecs are BW2300 rec3 (92-byte frame, 4.36 s), BW2750's own base record (the
same 92-byte frame on a 20-bin comb, 4.36 s) or BW500 level 4 (46-byte frame,
403 columns / 4.31 s). The bandwidth changes the comb, not the frame.

    TX: :func:`vara_frames.link_setup_frame` -> ``tx.varahf2300_tx.synth_frame``
        or ``tx.varahf500_tx.synth_burst``
        (byte-exact against real captured VARA link-setup overs at both)
    RX: ``rx.varahf2300.decode_over`` (payload-blind reference-column alignment,
        turbo decode, CRC-16/GENIBUS) or ``rx.varahf500.decode_burst``
        -> :func:`vara_frames.is_link_setup`
        -> :func:`vara_frames.caller_from_link_setup`

The BW500 side is closed against the real modem, not only against this decoder:
on 2026-08-15 a VARA HF 4.9.0 answering a replayed BW500 connect-request took
the burst :func:`link_setup_tx` renders and reported ``CONNECTED <caller> W1AW
500`` for W9SSJ, for KD9ZZZ — a callsign in no recording — and for G4ABC-7.

An earlier revision of this module modelled the spec's *internal* 128x940
cell-raster law  [spec 01 §2300] and held both directions PENDING on numeric
tables the spec then lacked. That model described the modulator's resident grid,
not the on-air burst; the promoted rec3 placement law and the
``spec/tables/bw2300`` tables superseded it, and RX is validated on real
captured VARA audio (``tests/kestrel/test_bw2300_real_audio``,
``test_vara_mail_rx``). The raster/twiddle geometry remains documented in
spec/01 §2300.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..arq import phy as _phy
from ..coding.crc import crc16_genibus
from ..rx import varahf500 as _rx500
from ..rx import varahf2300 as _rx
from ..tx import varahf500_tx as _tx500
from ..tx import varahf2300_tx as _tx
from . import vara_frames as _vf

FS = _rx.FS

# Where BW500 emission column 0 sits in the rendered burst: the synthesis pulse
# reaches back before its own symbol centre, so a column keyed at 0 would be
# dropped for want of room. Matches ``arq.phy.ONSET``.
ONSET_500 = 3000

# 403 columns is what a real VARA keys, and what is asked for here — but only 394
# of them come out. ``grid480`` reads 0 at the nine indices the lead-in columns
# address, so kestrel cannot render them and the recovered table has never held a
# value there. Those nine are pure reference cells ahead of the frame: a real
# VARA HF 4.9.0 completed the connect on a burst without them three times out of
# three (2026-08-15). What the gap does cost is that an energy detector puts OUR
# burst's start nine columns late, which is why the receive path below reads a
# peer's burst by its detected span and kestrel's own by :data:`ONSET_500`.
_NCOL_500 = _tx500.C0 + _tx500.NSYM

#: ``level`` names the RECORD here, as it does at BW2300: 3 is the base one and 2
#: the gear below it. BW500's own modules count in the host's ``BITRATE`` numbers
#: instead, where the base record is 4 — which is what ``arq.phy.base_level``
#: returns for it — so both spellings of the base land on the same waveform and
#: lower records 2, 1, 0 map to host levels 3, 2, 1.
_BW500_RECORDS = {4: _tx500.BASE_LEVEL, 3: _tx500.BASE_LEVEL,
                  2: _tx500.ROBUST_LEVEL, 1: 2, 0: 1}

#: Bandwidths whose link-setup burst this module speaks in both directions.
LINK_SETUP_BW = frozenset(_vf.LINK_SETUP_BODY)


def link_setup_tx(caller_callsign: str, over: int = 0,
                  bw: str = "2300", level: int = 4, ask: int = 4,
                  grade: int = 0x80, high_speed: bool = False,
                  lead: _tx.VB6Rnd | None = None,
                  compression: str = "OFF") -> np.ndarray:
    """Render caller identity at the offer's host speed level (1..4).

    ``level`` is the host speed, unlike ``data_over_tx``'s record index.
    The default preserves the established level-4 waveform at each bandwidth.
    ``lead`` is the session's lead stream  [see data_over_tx].
    """
    frame = _vf.link_setup_frame(caller_callsign, bw, level, ask, grade, high_speed, compression)
    if str(bw) == "500":
        return _tx500.synth_burst(frame, onset=ONSET_500, ncol=_NCOL_500,
                                  level=level, lead=lead)
    return _tx.synth_frame(frame, _rx.BASE_LEVELS[str(bw)] - 4 + level,
                           over if lead is None else lead)


def data_over_tx(body: bytes | Sequence[bytes], over: int = 0,
                 level: int = _rx.BASE_LEVEL, bw: str = "2300",
                 lead: _tx.VB6Rnd | None = None) -> np.ndarray:
    """Synthesize one DATA over carrying one speed level's VARA body.

    The same burst as the link-setup at either bandwidth, so this is
    :func:`link_setup_tx` with a body the caller composed instead of the
    caller-ID frame. At BW2300 ``over`` is the frame's position in the session's
    preamble stream, counting the link-setup as over 0, and ``level`` names the
    record the body has to be the length of: 90 bytes at the base level, 48 one
    gear down at record 2. BW500 bodies contain 44, 35, 23 or 10 bytes at
    records 4 (legacy alias 3), 2, 1 or 0 respectively. ``over`` does not
    reach its waveform, which carries no per-over preamble stream.

    Several bodies make one burst of several frames, which a BW500 sender keys
    at its base record once a delivery has run six single overs.

    ``lead`` is the session's lead stream, owned by the ARQ: each renderer
    draws its row's lead from it, advancing it by the row's cost (12 at the
    wideband L4 preamble, 18 at BW500's, 9 per band at a tone row, lead x
    carriers x frames at an OFDM row), and ``over`` is then unused. A record
    above the base one is a host level above 4, keyed by its family
    [see arq.phy.render_level], and it needs the stream.

    The CRC-16 trailer is appended here so a caller only ever handles the body
    (``arq.phy.vara_body`` builds one)."""
    bodies = [bytes(body)] if isinstance(body, (bytes, bytearray)) else list(body)
    if level not in (_BW500_RECORDS if str(bw) == "500" else _rx.RECORDS):
        return _phy.render_level(bodies, bw, _phy.host_level(bw, level), lead)
    frames = [b + crc16_genibus(b).to_bytes(2, "big") for b in bodies]
    if str(bw) == "500":
        return _tx500.synth_burst(frames if len(frames) > 1 else frames[0],
                                  onset=ONSET_500, ncol=_NCOL_500,
                                  level=_BW500_RECORDS[level], lead=lead)
    return _tx.synth_frame(frames[0], level, over if lead is None else lead)


def link_setup_rx(samples: np.ndarray, tries: int = 6,
                  bw: str = "2300") -> str | None:
    """Decode a received link-setup burst to the caller callsign, or None.

    ``samples`` is one segmented burst (lead-in/tail tolerated: the frame start
    is found by the payload-blind reference-column alignment search, and at BW500
    by the burst-energy span the decoder locates for itself). Acceptance is three
    independent gates, in order:

      1. reference-column alignment (24 payload-blind columns; noise tops out at
         9 matches where a real frame scores 20-24 — measured, see
         ``rx.varahf2300._guard_scores``),
      2. turbo decode converging on a CRC-16/GENIBUS that passes twice on
         identical bits (``coding.turbo``),
      3. the frame's fixed link-setup structure bytes  [spec 04 §4.2A] — a DATA
         over with a clean CRC is *not* a link-setup and returns None.

    Validated at both bandwidths on real captured VARA link-setup overs (the
    recovered caller matches the recording station's) and on kestrel's own TX
    round-trip."""
    frame = link_setup_read(samples, tries, bw)
    return None if frame is None else _vf.caller_from_link_setup(frame)


def link_setup_read(samples: np.ndarray, tries: int = 6,
                    bw: str = "2300", on_quality=None) -> bytes | None:
    """The link-setup frame :func:`link_setup_rx` reads the caller from, or None."""
    x = np.asarray(samples, float)
    if str(bw) == "500":
        spans = _rx500.burst_spans(x)
        if not spans:
            return None
        fr = _rx500.decode_burst(x, *spans[0])
    else:
        fr = _rx.decode_over(x, 0, len(x), tries=tries,
                             level=_rx.BASE_LEVELS[str(bw)])
    if not fr.crc_ok or not _vf.is_link_setup(fr.frame_bytes):
        return None
    if on_quality is not None and fr.quality is not None:
        on_quality(fr.quality)
    return bytes(fr.frame_bytes)
