# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW500 record 2 — the speed level below base — decoded off real VARA audio.

Host ``BITRATE (3)``, 61 bps, 34 payload bytes a block, 4.94 s. It is the level a
stock BW500 pair opens a delivery at, so until it was read no BW500 transfer could
start; and it is not the base waveform one gear slower but one-hot index
modulation, 226 columns of 1024 samples lighting one bin of eleven apiece.

The one session that holds it keyed two such overs, of 34 and 7 payload bytes,
both zeros. Both decode CRC-clean to those bytes here, and the base level reads
neither. Nothing in the chain is measured: the 226 base bins and 24 reference
classes are the draws of the record's own interleaver gap.

Its 4 training columns are not reversed — they differ between the two overs, and
no VB6 Rnd stream anchored anywhere reproduces both. They are training a peer
cannot check, so the transmitter keys a stand-in ahead of a body that is the
recording's own columns bin for bin.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import tablegen
from hfmodem.kestrel.rx import varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara import vara_ofdm as OF
from hfmodem.tests.kestrel import corpora

_ROBUST, _BASE = 2, 3                   # BW500 record indices

#: What the session's host log says each record-2 over delivered, in order: a full
#: block and the short one that ends the 256-byte transfer. The payload is zeros,
#: so the bytes past the short block's length are the frame's own padding and are
#: not the transfer's.
_DELIVERED = (34, 7)


@pytest.fixture(scope="module")
def session():
    return corpora.wav_mono(corpora.BW500_HANDSHAKE)


@pytest.fixture(scope="module")
def robust_spans(session):
    """The two record-2 bursts of the session, by their own reference score."""
    return [(a, b) for a, b in rx.burst_spans(session)
            if rx._l3_align(session, a, b)[2] >= rx._L3_GUARD_MIN]


@corpora.requires_bw500_handshake
def test_the_two_recorded_overs_decode_to_their_known_plaintext(session):
    result = rx.decode_stream(session)
    robust = [f for f in result.frames if f.level == 3]

    assert len(robust) == 2, [(f.start / rx.FS, f.level) for f in result.frames]
    assert all(f.crc_ok for f in robust), [f.crc_ok for f in robust]
    assert all(len(f.payload) == rx.L3_PAYLOAD for f in robust)
    for f, n in zip(robust, _DELIVERED):
        assert f.payload[:n] == bytes(n), f.payload.hex()
    # the frame start locks at the same lead on both, and every base-level over of
    # the same recording still decodes at its own c0
    assert [f.c0 for f in robust] == [rx.L3_LEAD] * 2
    assert all(f.c0 == 9 for f in result.frames if f.level == 4 and f.crc_ok)


@corpora.requires_bw500_handshake
def test_the_base_level_reads_a_record_2_over_as_nothing(session, robust_spans):
    """Why the gap was invisible rather than merely unread: asked to read one of
    these bursts at the base level, the receiver returns a frame that fails CRC —
    there is no report anywhere in that path saying a burst was a different gear."""
    assert len(robust_spans) == 2
    for a, b in robust_spans:
        frames = rx.decode_burst_frames(session, a, b, level=4)
        assert not any(f.crc_ok for f in frames)


@corpora.requires_bw500_handshake
def test_the_reference_score_tells_the_levels_apart_before_any_decode(session):
    """The payload-blind guard, over every burst of the session.

    The two record-2 overs match all 24 reference columns; the six base-level
    bursts of the same recording never match more than 7 of them at any lead. The
    threshold sits in a gap three times its own width."""
    scored = [rx._l3_align(session, a, b)[2] for a, b in rx.burst_spans(session)]
    robust = sorted(s for s in scored if s >= rx._L3_GUARD_MIN)
    base = sorted(s for s in scored if s < rx._L3_GUARD_MIN)

    assert robust == [24, 24], scored
    assert max(base) <= 7, scored


def test_noise_is_not_read_as_a_record_2_over():
    rng = np.random.default_rng(5)
    n = rng.standard_normal((rx.L3_NCOL + rx.L3_LEAD) * rx.L3_DW)
    assert rx._l3_align(n, 0, len(n))[2] < rx._L3_GUARD_MIN
    assert not rx.decode_l3_burst(n, 0, len(n)).crc_ok


def test_the_column_count_is_what_the_interleaver_gap_pays_for():
    """226 columns and 24 reference columns are not two measurements but one.

    Record 3's BW2300 bin tables are the draws between its own interleaver
    permutations and record 4's seed — one column allocation each, then one
    reference class each. BW500 record 2's gap is 250 draws and closes the same
    way on record 3's seed, so its columns and its references have to add to that;
    606 coded bits at 3 bits a column fixes the data half at 202, and 202 + 24 + 24
    is the only split that pays for it. The 24 reference columns then land where
    the tape has them, which is what makes the arithmetic evidence rather than
    numerology.
    """
    ndata = rx.L3_CODED // rx.L3_BPC
    seed, coded, n_info = tablegen.BW500[_ROBUST]
    assert (coded, n_info) == (rx.L3_CODED, rx.L3_N_SRC)

    _, s = tablegen.permutation(coded, seed)
    _, s = tablegen.permutation(n_info, s)
    for _ in range(ndata + 2 * 24):
        s = tablegen.step(s)

    assert s == tablegen.BW500[_BASE][0]
    assert rx.L3_NCOL == ndata + 24
    assert len(tablegen.bw500_ref_cols(2)) == 24


def test_the_reference_columns_light_bins_the_data_columns_never_could():
    """The eleven-bin wheel carries an eight-value alphabet, so three offsets of
    every data column mean nothing at all. That is what a wrong alignment runs
    into, and it is why the guard reads 24 or nothing rather than degrading."""
    _, _, _, _, valid, _, _ = rx._l3_tables()
    assert valid.shape == (rx.L3_CODED // rx.L3_BPC, rx.L3_SPAN)
    assert (valid.sum(1) == 2 ** rx.L3_BPC).all()


# --------------------------------------------------------------------------- #
# transmit: the same law forwards
_MARKER = 0x81


@pytest.mark.parametrize("payload", [
    bytes(rx.L3_PAYLOAD),
    bytes(range(rx.L3_PAYLOAD)),
    bytes(np.random.default_rng(11).integers(0, 256, rx.L3_PAYLOAD, dtype=np.uint8)),
])
def test_a_rendered_record_2_burst_reads_back(payload):
    """Round trip, payload-independent: the renderer is the decoder's inverse."""
    frame = tx.build_l3_frame(payload, _MARKER)
    audio = tx.synth_l3_burst(frame, onset=OF.ONSET_500)
    fr = rx.decode_l3_burst(audio, OF.ONSET_500, len(audio))

    assert fr.crc_ok
    assert fr.payload == payload
    assert fr.marker == _MARKER
    assert fr.c0 == rx.L3_LEAD
    # the guard a peer aligns on scores full marks on our own burst
    assert fr.self_consistency == 1.0


def test_the_burst_is_the_records_own_length():
    frame = tx.build_l3_frame(bytes(rx.L3_PAYLOAD), _MARKER)
    audio = tx.synth_l3_burst(frame)
    assert len(audio) == (rx.L3_LEAD + rx.L3_NCOL) * rx.L3_DW      # 4.93 s


def test_the_training_columns_are_the_records_own_allocation():
    """The stand-in, stated: the allocation D's law offsets by a draw, drawn at
    zero, because BW500 has no anchor to locate the draw from."""
    assert tx._l3_training_bins() == [
        int(b) for b in tablegen.bw500_alloc(2)[1:1 + rx.L3_LEAD]]


@corpora.requires_bw500_handshake
def test_the_rendered_body_is_the_recordings_own_columns(session, robust_spans):
    """Against the tape, bin for bin and sample for sample.

    The body is the recording's; the four training columns are not, and the
    difference between the two correlations is the whole of what is unfixed."""
    for start, stop in robust_spans:
        fr = rx.decode_l3_burst(session, start, stop)
        off = rx._l3_grid(session, start, stop)
        ncol = rx.L3_LEAD + rx.L3_NCOL
        tape = session[off:off + ncol * rx.L3_DW]
        ours = tx.synth_l3_burst(fr.frame_bytes)
        body = slice(rx.L3_LEAD * rx.L3_DW, None)

        lit = rx._l3_band(session, off, ncol).argmax(1) + rx.L3_FIRST_BIN
        assert (lit[rx.L3_LEAD:] == tx.l3_column_bins(fr.frame_bytes)).all()

        corr = lambda u, v: float(u @ v / np.sqrt((u @ u) * (v @ v)))
        assert corr(tape[body], ours[body]) > 0.9999
        assert corr(tape, ours) > 0.98


def test_the_bandwidth_dispatch_keys_the_record_the_caller_names():
    """``level`` at this seam is the record, at BW500 as at BW2300."""
    body = bytes(rx.L3_PAYLOAD) + bytes([_MARKER])
    audio = OF.data_over_tx(body, level=2, bw="500")
    fr = rx.decode_burst_frames(audio, OF.ONSET_500, len(audio))[0]

    assert fr.level == rx.ROBUST_LEVEL and fr.crc_ok
    assert fr.payload == bytes(rx.L3_PAYLOAD)
    # the base record answers to both its spellings — the record index, and the
    # host BITRATE number the ARQ hands down through `arq.phy.base_level`
    for lv in (None, 3, phy.base_level("500")):
        kw = {} if lv is None else {"level": lv}
        base = OF.data_over_tx(bytes(44), bw="500", **kw)
        assert rx.decode_burst(base, OF.ONSET_500).level == rx.BASE_LEVEL
    with pytest.raises(ValueError):
        OF.data_over_tx(body, level=9, bw="500")


# --------------------------------------------------------------------------- #
# The session keys it: the over that closes a delivery.
_MYCALL, _CALLED = "W9SSJ", "W1AW"


class _TxIO(VA.VaraIO):
    """Records what was keyed. Nothing here opens a device or reaches a radio."""

    def __init__(self):
        self.msgs: list[str] = []
        self.sent: list[np.ndarray] = []

    def key(self, on): ...

    def tx(self, samples): self.sent.append(np.asarray(samples, float))

    def pending(self): ...

    def connected(self, *a): ...

    def log(self, msg): self.msgs.append(msg)


def _delivery(payload: bytes) -> list:
    """The overs a connected BW500 initiator keys for ``payload``, each read back
    by this station's own narrow receiver, with the peer's control burst granting
    the turn and answering every over."""
    io = _TxIO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw="500")
    hs.role, hs.called, hs.caller = "initiator", _CALLED, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.send(payload)
    for _ in range(3):
        hs.on_rx_audio(MK.synth_tone_pairs(VF.CONTROL_BURST_RESPONDER_500))
    overs = [s for s in io.sent if len(s) > 2 * rx.FS]
    return [rx.decode_burst_frames(s, OF.ONSET_500, len(s))[0] for s in overs]


def test_the_close_after_a_full_over_is_keyed_at_record_2():
    """The record a stock BW500 pair closes on: its 200-byte delivery is four
    level-4 bodies and then a 35-byte record-2 body  [spec 04 §4.2B]. 47 bytes
    here is one full over and a 4-byte close, and the close goes out at record 2
    carrying the same trailer the base level would have."""
    assert phy.close_level("500") == 2
    assert phy.body_size("500", 2) == rx.L3_PAYLOAD + 1 == 35
    full, close = _delivery(b"x" * rx.FRAME_PAYLOAD + b"tail")

    assert full.level == rx.BASE_LEVEL and full.crc_ok
    assert full.payload == b"x" * rx.FRAME_PAYLOAD and full.marker == 0x81

    assert close.level == rx.ROBUST_LEVEL and close.crc_ok
    body = bytes(close.payload) + bytes([close.marker])
    assert len(body) == phy.body_size("500", 2)
    assert phy.vara_payload(body, caller=_MYCALL, body_len=len(body)) == b"tail"
    assert phy.over_is_last(body, _MYCALL)


def test_a_delivery_that_fits_one_over_stays_at_level_4():
    """No full over in front of it announces anything, so the close is the base
    level's — the login block a stock responder read byte-exact is one of these."""
    (only,) = _delivery(b"y" * 30)
    assert only.level == rx.BASE_LEVEL and only.crc_ok
    body = bytes(only.payload) + bytes([only.marker])
    assert phy.vara_payload(body, caller=_MYCALL, body_len=len(body)) == b"y" * 30
