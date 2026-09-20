# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Record 2 — the speed level below base — decoded off real VARA audio.

Host ``BITRATE (3)``, 82 bps, 5.03 s. On the two-sided bench session of
2026-08-14 both unmodified VARA HF 4.9.0 instances dropped to it for the 37-byte
tail of their 126-byte transfer, and until this level was reversed nothing in
kestrel could read either over: they score 5 and 6 of the 24 base-level reference
columns, under the 11 that recording's own noise reaches, so the receive path
declined them without a word.

The bin-placement table is measured, so these are the tests that hold it: the two
overs decode to their known plaintext, and the base level still cannot read them.

The transmit half closes on the same tape. Each over opens with six 1024-sample
training symbols — 6144 samples, the same span the base level fills with twelve of
512 — and their bins are the base preamble's law at record 2's geometry, drawn from
the one continuous VB6 Rnd stream the session's base-level overs draw from.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.coding.crc import crc16_genibus
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.rx import tablegen
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.tests.kestrel import corpora

_ROBUST = 2
_GUARD_MIN = 16                     # `vara_arq._OVER_GUARD_MIN`

#: Each side's Rnd state before the first preamble on tape, and how many base-level
#: overs precede its record-2 one. Recovered by inverting that side's base-level
#: preambles against the published base allocation table — the same measurement
#: `PREAMBLE_SEED` is for the staged capture, once per direction because the two
#: stations run their own generators.
_REF_STREAM = {"a2b": (2767643, 2), "b2a": (3653944, 1)}


def _side(name: str):
    p = next(q for q in corpora.BW2300_REF_SIDES if q.name.endswith(f"__{name}.wav"))
    a = corpora.wav_mono(p)
    peak = np.abs(a).max()
    return a / peak if peak else a


def _tape_preamble_bins(side: str) -> list:
    """The record-2 preamble bins the session's own Rnd stream reaches.

    One draw per training symbol, overs back to back, with the single extra draw
    between overs 0 and 1 the base level's stream also carries."""
    seed, before = _REF_STREAM[side]
    rnd = tx.VB6Rnd(seed)
    for o in range(before):
        if o == 1:
            rnd.draw16()
        tx.preamble_bins(rnd, rx.BASE_LEVEL)
    return tx.preamble_bins(rnd, _ROBUST)


def _column_one(audio, t0, t1) -> tuple:
    """(sample index of data column 1, decoded frame) for one record-2 over."""
    r = rx.RECORDS[_ROBUST]
    s = int((t0 - 0.4) * rx.FS)
    seg = np.asarray(audio[s:int((t1 + 0.4) * rx.FS)], float)
    for onset, g, mag in rx._alignments(seg, rx._GUARD_TRIES, _ROBUST):
        fr = rx.check_frame(rx.onair_to_frame(rx._onair_llr(mag[g:], _ROBUST),
                                              _ROBUST), _ROBUST)
        if fr.crc_ok:
            return s + onset + g * r.dw50, fr
    return None, None


def _best_corr(x, audio, base):
    """Normalised correlation at the best sample lag — the over's onset is not
    locked to the block grid [see test_bw2300_preamble]."""
    best = -1.0
    for lag in range(-64, 65):
        y = audio[base + lag:base + lag + len(x)]
        if len(y) < len(x):
            continue
        best = max(best, float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))))
    return best


@corpora.requires_bw2300_reference
@pytest.mark.parametrize("side,t0,t1,payload", corpora.BW2300_REF_ROBUST)
def test_the_robust_over_decodes_to_its_known_plaintext(side, t0, t1, payload):
    a = _side(side)
    fr = rx.decode_over(a, int((t0 - 0.2) * rx.FS), int((t1 + 0.2) * rx.FS),
                        level=_ROBUST)
    assert fr.crc_ok
    assert fr.payload[:len(payload)] == payload


@corpora.requires_bw2300_reference
@pytest.mark.parametrize("side,t0,t1,payload", corpora.BW2300_REF_ROBUST)
def test_the_base_level_reads_the_robust_over_as_noise(side, t0, t1, payload):
    """Why the gap was invisible rather than merely unread.

    A wrong-level score is not a low score with a warning in it — it is inside the
    population the guard was tuned to reject, so no threshold anywhere between
    noise and a frame separates the two."""
    seg = _side(side)[int((t0 - 0.2) * rx.FS):int((t1 + 0.2) * rx.FS)]
    (_, hits, of, _), = rx.index_guard(seg, (rx.BASE_LEVEL,))
    assert of >= 20 and hits * 24 < 9 * of      # under the 9 of 24 noise reaches
    (_, own, own_of, _), = rx.index_guard(seg, (_ROBUST,))
    assert own == own_of == 24


@corpora.requires_bw2300_reference
@pytest.mark.parametrize("side,t0,t1,payload", corpora.BW2300_REF_ROBUST)
def test_the_robust_over_is_named_in_a_live_sized_buffer(side, t0, t1, payload):
    """What `_stream_over` now says instead of nothing.

    The ARQ layer scores a buffer one base over long — shorter than a record-2
    over — so the report has to work on the columns it reaches. Ending the buffer
    just inside the over's tail is the worst case that still has to speak."""
    need = rx.RECORDS[rx.BASE_LEVEL].ncols * rx.RECORDS[rx.BASE_LEVEL].dw50
    end = int((t1 - 0.4) * rx.FS)
    said = rx.unread_over(_side(side)[max(0, end - need - 24000):end], _GUARD_MIN, 0)
    assert said and "record 2" in said


@corpora.requires_bw2300_reference
@pytest.mark.parametrize("start", [1.0, 12.0, 19.0, 30.0, 50.0])
def test_nothing_else_in_the_session_is_called_an_unread_over(start):
    """The same buffer walked over the base-level overs, the control bursts and the
    empty band of the same recording. A report that fires on those is worse than
    the silence it replaces."""
    need = rx.RECORDS[rx.BASE_LEVEL].ncols * rx.RECORDS[rx.BASE_LEVEL].dw50
    end = int(start * rx.FS)
    assert rx.unread_over(_side("a2b")[max(0, end - need - 24000):end],
                          _GUARD_MIN, 0) is None


def test_noise_is_not_called_an_unread_over():
    rng = np.random.default_rng(3)
    n = rng.standard_normal(rx.RECORDS[_ROBUST].ncols * rx.RECORDS[_ROBUST].dw50)
    assert rx.unread_over(n, _GUARD_MIN, 4) is None


def test_the_column_count_is_what_the_interleaver_gap_pays_for():
    """228 columns and 24 reference columns are not two measurements but one.

    Record 3's bin tables are the 419 `Rnd` draws between its own interleaver
    permutations and record 4's seed — 395 column allocations then 24 reference
    classes. Record 2's gap is 252 draws and closes the same way on record 3's
    seed, so its columns and its references have to add to that; 816 coded bits at
    4 bits a column fixes the data half at 204, and 204 + 2*24 = 252 is the only
    split that pays for it. The 24 reference columns then land where the tape has
    them, which is what makes the arithmetic evidence rather than numerology.
    """
    r = rx.RECORDS[_ROBUST]
    ndata = r.coded // r.bpc
    _, s = tablegen.permutation(r.coded, tablegen.BW2300[_ROBUST][0])
    _, s = tablegen.permutation(r.n_info, s)
    for _ in range(ndata + 2 * 24):
        s = tablegen.step(s)
    assert s == tablegen.BW2300[rx.BASE_LEVEL][0]
    assert r.ncols == ndata + 24
    assert len(tablegen.map1_col2().nonzero()[0]) == 24


class _IO(VA.VaraIO):
    """Records what was keyed and what reached the host. Opens nothing."""

    def __init__(self):
        self.keyed: list[int] = []
        self.msgs: list[str] = []
        self.payloads: list[bytes] = []

    def key(self, on):
        if on:
            self.keyed.append(len(self.msgs))

    def tx(self, samples): ...

    def pending(self): ...

    def connected(self, *a): ...

    def data(self, payload): self.payloads.append(bytes(payload))

    def log(self, msg): self.msgs.append(msg)


def _linked():
    """A station as it stands mid-session: BW2300, linked, initiator, the turn
    with the peer and nothing queued. W9SSJ is the caller both ways here — A
    originated the session — which is the key the payload trailer is written to."""
    io = _IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", "W1AW", "W9SSJ"
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs, io


@corpora.requires_bw2300_reference
@pytest.mark.parametrize("side,t0,t1,payload", corpora.BW2300_REF_ROBUST)
def test_the_robust_over_reads_through_the_live_arq_path(side, t0, t1, payload):
    """The over as the receive stream delivers it, not as a decoder is handed it.

    Everything above this reads the tape offline with the extent already known.
    This feeds the same seconds to `on_rx_stream` in 0.1 s device blocks, with
    nothing telling the station where the burst is or which record it is at, and
    asks for the two things a session owes a gateway: the plaintext to the host,
    and one answer on the air.

    Reading a record-2 over needs both halves of the sizing. `_OVER_CARRY` is
    what makes a 4.86 s burst land whole in some scanned buffer — carrying the
    base level's 4.21 s forward leaves it straddling every one of them — and
    `_OVER_NEED` staying at the base level is what keeps the first scan where it
    was, so the answer still lands inside the turnaround.
    """
    hs, io = _linked()
    x = _side(side)
    n = int(0.1 * rx.FS)
    for i in range(int((t0 - 0.5) * rx.FS), int((t1 + 1.5) * rx.FS), n):
        hs.on_rx_stream(x[i:i + n])

    assert io.payloads == [payload], (
        f"{len(io.payloads)} payload(s) came out of the record-2 over", io.msgs)
    assert len(io.keyed) == 1, (f"keyed {len(io.keyed)}x for one over", io.msgs)
    assert any("tx per-over response" in m for m in io.msgs), io.msgs
    assert hs.turn == VA._TURN_PEER, "took a turn nobody offered"


@corpora.requires_bw2300_reference
@pytest.mark.parametrize("side,t0,t1,payload", corpora.BW2300_REF_ROBUST)
def test_the_robust_body_is_trimmed_at_its_own_record_length(side, t0, t1, payload):
    """Why the delivered payload is 37 bytes and not 48.

    Record 2's body is 48 bytes and carries the base level's trailer laid out
    from its own end: 0x14, the caller callsign's CRC byte, zero fill, then the
    two-byte per-frame field last. Read against the base level's 90 the zero run
    never closes — the 0x04 0x82 is at 46 rather than 88 — and the trailer goes
    to the host as eleven bytes of mail.
    """
    fr = rx.decode_over(_side(side), int((t0 - 0.2) * rx.FS),
                        int((t1 + 0.2) * rx.FS), level=_ROBUST)
    body = bytes(fr.payload)
    assert len(body) == rx.payload_bytes(_ROBUST) == 48
    assert phy.vara_payload(body, caller="W9SSJ", body_len=len(body)) == payload
    assert phy.vara_payload(body, caller="W9SSJ") == body      # read as a base body


def _link_setup_at(level: int, caller: str) -> bytes:
    """A link-setup frame laid out at ``level``'s body length.

    The layout is the one `link_setup_frame` builds and
    `test_linksetup_frame.test_bw500_zero_region_is_32_bytes` pins across the two
    bandwidths: the structure bytes keep their offsets from each end and only the
    zero fill between them changes. Record 2's body is 48 bytes rather than 90,
    and one such over is on the 2026-07 bench recording the bin table was measured
    against — but that recording is not in this tree, so this is the layout
    carried over rather than the frame itself.
    """
    wide = VF.link_setup_frame(caller)
    body = bytearray(rx.RECORDS[level].frame_bytes - 2)
    body[:10], body[-2:] = wide[:10], wide[-4:-2]
    return bytes(body) + crc16_genibus(bytes(body)).to_bytes(2, "big")


def test_a_caller_who_dropped_a_gear_is_still_a_caller():
    """The hole reading record 2 would otherwise open.

    `_peer_data_over` refuses to answer a link-setup, because a link-setup is a
    connect and answering one keys the transmitter at a station opening a session
    with somebody. That refusal used to run off a whitelist of the two base
    levels' frame lengths, which a 50-byte record-2 frame is not in — so the
    moment this receiver could read record 2, a caller who dropped a gear would
    have read as a DATA over of our own session.
    """
    fb = _link_setup_at(_ROBUST, "KC9GHZ")
    assert len(fb) == 50
    assert VF.is_link_setup(fb)
    assert VF.caller_from_link_setup(fb) == "KC9GHZ"


@corpora.requires_bw2300_reference
@pytest.mark.parametrize("side,t0,t1,payload", corpora.BW2300_REF_ROBUST)
def test_a_real_robust_over_is_not_read_as_a_link_setup(side, t0, t1, payload):
    """The control: bytes 7 and 8 are payload in a DATA over and fixed in a
    link-setup, and that is the whole of what separates the two at any length."""
    fr = rx.decode_over(_side(side), int((t0 - 0.2) * rx.FS),
                        int((t1 + 0.2) * rx.FS), level=_ROBUST)
    assert fr.crc_ok and not VF.is_link_setup(bytes(fr.frame_bytes))


def test_the_preamble_allocation_is_the_records_own_column_table():
    """Why record 2 needed no table of its own.

    The base level's published 12-entry preamble allocation is its column
    allocation table read from index 1 — a VB6 one-based walk of a zero-based
    array — and record 2's six are the same slice of its own.

    The count is the record's own field, not a sample span divided out. It comes
    to a fixed 6144 samples at records 3 and 2 and to 4096 at records 1 and 0,
    which is the 2 / 4 / 6 / 12 the 2026-09-04 ladder tapes read off the air.
    """
    assert tx.preamble_alloc(rx.BASE_LEVEL).tolist() == [
        24, 18, 18, 9, 12, 22, 11, 22, 17, 23, 12, 14]
    assert tx.preamble_alloc(_ROBUST).tolist() == \
        tablegen.base_bins_col2()[1:7].tolist() == [35, 40, 39, 36, 31, 24]
    symbols = {lv: len(tx.preamble_alloc(lv)) for lv in rx.INDEX_LEVELS}
    assert symbols == {0: 2, 1: 4, 2: 6, 3: 12}
    for lv, n in symbols.items():
        assert n * rx.RECORDS[lv].dw50 == (tx.PREAMBLE_SAMPLES if lv >= 2 else 4096)


@corpora.requires_bw2300_reference
@pytest.mark.parametrize("side,t0,t1,payload", corpora.BW2300_REF_ROBUST)
def test_the_robust_preamble_bins_are_the_ones_on_the_tape(side, t0, t1, payload):
    """Six one-hot symbols ahead of column 1, read against the generated stream.

    Two directions, two independent generators, one allocation table: that the
    same six allocations fall out of both streams is what makes them the record's
    and not this session's."""
    a = _side(side)
    c1, fr = _column_one(a, t0, t1)
    assert fr is not None
    r = rx.RECORDS[_ROBUST]
    heard = [int(np.abs(np.fft.rfft(np.asarray(
        a[c1 - i * r.dw50:c1 - (i - 1) * r.dw50], float))).argmax())
        for i in range(6, 0, -1)]
    assert heard == _tape_preamble_bins(side)


@corpora.requires_bw2300_reference
@pytest.mark.parametrize("side,t0,t1,payload", corpora.BW2300_REF_ROBUST)
def test_the_synthesised_robust_over_reproduces_the_recording(side, t0, t1, payload):
    """The transmit-side claim, the way the base level makes it: preamble plus
    body against the captured samples, and the preamble-free baseline that says
    the six symbols are doing the work."""
    a = _side(side)
    c1, fr = _column_one(a, t0, t1)
    r = rx.RECORDS[_ROBUST]
    pre = tx.synth_preamble(_tape_preamble_bins(side), _ROBUST)[rx._BASE_LEADIN * r.dw50:]
    data = tx.synth_frame(fr.frame_bytes, _ROBUST, over=None)
    assert len(pre) == tx.PREAMBLE_SAMPLES
    start = c1 - len(pre)
    without = _best_corr(np.concatenate([np.zeros(len(pre)), data]), a, start)
    with_pre = _best_corr(np.concatenate([pre, data]), a, start)
    assert without < 0.99, f"preamble-free baseline unexpectedly high: {without:.5f}"
    assert with_pre > 0.9999, f"correlation {with_pre:.6f}"


def test_a_synthesised_robust_over_decodes_at_its_own_level():
    """Round trip through the receive path a session uses, not through
    `decode_burst`: the burst is placed in a longer buffer with the frame start
    unknown, and `decode_over` has to find it."""
    pl = bytes(range(rx.payload_bytes(_ROBUST)))
    burst = tx.synth_burst(pl, _ROBUST, over=2)
    assert len(burst) == rx.burst_length(_ROBUST)
    buf = np.concatenate([np.zeros(7000), burst, np.zeros(7000)])
    fr = rx.decode_over(buf, 0, len(buf), level=_ROBUST)
    assert fr.crc_ok and fr.payload == pl


def test_the_robust_level_is_keyable_and_nothing_shifts_into_it():
    """What this level's transmit path is and is not.

    `KEYABLE_LEVELS` says the waveform renders. The VARA station engine keys
    this level for the one over that closes a delivery and the base level for
    every other, which is a decision about the block rather than a gear-shift
    [see `arq.phy.close_level`]. Kestrel's own ARQ never reaches it, for a harder
    reason: `speed_ladder[0]` is where `ArqFsm` opens and where a down-shift
    lands, and the ARQ block is sized once at the base level's capacity. A block
    rendered into record 2's 48-byte frame loses 42 of its 89 bytes and arrives
    with a clean CRC, which is what 180 bytes through the BW2300 loopback read as
    when the ladder started here."""
    assert _ROBUST in rx.KEYABLE_LEVELS
    assert _ROBUST not in phy.speed_ladder("2300")
    assert phy.speed_ladder("2300")[0] == rx.BASE_LEVEL
    assert (inspect.signature(VA.OF.data_over_tx)
            .parameters["level"].default == rx.BASE_LEVEL)


@corpora.requires_bw2300_reference
@pytest.mark.parametrize("side,base_overs", [("a2b", [4.0, 14.5]), ("b2a", [31.6])])
def test_the_session_seed_is_the_base_levels_too(side, base_overs):
    """What keeps the record-2 fit from being circular.

    `_REF_STREAM`'s two numbers are not fitted to the record-2 preambles they are
    then used to check. Each was solved from that side's *base-level* preambles
    against the published base allocation table, so the same seed walked forward
    has to reproduce those first — and the six record-2 allocations then fall out
    of two streams that were never told about each other.
    """
    a = _side(side)
    seed, before = _REF_STREAM[side]
    assert before == len(base_overs)
    rnd = tx.VB6Rnd(seed)
    r = rx.RECORDS[rx.BASE_LEVEL]
    for o, t in enumerate(base_overs):
        if o == 1:
            rnd.draw16()
        s = int(t * rx.FS)
        seg = np.asarray(a[s:s + int(5.5 * rx.FS)], float)
        c1 = next(s + onset + g * r.dw50
                  for onset, g, mag in rx._alignments(seg, rx._GUARD_TRIES)
                  if rx.check_frame(rx.onair_to_frame(
                      rx._onair_llr(mag[g:], rx.BASE_LEVEL), rx.BASE_LEVEL),
                      rx.BASE_LEVEL).crc_ok)
        heard = [int(np.abs(np.fft.rfft(np.asarray(
            a[c1 - i * r.dw50:c1 - (i - 1) * r.dw50], float))).argmax())
            for i in range(12, 0, -1)]
        assert heard == tx.preamble_bins(rnd, rx.BASE_LEVEL), f"over {o}"


# --------------------------------------------------------------------------- #
# Record 2 is what closes a delivery.
_MYCALL, _CALLED = "W9SSJ", "KC9GHZ"


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


def _connected():
    io = _TxIO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", _CALLED, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs, io


def _peer_control():
    """The peer's 11-symbol control burst — the grant, and the answer to an over."""
    return MK.synth_tone_pairs(VF.CONNECTED_ACK_2300)


def _keyed_overs(io):
    return [s for s in io.sent if len(s) > rx.burst_length(rx.BASE_LEVEL) // 2]


@corpora.requires_over_close_records
@pytest.mark.parametrize("path,overs", corpora.OVER_CLOSE_RECORDS,
                         ids=lambda v: getattr(v, "stem", ""))
def test_a_stock_close_drops_a_record_where_the_full_over_said_0x81(path, overs):
    """The record a stock delivery closes at, and the field that goes with it.

    Both are read off the audio: `index_guard` scores the reference columns of
    the two index records — the one the over is keyed at reads 24 of 24 and the
    other 7 or less, against a 9 of 24 ceiling for noise — and the body then
    decodes at that record and carries its own per-frame field.

    Eight deliveries, six sessions, both stations, both directions. Every close
    behind a full over announcing ``0x81`` is record 2, and the two behind
    ``0x89`` stay at the base level. ``0x89`` is not a value `_frame_field`
    produces, so the pair this station keys has one record to go to.
    """
    a = corpora.wav_mono(path)
    seen = []
    for start, want, field in overs:
        s = int((start - 0.08) * rx.FS)
        seg = np.asarray(a[s:s + int(5.4 * rx.FS)], float)
        got = {lv: (hits, of) for lv, hits, of, _ in rx.index_guard(seg)}
        assert got[want] == (24, 24), f"{start}s at record {want}: {got}"
        other = rx.BASE_LEVEL if want == _ROBUST else _ROBUST
        assert got[other][0] <= 7, f"{start}s reads at record {other} too: {got}"
        fr = rx.decode_over(seg, 0, len(seg), level=want)
        assert fr.crc_ok, f"{start}s does not decode at record {want}"
        assert fr.payload[-1] == field, f"{start}s field 0x{fr.payload[-1]:02x}"
        if field == 0x82:                       # a close, and what preceded it
            assert phy.over_is_last(fr.payload, "W9SSJ")
            assert want == (_ROBUST if seen[-1] == 0x81 else rx.BASE_LEVEL)
        seen.append(field)


def test_the_close_after_a_full_over_is_keyed_at_record_2():
    """What this station keys for a delivery that runs past one over.

    93 bytes is one full over and a 4-byte close, and the close carries its
    payload into record 2's 48-byte body. Decoded back at each over's own level,
    because a record-2 over is not readable at the base one — which is the whole
    reason the level has to be right.
    """
    hs, io = _connected()
    hs.send(b"x" * 89 + b"tail")
    hs.on_rx_audio(_peer_control())
    for _ in range(2):
        hs.on_rx_audio(_peer_control())
    overs = _keyed_overs(io)
    assert len(overs) == 2, io.msgs

    full = rx.decode_over(overs[0], 0, len(overs[0]), level=rx.BASE_LEVEL)
    assert full.crc_ok and full.payload[:89] == b"x" * 89

    close = rx.decode_over(overs[1], 0, len(overs[1]), level=_ROBUST)
    assert close.crc_ok, "the closing over does not decode at record 2"
    assert len(close.payload) == rx.payload_bytes(_ROBUST)
    assert phy.vara_payload(close.payload, caller=_MYCALL,
                            body_len=len(close.payload)) == b"tail"
    assert phy.over_is_last(close.payload, _MYCALL)


def test_a_delivery_that_fits_one_over_stays_at_the_base_level():
    """The close drops a level only where a full over is in front of it.

    A lone short over closes its own delivery and no recording holds a stock
    station dropping for one: the 65-byte login block this station has had read
    back byte-exact off a stock responder's data port is one of these.
    """
    hs, io = _connected()
    hs.send(b"y" * 65)
    hs.on_rx_audio(_peer_control())
    overs = _keyed_overs(io)
    assert len(overs) == 1, io.msgs
    fr = rx.decode_over(overs[0], 0, len(overs[0]), level=rx.BASE_LEVEL)
    assert fr.crc_ok
    assert phy.vara_payload(fr.payload, caller=_MYCALL,
                            body_len=len(fr.payload)) == b"y" * 65


def test_a_close_too_long_for_record_2_stays_at_the_base_level():
    """Record 2 carries 47 payload bytes and a close may be up to 88.

    Nothing on tape holds a stock station closing on one that does not fit, so
    the over goes where it went before rather than losing 41 bytes to a frame it
    is too big for.
    """
    hs, io = _connected()
    hs.send(b"z" * (89 + 60))
    hs.on_rx_audio(_peer_control())
    for _ in range(2):
        hs.on_rx_audio(_peer_control())
    overs = _keyed_overs(io)
    assert len(overs) == 2, io.msgs
    fr = rx.decode_over(overs[1], 0, len(overs[1]), level=rx.BASE_LEVEL)
    assert fr.crc_ok
    assert phy.vara_payload(fr.payload, caller=_MYCALL,
                            body_len=len(fr.payload)) == b"z" * 60
