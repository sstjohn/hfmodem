# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW2750 interop proof — kestrel reads and keys REAL VARA HF BW2750 base overs.

BW2750's base level is BW2300 record 3's index law on a wider comb: 20 bins at
6..25 instead of 16 at 9..24, the same 512-sample no-CP symbol, the same 395
emission columns, the same 24 reference columns, the same coding chain and the
same 92-byte frame. Its bin-allocation table is the same LCG stream quantised to
20 bins instead of 16 (``tablegen.alloc_col3(span=20, first_bin=6)``). Nothing in
the record is measured — the table is closed form — so what these fixtures test is
that the closed form is VARA's.

Two independent bodies of audio, and the argument needs both:

  * the 2026-07-21 Wine loopback, whose link-setup frame is known in advance. The
    same tape holds a second, unlogged session 50 s later that is BW2300 — same
    modems, same callsigns — so one file carries a 20-bin over the base record
    cannot read and a 16-bin over it can.
  * three 2026-07-24 off-air Winlink sessions that VARA itself reported as
    ``CONNECTED ... 2750``, whose plaintext was not known in advance. Seven overs
    from three gateways, all CRC-clean, reading as continuous B2F greeting text.

The MFSK session frames of the same loopback are the second half: every one of
them regenerates from the descriptor ``vara_frames`` already holds, over an
alphabet of BW2750's own — 84 carriers at bins 22..105 — with the connect pair
carrying its own state the way BW500's does. Both cables, so a burst is
attributed by the file that holds it, and the BW2300 session on the same tape is
the control at every step.
"""
from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_frames as vf
from hfmodem.kestrel.vara import vara_mfsk as mk
from hfmodem.kestrel.vara import vara_ofdm as of
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.corpora import (BW2750_CAPTURE, REGRESS_FIXTURES,
                                           requires_bw2750_capture,
                                           requires_over_continue_2750,
                                           requires_regress_fixtures)

_BASE = rx.BASE_LEVELS["2750"]

#: The greeting each off-air session opens with, and how many of its overs are
#: BW2750 base overs. The rest of each recording's brackets are tail-of-burst and
#: control-plus-noise: their max-bin energy fraction is 0.18-0.31 against 0.99 for
#: a base over, at every symbol length tried.
_OFFAIR = (("qso_vara_ns0a", 3, b"Welcome to the NS0A Gateway"),
           ("qso_vara_kc9ghz", 2, b"daily minutes remaining with KC9GHZ (EN62BK)"),
           ("qso_vara_ko2f", 2, b"Welcome to the KO2F HF Gateway System"))


@cache
def _loopback():
    # The capture harness died before closing this file, so its RIFF data-size
    # field is 0 and every wav reader returns an empty array. The samples are
    # there; they start at byte 44.
    a = np.fromfile(str(BW2750_CAPTURE), dtype=np.float32, offset=44).astype(float)
    return a / np.abs(a).max()


@cache
def _offair(name: str):
    from scipy.io import wavfile
    _, a = wavfile.read(str(REGRESS_FIXTURES / f"{name}.wav"))
    a = np.asarray(a, float)
    return a / np.abs(a).max()


@cache
def _decoded(name: str):
    """``[(2750 frame, rec3 frame)]``, one pair per detected over."""
    a = _offair(name)
    return [(rx.decode_over(a, s, e, level=_BASE), rx.decode_over(a, s, e))
            for s, e in rx.detect_overs(a)]


# --------------------------------------------------------------------------- #
@requires_bw2750_capture
def test_loopback_link_setup_reads_only_at_the_2750_comb():
    """The one wideband OFDM burst of the BW2750 session, against a known frame."""
    a = _loopback()
    s, e = rx.detect_overs(a)[0]
    assert not rx.decode_over(a, s, e).crc_ok, "BW2300 record 3 read a 20-bin over"
    fr = rx.decode_over(a, s, e, level=_BASE)
    assert fr.crc_ok
    assert fr.frame_bytes == vf.link_setup_frame("AAAA1")
    assert vf.is_link_setup(fr.frame_bytes)
    assert vf.caller_from_link_setup(fr.frame_bytes) == "AAAA1"


@requires_bw2750_capture
def test_the_bw2300_control_on_the_same_tape_reads_only_at_rec3():
    """Same file, same modems, same callsigns, 50 s later — and 16 bins wide.

    Without it the finding is one over that one record reads; with it the two
    records are separated on one recording."""
    a = _loopback()
    s, e = rx.detect_overs(a)[1]
    assert not rx.decode_over(a, s, e, level=_BASE).crc_ok
    fr = rx.decode_over(a, s, e)
    assert fr.crc_ok and fr.frame_bytes == vf.link_setup_frame("AAAA1")


@requires_bw2750_capture
def test_the_two_records_separate_before_anything_is_decoded():
    """The reference columns alone tell the combs apart, on both overs.

    24 of 24 at the record that owns the over and 6 at the other, in both
    directions — so a receiver that scores the wrong comb reads a BW2750 over as
    band noise rather than as a frame it failed on."""
    a = _loopback()
    wide, narrow = (a[s:e] for s, e in rx.detect_overs(a)[:2])
    assert rx._partial_scores(wide, _BASE)[:2] == (24, 24)
    assert rx._partial_scores(wide, rx.BASE_LEVEL)[0] <= 9
    assert rx._partial_scores(narrow, rx.BASE_LEVEL)[:2] == (24, 24)
    assert rx._partial_scores(narrow, _BASE)[0] <= 9


@requires_bw2750_capture
def test_the_2750_preamble_is_the_records_own_columns():
    """The 12 training symbols, off the tape, against the base level's own law.

    The preamble has no table of its own at BW2750 either: it walks the record's
    column table from index 1 and offsets each by ``Int(Rnd*16)`` — the draw is 16
    wide on a 20-bin comb — reduced into the record's span. Twelve of twelve, at
    offset 5 in the ``PREAMBLE_SEED`` stream; the BW2300 link-setup the same
    modem keyed 50 s later sits at offset 0, so the five draws are the
    bandwidth's lead and not the process's history  [see tx.STREAM_LEAD]."""
    a = _loopback()
    for idx, level in ((0, _BASE), (1, rx.BASE_LEVEL)):
        r = rx.RECORDS[level]
        s, e = rx.detect_overs(a)[idx]
        seg = a[s:e]
        onset, g, _ = next(rx._alignments(seg, 1, level))
        pre = seg[onset + (g - 12) * r.dw50:onset + g * r.dw50].reshape(12, r.dw50)
        observed = np.abs(np.fft.rfft(pre, axis=1))[:, :r.first_bin + r.span].argmax(1)
        assert tx.over_preamble_bins(0, level=level) == list(observed)
    # The lead is the bandwidth's, so every BW2750 record carries it and no
    # BW2300 one does.
    assert set(tx.STREAM_LEAD) == set(rx.INDEX_LEVELS_BW["2750"])
    assert set(tx.STREAM_LEAD.values()) == {5}


# --------------------------------------------------------------------------- #
@requires_regress_fixtures
@pytest.mark.parametrize("name,nclean,text", _OFFAIR)
def test_offair_gateway_overs_decode_at_2750(name, nclean, text):
    """Real Winlink gateways, off air, plaintext not known in advance."""
    pairs = _decoded(name)
    clean = [f.frame_bytes for f, _ in pairs if f.crc_ok]
    assert len(clean) == nclean, f"{name}: {len(clean)} CRC-clean overs at BW2750"
    assert any(text in b for b in clean)
    assert b"[WL2K-5.0-B2FWIHJM$]" in b"".join(clean)


@requires_regress_fixtures
@pytest.mark.parametrize("name,nclean,text", _OFFAIR)
def test_the_bw2300_base_record_reads_none_of_them(name, nclean, text):
    assert not any(f.crc_ok for _, f in _decoded(name))


@requires_regress_fixtures
@pytest.mark.parametrize("name", ["pos_vara_session", "pos_vara_netherlands",
                                  "neg_noise_chatham", "neg_wide_560hz",
                                  "edge_long_80m", "edge_winlink_oceania"])
def test_what_the_new_record_costs_the_recordings_it_should_decline(name):
    """The false-accept floor for the 20-bin comb, on the corpus every wide search
    here is measured against.

    Six recordings that hold no BW2750 over — a real BW2300 VARA session among
    them, which is the adversarial cell — score at most 9 of 24 reference columns
    at the new record, against 24 of 24 for the overs it is for. Measured across
    the whole regression corpus, 30 brackets in 17 recordings, the worst is the
    same 9."""
    from scipy.io import wavfile
    _, a = wavfile.read(str(REGRESS_FIXTURES / f"{name}.wav"))
    a = np.asarray(a, float)
    if a.ndim > 1:
        a = a[:, 0]
    a /= np.abs(a).max()
    brackets = rx.detect_overs(a)
    assert brackets, f"{name}: nothing to score"
    assert max(rx._partial_scores(a[s:e], _BASE)[0] for s, e in brackets) <= 9


# --------------------------------------------------------------------------- #
def test_the_2750_record_is_rec3_on_a_wider_comb():
    r, base = rx.RECORDS[_BASE], rx.RECORDS[rx.BASE_LEVEL]
    assert (r.span, r.first_bin, r.stride) == (20, 6, 1)
    assert (r.dw50, r.cp, r.ncols, r.lead) == (base.dw50, base.cp, base.ncols, base.lead)
    assert (r.n_info, r.coded, r.frame_bytes, r.bpc) == (base.n_info, base.coded,
                                                         base.frame_bytes, base.bpc)
    # the same coding chain, reached through the record's own interleaver column
    assert r.il_col == base.level
    assert np.array_equal(rx.turbo_perm(_BASE), rx.turbo_perm(rx.BASE_LEVEL))
    assert np.array_equal(rx.chan_perm(_BASE), rx.chan_perm(rx.BASE_LEVEL))
    # and the same 24 reference columns in the same places
    assert np.array_equal(rx._ROLES[_BASE][0], rx._ROLES[rx.BASE_LEVEL][0])
    # not a rung of the BW2300 ladder
    assert _BASE not in rx.LEVELS and _BASE not in rx.INDEX_LEVELS
    assert rx.INDEX_LEVELS_BW["2750"] == (100, 101, 102, _BASE)


def test_a_2750_over_lights_the_twenty_bins_and_round_trips():
    frame = vf.link_setup_frame("W9SSJ")
    x = tx.synth_frame(frame, _BASE, over=0)
    assert len(x) == rx.burst_length(_BASE) == 209408          # 4.363 s
    r = rx.RECORDS[_BASE]
    blocks = x[r.lead * r.dw50:].reshape(-1, r.dw50)
    lit = np.abs(np.fft.rfft(blocks, axis=1)).argmax(1)
    assert set(lit.tolist()) == set(range(6, 26))
    assert rx.decode_burst(x, _BASE).frame_bytes == frame


def test_link_setup_rides_the_bandwidths_base_level():
    assert "2750" in of.LINK_SETUP_BW
    assert vf.LINK_SETUP_BODY["2750"] == 90
    x = of.link_setup_tx("W9SSJ", bw="2750")
    assert of.link_setup_rx(x, bw="2750") == "W9SSJ"
    # the comb is the bandwidth's, so the other one cannot read it
    assert of.link_setup_rx(x, bw="2300") is None
    assert of.link_setup_rx(of.link_setup_tx("W9SSJ", bw="2300"), bw="2750") is None


def test_a_2750_data_over_carries_the_same_ninety_byte_body():
    from hfmodem.kestrel.arq import phy
    assert phy.base_level("2750") == _BASE
    assert phy.payload_size("2750") == phy.payload_size("2300") == 89
    body = bytes(range(90))
    x = of.data_over_tx(body, over=1, level=_BASE)
    assert rx.decode_burst(x, _BASE).payload == body


def test_a_session_at_2750_identifies_its_peers_over_and_a_2300_one_does_not():
    """The wiring, not the codec: a CONNECTED station reads the peer's over at
    the record its own bandwidth's base level names, and at no other."""
    from hfmodem.kestrel.vara import vara_arq as va

    body = b"CMS via KO2F >\r".ljust(90, b"\x00")
    burst = of.data_over_tx(body, over=1, level=_BASE)

    def station(bw):
        hs = va.VaraStationHandshake(["W9SSJ"], va.VaraIO(), bw=bw)
        hs.role, hs.called, hs.caller = "initiator", "KO2F", "W9SSJ"
        hs.state, hs.step = va.VaraState.CONNECTED, va._I_CONNECTED
        return hs

    assert station("2750")._peer_data_over(burst) == [body]
    assert station("2300")._peer_data_over(burst) == []
    assert station("500")._peer_data_over(burst) == []


# --------------------------------------------------------------------------- #
# The MFSK session frames of the same loopback, both cables. Each row is a burst
# read off the tape by ``vara_mfsk.demod_tones`` — the modem's own symbols, not a
# round trip through kestrel's generator — at the second it starts, on the cable
# of the station that keyed it, and the kind it regenerates from over
# ``BW2750_TONES``. The connect pair is the BW2750 descriptor pair; the rest are
# the BW2300 descriptors unchanged.
_LINK = ("AAAA1", "BBBB2")
_SESSION_2750 = (
    ("a2b", 0.597, vf.CR2750),
    ("b2a", 2.448, vf.CONNECT_RESPONSE_2750),
    ("a2b", 8.779, vf.SESSION_CONFIRM),
    ("a2b", 10.171, vf.SESSION_KEEPALIVE_A),
    ("a2b", 12.256, vf.SESSION_KEEPALIVE_B),
)
#: The BW2300 session 50 s later, same modems, same callsigns: the control.
_SESSION_2300 = (
    ("a2b", 56.034, vf.CR),
    ("b2a", 57.899, vf.CONNECT_RESPONSE),
    ("b2a", 66.459, vf.SESSION_TURN_REQUEST_RESPONDER),
)
_IDS = {row: f"{row[0]}-{row[1]}-{row[2].name}" for row in _SESSION_2750 + _SESSION_2300}


@cache
def _cable(name: str):
    path = BW2750_CAPTURE.with_name(BW2750_CAPTURE.name.replace("a2b", name))
    a = np.fromfile(str(path), dtype=np.float32, offset=44).astype(float)
    return a / np.abs(a).max()


#: The band a BW2750 receiver searches: 84 carriers at bins 22..105, seven below
#: and seven above BW2300's 29..98. It is the shipped reader that reads these
#: bursts, at the band a session armed at 2750 gives it  [vara_mfsk, band_for];
#: at BW2300's band the response's own alphabet reaches past both ends.
_BAND = mk.band_for("2750")


def _read(cable: str, t: float, kind: vf.BurstKind) -> list[int]:
    """The burst's tones over the BW2750 search band, at the alignment that locks
    its fixed preamble."""
    x = _cable(cable)
    n_sym = len(kind.preamble) + kind.n_payload
    at = int(t * mk.FS)
    env = np.sqrt(np.convolve(x[at - 2048:at + 4096] ** 2, np.ones(64) / 64, "same"))
    onset = at - 2048 + int(np.argmax(env > 0.3 * env.max()))
    best = None
    for start in range(onset - 512, onset + 1024, 64):
        tones = mk.demod_tones(x[start:start + n_sym * mk.HOP + mk.STRIDE],
                               n_sym, _BAND)
        hit = sum(1 for a, b in zip(tones, kind.preamble) if a == b)
        if best is None or hit > best[0]:
            best = (hit, tones)
    assert best[0] == len(kind.preamble), f"preamble never locked at {cable} {t}"
    return best[1]


def _keyed_to(kind: vf.BurstKind) -> str:
    return _LINK[0] if kind.keyed_by == "caller" else _LINK[1]


@requires_bw2750_capture
@pytest.mark.parametrize("cable,t,kind", _SESSION_2750, ids=_IDS.get)
def test_a_2750_session_frame_regenerates_on_the_2750_alphabet(cable, t, kind):
    tones = _read(cable, t, kind)
    kind = vf.for_bw(kind, "2750")
    assert vf.payload_match(tones[len(kind.preamble):], _keyed_to(kind), kind) \
        == (kind.n_payload, kind.n_payload)
    assert vf.recognize(tones, _keyed_to(kind), kind)


@requires_bw2750_capture
@pytest.mark.parametrize("cable,t,kind", _SESSION_2750, ids=_IDS.get)
def test_the_bw2300_reading_of_the_same_burst_is_chance(cable, t, kind):
    """The same descriptor over the BW2300 alphabet, or the BW2300 connect pair
    for the two handshake bursts — what a station armed at 2300 would look for."""
    tones = _read(cable, t, kind)
    wide = {vf.CR2750: vf.CR, vf.CONNECT_RESPONSE_2750: vf.CONNECT_RESPONSE}
    at2300 = vf.for_bw(wide.get(kind, kind), "2300")
    m, n = vf.payload_match(tones[len(kind.preamble):], _keyed_to(kind), at2300)
    assert m <= 3, f"{at2300.name} reads a BW2750 burst {m}/{n}"


@requires_bw2750_capture
@pytest.mark.parametrize("cable,t,kind", _SESSION_2300, ids=_IDS.get)
def test_the_bw2300_session_on_the_same_tape_reads_at_2300_and_not_at_2750(cable, t, kind):
    tones = _read(cable, t, kind)
    assert vf.recognize(tones, _keyed_to(kind), kind)
    wide = vf.for_bw(kind, "2750")
    m, _ = vf.payload_match(tones[len(kind.preamble):], _keyed_to(kind), wide)
    assert m <= 3


@requires_bw2750_capture
def test_the_2750_request_is_the_wide_alphabet_in_a_state_of_its_own():
    """A station listening at any bandwidth reads the request; the state behind
    the preamble says which bandwidth is being asked for."""
    tones = _read("a2b", 0.597, vf.CR2750)
    assert vf.CR2750.tones is vf.BW2300_TONES
    assert vf.payload_states(tones[10:], vf.CR2750)
    assert not vf.recognize(tones, "BBBB2", vf.CR)
    assert not vf.recognize(tones, "BBBB2", vf.CR500)
    assert vf.recognize(tones, "BBBB2", vf.CR2750)
    assert (vf.CR.seed_off, vf.CR500.seed_off, vf.CR2750.seed_off) == (50, 57, 58)


def test_the_2750_alphabet_is_the_wide_one_moved_seven_bins():
    """Every BW2750 carrier is a BW2300 carrier seven bins up or down, and the
    alphabet is one first-draw value wider: 84 carriers at 22..105 against 70 at
    29..98."""
    wide, wider = vf.BW2300_TONES.carriers, vf.BW2750_TONES.carriers
    assert (min(wider), max(wider), len(wider)) == (22, 105, 84)
    assert all(c - 7 in wide or c + 7 in wide for c in wider)
    assert vf.BW2750_TONES == vf.ToneAlphabet(22, 6)
    assert vf.ALPHABETS["2750"] is vf.BW2750_TONES
    assert mk.band_for("2750") == (22, 105)
    assert mk.band_for("2300") == mk.band_for("500") == (29, 98)


def test_a_2750_session_keys_its_own_pair_and_the_shared_frames_on_its_alphabet():
    assert vf.connect_request("2750") is vf.CR2750
    assert vf.connect_response("2750") is vf.CONNECT_RESPONSE_2750
    assert vf.for_bw(vf.CR, "2750") is vf.CR2750
    assert vf.for_bw(vf.CONNECT_RESPONSE, "2750") is vf.CONNECT_RESPONSE_2750
    for kind in (vf.SESSION_CONFIRM, vf.SESSION_KEEPALIVE_A, vf.SESSION_TURN_RELEASE,
                 vf.SESSION_TURN_REQUEST_RESPONDER, vf.SESSION_DRAINED_RESPONDER):
        at = vf.for_bw(kind, "2750")
        assert at.tones is vf.BW2750_TONES
        assert (at.seed_off, at.preadv, at.keyed_by, at.preamble) \
            == (kind.seed_off, kind.preadv, kind.keyed_by, kind.preamble)
    # Both the final ACK and continue frame use this bandwidth's measured tails.
    assert vf.control_bursts("W9SSJ", "2750") == (
        vf.CONTROL_BURST_CALLER_2750, vf.CONTROL_BURST_RESPONDER_2750)
    assert vf.control_bursts("W9SSJ", "2750") != vf.control_bursts("W9SSJ", "2300")
    assert vf.over_continue("W9SSJ", "2750") == vf.OVER_CONTINUE_CALLER_2750
    assert vf.over_continue("W9SSJ", "2750") != vf.over_continue("W9SSJ", "2300")


# --------------------------------------------------------------------------- #
# The 8-symbol answer to an intermediate over, off two BW2750 cables.
def _two_tone(x: np.ndarray, at: float, n_sym: int, band) -> tuple:
    """The burst's pairs at the offset that puts the most energy in its own peak
    bins — a criterion that knows nothing about the frame, which is what lets two
    copies of an unnamed one be compared at all."""
    seg = x[int((at - 0.20) * mk.FS):int(at * mk.FS) + n_sym * mk.HOP + 2 * mk.NFFT]
    best = (-1.0, ())
    last = len(seg) - mk._WOFF - (n_sym - 1) * mk.HOP - mk.NFFT
    for off in range(0, last, 4):
        score = 0.0
        for i in range(n_sym):
            lo = off + mk._WOFF + i * mk.HOP
            S = np.abs(np.fft.rfft(seg[lo:lo + mk.NFFT] * np.hanning(mk.NFFT))) ** 2
            score += S.max() / S.sum() if S.any() else 0.0
        if score > best[0]:
            best = (score, tuple(tuple(sorted(p)) for p in
                                 mk.demod_tone_pairs(seg[off:], n_sym, band)))
    return best[1]


@requires_over_continue_2750
@pytest.mark.parametrize("path,at", [(p, t) for p, ts in corpora.OVER_CONTINUE_2750
                                     for t in ts])
def test_the_2750_continue_burst_regenerates_off_both_callers_cables(path, at):
    """Six copies over two sessions, and the constant is what every one of them
    reads. Padded past one over each way, so the caller had three intermediate
    overs to answer in each."""
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    read = _two_tone(x, at, vf.OVER_CONTINUE_NSYM, mk.band_for("2750"))
    assert read == tuple(tuple(sorted(p)) for p in vf.OVER_CONTINUE_CALLER_2750)


@requires_over_continue_2750
def test_the_responders_copies_carry_symbols_no_bw2300_burst_can_reach():
    """What says the seven behind the lead are drawn on the wide alphabet rather
    than borrowed. The caller's own copy happens to fall inside BW2300's band;
    the responder's does not, and neither end's is the other's."""
    band = mk.band_for("2750")
    seen = set()
    for path, times in corpora.OVER_CONTINUE_2750_RESPONDER:
        x = corpora.wav_mono(path)
        x = x / (np.abs(x).max() or 1.0)
        for at in times:
            pairs = _two_tone(x, at, vf.OVER_CONTINUE_NSYM, band)
            assert pairs[0] == tuple(sorted(vf.OVER_CONTINUE_CALLER_2750[0]))
            assert pairs[1:] != tuple(tuple(sorted(p)) for p in
                                      vf.OVER_CONTINUE_CALLER_2750[1:])
            seen.update(b for p in pairs for b in p)
    lo, hi = mk.band_for("2300")
    assert seen - set(range(lo, hi + 1)), "every symbol sits inside BW2300's band"
