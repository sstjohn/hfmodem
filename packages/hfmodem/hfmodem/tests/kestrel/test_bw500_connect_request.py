# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Calling a BW500-only station: the request kestrel keys, and what reads it back.

203 of the 1144 rows of the Winlink VARA list held here are published as VARA 500,
and 200 of its callsign/frequency channels carry VARA 500 and nothing else. Until the alphabet in :data:`VF.BW500_TONES` was solved,
kestrel could not so much as raise one: it keyed the BW2300 request, which a
BW500 responder ignores.

Three things are pinned here, in the order they have to hold:

  * the tones. A generated request lands entirely on the fourteen-carrier lattice,
    and reproduces a real VARA's own request off the loopback tape at both
    callsigns the corpus holds — the arbiter, not a round trip through ourselves.
  * the length. 31 payload tones, 41 symbols, 1.749 s. The count was open until
    2026-08-15 because the only on-air burst held ran 38 slots and stopped, on a
    recorder independently measured dropping ~9% of its samples; the loopback
    keying settles it.
  * the reading. ``HandshakeScanner`` names the callsign out of the synthesised
    audio, and still calls a BW2300 request BW2300 — a scanner that relabelled
    ordinary traffic would be worse than one that could not read BW500 at all.

What settles it is not any of the above but a real modem. Real VARA HF v4.9.0
over BlackHole, 2026-08-15, MYCALL W1AW, LISTEN ON, three runs:

    kestrel BW500  -> VARA armed BW500   PENDING x4, VARA keyed x4  (and x2, x2)
    kestrel BW2300 -> VARA armed BW500   PENDING x0, VARA keyed x0
    kestrel BW2300 -> VARA armed BW2300  PENDING x2, CONNECTED

The middle row is the control and it is the whole thesis: the same tool, the same
audio path, the same responder, one waveform answered and the other ignored. The
bottom row is the regression — parameterising the alphabet did not disturb the
sessions kestrel already completes.

The step behind the request — the BW500 link-setup that carries the caller's own
callsign, and with it a connect that reaches CONNECTED — is a separate solution
against separate evidence, and lives in ``test_bw500_handshake`` and
``test_linksetup_frame``. This file is the request and what reads it back.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

VM = corpora.harness("vara_monitor")

CALLS = ["W1AW", "KC2OUR", "W9SSJ", "NS0A-10", "KB9MMT", "WE3D-10", "VE1YZ",
         "AAAA1", "BBBB2"]

#: PTT ledger sample of the first keying in each loopback session, and the callsign
#: that session's request is keyed to (the CALLED station).
LOOPBACK = ((corpora.BW500_HANDSHAKE, 24576, 110592, "BBBB2"),
            (corpora.BW500_HANDSHAKE_REV, 24576, 110592, "AAAA1"))


def test_every_bw500_payload_tone_lands_on_the_fourteen_carrier_lattice():
    """The alphabet is 14 even carriers, bins 50..76 — 1171.9 to 1781.3 Hz on a
    46.9 Hz lattice. A tone outside it is 500 Hz of channel used wrongly."""
    lattice = VF.BW500_TONES.carriers
    assert lattice == frozenset(range(50, 77, 2))
    for cs in CALLS:
        bins = VF.payload_bins(cs, VF.CR500)
        assert len(bins) == 31
        assert set(bins) <= lattice, cs
    # Fourteen carriers is the whole alphabet and not a slice of a wider one: over
    # this many callsigns every one of them is reached.
    seen = {b for cs in CALLS for b in VF.payload_bins(cs, VF.CR500)}
    assert seen == lattice


def test_the_bw500_request_is_41_symbols_of_audio():
    """10 preamble + 31 payload at the 2048-sample advance = 1.749 s. The burst a
    BW500 responder is listening for is this long and not the 38 slots the one
    on-air recording could support."""
    tones = VF.handshake_tones("W1AW", VF.CR500)
    assert len(tones) == 41
    assert tones[:10] == list(VF.CR.preamble)          # preamble is bandwidth-free
    audio = MK.synth_burst("W1AW", VF.CR500)
    assert len(audio) == 40 * MK.HOP + MK.STRIDE
    assert abs(len(audio) / MK.FS - 1.7493) < 0.001


def test_bw2300_tones_are_untouched_by_the_bw500_alphabet():
    """The alphabet became a parameter to make room for BW500; BW2300 must come
    out of that byte for byte, or every session kestrel already completes breaks."""
    assert VF.TONE_ALPHABET == frozenset(29 + p + 14 * P + 2 * D
                                         for p in (0, 1) for P in range(5)
                                         for D in range(7))
    assert VF.payload_bins("W1AW", VF.CR)[:6] == [42, 65, 94, 31, 92, 87]
    assert VF.connect_request("2300") is VF.CR
    assert VF.connect_request("500") is VF.CR500


@corpora.requires_bw500_handshake
@pytest.mark.parametrize("path, lo, hi, called", LOOPBACK)
def test_the_generator_reproduces_a_real_varas_bw500_request(path, lo, hi, called):
    """The arbiter: a real VARA HF keyed these, and the PTT ledger in the session
    manifest says which station and to whom. Two sessions, two called callsigns —
    one would not separate the alphabet from a lucky seeding."""
    x = corpora.wav_mono(path)
    seg = x[max(0, lo - 4096):hi + 4096]
    at = MK.lock_preamble(seg, VF.CR500)
    assert at is not None, "the fixed preamble is bandwidth-independent"
    heard = MK.demod_tones(seg[at:], 41)
    assert heard == VF.handshake_tones(called, VF.CR500)
    # And the callsign is what makes it match: the other end of the same link
    # scores at chance on the same tones.
    other = "AAAA1" if called == "BBBB2" else "BBBB2"
    m, n = VF.payload_match(heard[10:], other, VF.CR500)
    assert m <= 4, f"{other} scored {m}/{n} on {called}'s request"


@corpora.requires_bw500_handshake
@pytest.mark.parametrize("path, lo, hi, called", LOOPBACK)
def test_a_real_bw500_request_keys_41_symbol_advances_of_audio(path, lo, hi, called):
    """Where the 31-vs-28 payload count is actually settled. Keyed audio, measured
    between the first and last sample above 2% of the burst's own peak, comes to
    83967 samples = 41.000 advances — the recorder in these sessions drops
    nothing, unlike the one that carried the 2026-08-14 on-air burst."""
    x = corpora.wav_mono(path)
    seg = x[lo:hi + 8192]
    lit = np.flatnonzero(np.abs(seg) > np.abs(seg).max() * 0.02)
    assert lit[0] > 0, "the burst starts inside the window, so its length is real"
    assert round((lit[-1] - lit[0] + 1) / MK.HOP, 2) == 41.00


def _scan(audio: np.ndarray, calls=CALLS):
    """Every handshake the scanner finds in AUDIO, driven as the monitor drives
    it. Padded either side because the scanner needs a lattice run around a burst,
    and with noise rather than silence so nothing rests on a digital-black floor."""
    rng = np.random.default_rng(20260815)
    pad = rng.normal(0, 0.02, 6 * MK.FS)
    x = np.concatenate([pad, audio, pad])
    scanner = VM.HandshakeScanner(calls)
    out = []
    for i in range(0, len(x), 4800):
        out += scanner.push(x[i:i + 4800])
    return [r for _, r in out]


def test_the_scanner_reads_the_callsign_out_of_a_generated_bw500_request():
    """The round trip the transmitter is graded on: synthesise a request for a
    station, and the receiver names that station back."""
    for cs in ("W1AW", "WE3D-10", "NS0A-10"):
        found = _scan(MK.synth_burst(cs, VF.CR500))
        named = [r for r in found if r.gateway == cs]
        assert named, f"{cs}: scanner found {[(r.kind, r.info) for r in found]}"
        assert named[0].kind == "CR"
        assert "BW500" in named[0].info
        assert "31/31 payload" in named[0].quality


def test_the_scanner_still_calls_a_bw2300_request_bw2300():
    """The regression that matters more than the feature: teaching the scanner a
    second alphabet must not relabel the traffic it already reads. A BW2300
    request cannot beat chance on the BW500 lattice, so the label follows the
    tones rather than a guess."""
    found = _scan(MK.synth_burst("W1AW", VF.CR))
    named = [r for r in found if r.gateway == "W1AW"]
    assert named, [(r.kind, r.info) for r in found]
    assert "BW500" not in named[0].info
    assert "31/31 payload" in named[0].quality


#: The 15 payload tones a real VARA HF v4.9.0 armed at BW500 keyed back at
#: kestrel's request on 2026-08-15, read off the BlackHole recording of its own
#: output. Kept as tones rather than as audio: it is what the descriptor is
#: graded against, and W1AW is a second callsign for it, the loopback corpus
#: having supplied only BBBB2.
BENCH_RESPONSE_W1AW = [54, 66, 74, 52, 68, 54, 76, 68, 62, 58, 54, 64, 58, 56, 68]


def test_the_generator_reproduces_a_real_varas_bw500_connect_response():
    """The answer, not just the call. A responder that hears a BW500 request keys
    this back, and reading it is how an operator learns the gateway is alive."""
    assert VF.payload_bins("W1AW", VF.CONNECT_RESPONSE_500) == BENCH_RESPONSE_W1AW
    assert set(BENCH_RESPONSE_W1AW) <= VF.BW500_TONES.carriers
    # BW2300's own pre-advance is 210 draws short of this one and lands nowhere
    # near it, so the pair is doing work rather than inheriting.
    m, _ = VF.payload_match(BENCH_RESPONSE_W1AW, "W1AW", VF.CONNECT_RESPONSE)
    assert m == 0


def test_the_scanner_names_a_bw500_connect_response():
    """Same round trip for the responder's half, so a monitor logs a BW500
    exchange as an exchange rather than as two unidentified bursts."""
    found = _scan(MK.synth_burst("W1AW", VF.CONNECT_RESPONSE_500))
    named = [r for r in found if r.gateway == "W1AW"]
    assert named, [(r.kind, r.info) for r in found]
    assert named[0].kind == "connect-response"
    assert "BW500" in named[0].info


def test_a_bw500_request_for_nobody_on_the_list_is_still_reported_as_bw500():
    """A monitor hears stations it was not told about. The lattice says the sender
    was at BW500 whether or not any candidate callsign regenerates its payload,
    and that is worth logging — it is how the 2026-08-14 recording was read at all."""
    found = _scan(MK.synth_burst("K7ABC-9", VF.CR500), calls=["W1AW", "KC2OUR"])
    assert found, "a burst addressed to nobody on the list is still a burst"
    assert any("BW500" in r.info and not r.gateway for r in found), \
        [(r.kind, r.info, r.quality) for r in found]
