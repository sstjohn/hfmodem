# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The tone search is confined to the MFSK alphabet's band, and has to be.

A received burst carries images of its own tones at odd multiples of each
carrier. They cost nothing while the wanted signal dominates the whole spectrum,
and everything once it does not: on the 2026-07-26 recording of a gateway
answering kestrel, the in-band power sits only 9.0 dB above the out-of-band power
and the 3f image beats the fundamental for every tone above about 1.1 kHz. An
argmax over the full 0-24 kHz rfft then reads 3x the true carrier for most
symbols of the connect-response, ``lock_preamble`` returns None, and a gateway
that answered on the air presents to the operator as silence.

The BW2300 alphabet is 29 + parity + 14P + 2D with P in 0..4 and D in 0..6, so
no legitimate carrier of a BW2300 or BW500 session lies outside bins 29..98
(679.7-2296.9 Hz); a BW2750 session reads its own alphabet as well and searches
22..105. The first test asserts each band against the generator and every fixed
preamble, so neither can drift away from the waveform it is derived from.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

_CALL = "KB9MMT"
# The gateway's connect-response, 56.03 s into the recording; the bracket is the
# couple of seconds around it, as the segmenter would hand it over.
_RESPONSE_SPAN = (55.5, 58.0)


def _naive_tones(samples: np.ndarray, n_sym: int) -> list[int]:
    """demod_tones as it was: peak of the whole rfft, no band limit."""
    win = np.hanning(MK.NFFT)
    off = (MK.STRIDE - MK.NFFT) // 2
    out = []
    for k in range(n_sym):
        a = k * MK.HOP + off
        seg = np.asarray(samples[a:a + MK.NFFT], dtype=np.float64)
        seg = np.concatenate([seg, np.zeros(MK.NFFT - seg.size)])
        out.append(int(np.argmax(np.abs(np.fft.rfft(seg * win)))))
    return out


_CALLS = ("W9SSJ", "KB9MMT", "NS0A", "KC9GHZ", "W1AW-10")


def test_the_band_covers_every_carrier_the_waveform_can_emit():
    """Per bandwidth, because the band is the session's  [vara_mfsk.band_for]."""
    for bw, alphabet in VF.ALPHABETS.items():
        every = set(alphabet.carriers) | set(VF.BW2300_TONES.carriers)
        every.update(*(k.preamble for k in VF.BURSTS.values()))
        every.update(VF.SESSION_RESPONSE_2300)
        assert MK.band_for(bw) == (min(every), max(every)), (
            f"the BW{bw} tone band no longer matches the tones that bandwidth "
            "reads; a burst whose carrier falls outside it can never be "
            "demodulated")
    assert (MK.BIN_LO, MK.BIN_HI) == MK.band_for("2300")

    of_bw = {a: bw for bw, a in VF.ALPHABETS.items()}
    for name, kind in VF.BURSTS.items():
        lo, hi = MK.band_for(of_bw[kind.tones])
        emitted = set(kind.preamble).union(
            *(VF.payload_bins(c, kind) for c in _CALLS))
        assert min(emitted) >= lo and max(emitted) <= hi, (
            f"{name} emits outside the band its own bandwidth searches")


def test_odd_harmonic_images_do_not_capture_the_demodulator():
    """Images at 3f and 5f, each louder than the fundamental, must not be read.

    Synthetic twin of the off-air failure below, so the protection survives the
    corpus being absent: the naive full-band peak is asserted to fall for the
    images, and the demodulator is required not to.

    The band limit separates an image from its fundamental only while the image
    lands outside the alphabet, and for the four lowest carriers it does not:
    3 x 29..32 is 87..96, which are legitimate tones. Those symbols are lost to a
    strong 3f whatever the search does — the burst still recognizes, because one
    tone in fifteen is what the acceptance threshold has margin for. Widening the
    band cannot help them and would cost every symbol above 1.1 kHz.
    """
    kind = VF.CONNECT_RESPONSE
    tones = VF.handshake_tones(_CALL, kind)
    audio = (MK.synth_tones(tones, amplitude=0.3)
             + MK.synth_tones([3 * c for c in tones], amplitude=0.6)
             + MK.synth_tones([5 * c for c in tones], amplitude=0.45))

    captured = sum(1 for got, want in zip(_naive_tones(audio, len(tones)), tones)
                   if got != want)
    assert captured >= len(tones) - 1, (
        "the images are not strong enough to fool an unbounded search, so this "
        "test would pass with or without the band limit")

    got = MK.demod_tones(audio, len(tones))
    aliased = [i for i, t in enumerate(tones) if 3 * t <= MK.BIN_HI]
    assert [i for i, (g, t) in enumerate(zip(got, tones)) if g != t] == aliased
    at = MK.lock_preamble(audio, kind)
    assert at is not None and at < MK.HOP, "the preamble did not lock inside its own burst"
    assert VF.recognize(MK.demod_tones(audio[at:], len(tones)), _CALL, kind)


@corpora.requires_onair_connect_attempt
def test_a_real_gateway_reply_survives_its_own_harmonics():
    """The recording the defect was found on: KB9MMT answering hfmodem.kestrel.

    Unpatched this locks nothing at all — 16 of the 23 symbols demodulate to
    exactly 3x their true carrier — which the connect tool reports as "not an
    MFSK handshake burst" and the operator sees as no answer.
    """
    fs = 48000
    x = corpora.wav_mono(corpora.ONAIR_CONNECT_ATTEMPT)
    x = x / (np.abs(x).max() or 1.0)
    seg = x[int(_RESPONSE_SPAN[0] * fs):int(_RESPONSE_SPAN[1] * fs)]

    kind = VF.CONNECT_RESPONSE
    at = MK.lock_preamble(seg, kind)
    assert at is not None, "no lock on a bracket known to hold the gateway's answer"

    n = len(kind.preamble) + kind.n_payload
    tones = MK.demod_tones(seg[at:], n)
    m, total = VF.payload_match(tones[len(kind.preamble):], _CALL, kind)
    assert (m, total) == (15, 15), f"payload match is {m}/{total} for the station we called"
    assert VF.recognize(tones, _CALL, kind)
    assert not VF.recognize(tones, "W9SSJ", kind), "the answer matched our own call"

    naive = _naive_tones(seg[at:], n)
    tripled = sum(1 for got, want in zip(naive, tones) if got == 3 * want)
    assert tripled >= n // 2, (
        f"only {tripled}/{n} symbols peak at 3x their carrier; this recording is "
        "the evidence that the band limit is load-bearing, so a change in its "
        "harmonic content needs re-measuring, not a weaker assertion")


def test_the_clean_corpus_is_unaffected():
    """Band-limiting must not perturb the bursts that always worked — each read
    at the band of the bandwidth whose alphabet it is keyed on."""
    of_bw = {a: bw for bw, a in VF.ALPHABETS.items()}
    for name, kind in VF.BURSTS.items():
        tones = VF.handshake_tones("NS0A", kind)
        audio = MK.synth_tones(tones)
        band = MK.band_for(of_bw[kind.tones])
        assert MK.demod_tones(audio, len(tones), band) == tones, name


if __name__ == "__main__":                       # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
