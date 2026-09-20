# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""VARA BW500 control-token codec: round-trip, and recognition of real VARA audio.

The real-audio test is the interop proof — the detector, built only from the
promoted spec-02 token patterns, must classify the control bursts a real VARA
keyed on the loopback corpus.
"""
import numpy as np

from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.corpora import (
    CONTROL_BURSTS, requires_control_bursts, wav_mono,
)
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_control as vc


def test_round_trip_every_token():
    for name in vc.TOKENS:
        m = vc.detect_token(vc.synth_token(name))
        assert m is not None and m.name == name and m.hamming == 0, name


def test_synth_reproduces_canonical_bits():
    for name, patt in vc.TOKENS.items():
        bits, q, _ = vc._demod_bits(vc.synth_token(name), len(patt) + 1)
        assert "".join(map(str, bits[:len(patt)])) == patt
        assert q > 0.99


def test_connected_ack_and_alive_differ_only_in_the_flag():
    """The two share the 40-bit template and differ only in the 4-bit tail flag
    (0000 vs 1111) — a 1-bit field, repetition-coded x4 (spec 02 §2.6)."""
    ca = np.array([int(c) for c in vc.TOKENS["connected-ack"]])
    al = np.array([int(c) for c in vc.TOKENS["alive"]])
    diff = np.flatnonzero(ca != al)
    assert list(diff) == [40, 41, 42, 43]


def test_distinct_function_tokens_are_well_separated():
    """Tokens with different ARQ functions (not the flag pair) stay far apart, so
    the detector's max_hamming=2 can never cross-classify them."""
    families = ["data-ack", "nak", "connected-ack", "ready"]   # distinct functions
    for i, a in enumerate(families):
        for b in families[i + 1:]:
            pa = np.array([int(c) for c in vc.TOKENS[a]])
            pb = np.array([int(c) for c in vc.TOKENS[b]])
            n = min(len(pa), len(pb))
            hd = min(int((pa[:n] != pb[:n]).sum()), n - int((pa[:n] != pb[:n]).sum()))
            assert hd > 4, f"{a} vs {b} too close ({hd})"


def _bursts(a, thr=0.12, lo_s=0.15):
    win = vc.FS // 50
    env = np.sqrt(np.convolve(a * a, np.ones(win) / win, "same"))
    on = env > env.max() * thr
    edges = np.diff(on.astype(int))
    s = list(np.flatnonzero(edges == 1) + 1)
    e = list(np.flatnonzero(edges == -1) + 1)
    if on[0]:
        s.insert(0, 0)
    if on[-1]:
        e.append(len(a))
    return [(x, y) for x, y in zip(s, e) if (y - x) / vc.FS >= lo_s]


@requires_control_bursts
def test_detects_real_vara_control_bursts():
    a = wav_mono(CONTROL_BURSTS)
    bursts = _bursts(a)
    recognised = sum(vc.detect_token(a[max(0, s - 128):e + 128]) is not None
                     for s, e in bursts)
    # A long transfer is dominated by data-ACKs; the only expected misses are the
    # ~1 s MFSK connect-response and the long idle-keepalive.
    assert recognised >= len(bursts) - 3
    # and the constant data-ACK must be recognised many times over
    acks = sum((m := vc.detect_token(a[max(0, s - 128):e + 128])) is not None
               and m.name == "data-ack" for s, e in bursts)
    assert acks >= 40


# The vocabulary's own column span: `detect_token` demodulates len(pattern)+1
# columns, so a burst outside this cannot be a token whatever its SNR.
_TOKEN_COLS = sorted(len(bits) + 1 for bits in vc.TOKENS.values())


@corpora.requires_clear_channel
@corpora.requires_onair_silent_calls
def test_band_noise_resolves_no_token_and_a_buried_one_still_does():
    """The false-accept floor, and the sensitivity it was bought at, on one corpus.

    Both halves are needed together, and one night's report is why. On
    2026-08-23 a KD0PYG arm logged eight bursts inside this column span, every
    one of them `tokens unresolved`, and that was read as the decoder declining
    what a gateway had sent. It was band noise: `vara_monitor._unnamed_burst`
    routed a bracket to that line on a DBPSK-collapse test that 84% of these
    same noise windows passed at 32-44 columns, because the collapse statistic
    is a max over 5 carriers x 32 onsets and had nothing to beat it down at that
    length. That gate is now measured in units of its own noise
    [test_monitor_traffic]; a decoder that declines noise looks identical to one
    that is broken unless the second assert is standing next to the first.

    Sampled over 30 s of a verified-clear 40 m channel and the two 2026-08-06
    calls nobody answered, at each column count the vocabulary occupies.
    """
    rng = np.random.default_rng(3)
    noise = [wav_mono(p) for p in
             (corpora.CLEAR_CHANNEL, *corpora.ONAIR_SILENT_CALLS)]
    matched = []
    for a in noise:
        for cols in _TOKEN_COLS:
            n = cols * vc.H
            for _ in range(40):
                i = int(rng.integers(0, len(a) - n))
                m = vc.detect_token(a[i:i + n])
                if m is not None:
                    matched.append((cols, m.name, m.hamming))
    assert not matched, matched

    a = noise[0]
    for name in vc.TOKENS:
        token = vc.synth_token(name)
        for snr_db in (6, 3, 0):
            i = int(rng.integers(0, len(a) - len(token)))
            nz = a[i:i + len(token)]
            amp = (np.sqrt(np.mean(nz * nz)) * 10 ** (snr_db / 20)
                   / np.sqrt(np.mean(token * token)))
            m = vc.detect_token(token * amp + nz)
            assert m is not None and m.name == name, (name, snr_db)


def _ref_burst(side: str, t0: float, cols: int, pad: int = 128):
    p = next(q for q in corpora.BW2300_REF_SIDES if q.name.endswith(f"__{side}.wav"))
    a = wav_mono(p)
    s = int(t0 * vc.FS)
    return a[max(0, s - pad):s + cols * vc.H + pad]


@corpora.requires_bw2300_reference
@corpora.requires_control_bursts
def test_this_vocabulary_names_nothing_in_a_real_vara_bw2300_session():
    """The table this module used to carry for BW2300, and why it is gone.

    Four patterns read on 2026-07-23 off a dual Wine session that is in no corpus,
    matching nothing since — not on either successful connect of 2026-08-23, not on
    the KB3AC-10 arm that reached CONNECTED and read three clean-CRC overs, not in
    any of the four monitor runs. That is a negative an HF channel can always
    explain away, so what settles it is asserted here instead: VARA HF v4.9.0
    against itself at BW2300 over a cable, both directions on tape, every short
    frame in the session, no noise and no offset.

    What stands in its place is this: the BW500 vocabulary, which IS a promoted
    spec fact, resolves none of those thirteen bursts either. A real VARA keys no
    DBPSK control token at BW2300 — of any vocabulary — and callers at that
    bandwidth are meant to reach `vara_arq._ack_plateau` and nothing here.

    The BW500 assertion beside it is what makes that mean something. The same
    detector, the same kind of session from the same bench, finds forty-plus
    data-ACKs at Hamming 0. The instrument works; at BW2300 there is nothing for it
    to find.
    """
    got = [(side, t0, cols, m)
           for side, t0, cols in corpora.BW2300_REF_CONTROL
           if (m := vc.detect_token(_ref_burst(side, t0, cols)))]
    assert not got, got

    a = wav_mono(corpora.CONTROL_BURSTS)
    acks = sum(vc.detect_token(a[i:i + 34 * vc.H]) is not None
               for i in range(0, len(a) - 34 * vc.H, vc.H))
    assert acks >= 40, (
        f"the BW500 side of the same detector found {acks} tokens, so this test "
        "is measuring a broken detector rather than an absent waveform")


@corpora.requires_bw2300_reference
def test_the_real_bw2300_acknowledgement_is_the_connected_ack_waveform():
    """What a real VARA BW2300 acknowledgement is, since it is not a token.

    The burst spec 04 §4.2C specifies for CONNECT setup — eleven two-tone MFSK
    symbols of four 512-sample columns, on a fixed four-symbol preamble — and the
    burst a real VARA answers each over with are one waveform. `_ack_plateau` is
    the reader for it, so it is what asserts here: all eleven 45-column bursts of
    the reference session hold that preamble, on both sides, and neither of the two
    33-column turn bursts does.

    The responder's connect-time burst at 8.567 s and its first per-over answer at
    9.943 s are the same waveform to correlation 1.00 — so `connected-ack` and
    `data-ack` are not two things at this bandwidth.
    """
    plateau = {(side, t0, cols): VA._ack_plateau(_ref_burst(side, t0, cols))
               for side, t0, cols in corpora.BW2300_REF_CONTROL}
    acks = {k: v for k, v in plateau.items() if k[2] == 45}
    assert len(acks) == 11, acks
    assert all(v >= VA._ACK_PLATEAU for v in acks.values()), acks
    assert all(v < VA._ACK_PLATEAU for k, v in plateau.items() if k[2] == 33), \
        f"the turn burst holds the acknowledgement preamble too: {plateau}"

    a = _ref_burst("b2a", 8.567, 45, pad=0)
    b = _ref_burst("b2a", 9.943, 45, pad=0)
    n = min(len(a), len(b))
    r = float(np.abs(np.correlate(a[:n], b[:n], "full")).max()
              / np.sqrt((a[:n] ** 2).sum() * (b[:n] ** 2).sum()))
    assert r > 0.98, (
        "the connect-time ack and the first per-over answer are no longer the same "
        f"waveform (correlation {r:.2f}), so BW2300 does distinguish them")
