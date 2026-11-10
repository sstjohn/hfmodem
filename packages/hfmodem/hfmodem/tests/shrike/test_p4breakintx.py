# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Changeover chain pinned to independent recorded bits and located audio."""

from pathlib import Path
import numpy as np
import pytest
from hfmodem.shrike import p4breakin as rx, p4breakintx as tx, p4frame, p4rx


@pytest.mark.parametrize(
    "name,data,distance",
    [
        ("kb5lzk", b"RMS Trimode 1.4.4.0 KB5LZK at t", 0),
        ("ve3kpg", b"RMS Trimode 1.4.4.0\r\nW9SSJ has ", 1),
    ],
)
def test_recorded_channel_bits(name, data, distance):
    bits = np.array(
        [
            int(x)
            for x in (Path(__file__).parent / "fixtures/p4-breakin" / (name + ".bits"))
            .read_text()
            .strip()
        ]
    )
    field = p4frame.encode_field(data, p4frame.Status(0))
    assert np.count_nonzero(tx.channel_bits(field) != bits) == distance


@pytest.mark.parametrize("fs", [12000, 48000])
@pytest.mark.parametrize(
    "data,status", [(b"hello", 0), (b"", 0x40), (b"FS YY\r", 0x40), (b"", 0x20)]
)
def test_located_roundtrip(fs, data, status):
    x = tx.changeover_packet(data, status, fs=fs)
    assert len(x) / fs == pytest.approx(
        tx.EMISSION_S, abs=1 / fs + 1 / p4rx.SAMPLE_RATE
    )
    z = p4rx.audio_to_baseband(np.r_[x, np.zeros(fs // 10)], fs)
    frame = rx.decode_body(z, first_training_start=432 * 16)
    assert frame is not None
    assert frame.data == data.ljust(31, b"\x1e")
    assert frame.status.byte == status
    assert rx.decode_body(z, first_training_start=482 * 16) is None
    llr = rx.body_scores(z, first_training_start=432 * 16)
    assert llr.shape == (840,)
    assert (
        np.mean(
            (llr < 0) == tx.coded_bits(p4frame.encode_field(frame.data, frame.status))
        )
        > 0.99
    )


def test_permutation_and_no_signal():
    assert np.array_equal(np.sort(rx.component_permutation()), np.arange(280))
    assert rx.decode_body(np.zeros(30000), first_training_start=0) is None
    assert rx.decode_body(np.ones(20), first_training_start=0) is None
    rng = np.random.default_rng(720)
    assert (
        rx.decode_body(
            rng.normal(size=30000) + 1j * rng.normal(size=30000), first_training_start=0
        )
        is None
    )


@pytest.mark.parametrize(
    "name,expected,status",
    [
        ("kb5lzk-audio", b"RMS Trimode 1.4.4.0 KB5LZK at t", 0),
        ("ve3kpg-audio", b"RMS Trimode 1.4.4.0\r\nW9SSJ has ", 0),
        ("n9vc-sid", b"[BPQ-6.0.25.1-B2FWIHJM$]\r\rde N9", 0),
        ("n9vc-idle", b"\x1e" * 31, 0x40),
        ("n9vc-close", b"\x1e" * 31, 0x20),
    ],
)
def test_recorded_body_from_its_word_origin(name, expected, status):
    import hashlib, json
    from scipy.io import wavfile
    from hfmodem.shrike import p4oracle

    path = Path(__file__).parent / "fixtures/p4-breakin" / name
    meta = json.loads(path.with_suffix(".json").read_text())
    assert (
        hashlib.sha256(path.with_suffix(".wav").read_bytes()).hexdigest()
        == meta["sha256"]
    )
    fs, pcm = wavfile.read(path.with_suffix(".wav"))
    audio = pcm.astype(float) / 32768
    (head,) = p4oracle.word_scan(audio, fs)
    assert head.word == "Y27"
    frame = p4oracle.body_after(audio, fs, 0, head)
    assert frame is not None and frame.data == expected and frame.status.byte == status


def test_polarity_symmetry_preserves_both_crc_candidates():
    from hfmodem.shrike import p4decode

    # Exhaust the component trellis before checking the iterative decoder.
    for state in range(8):
        for bit in range(2):
            assert p4decode._NEXT[7 - state, 1 - bit] == 7 - p4decode._NEXT[state, bit]
            assert (
                p4decode._PARITY[7 - state, 1 - bit] == 1 - p4decode._PARITY[state, bit]
            )
    rng = np.random.default_rng(327)
    for scale in (0.0, 0.5, 4.0, 20.0):
        fields = rng.normal(size=(3, 280)) * scale
        fields[1:, -8:] = 0
        positive = p4decode.turbo_decode(*fields, rx.component_permutation())
        negative = p4decode.turbo_decode(*(-fields), rx.component_permutation())
        assert np.allclose(positive.posterior, -negative.posterior, atol=1e-10)
        assert np.array_equal(negative.bits, (-positive.posterior < 0))
