# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The branch-B connect family: Robust Call and the two Free Signals.

One frame, three kinds, encoder in `pactor1` and decoder in `p1rx`. What holds it
to the air:

  1. ROUND TRIP over all three kinds, both polarities, callsigns of 3 to 8
     characters, at this station's own tones and at a shifted pair, in noise.
  2. OFF AIR. Eight trains from real SCS modems -- seven heard here and SCS's own
     published Free Signal -- each read back with the ident and kind an
     independent monitor gives it, at least as often as the monitor reports it.
     The monitor's counts are a floor and nothing here asks for more: the decoder
     is not being graded on sensitivity.
  3. BIT FOR BIT. The eleven bytes K6SDR's modem keyed on 10.145 MHz on
     2026-07-24, demodulated off the tape, are what `call_b_bytes` builds for
     `("K6SDR", "robust")`. That is what pins the pad convention -- the callsign
     repeated on a space, cyclically, filling all eight characters -- which no
     published text states and a CRC-16 leaves no room to guess at.
  4. NEGATIVES. Real Normal connects off the air and published, this station's
     own keyed PACTOR-1, and the shared regression corpus: no frame anywhere.

The monitor arm is not here. Six bursts of each kind for `W9SSJ`, rendered by
`build_call_b` alone, are read by SCS's own monitor, version 1.0, as
`[Robust Call: W9SSJ]`, `[Free Signal Normal: W9SSJ]` and
`[Free Signal Encrypted: W9SSJ]`, six each; it runs in a VM and costs four
minutes, so it is a harness step rather than a test.

Run:  pytest packages/hfmodem/hfmodem/tests/shrike/test_callb.py
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p1rx, pactor1, session
from hfmodem.tests import evidence
from hfmodem.tests.kestrel import corpora

FS = p1rx.FS
KINDS = {"robust": "Robust",
         "fs_normal": "FreeSignalNormal",
         "fs_encrypted": "FreeSignalEncrypted"}

OFFAIR = corpora.RF_CORPUS.parent / "offair" / "captures"

# The two published SCS samples, banked beside the corpus. They are the only Free
# Signal on tape anywhere in the record, and the only connect of either branch
# this station did not hear itself.
SIGIDWIKI = corpora.RF_CORPUS / "sigidwiki"
FREE_SIGNAL = SIGIDWIKI / "PACTOR-FS.mp3"
SELCALL = SIGIDWIKI / "PACTOR_SELCALLaudio.mp3"

# (recording, slice, tone-pair centre, ident, kind, how often the monitor reports it)
TRAINS = [
    (corpora.RF_CORPUS / "10145k_055704.wav", None, 707.0, "K6SDR", "Robust", 10),
    (corpora.RF_CORPUS / "10145k_055923.wav", None, 2006.0, "AJ7C", "Robust", 5),
    (corpora.RF_CORPUS / "14109k_131254.wav", None, 1013.0, "W5STX", "Robust", 10),
    (corpora.RF_CORPUS / "7101k_192259.wav", None, 806.0, "KD4JWF", "Robust", 5),
    (OFFAIR / "20260722T025051Z_KB8AY-7.101MHz-VARA2750.wav", None, 1500.0,
     "KY4RY", "Robust", 12),
    (OFFAIR / "20260722T024936Z_KB8AY-7.101MHz-VARA2750.wav", None, 1200.0,
     "WM4RB", "Robust", 12),
    # The one train this station heard itself, on 40 m on 2026-08-03, up against
    # the top of the passband where the sweep has one centre left.
    (evidence.CAPTURES / "onair-0803-201543-passive" / "listen.wav", (325.0, 336.0),
     2500.0, "K4MSU", "Robust", 2),
]

NEGATIVES = [
    corpora.RF_CORPUS / "7101k_054347.wav",                     # a Normal connect
    corpora.REGRESS_FIXTURES / "pos_p1_data_jn36lf.wav",        # real P1 ARQ data
    corpora.REGRESS_FIXTURES / "pos_p1_twosided_14110.wav",
    corpora.REGRESS_FIXTURES / "pos_p1cs_ws8eoc.wav",
    corpora.REGRESS_FIXTURES / "pos_pactor1_local.wav",
    corpora.REGRESS_FIXTURES / "oracle_pactor3_dl6maa.wav",
    corpora.REGRESS_FIXTURES / "neg_noise_chatham.wav",
]
"""`pos_p1_kb8ay_float.wav` is deliberately NOT here. It is a cut of the 7.101 MHz
recording KY4RY was calling over, and 23 of its frames are that call: the fixture
is a PACTOR-1 positive and a branch-B positive at once."""


def _load(path: Path, span: tuple[float, float] | None = None) -> np.ndarray:
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    audio = session.load_wav(str(path), FS)
    return audio if span is None else audio[int(span[0] * FS):int(span[1] * FS)]


def _load_mp3(path: Path, tmp_path: Path) -> np.ndarray:
    """The sigidwiki samples are mp3, and nothing in the package reads one."""
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not on PATH, and the sample is mp3")
    wav = tmp_path / f"{path.stem}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                    "-ac", "1", "-ar", str(FS), str(wav)], check=True)
    return session.load_wav(str(wav), FS)


def _shifted(audio: np.ndarray, hz: float) -> np.ndarray:
    """The same burst on a tone pair `hz` away, as a receiver off frequency hears it."""
    from scipy.signal import hilbert
    n = np.arange(audio.size)
    return np.real(hilbert(audio) * np.exp(2j * np.pi * hz * n / FS))


def _noisy(burst: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    audio = np.concatenate([np.zeros(2400), burst, np.zeros(2400)])
    return audio + rng.normal(0, sigma, audio.size)


def _slot(audio: np.ndarray, centre: float, start_s: float) -> bytes:
    """The eleven on-air bytes at `start_s`, demodulated rather than decoded."""
    sps = FS // 100
    tones = p1rx._Tones(audio, sps, centre - 100, centre + 100)
    start = np.array([int(start_s * FS)], np.int64)
    return tones.read(start, sps, pactor1.CALL_B_LEN, False)[0].tobytes()


@pytest.mark.parametrize("call", ["W9SSJ", "K6SDR", "AJ7C", "KD4JWF", "N0CALL/9",
                                  "ABCDEFGH", "W1A"])
@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("invert", [False, True])
def test_round_trip(call: str, kind: str, invert: bool) -> None:
    audio = _noisy(pactor1.build_call_b(call, kind, invert, amp=0.3), 0.06, 11)
    got = p1rx.decode_call_b(audio)
    assert got == p1rx.Connect(KINDS[kind], call, invert)


@pytest.mark.parametrize("hz", [-700.0, 550.0])
def test_round_trip_off_frequency(hz: float) -> None:
    """The sweep finds the pair wherever the caller's dial put it -- 1400/1600 is
    only where THIS station keys, and not one of the seven real trains is on it."""
    burst = pactor1.build_call_b("W9SSJ", "fs_normal", amp=0.3)
    got = p1rx.decode_call_b(_noisy(_shifted(burst, hz), 0.06, 12))
    assert got == p1rx.Connect("FreeSignalNormal", "W9SSJ", False)


def test_reports_are_the_monitors_strings() -> None:
    calls = [p1rx.decode_call_b(pactor1.build_call_b("W9SSJ", k, amp=0.3))
             for k in KINDS]
    assert [c.report() for c in calls] == [
        "###CONNECT: [Robust Call: W9SSJ]",
        "###CONNECT: [Free Signal Normal: W9SSJ]",
        "###CONNECT: [Free Signal Encrypted: W9SSJ]"]
    assert (p1rx.Connect("Normal", "W1AW", False).report()
            == "###CONNECT: [Normal Call: W1AW]")


def test_the_connect_entry_point_keys_the_robust_frame() -> None:
    """`connect_signal` is what arq and the radio io reach the air through."""
    audio = pactor1.connect_signal("W9SSJ", variant="robust")
    assert p1rx.decode_call_b(audio) == p1rx.Connect("Robust", "W9SSJ", False)
    assert p1rx.decode_connect(audio) is None
    assert "robust" in pactor1.CONNECT_VARIANTS


def test_address_field_is_the_callsign_repeated_on_a_space() -> None:
    assert pactor1.call_b_address("K6SDR") == "K6SDR K6"
    assert pactor1.call_b_address("AJ7C") == "AJ7C AJ7"
    assert pactor1.call_b_address("KD4JWF") == "KD4JWF K"
    assert pactor1.call_b_address("ABCDEFGH") == "ABCDEFGH"
    assert pactor1.call_b_address("W1A") == "W1A W1A "


def test_bytes_are_k6sdrs_own() -> None:
    """The air, byte for byte: what an SCS modem keyed is what the encoder builds."""
    path, _, centre, call, _, _ = TRAINS[0]
    audio = _load(path)
    hits = p1rx.decode_call_b_all(audio, centre=centre)
    assert hits, "no frame at the pair the train is known to sit on"
    t, connect = hits[0]
    want = pactor1.call_b_bytes(call, "robust")
    if connect.inverted:
        want = bytes(b ^ 0xFF for b in want)
    assert _slot(audio, centre, t) == want


@pytest.mark.parametrize("path, span, centre, call, variant, floor", TRAINS,
                         ids=[t[3] for t in TRAINS])
def test_offair_train(path: Path, span, centre: float, call: str, variant: str,
                      floor: int) -> None:
    hits = p1rx.decode_call_b_all(_load(path, span))
    assert len(hits) >= floor
    assert {(c.variant, c.callsign) for _, c in hits} == {(variant, call)}


def test_offair_free_signal(tmp_path: Path) -> None:
    """SCS's own published Free Signal, which its own monitor reads as `Free
    Signal Encrypted: DAO8` five times over -- and a sixth time as a garbage
    ident that closes the CRC, which this decoder's character check refuses and
    the floor does not ask for."""
    hits = p1rx.decode_call_b_all(_load_mp3(FREE_SIGNAL, tmp_path))
    assert len(hits) >= 5
    assert {(c.variant, c.callsign) for _, c in hits} == {
        ("FreeSignalEncrypted", "DAO8")}


@pytest.mark.parametrize("path", NEGATIVES, ids=lambda p: p.stem)
def test_no_frame_where_there_is_none(path: Path) -> None:
    assert p1rx.decode_call_b_all(_load(path)) == []


def test_no_frame_in_the_published_selcall(tmp_path: Path) -> None:
    """The other sigidwiki sample, the branch-A control: a monitor reads ten
    `Normal Call: OL1A` out of it and this decoder reads no branch-B frame."""
    assert p1rx.decode_call_b_all(_load_mp3(SELCALL, tmp_path)) == []


def test_our_own_pactor1_is_not_a_branch_b_frame() -> None:
    """Both connect variants and a data packet, at the length and rate that are
    closest to this frame's: 0.96 s of 100 Bd is where a false accept would come
    from, and the three gates plus the CRC refuse it."""
    for variant in ("normal", "longpath"):
        assert p1rx.decode_call_b(
            pactor1.connect_signal("W9SSJ", variant=variant)) is None
    assert p1rx.decode_call_b(pactor1.packet_signal(b"1W9SSJ", 100)) is None
