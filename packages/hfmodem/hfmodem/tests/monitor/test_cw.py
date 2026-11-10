# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The CW reader against our own Morse renderer and against WS8EOC on the air.

SNR is signal power over noise power in 2500 Hz. The fade is a Rayleigh
envelope with 0.5 Hz Doppler bandwidth, unit RMS, held no deeper than 15 dB
below that.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from hfmodem.core import cwid
from hfmodem.tests import evidence
from hfmodem.monitor.cw import CwReader

FS = 48000
REPO = Path(__file__).resolve().parents[5]
WS8EOC_TAPE = (REPO / "working/pactor-level-air-0916/arms"
               / "pactor-current-ws8eoc-80-force-20260917T143320Z/tap.wav")


def _read(x: np.ndarray, centre_hz: float, chunk: int = 4096) -> list[dict]:
    reader = CwReader(FS, centre_hz)
    out = []
    for i in range(0, len(x), chunk):
        out += reader.push(x[i:i + chunk])
    return out + reader.flush()


def _fade(n: int, rng: np.random.Generator) -> np.ndarray:
    rate = 50
    k = n * rate // FS + 2
    g = np.convolve(rng.normal(size=k) + 1j * rng.normal(size=k),
                    signal.firwin(101, 0.5, fs=rate), "same")
    t = np.arange(n) * rate / FS
    a = np.abs(np.interp(t, np.arange(k), g.real) + 1j * np.interp(t, np.arange(k), g.imag))
    return np.maximum(a / np.sqrt(np.mean(a ** 2)), 10 ** (-15 / 20))


def _on_air(text: str, wpm: float, tone_hz: float, faded: bool, seed: int,
            monkeypatch) -> np.ndarray:
    monkeypatch.setattr(cwid, "MAX_WPM", 40.0)
    rng = np.random.default_rng(seed)
    x = np.r_[np.zeros(FS), cwid.audio(text, wpm, tone_hz=tone_hz), np.zeros(FS)]
    if faded:
        x = x * _fade(len(x), rng)
    noise_rms = np.sqrt(0.5 ** 2 / 2 / 10 ** (20 / 10) * (FS / 2) / 2500)
    return x + rng.normal(scale=noise_rms, size=len(x))


@pytest.mark.parametrize("faded", [False, True], ids=["steady", "faded"])
@pytest.mark.parametrize("wpm", [12, 20, 30])
@pytest.mark.parametrize("seed", range(4))
def test_a_rendered_callsign_reads_back_exactly(wpm, faded, seed, monkeypatch):
    tone = 600.0 + 450.0 * seed
    recs = _read(_on_air("WS8EOC", wpm, tone, faded, seed, monkeypatch), tone)
    assert [r["text"] for r in recs] == ["WS8EOC"]
    rec = recs[0]
    assert rec["grade"] == "confirmed"
    assert rec["station"] == "WS8EOC" and rec["role"] == "sender"
    assert (rec["modem"], rec["protocol"], rec["kind"]) == ("cw", "CW", "text")
    assert rec["t"] == pytest.approx(1.0, abs=0.05)
    assert abs(float(rec["detail"].split()[0]) - wpm) <= 1.5


def test_words_are_emitted_at_their_gaps(monkeypatch):
    recs = _read(_on_air("DE WS8EOC K", 20, 1000.0, False, 7, monkeypatch), 1000.0)
    assert [r["text"] for r in recs] == ["DE", "WS8EOC", "K"]
    assert [r["grade"] for r in recs] == ["tentative", "confirmed", "tentative"]


def _pactor1_control_signal(rng: np.random.Generator) -> np.ndarray:
    """120 ms of 100 Bd FSK on 1400/1600 Hz: a PACTOR-1 control signal's shape."""
    bits = rng.integers(0, 2, 12)
    freq = np.repeat(np.where(bits, 1600.0, 1400.0), FS // 100)
    return 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / FS)


@pytest.mark.parametrize("centre", [1400.0, 1500.0, 1600.0])
def test_a_pactor1_control_signal_is_not_confirmed_text(centre):
    rng = np.random.default_rng(3)
    cycle = np.zeros(int(1.25 * FS))
    x = np.concatenate([np.zeros(FS)] + [
        np.r_[_pactor1_control_signal(rng), cycle[int(0.12 * FS):]] for _ in range(12)])
    x = x + rng.normal(scale=0.005, size=len(x))
    assert not [r for r in _read(x, centre) if r["grade"] == "confirmed"]


@pytest.mark.skipif(not WS8EOC_TAPE.exists(), reason=f"{WS8EOC_TAPE} is not on this machine")
def test_ws8eoc_identifies_itself_after_the_link_goes_down():
    """One ID, 0.26 s after our twentieth and last keying (QRT) drops.

    The tape also carries all twenty of our keyings, whose 1600 Hz tone sits
    inside the reader's band; they may read as text but never as confirmed.
    """
    with wave.open(str(WS8EOC_TAPE)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0
    confirmed = [r for r in _read(x, 1605.5, chunk=FS // 4) if r["grade"] == "confirmed"]
    assert [(r["text"], r["station"], r["detail"]) for r in confirmed] == [
        ("WS8EOC", "WS8EOC", "20 WPM")]
    assert confirmed[0]["t"] == pytest.approx(62.54, abs=0.05)


CORPUS_SESSION = evidence.CORPUS / "regress" / "fixtures" / "pos_vara_session.wav"


@pytest.mark.skipif(not CORPUS_SESSION.exists(), reason=f"{CORPUS_SESSION} is not on this machine")
def test_an_id_with_a_clipped_head_is_not_confirmed():
    """A CW ID at 606 Hz, 31.6 s into a VARA session, reads D5YTA.

    The same rig's PACTOR-1 capture names KE5YTA. The head's last dit is keyed
    40 % narrower than the word's other eight, so it was not sent as a dit: the
    read is garbled even though every dah/dit ratio and gap class holds.
    """
    with wave.open(str(CORPUS_SESSION)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0
    recs = _read(x, 606.0)
    assert [r["grade"] for r in recs if r["text"] == "D5YTA"] == ["tentative"]
    assert not [r for r in recs if r["grade"] == "confirmed"]
