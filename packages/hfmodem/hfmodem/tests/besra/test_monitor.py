# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Monitor mode over a multi-frame capture: the demodulator locates every frame
by its leader in one pass, so the monitor decodes a whole capture without
pre-segmentation — the real-channel case the single-frame fixtures don't cover.

The live path gets the same scrutiny here, driven by an injected fake audio source
instead of a sound card: `LiveMonitor.push` takes exactly what `capture_stream`'s
callback hands over, so feeding it 48 kHz float32 blocks exercises the recorder,
the rolling decode, the level maths and the heartbeat with no hardware and no
radio anywhere near the test."""

from __future__ import annotations

import json
import re
import signal
import subprocess
import sys
import textwrap
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

import hfmodem.besra.radio as R
from hfmodem.tests import evidence
from hfmodem.core.resample import to_card
from hfmodem.besra import monitor
from hfmodem.besra.phy import demodulator as D
from hfmodem.besra.phy import modulator as M


def _stream():
    gap = np.zeros(3600, dtype="<i2")
    frames = [
        M.render_frame(0x31, caller="W9SSJ", target="K7ABC", session_id=0xFF),
        M.render_frame(0x3A, payload=bytes([24, 24, 24]), session_id=0x5E),
        M.render_frame(0x4A, payload=b"monitoring besra"[:64], session_id=0x5E),
        M.render_frame(0x30, caller="W9SSJ", grid="EN63", session_id=0xFF),
    ]
    return np.concatenate([gap] + sum(([f, gap] for f in frames), []))


def test_monitor_decodes_every_frame_in_one_pass():
    decoded = D.decode(_stream())

    assert [f.type for f in decoded] == [0x31, 0x3A, 0x4A, 0x30]
    assert [f.offset for f in decoded] == sorted(f.offset for f in decoded)  # increasing
    assert decoded[0].caller == "W9SSJ" and decoded[0].target == "K7ABC"
    assert decoded[2].payload == b"monitoring besra"
    assert decoded[3].caller == "W9SSJ" and decoded[3].grid == "EN63"


def test_cli_decodes_a_wav_and_a_stream(tmp_path):
    # Exercise the actual CLI paths (WAV + --stdin) end to end, not just decode().
    stream = _stream()
    wav = tmp_path / "cap.wav"
    with wave.open(str(wav), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(12000)
        w.writeframes(stream.tobytes())

    for argv, stdin in [([str(wav)], None), (["--stdin"], stream.tobytes())]:
        r = subprocess.run([sys.executable, "-m", "hfmodem.besra.monitor", *argv],
                           input=stdin, capture_output=True, timeout=120)
        assert r.returncode == 0, r.stderr.decode()
        out = r.stdout.decode()
        assert "ConReq200M" in out and "4FSK.500.100" in out and "IDFrame" in out


# --------------------------------------------------------------------------- #
# The live path, on an injected fake audio source.
# --------------------------------------------------------------------------- #

def _card(samples12: np.ndarray, block_s: float = 0.1) -> list[np.ndarray]:
    """12 kHz int16 as the sound card would deliver it: mono float32 at 48 kHz in
    `block_s` blocks — precisely what `capture_stream`'s callback passes on."""
    au = to_card(np.asarray(samples12, dtype="<i2"), D.SAMPLE_RATE)
    n = int(block_s * R.FS_RADIO)
    return [au[i:i + n] for i in range(0, len(au), n)]


def _live(blocks, tmp_path, capsys):
    mon = monitor.LiveMonitor(D.Demodulator(), tmp_path / "cap.wav", source="fake")
    for b in blocks:
        mon.push(b)
        mon.pump()
    mon.close()
    return capsys.readouterr().out, json.loads((tmp_path / "cap.json").read_text())


def _frame_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if "  0x" in ln]


#: Frames straddling the rolling decode's step boundaries and its 6 s carry-over,
#: which is the arrangement an energy gate loses and a whole-capture pass does not.
#: Durations are 1.745 s (0x31/0x34/0x30) and 3.785 s (0x4A).
_STRADDLE = [(1.90, 0x31, dict(caller="W9SSJ", target="K7ABC")),
             (5.90, 0x4A, dict(payload=b"across a window edge")),
             (11.05, 0x34, dict(caller="KC3OWM", target="AB4NW")),
             (13.95, 0x30, dict(caller="W9SSJ", grid="EN63"))]


def _straddling_stream() -> np.ndarray:
    parts, at = [], 0
    for start, ftype, kw in _STRADDLE:
        parts.append(np.zeros(int(start * D.SAMPLE_RATE) - at, dtype="<i2"))
        parts.append(M.render_frame(ftype, session_id=0x5E, **kw))
        at = int(start * D.SAMPLE_RATE) + len(parts[-1])
    parts.append(np.zeros(int(2.0 * D.SAMPLE_RATE), dtype="<i2"))
    return np.concatenate(parts)


def test_live_rolling_decode_matches_the_file_path(tmp_path, capsys):
    """Defect: the live path used to pre-segment on an energy gate, so a frame
    straddling the gate's open or close was lost while the file path — which the
    module docstring already promised was gapless — found it. The rolling window
    must reach the same verdict, frame for frame and position for position."""
    stream = _straddling_stream()
    out, _ = _live(_card(stream), tmp_path, capsys)

    expect = D.decode(stream)
    lines = _frame_lines(out)
    assert len(lines) == len(expect) == len(_STRADDLE), f"{lines}\nvs {expect}"
    for ln, f in zip(lines, expect):
        assert f"0x{f.type:02X}" in ln
        assert abs(float(ln.split()[1]) - f.offset / D.SAMPLE_RATE) < 0.05
    assert "W9SSJ > K7ABC" in out and "KC3OWM > AB4NW" in out and "W9SSJ  EN63" in out
    assert b"across a window edge".hex()[:16] in out
    assert "CRC/RS FAIL" not in out


def test_live_records_one_unbroken_wav(tmp_path, capsys):
    """The recording is the artefact the window exists to produce, so it is the whole
    stream: every sample the card delivered, gaps included, in one file. Chopping it
    per burst would not merely lose the quiet — the sibling PACTOR work measured the
    same audio reading a *different* well-formed callsign once windowed."""
    stream = _straddling_stream()
    blocks = _card(stream)
    _live(blocks, tmp_path, capsys)

    with wave.open(str(tmp_path / "cap.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (R.FS_RADIO, 1, 2)
        rec = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    want = np.concatenate(blocks)
    assert rec.size == want.size, "recording is not the full capture"
    assert np.array_equal(rec, np.clip(np.round(want * 32768.0), -32768, 32767).astype("<i2"))

    # And it is a usable fixture: re-decoding the file finds what the live pass did.
    back = np.clip(np.round(resample_poly(rec.astype(np.float64), 12000, R.FS_RADIO)),
                   -32768, 32767).astype("<i2")
    assert [f.type for f in D.decode(back)] == [f.type for f in D.decode(stream)]


def test_sidecar_levels_are_against_full_scale(tmp_path, capsys):
    """The level maths, and the way to get it wrong: measured against the capture's
    own peak every recording is 100% railed by construction, and a quiet band reads
    as a clipped input. -26 dBFS is -26 dBFS."""
    quiet = 0.05 * np.sin(2 * np.pi * 1500 * np.arange(int(3 * R.FS_RADIO)) / R.FS_RADIO)
    _, sc = _live([quiet.astype(np.float32)], tmp_path, capsys)

    assert sc["railed_pct"] == 0.0
    assert abs(sc["peak"] - 0.05) < 1e-3
    assert abs(sc["peak_dbfs"] - -26.02) < 0.1            # 20*log10(0.05)
    assert abs(sc["rms_dbfs"] - -29.03) < 0.1            # a sine is 3.01 dB below peak
    assert sc["samplerate"] == R.FS_RADIO and sc["channels"] == 1
    assert abs(sc["duration_s"] - 3.0) < 1e-6


def test_sidecar_reports_a_railed_input(tmp_path, capsys):
    half = np.full(int(1 * R.FS_RADIO), 0.5, dtype=np.float32)
    _, sc = _live([np.ones(int(1 * R.FS_RADIO), dtype=np.float32), half], tmp_path, capsys)

    assert abs(sc["railed_pct"] - 50.0) < 0.01           # one second of two at full scale
    assert sc["peak_dbfs"] == 0.0


def test_heartbeat_reports_levels_and_clipping(tmp_path, capsys):
    """A silent monitor is otherwise indistinguishable from a dead audio path: the
    heartbeat is how the operator tells a quiet band from no audio at all."""
    n = int(monitor._HEARTBEAT_S * R.FS_RADIO)
    hot = np.ones(n, dtype=np.float32)
    quiet = (0.01 * np.random.default_rng(7).normal(size=n)).astype(np.float32)
    out, _ = _live([hot, quiet], tmp_path, capsys)

    beats = [ln for ln in out.splitlines() if " -- peak " in ln]
    assert len(beats) == 2, out
    assert "CLIPPING 100.00%" in beats[0] and "0.0 dBFS" in beats[0]
    assert "no clip" in beats[1] and "-40." in beats[1]      # 0.01 rms -> -40 dBFS
    # A heartbeat must never read as a frame to rf-corpus/regress/run.py, whose
    # detection-line test is exactly this regex.
    assert not any(re.search(r"0x[0-9A-F]{2} ", ln) for ln in beats)


@pytest.mark.realtime
def test_live_lines_carry_wall_clock_and_elapsed(tmp_path, capsys):
    """Live decodes used to print the literal string `live`, which makes a decode
    impossible to correlate with operator action and inter-frame cadence impossible
    to measure — the very measurement that corroborated the KC3OWM capture."""
    gap = np.zeros(int(4.0 * D.SAMPLE_RATE), dtype="<i2")
    frame = M.render_frame(0x31, caller="W9SSJ", target="K7ABC", session_id=0xFF)
    stream = np.concatenate([gap, frame, gap, frame, gap])
    out, _ = _live(_card(stream), tmp_path, capsys)

    lines = [ln for ln in out.splitlines() if "0x31" in ln]
    assert len(lines) == 2
    stamps = []
    for ln, f in zip(lines, D.decode(stream)):
        clock, elapsed = ln.split()[0], float(ln.split()[1])
        assert len(clock) == 9 and clock[2] == clock[5] == ":" and clock.endswith("Z")
        assert abs(elapsed - f.offset / D.SAMPLE_RATE) < 0.05
        stamps.append(elapsed)
    # Cadence: 4 s gap + the frame's own duration, recovered from the log alone.
    assert abs((stamps[1] - stamps[0]) - (4.0 + len(frame) / D.SAMPLE_RATE)) < 0.05


def test_monitor_never_transmits():
    """Receive only, and visibly so: no rig, no PTT, no output device in the module."""
    src = Path(monitor.__file__).read_text()
    for forbidden in ("ptt", "Rig", "OutputStream", "sd.play", "out_device"):
        assert forbidden not in src, f"monitor.py references {forbidden!r}"


_CORPUS = evidence.CORPUS
_KC3OWM = {"7102k_065457": ("KC3OWM", "K4PAR-2"), "7101k_065546": ("KC3OWM", "AB4NW")}


#: Extra ConReqs the rolling window finds that a whole-capture pass over the same
#: recording does not. Not false ones: this station repeats its ConReq on a 3.59 s
#: grid (a 1.745 s frame plus the 2.0 s interval), these fill it in unbroken from
#: 1.78 to 16.13 s, and every one decodes the same RS-validated KC3OWM > K4PAR-2
#: callsign block. A 44 s pass buries them because the demodulator's silence gate
#: is a fraction of the *capture's* peak, where a 6 s window sets a far lower bar.
_EXTRA = {"7102k_065457": [1.78, 5.37, 12.55]}


@pytest.mark.parametrize("name,call", sorted(_KC3OWM.items()))
def test_live_path_finds_the_offair_conreqs_the_file_path_does(name, call, tmp_path, capsys):
    """The acceptance test for the live decode: real off-air ARDOP, played into the
    live path as the sound card would deliver it, must yield everything a
    whole-capture pass over the same recording yields, at the same positions."""
    wav = _CORPUS / f"{name}.wav"
    if not wav.exists():
        pytest.skip(f"{name}.wav absent")
    with wave.open(str(wav)) as w:
        assert w.getframerate() == R.FS_RADIO
        raw = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    blocks = [b for b in np.array_split(raw.astype(np.float32) / 32768.0,
                                        max(1, raw.size // int(0.1 * R.FS_RADIO)))]
    out, sc = _live(blocks, tmp_path, capsys)

    at12 = np.clip(np.round(resample_poly(raw.astype(np.float64), 12000, R.FS_RADIO)),
                   -32768, 32767).astype("<i2")
    expect = sorted([f.offset / D.SAMPLE_RATE for f in D.decode(at12) if f.ok]
                    + _EXTRA.get(name, []))
    caller, target = call
    assert len(expect) >= 2, f"corpus changed: {name} now decodes {len(expect)} frames"

    lines = [ln for ln in out.splitlines() if f"{caller} > {target}" in ln]
    assert len(lines) == len(expect), out
    assert "CRC/RS FAIL" not in out
    for ln, at in zip(lines, expect):
        assert abs(float(ln.split()[1]) - at) < 0.05
    assert sc["frames_decoded"] == len(expect)


def test_control_frame_immediately_before_data():
    # A ConReq's non-monotonic 4FSK trailer plants a false leader; the scan must
    # step over it and still find the following data frame (regression).
    gap = np.zeros(3600, dtype="<i2")
    stream = np.concatenate([
        gap, M.render_frame(0x31, caller="W9SSJ", target="K7ABC", session_id=0xFF),
        gap, M.render_frame(0x4A, payload=b"after a control frame"[:64], session_id=0x5E),
        gap])
    decoded = D.decode(stream)
    assert [f.type for f in decoded] == [0x31, 0x4A]
    assert decoded[1].payload == b"after a control frame"


def test_sigterm_still_writes_the_level_sidecar(tmp_path):
    # A window ends however it ends. `timeout`, a session teardown and Ctrl-C are
    # all routine, and only the last used to reach close(): SIGTERM took the audio
    # and dropped the sidecar. That is the one file that can show a capture was not
    # railed, and a WAV alone provably cannot — so losing it loses the evidence.
    wav = tmp_path / "sigterm.wav"
    script = textwrap.dedent(f"""
        import numpy as np, contextlib, threading, hfmodem.besra.monitor as mon
        from hfmodem.besra import radio
        stop = threading.Event()

        @contextlib.contextmanager
        def fake(device, on_block):
            t = threading.Thread(
                target=lambda: [on_block(np.zeros(4800, "float32")) or stop.wait(0.05)
                                for _ in iter(lambda: not stop.is_set(), False)],
                daemon=True)
            t.start()
            try:
                yield None
            finally:
                stop.set()

        radio.capture_stream = fake
        print("ready", flush=True)
        mon._live(mon.Demodulator(), "fake", {str(wav)!r})
    """)
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    time.sleep(1.5)
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=15)

    sidecar = wav.with_suffix(".json")
    assert sidecar.exists(), "SIGTERM dropped the sidecar"
    assert wav.exists() and wav.stat().st_size > 44
    assert json.loads(sidecar.read_text())["railed_pct"] == 0.0
