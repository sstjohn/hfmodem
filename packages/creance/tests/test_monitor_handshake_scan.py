# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One creance pass has to find the connect request, not two.

Until 2026-08-14 the kestrel runner ran ``MonitorGate`` + ``classify`` and nothing
else, so ``creance.cli monitor --deep`` could not report a handshake burst the
energy gate never bracketed — and by kestrel's own measurement that is most of
them, because a connect request arrives level with the band noise or inside the
mute under somebody's transmission. The cost was not hypothetical: the morning
slot of 2026-08-14 published "no VARA connect handshake captured" as a headline
negative, having run only the gate pass, and its own afternoon recording holds
two connect requests that ``tools/vara_monitor.py`` reads off the same audio.

So the runner now drives ``HandshakeScanner`` over every chunk as well, and these
are the tests that keep it there: a request buried under the gate's threshold, the
two on the real recording, and the two negative controls that must stay silent.
The recordings live outside the tree like the rest of the off-air evidence; the
skips name what is missing.
"""

import json
import subprocess
from pathlib import Path

import pytest

from creance.monitor import Detection, default_specs
from creance.monitor.driver import MODEM_ROOT

KESTREL = next(s for s in default_specs() if s.name == "kestrel")

#: 16:34 UTC on 7096.5 kHz: a VARA 500 exchange whose two callsigns the Winlink
#: RMS feed names independently of the audio, and whose spectrum is 600 Hz wide
#: and centred on the registered channel. Two connect requests in it, at wav
#: 4.35 s and 286.18 s, each 9/10 on the preamble at zero offset with 24 of 28
#: payload tones on the BW500 lattice — where band noise means 0.23 and never
#: reached 0.58 across 1154 sampled alignments of the two controls below.
SLOT = MODEM_ROOT / "working"
GROUND_TRUTH = SLOT / "rx-20260814-113359-7096500" / "audio" / "rx-000.wav"
#: 6800 kHz, no amateur allocation and no Winlink channel, 211 s.
CONTROL_OUT_OF_BAND = SLOT / "rx-20260814-114505-6800000" / "audio" / "rx-000.wav"
#: 7103.5 kHz over the ten minutes every classifier and the feed agree were empty.
CONTROL_QUIET = SLOT / "rx-20260814-111357-7103500" / "audio" / "rx-000.wav"

GATEWAY_LIST = Path(KESTREL.cwd) / "winlink-vara-gateways.csv"

live = pytest.mark.skipif(
    not (KESTREL.available() and Path(KESTREL.interpreter).exists()),
    reason="the kestrel runner or its interpreter is not here")


def _requires(path: Path):
    return pytest.mark.skipif(
        not path.exists(),
        reason=f"2026-08-14 monitoring slot recording not present ({path})")


def _detections(pcm: bytes) -> list[Detection]:
    """The real runner over raw audio, driven exactly as ``ModemMonitor`` drives
    it: 48 kHz s16le on stdin, detection JSON out."""
    done = subprocess.run([KESTREL.interpreter, KESTREL.runner], input=pcm,
                          cwd=KESTREL.cwd, stdout=subprocess.PIPE, timeout=900,
                          check=True)
    return [Detection.from_json(json.loads(ln))
            for ln in done.stdout.splitlines() if ln.strip()]


def _of_wav(path: Path) -> list[Detection]:
    from hfhost.audio import WavReplaySource
    return _detections(b"".join(
        p for _, p in WavReplaySource(path, realtime=False).frames()))


#: Burst amplitude against a 0.05 band-noise sigma, for each of the two ways a
#: connect request goes under an energy gate. Both are measured against the gate
#: as it stands, and the fixture re-checks them rather than trusting these figures.
#:
#: ``noise`` was 0.10 until 2026-08-14, when ``MonitorGate`` gained a second way in
#: — a lift HELD over MONITOR_HOLD_S rather than a peak — and started bracketing
#: this very fixture at 6.95 s. A connect request is 1.73 s of one tone, which is
#: exactly what a held test is built to find, so the old figure stopped being under
#: the gate. Measured on the sweep that replaced it: the gate opens at 0.10 and is
#: silent at 0.06 and below, while the scanner still reads 31/31 payload and 10/10
#: preamble down to 0.02 and holds 10/10 preamble at 0.015. 0.06 is the knee minus
#: 4.4 dB, and it is the figure to move if the gate improves again.
#:
#: ``mute`` needs no such margin and is the stronger case: a request that arrives
#: inside the 2.1 s hole under somebody's transmission is dropped by the gate
#: before any threshold is applied, because the gate refuses to judge a band it
#: cannot hear. It stays under at 0.10, 0.05 and 0.02 alike. That is a property of
#: the mute rule rather than of a threshold, so no future calibration can erode it.
BURST_AMP = {"noise": 0.06, "mute": 0.10}


def _quiet_connect_request(call: str, how: str = "noise",
                           at_s: float = 6.0, secs: float = 20.0):
    """Band noise with one connect request in it, under the gate — see BURST_AMP.

    The assertion at the end is the whole point of the fixture: it fails loudly if
    the request stops being invisible to the gate, rather than letting the test
    keep passing for a reason it no longer holds."""
    import sys

    import numpy as np
    from hfmodem.kestrel.vara import vara_frames as VF
    from hfmodem.kestrel.vara import vara_mfsk as MK

    sys.path.insert(0, str(Path(KESTREL.cwd) / "tools"))
    import vara_monitor as vm

    fs = vm.FS
    rng = np.random.default_rng(20260814)
    x = rng.standard_normal(int(secs * fs)) * 0.05
    if how == "mute":
        # our own transmission: the rig mutes its receiver 20 dB down under it,
        # and the request arrives inside that hole [corpus, 2026-07-26]
        x[int((at_s - 0.3) * fs):int((at_s + 1.8) * fs)] *= 10 ** (-20 / 20)
    burst = MK.synth_burst(call, VF.CR)
    x[int(at_s * fs):int(at_s * fs) + len(burst)] += burst * BURST_AMP[how]
    x = x / (np.abs(x).max() or 1.0)

    gate = vm.MonitorGate()
    opened = [s for i in range(0, len(x), 4800) for s, _ in gate.push(x[i:i + 4800])]
    assert not opened, ("the fixture is meant to sit under the gate, and the gate "
                        f"opened at {[round(s / fs, 2) for s in opened]} s")
    return (x * 32767).astype("<i2").tobytes()


@live
@pytest.mark.skipif(not GATEWAY_LIST.exists(),
                    reason="no cached Winlink gateway list to attribute against")
@pytest.mark.parametrize("how", ["noise", "mute"],
                         ids=["level with the band", "inside our own mute"])
def test_a_connect_request_under_the_gate_reaches_a_creance_pass(how):
    """The defect, at its smallest, in both the ways the module docstring names. A
    request that opens no gate produced an empty log before the scanner was wired
    in — which is what "no handshake captured" meant in the morning slot's report.
    """
    got = _detections(_quiet_connect_request("KB9MMT", how))
    crs = [d for d in got if d.kind == "CR"]
    assert len(crs) == 1, ("expected one connect request, got "
                           + (", ".join(f"{d.kind}@{d.t:.2f}" for d in got) or "nothing"))
    assert abs(crs[0].t - 6.0) <= 0.05, crs[0].t
    assert crs[0].protocol == "VARA" and crs[0].station == "KB9MMT", crs[0]


@live
@_requires(GROUND_TRUTH)
def test_both_recorded_connect_requests_survive_one_creance_pass():
    """The two the afternoon slot needed two passes to see. Both land immediately
    before a measured burst — 4.35 s against a run starting at 4.69 s, 286.18 s
    against one at 287.06 s — on the one recording in this project where an
    independent source names the mode, the channel and both stations."""
    crs = [d for d in _of_wav(GROUND_TRUTH) if d.kind == "CR"]
    at = sorted(round(d.t, 2) for d in crs)
    assert len(crs) == 2, f"expected two connect requests, got {at}"
    assert abs(at[0] - 4.35) <= 0.05 and abs(at[1] - 286.18) <= 0.05, at
    for d in crs:
        assert d.protocol == "VARA", d
        # Named by bandwidth and not by station: the payload sits on the BW500
        # tone lattice, which kestrel cannot regenerate a callsign from.
        assert "BW500" in d.detail, d.detail
        assert not d.station, d


@live
@pytest.mark.parametrize("wav", [CONTROL_OUT_OF_BAND, CONTROL_QUIET],
                         ids=["out-of-band 6800 kHz", "quiet 7103.5 kHz"])
def test_the_negative_controls_stay_silent(wav):
    """Neither control may speak. The scanner reads a raw stream rather than a
    bracketed burst, so the question it has to answer is whether it chatters on
    band noise.

    Measured 2026-08-14 over every alignment in both, at all seven frequency
    shifts: the best connect-request preamble in the out-of-band control locks 3
    of 10 tones over 314,705 alignments, and in the quiet 40 m window 7 of 10 over
    811,313 — one tone short of the 8 a detection takes, against the 9 and 10 the
    three real requests lock. The margin on the quiet window is one tone and not
    more, which is the number to watch if this ever fires."""
    if not wav.exists():
        pytest.skip(f"2026-08-14 negative control not present ({wav})")
    spoke = [d for d in _of_wav(wav) if d.kind in ("CR", "connect-response")]
    assert not spoke, ("a negative control named a handshake burst: "
                       + "; ".join(f"{d.kind}@{d.t:.2f} {d.detail}" for d in spoke))
