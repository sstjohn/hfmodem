# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The monitor must hear a wideband VARA session, not just handshake bursts.

On 2026-08-10 this station's receiver sat on 7101.5 kHz — a channel a KiwiSDR
had just decoded 14 VARA frames on, and where ``core.busy`` read every window
occupied — while the merged monitor printed "quiet -> nothing heard in the
window" nine windows running. The kestrel runner was bracketing the stream with
the connect tool's ``Segmenter``, whose 4.0x enter threshold is calibrated for
MFSK handshake bursts standing ~12 dB proud of the broadband floor; a wideband
over spreads the same power across the whole passband and never opened it. The
fix is the gate ``vara_monitor``'s own traffic log runs (``MonitorGate``,
2.5x and mute-aware): on the 30 s 7101.5 capture it brackets 13 bursts where
``Segmenter`` bracketed one.

The recordings are driven through the real runner subprocess exactly as
``ModemMonitor`` drives it, and the assertions are made on ``ActivityView``
summaries at the aggregator's own cadence, so what is tested is the verdict an
operator reads. Like the rest of the off-air evidence the captures live outside
the tree; the skips below name what is missing, the bargain
hfmodem/tests/kestrel/corpora.py strikes.
"""

import json
import subprocess
from pathlib import Path

import pytest

from creance.monitor import ActivityView, Detection, QUIET, default_specs
from hfmodem.tests import evidence

KESTREL = next(s for s in default_specs() if s.name == "kestrel")

EVIDENCE = evidence.WORKING / "detector-evidence-20260811"
OWN_RX_7101K5 = EVIDENCE / "channel-7101k5.wav"
OWN_RX_7106K5 = EVIDENCE / "channel-7106k5.wav"
#: the KiwiSDR capture of the same evening's live session: 14 VARA frames
#: calling KO2F in its decode sidecars, 18/18 occupancy windows busy
KIWI_SESSION = (evidence.WORKING / "gwsurvey" / "run-20260811-0105Z"
                / "WS8EOC_7101.5k.wav")

WINDOW_S = 20.0
SUMMARY_EVERY_S = 10.0

live = pytest.mark.skipif(
    not (KESTREL.available() and Path(KESTREL.interpreter).exists()),
    reason="the kestrel runner or its interpreter is not here")


def _requires(path: Path):
    return pytest.mark.skipif(
        not path.exists(),
        reason=f"2026-08-10 monitor evidence capture not present ({path})")


def _detections(path: Path) -> list[Detection]:
    """The real runner over one recording, driven exactly as ``ModemMonitor``
    drives it: 48 kHz s16le on stdin, detection JSON out."""
    from hfhost.audio import WavReplaySource

    pcm = b"".join(p for _, p in WavReplaySource(path, realtime=False).frames())
    done = subprocess.run([KESTREL.interpreter, KESTREL.runner], input=pcm,
                          cwd=KESTREL.cwd, stdout=subprocess.PIPE, timeout=300,
                          check=True)
    return [Detection.from_json(json.loads(ln))
            for ln in done.stdout.splitlines() if ln.strip()]


def _summaries(dets: list[Detection], until_s: float) -> list[dict]:
    """The aggregator's periodic summaries, at its own default cadence."""
    view = ActivityView(window_s=WINDOW_S)
    out = []
    t = SUMMARY_EVERY_S
    while t <= until_s + 1e-9:
        for d in dets:
            if t - SUMMARY_EVERY_S < d.t <= t:
                view.add(d)
        out.append(view.summary(t))
        t += SUMMARY_EVERY_S
    return out


@live
@_requires(OWN_RX_7101K5)
def test_the_channel_the_monitor_called_quiet_is_not_quiet():
    """The defect itself: our own receiver on 7101.5 kHz during a live VARA
    session, and the monitor heard nothing for two windows running. Every
    summary over this capture must now report something on the channel."""
    summaries = _summaries(_detections(OWN_RX_7101K5), 30.0)
    verdicts = [s["overall"] for s in summaries]
    assert QUIET not in verdicts, (
        f"the monitor called a channel carrying a live VARA session quiet: "
        f"{verdicts}")


@live
@_requires(OWN_RX_7106K5)
def test_the_16db_wideband_over_is_reported_as_vara():
    """7106.5 kHz the same evening: the +16 dB wideband occupant of the working
    notes. Its over runs longer than any handshake burst and cannot be decoded
    on the live path, but it must still surface — bracketed whole and graded
    tentative VARA, not fragmented into unclassified energy or missed."""
    dets = _detections(OWN_RX_7106K5)
    assert any(d.protocol == "VARA" and d.kind == "long burst" for d in dets), (
        "no wideband over was reported on the channel carrying one: "
        + ", ".join(f"{d.kind}@{d.t:.1f}" for d in dets))
    assert QUIET not in [s["overall"] for s in _summaries(dets, 30.0)]


@live
@_requires(KIWI_SESSION)
def test_the_kiwi_witnessed_session_is_never_quiet():
    """The positive control: 180 s of the same channel through a KiwiSDR, with
    14 decoded VARA frames in its sidecar logs. If the monitor calls any window
    of this quiet it is not a monitor."""
    dets = _detections(KIWI_SESSION)
    summaries = _summaries(dets, 180.0)
    quiet = [s["t"] for s in summaries if s["overall"] == QUIET]
    assert not quiet, f"quiet windows at {quiet} s over a decoded live session"
    assert any(d.protocol == "VARA" for d in dets), (
        "nothing on the channel was graded VARA in 180 s of a VARA session")
