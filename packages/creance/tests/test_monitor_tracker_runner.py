# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The tracker runner: its pure ``normalize``, and the runner itself over a
passband built to hold a PACTOR-shaped link and a CW ID."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from creance.monitor.activity import TENTATIVE, Detection
from creance.monitor.runners import tracker_runner as tracker

REPORT = dict(t=12.5, kind="track", track=3, t_start=4.02, centre_hz=1614.83,
              lo_hz=1394.5, hi_hz=1828.1, level_dbfs=-41.2, over_db=24.0, crest_db=9.8,
              cadence_s=1.251, label="FSK-shaped", bursts=7, active_s=6.3)


def test_a_track_is_tentative_and_names_a_shape_never_a_protocol():
    d = tracker.normalize(**REPORT)
    assert d["modem"] == "tracker" and d["grade"] == TENTATIVE
    assert d["protocol"] == "FSK-shaped" and d["kind"] == "track"
    assert "station" not in d


def test_the_record_carries_the_designs_fields_through_detection():
    det = Detection.from_json(tracker.normalize(**REPORT))
    assert (det.centre_hz, det.level_dbfs, det.cadence_s, det.track) == (
        1614.83, -41.2, 1.251, 3)
    assert det.bw_hz == pytest.approx(1828.1 - 1394.5 + 48000 / 4096, abs=0.1)
    assert det.rf_hz is None and det.offset_hz is None and det.text is None


def test_no_cadence_is_left_out_rather_than_written_as_one():
    d = tracker.normalize(**(REPORT | {"cadence_s": None, "kind": "track-end"}))
    assert "cadence_s" not in d and d["kind"] == "track-end"
    assert Detection.from_json(d).cadence_s is None


def _passband():
    import numpy as np
    from hfmodem.core import cwid

    fs, rng = 48000, np.random.default_rng(1)
    n = 40 * fs
    x = rng.normal(0, 0.003, n)
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[(f < 1300) | (f > 1700)] = 0
    link = np.fft.irfft(spec, n)
    t = np.arange(n) / fs
    x += 0.03 * link / link.std() * ((t % 1.25) < 0.96) * (t > 2) * (t < 25)
    cw = cwid.audio("WS8EOC", 20, tone_hz=2205.0)
    x[28 * fs:28 * fs + len(cw)] += 0.002 * cw
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def test_the_runner_reports_the_link_and_reads_the_id():
    done = subprocess.run([sys.executable, str(Path(tracker.__file__))], input=_passband(),
                          stdout=subprocess.PIPE, timeout=120, check=True)
    recs = [json.loads(ln) for ln in done.stdout.splitlines() if ln.strip()]
    dets = [Detection.from_json(r) for r in recs]
    link = [d for d in dets if d.modem == "tracker" and d.cadence_s
            and d.centre_hz is not None and 1300 <= d.centre_hz <= 1700]
    assert link and all(abs(d.cadence_s - 1.25) <= 0.015 for d in link)
    assert {d.protocol for d in link} == {"FSK-shaped"}
    [cw_track] = {d.track for d in dets
                  if d.modem == "tracker" and d.protocol == "CW-shaped"}
    text = [d for d in dets if d.modem == "cw"]
    assert [(d.text, d.station, d.track) for d in text] == [("WS8EOC", "WS8EOC", cw_track)]
    assert text[0].t == pytest.approx(28.0, abs=0.1)
