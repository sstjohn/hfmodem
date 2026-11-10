# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The PACTOR-1 legs take `--p1-drive`; the PACTOR-3 legs never do.

`levels.at_drive` normalises every burst to one peak, which puts our
constant-envelope PACTOR-1 about 6 dB ABOVE the speed-level-1 entry packet in
average power. Both modems of the reference session run the ratio the other way:
DL6MAA's SL1 entry carries +7.0 dB over its own PACTOR-1 average
(`rf-corpus/PIII_Complete_1`, measured 2026-08-29 by squaring-law carrier
estimation), and the PTC-II answering it keys its own
PACTOR-1 control signals below its PACTOR-3 greeting by more. `p1_drive` is the
knob that lets an arm fly that convention: it scales ONLY the PACTOR-1
renderings, so the entry packet -- the artefact under test on the air -- goes
out exactly as before.

Default 1.0 is the shipped behaviour, asserted here so the knob cannot drift
into being a silent transmit-constant change.

The peak is read off the audio `_tx` hands the sink, after `at_drive`, which is
the sample stream a live arm would key; the dry-run WAV itself renormalises on
write and cannot carry the measurement.

Run:  pytest hfmodem/tests/shrike/test_p1drive.py
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import onair

ENTRY_STATUS = 0x1A     # the reference entry packet's own status byte

LEGS = ("connect", "p1 packet", "p1 cs", "p1 breakin", "entry", "p3 cs")
P1_LEGS = LEGS[:4]
P3_LEGS = LEGS[4:]


def _peaks(tmp_path, monkeypatch, p1_drive):
    keyed = []
    monkeypatch.setattr(onair.session, "write_wav",
                        lambda path, audio: keyed.append(np.asarray(audio)))
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path,
                       drive=0.8, p1_drive=p1_drive)
    tx.connect_burst("W9SSJ", "KB5LZK")
    tx.send_p1_packet(b"1W9SSJ\r", 100, 1)
    tx.send_p1_cs(0)
    tx.send_p1_breakin(b"", 100, 0)
    tx.send_entry_packet(1, b"", ENTRY_STATUS)
    tx.send_cs(0)
    assert len(keyed) == len(LEGS)
    return dict(zip(LEGS, (float(np.abs(a).max()) for a in keyed)))


def test_default_keys_every_leg_at_the_same_peak(tmp_path, monkeypatch):
    peaks = _peaks(tmp_path, monkeypatch, p1_drive=1.0)
    for name, peak in peaks.items():
        assert abs(peak - 0.8) < 1e-6, (name, peak)


def test_p1_drive_scales_only_the_pactor1_legs(tmp_path, monkeypatch):
    peaks = _peaks(tmp_path, monkeypatch, p1_drive=0.35)
    for name in P1_LEGS:
        assert abs(peaks[name] - 0.8 * 0.35) < 1e-6, (name, peaks[name])
    for name in P3_LEGS:
        assert abs(peaks[name] - 0.8) < 1e-6, (name, peaks[name])
