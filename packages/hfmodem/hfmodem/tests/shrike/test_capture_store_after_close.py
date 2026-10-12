# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""No process is spawned while a session's audio stream is open.

From 1a58d440 every rx/hold capture was stored as FLAC as soon as it was
written, which is two `flac` runs -- two forks of the whole modem -- every
1.25 s cycle with CoreAudio live. 7 of the next 50 PACTOR sessions lost ~10 ms
of capture to it, against 0 of the 190 before, and one of those losses moved a
transmission. The captures are written as WAV during the session and become
FLAC once the card is closed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_armdefaults import _parsed
from hfmodem.tests.shrike.test_grid import _Rig
from hfmodem.tests.shrike.test_identify import Clock

pytestmark = pytest.mark.skipif(shutil.which("flac") is None, reason="no flac on PATH")


class _Card(Clock):
    live = False

    def __init__(self, *a, **kw):
        super().__init__()
        _Card.live = True

    def close(self):
        # The writer keeps pace with the cycle on the air, so everything it was
        # handed is on the disk before the card goes away; here it has to be
        # waited for, or a spawn would slip past the close by luck.
        if onair._WRITER is not None:
            onair._WRITER.drain(30.0)
        _Card.live = False
        super().close()


def test_a_session_spawns_nothing_while_its_card_is_open_and_stores_after(
        tmp_path, monkeypatch):
    out = tmp_path / "out"
    spawned, spawned_live = [], []
    popen = subprocess.Popen

    class _Spy(popen):
        def __init__(self, args, *a, **kw):
            (spawned_live if _Card.live else spawned).append((args, kw))
            super().__init__(args, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", _Spy)
    monkeypatch.setattr(onair.ota, "Rig", lambda *a, **kw: _Rig())
    monkeypatch.setattr(onair, "find_device", lambda *a, **kw: 0)
    monkeypatch.setattr(onair, "_LiveInput", _Card)

    onair.run(_parsed("--transmit", "--serial", "/dev/null",
                      "--dial", "7100000", "--outdir", str(out),
                      "--pactor1-only", "--hold", "1", "--max-cycles", "3"))

    assert not _Card.live
    assert not spawned_live, f"spawned with the card open: {spawned_live}"
    # What CPython needs to posix_spawn instead of forking the modem.
    assert spawned and all(Path(args[0]).is_absolute() and kw["close_fds"] is False
                           for args, kw in spawned), spawned
    sides = [p for p in out.glob("*.json")
             if p.with_suffix(".flac").exists() or p.with_suffix(".wav").exists()]
    captures = [p for p in sides if "samples" in json.loads(p.read_text())]
    assert captures, sorted(p.name for p in out.iterdir())
    for side in captures:
        facts = json.loads(side.read_text())
        assert "kept" not in facts, facts
        assert facts["wav"] == side.with_suffix(".flac").name
        assert side.with_suffix(".flac").exists()
        assert not side.with_suffix(".wav").exists()
