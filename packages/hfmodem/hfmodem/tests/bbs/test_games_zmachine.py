# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Opt-in actual runtime acceptance; no payload or mock satisfies this test."""
import os
from pathlib import Path
import time

import pytest

from hfmodem.bbs.games.zmachine import ZMachineAdapter
from hfmodem.bbs.install import digest

INSTALL = os.environ.get("HFMODEM_BBS_ZORK_INSTALL")
pytestmark = pytest.mark.skipif(not INSTALL, reason="operator-installed pinned Zork/Frotz required")


def test_real_zork_native_save_reconnect_restore_and_file_mediation(tmp_path):
    installation = Path(INSTALL)
    executable, story = installation / "dfrotz", installation / "zork1.z3"
    assert digest(story) == "37084966477dff679282de42974b2077156b1bd68fad92a65d4ea94d8eb64d79"
    records = []
    def make():
        return ZMachineAdapter(executable, story, runtime_sha256=digest(executable), story_sha256=digest(story))
    def wait(adapter, kind="prompt"):
        limit = time.monotonic() + 5
        while time.monotonic() < limit:
            batch = adapter.poll()
            records.extend(batch)
            if any(event.kind == kind for event in batch):
                return batch
            assert not any(event.kind == "exited" for event in batch), records
            time.sleep(.005)
        pytest.fail(f"real Zork failed waiting for {kind}")
    first = make()
    state = tmp_path / "stable-player"
    try:
        first.start("stable-player", state)
        wait(first)
        assert not first.submit("\\help")
        assert not first.submit("north\nsouth")
        assert first.submit("open mailbox")
        wait(first)
        assert first.submit("save")  # native opcode uses the same managed handler
        wait(first)
        assert any(event.kind == "saved" for event in records)
        assert first.submit("north")
        wait(first)
    finally:
        first.close("local-to-VarAC switch")
        wait(first, "exited")
    second = make()
    try:
        second.start("stable-player", state)
        wait(second)
        assert second.restore()
        wait(second)
        assert any(event.kind == "restored" for event in records)
        assert second.submit("look")
        looked = wait(second)
        assert "The small mailbox contains" in "".join(event.text for event in looked)
        assert second.submit("script")
        wait(second)
        assert not list(state.glob("*.scr"))
        assert not list(state.glob("*.rec"))
        assert second.submit("quit")
        wait(second)
        assert second.submit("yes")
        wait(second, "exited")
    finally:
        if second.alive:
            second.close("test cleanup")
            wait(second, "exited")
