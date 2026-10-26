# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A clean line unkey needs no daemon forensics and must never reach CAT."""
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.fake_line_keyer import FakeLineKeyer

bridge = corpora.harness("vara_rig_bridge")


def test_a_clean_unkey_writes_no_forensics(monkeypatch):
    monkeypatch.setattr(bridge, "LineKeyer", FakeLineKeyer)

    def no_cat(*args, **kwargs):
        raise AssertionError("line PTT reached CAT")

    monkeypatch.setattr(bridge.Rig, "_cmd", no_cat)
    logged = []
    r = bridge.Rig(None, armed=True, log=logged.append,
                   line_ptt=True, ptt_device="/dev/fake")
    line = r._keyer
    try:
        assert r.key(True)
        assert line.high
        assert r.key(False)
        assert not line.high and not r.keyed
        assert not any("attempt" in m for m in logged), logged
    finally:
        r.shutdown()
