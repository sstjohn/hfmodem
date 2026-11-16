# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.shrike import onair, rxfront
from hfmodem.tests.shrike.p3_recovery_support import _RecoveryReplay


@pytest.mark.parametrize("stage", ["packet", "blind"])
def test_recovery_admission_sees_completed_packet_search_cost(tmp_path, monkeypatch, stage):
    class Costs:
        def seconds(self, name, args, kwargs, elapsed):
            return .1 if name == stage else 0.0

    replay = _RecoveryReplay("e08", tmp_path, scale=15, recovery=True,
                             measured_costs=Costs())
    receiver = replay.sessrx
    monkeypatch.setattr(receiver.sync, "packet", lambda *a, **kw: None)
    monkeypatch.setattr(rxfront, "decode_expected_packet", lambda *a, **kw: None)
    monkeypatch.setattr(receiver, "_corrected", lambda audio: audio)
    observed = []
    monkeypatch.setattr(receiver, "_p3_changeover_packet",
                        lambda audio: observed.append(replay.bench.sample_now()))
    before = replay.bench.sample_now()
    with replay._charging():
        receiver.upgrade_scan(np.zeros(onair.FS, np.float32))
    assert observed == [before + round(.1 * 15 * onair.FS)]


@pytest.mark.parametrize("p2_s,tail_s", [(.010156, .003953), (.003953, .010156)])
def test_recovery_preserves_mail_campaign_slots_with_slow_followup_reads(tmp_path, p2_s, tail_s):
    class Costs:
        def seconds(self, name, args, kwargs, elapsed):
            return {
                "control": .000974, "flush": .023228, "blind": .025673,
                "search": .001571, "cs-changeover": .000031,
                "p2": p2_s, "tail": tail_s, "upgrade": .000046,
                "scan": .000011, "recovery": .000024,
            }.get(name, 0.0)

    baseline = _RecoveryReplay("e08", tmp_path / "baseline", scale=15,
                               recovery=False, measured_costs=Costs()).run()
    recovered = _RecoveryReplay("e08", tmp_path / "recovery", scale=15,
                                recovery=True, measured_costs=Costs()).run()
    assert baseline.tx.slots_used == list(range(2, 13))
    assert recovered.tx.slots_used == baseline.tx.slots_used
    assert recovered.placements == baseline.placements
