# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The audio transport's unkey fence over the rigctld client's keying line.

A decision-only keyer double: no device, ioctl or physical-line evidence.
"""
from types import SimpleNamespace

import pytest

from . import corpora

rig_client = corpora.harness("rig_client")
kestrel_connect = corpora.harness("kestrel_connect")


class Keyer:
    def __init__(self, device, log):
        self.transport = SimpleNamespace(sense=lambda: self.reads)
        self.reads = False
        self.released_low = None
        self.must_retire = False

    def arm(self):
        pass

    def key(self, on, why=''):
        self.reads = on
        return True

    def hand_back(self):
        self.released_low = self.reads is False


@pytest.fixture
def fence(monkeypatch):
    monkeypatch.setattr(rig_client, "LineKeyer", Keyer)
    rig = rig_client.Rig(None, False, lambda _: None, ptt_device="/dev/fake", line_ptt=True)
    io = object.__new__(kestrel_connect.AudioVaraIO)
    io.rig, io._timing_key_active = rig, False
    yield rig, io.unkey_confirmed
    rig.retire()


def test_a_line_never_held_has_no_unkey_outstanding(fence):
    rig, confirmed = fence
    assert rig.line_state() is None
    assert confirmed() is True


@pytest.mark.parametrize("reads,expected", [(False, True), (None, False), (True, False)])
def test_a_held_line_is_confirmed_only_by_reading_low(fence, reads, expected):
    rig, confirmed = fence
    rig.armed = True
    rig._keyer.reads = reads
    assert confirmed() is expected


@pytest.mark.parametrize("reads,expected", [(False, True), (True, False)])
def test_a_handed_back_line_answers_with_its_hand_back_readback(fence, reads, expected):
    rig, confirmed = fence
    rig.armed = True
    keyer = rig._keyer
    keyer.key = lambda on, why='': True
    keyer.reads = reads
    rig.armed = False
    assert rig.released_low is expected and not rig.line_held
    assert confirmed() is expected


def test_a_retired_rig_never_confirms(fence):
    rig, confirmed = fence
    rig.retire()
    assert confirmed() is False
