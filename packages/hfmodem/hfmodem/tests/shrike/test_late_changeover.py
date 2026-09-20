# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A CS3 the post-key flush delivers must rotate the next cycle's grid.

KB5LZK 2026-09-07 21:57 CDT (`working/onair-0907-2057`): CS3 at 223.65 s changed
the role to IRS after the cycle had snapshotted it, and the next cycle compared
against the snapshot, read "role unchanged", and kept transmitting on a grid still
phased for sending. Every reply after it fell ~0.5 s outside the window.
"""
import re

import pytest

from hfmodem.shrike import arq, onair, pactor1, rxfront
from hfmodem.tests.shrike.test_grantslot import _log


@pytest.mark.parametrize("reader,tag", (
    ("upgrade_scan", "RX"),
    ("upgrade_scan", "HOLD RX"),
    ("codeword", "HOLD RX"),
))
def test_final_decoder_changeover_rotates_before_receive_grid_update(
        tmp_path, monkeypatch, reader, tag):
    """A pre-key correction next cycle is too late for its receive window."""
    injected = []
    checked = []
    update = onair._MasterGrid.update

    def inject(sessrx):
        if (not injected and sessrx.tag == tag
                and sessrx.host.arq.state == arq.State.CONNECTED
                and sessrx.host.arq.role == arq.ISS):
            injected.append(sessrx.host)
            sessrx._on(rxfront.Event(0.0, "cs", "final decoder CS3",
                                     protocol="PACTOR-1",
                                     cs=pactor1.CS_CHANGEOVER))
            assert sessrx.host.arq.role == arq.IRS

    if reader == "upgrade_scan":
        scan = onair._SessionRx.upgrade_scan

        def late_scan(self, audio):
            scan(self, audio)
            inject(self)

        monkeypatch.setattr(onair._SessionRx, "upgrade_scan", late_scan)
    else:
        read = onair._read_codeword_at_bursts

        def late_codeword(sessrx, *args):
            read(sessrx, *args)
            inject(sessrx)

        monkeypatch.setattr(onair, "_read_codeword_at_bursts", late_codeword)

    def check_grid(self, *args, **kwargs):
        if injected and not checked:
            assert not self.sending, "receive grid still phased for ISS after CS3"
            checked.append(True)
        return update(self, *args, **kwargs)

    monkeypatch.setattr(onair._MasterGrid, "update", check_grid)
    log = _log(tmp_path, "--pactor1-only", grant=False)
    assert injected and checked, log


@pytest.mark.parametrize("tag", ("RX", "HOLD RX"))
def test_post_key_changeover_rotates_before_the_next_transmission(tmp_path,
                                                                monkeypatch, tag):
    flush = onair._SessionRx.flush
    injected = []

    def late_cs3(self):
        flush(self)
        if (not injected and self.tag == tag
                and self.host.arq.state == arq.State.CONNECTED
                and self.host.arq.role == arq.ISS):
            injected.append(True)
            print("TEST late CS3")
            self._on(rxfront.Event(0.0, "cs", "late CS3", protocol="PACTOR-1",
                                  cs=pactor1.CS_CHANGEOVER))

    monkeypatch.setattr(onair._SessionRx, "flush", late_cs3)
    log = _log(tmp_path, "--pactor1-only", grant=False)
    assert injected, log
    after = log.split("TEST late CS3", 1)[1]
    next_tx = re.search(r"TX\[\d+\]", after)
    assert next_tx is not None, after
    assert "GRID REVERSED -> IRS" in after[:next_tx.start()], after


@pytest.mark.parametrize("tag", ("RX", "HOLD RX"))
def test_a_changeover_found_by_the_hand_back_rotates_before_the_key(tmp_path,
                                                                   monkeypatch, tag):
    regrid = onair._regrid
    injected = []

    def late_cs3(live, raster, tx, host, sessrx, *rest):
        out = regrid(live, raster, tx, host, sessrx, *rest)
        if (not injected and sessrx.tag == tag
                and host.arq.state == arq.State.CONNECTED
                and host.arq.role == arq.ISS):
            injected.append(True)
            print("TEST late CS3")
            sessrx._on(rxfront.Event(0.0, "cs", "late CS3", protocol="PACTOR-1",
                                     cs=pactor1.CS_CHANGEOVER))
        return out

    monkeypatch.setattr(onair, "_regrid", late_cs3)
    log = _log(tmp_path, "--pactor1-only", grant=False)
    if not injected:
        pytest.skip("the replay never handed a slot back on this path")
    after = log.split("TEST late CS3", 1)[1]
    next_tx = re.search(r"TX\[\d+\]", after)
    assert next_tx is not None, after
    assert "GRID REVERSED -> IRS" in after[:next_tx.start()], after
