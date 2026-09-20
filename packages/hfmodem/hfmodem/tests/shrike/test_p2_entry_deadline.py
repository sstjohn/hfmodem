# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The shaped compatibility entry must retain its 1.25-second cadence."""
from hfmodem.shrike import onair, pactor1, rxfront, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_grid import _Bench, _Rig
from hfmodem.tests.shrike.test_p3_offer import Keyed, cs_event


def test_entry_pulse_lead_is_reserved_before_the_renderer_runs(tmp_path):
    host = PtcHost(peer=Keyed(), mycall="W9SSJ")
    host.p1_grant_only = True
    host.arq.cfg.entry_ladder = ("p2sl1",)
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.on_rx_event(cs_event(pactor1.CS_SPEED))
    host.tick()
    host.on_rx_event(rxfront.Event(0.2, "unassigned", "0x59A",
                                   protocol=spec.Protocol.PACTOR1,
                                   spare=pactor1.CS_59A))
    assert host.arq.entry_pending
    tx = onair.RadioTx(_Rig(), transmit=True, out_dev=0, outdir=tmp_path,
                      settle=0.04)
    tx.host = host
    tx.live = bench = _Bench()
    fs = onair.FS
    grid = onair._MasterGrid(0, round(1.25 * fs), 0,
                            packet_n=round(0.810 * fs),
                            cs_n=round(0.210 * fs),
                            d_max_n=round(0.130 * fs))
    for slot in range(10, 14):
        tx.aim(grid, slot)
        # The live bridge closes against key_instant, then the renderer runs.
        # Charge a bounded 5 ms render on the real codec notice/clamp arithmetic.
        # Before the fix, the unreserved 18.75 ms pulse lead loses every slot.
        bench.now = bench.pos = tx.key_instant(grid, slot) - round(tx.settle * fs)
        bench.spend(round(0.005 * fs))
        tx.send_p2_entry_packet(b"", 0x1a)
        assert tx.slots_used[-1] == slot
        assert bench.emissions[-1][0] + onair.P2_KEY_LEAD_N == grid.boundary(slot)
