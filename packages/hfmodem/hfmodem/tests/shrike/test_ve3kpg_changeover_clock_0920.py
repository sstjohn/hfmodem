"""VE3KPG 21:42: fresh CS3 packets must not inherit an old receive period."""
import pytest
from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.tests.shrike.test_long_sl1 import linked


@pytest.mark.parametrize('long', [False, True])
def test_changeover_uses_current_cycle_and_emits_ack_for_each_repeat(long):
    host, tx = linked(role=arq.IRS, long=long)
    # Capture the rendered audio and report an emission at the device seam.
    render = tx._tx
    def emit(*args, **kwargs):
        render(*args, **kwargs)
        tx.n += 1
    tx._tx = emit
    period = 180000 if long else 60000
    phase = 6257790  # First final-turn CS3 in the 21:42 recording.
    g = onair._MasterGrid(phase + round(.890 * 48000), 60000, 4320,
                          packet_n=46080, cs_n=5760, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.regear(long)
    tx.aim(g, 0)
    receiver = onair._SessionRx(host)
    receiver._p3_cycle_n = 60000 if long else 180000
    for n in range(3):
        at = phase + n * period
        ev = rxfront.Event(at / 48000, 'packet', 'empty CS3',
                          protocol=spec.Protocol.PACTOR3,
                          packet=(1, 0, b'', True), breakin=True,
                          start=at, cycle_long=False)
        assert receiver._note_p3_frame(ev, at)
        tx.aim(g, n * (3 if long else 1))
        assert receiver._p3_cycle_n == period
        assert g.p3_control_refusal(tx.slot) is None
        assert tx.send_cs(0) != arq.REFUSED
        assert not tx.refused and len(tx.frames) == n + 1
    # Repairing stale state must not disable the guard for a clock on the
    # other cycle length. Age alone no longer ends a corroborated comb; the
    # ARQ's silence budget does (1653d8a8).
    assert g.p3_control_refusal(tx.slot + 100) is None
    g.regear(not long)
    assert g.p3_control_refusal(tx.slot + 1) is not None
