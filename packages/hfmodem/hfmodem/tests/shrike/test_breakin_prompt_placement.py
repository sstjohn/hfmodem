# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The changeover at the greeting prompt: 0913-2050, WS8EOC 40 m, arm v16.

The arm's own coordinates. Every phase below is read out of that launch log's
`[ack] ... CRC frame @` and `audio start=` lines -- the internal phase
references, not the `[p3] frame @` detection samples printed beside them -- so
the geometry under test is the one the arm flew: 1.25 s cycle, a 210 ms PACTOR-3
control, the peer's 815 ms packet, the reverse gap at 150 ms.
"""
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_p3_breakin_timing import transitioned, tx_at

CYCLE_N = 60000
WIDTH = 39120                       # The peer's 815 ms PACTOR-3 packet.
# Three CRC frames in a row, ending with the 59 bytes that carried the SID.
RUN = (3108674, 3168554, 3228794)
CONTROLS = tuple(range(3211463, 3211463 + 11 * CYCLE_N, CYCLE_N))
DEAF_CYCLES = 14                    # Nothing decoded between RUN[-1] and RESUMED.
RESUMED = 4068768                   # ' via WS8EOC >\r', the greeting's last piece.


def at_the_prompt(deaf: int = DEAF_CYCLES, run=RUN):
    """The grid as the arm left it when the peer's greeting resumed."""
    g = transitioned()
    for n, phase in enumerate(run):
        g.note_p3_control(CONTROLS[n])
        g.note_p3_packet(phase, WIDTH, CYCLE_N)
        g.cycles += 1
    for control in CONTROLS[len(run):len(run) + deaf - 3]:
        g.note_p3_control(control)
        g.cycles += 1
    g.cycles += 3                   # The refused cycles that keyed nothing.
    return g


def test_breakin_prompt_arm_v16_geometry_is_the_one_that_was_refused():
    g = at_the_prompt()
    assert g.cs_n == 10080 and g.cycle_n == CYCLE_N
    assert g._p3_raster_corroborated
    # The resumed greeting is on the peer's own comb to 2 ms...
    assert g._peer_raster_position(RESUMED, CYCLE_N) is not None
    # ...and fourteen of its cycles past the last frame that decoded.
    assert round((RESUMED - RUN[-1]) / CYCLE_N) == DEAF_CYCLES


def test_breakin_prompt_resumed_frame_corroborates_the_reply_position():
    g = at_the_prompt()
    assert g._p3_reply_timing() is None      # Nothing decoded for fourteen cycles.
    g.note_p3_packet(RESUMED, WIDTH, CYCLE_N)
    timing = g._p3_reply_timing()
    assert timing is not None
    assert timing.control_phase == CONTROLS[10]
    assert timing.packet_phase == RESUMED
    assert timing.reverse_gap == 7225        # 150.5 ms, the arm's reverse gap.


def test_breakin_prompt_changeover_places_on_the_first_eligible_cycle(tmp_path):
    g = at_the_prompt()
    g.note_p3_packet(RESUMED, WIDTH, CYCLE_N)
    slot = (RESUMED - g.anchor) // CYCLE_N + 1
    tx = tx_at(g, slot, tmp_path)
    assert g.breakin_refusal(g.peer_packet_end(slot)) is None
    assert tx._place_breakin() and tx.placed and tx.unplaceable is None
    # Behind the packet it answers, inside the gap the peer reads in.
    past = tx.boundary - g.peer_packet_end(slot).at_slot
    assert 0 < past <= round(onair.BREAKIN_LEAD_S * onair.FS) + g.d_max_n


@pytest.mark.parametrize("uncorroborated", [False, True])
def test_breakin_prompt_reach_is_the_corroborated_rasters_and_no_wider(
        uncorroborated):
    """NEGATIVE CONTROL. One reading of the peer's comb still buys eight cycles.

    The fix is the raster's reach, not a longer age: a single frame projected
    fourteen cycles is the reading outliving its evidence that ONSET_MAX_CYCLES
    exists to refuse, and it stays refused.
    """
    g = at_the_prompt(run=RUN[:1] if uncorroborated else RUN)
    assert g._p3_raster_corroborated is not uncorroborated
    g.note_p3_packet(RESUMED, WIDTH, CYCLE_N)
    assert (g._p3_reply_timing() is None) is uncorroborated


def test_breakin_prompt_a_deaf_stretch_past_the_rasters_reach_stays_refused(
        tmp_path):
    g = at_the_prompt(deaf=onair.RASTER_PROJECT_CYCLES + 2)
    resumed = RUN[-1] + (onair.RASTER_PROJECT_CYCLES + 2) * CYCLE_N
    g.note_p3_packet(resumed, WIDTH, CYCLE_N)
    assert g._p3_reply_timing() is None
    slot = (resumed - g.anchor) // CYCLE_N + 1
    tx = tx_at(g, slot, tmp_path)
    assert tx._place_breakin() == ""
    assert "no fresh corroborated" in tx.unplaceable


class Seam(arq.ArqIO):
    """The transmit seam of the arm: the changeover refuses, controls key."""

    def __init__(self):
        self.events = []
        self.refuse_cs = False

    def send_packet(self, sl, payload, status, breakin=False):
        self.events.append(("packet", bytes(payload), breakin))
        return arq.REFUSED

    def send_cs(self, index):
        self.events.append(("cs", index))
        return arq.REFUSED if self.refuse_cs else None


def owing_a_goodbye():
    io = Seam()
    a = arq.PactorArq(io)
    a.role = arq.IRS
    a._enter_connected()
    a.on_rx_packet(1, b"RMS", 0, True, breakin=True,
                   protocol=spec.Protocol.PACTOR3)
    a.on_cycle()
    a._give_up("no decodable traffic from the peer")
    io.events.clear()
    return a, io


def test_breakin_prompt_refused_goodbye_still_keys_a_control():
    """0913-2050's eleven silent cycles, and why they could not recover.

    The goodbye's changeover is refused for want of the reply position; the
    reply position is measured from an emitted control; the changeover occupies
    the control slot. Keying nothing closes that loop for good, and the arm died
    on the goodbye's own clock with the peer's greeting still arriving.
    """
    a, io = owing_a_goodbye()
    assert a._qrt_pending
    for cycle in range(a.cfg.max_retries + 1):
        a.on_cycle()
        assert io.events[-2][0] == "packet" and io.events[-2][2], cycle
        assert io.events[-1] == ("cs", arq.CS_REQUEST), cycle
        assert a.state is arq.State.CONNECTED and a.role == arq.IRS, cycle
    # Past the refusal budget and still holding, because the goodbye's own
    # clock is the longer one -- and that clock, not ours, is what ends it.
    assert a._refused_cycles == 0 and a._silent_cycles <= a.cfg.max_retries
    while a.state is arq.State.CONNECTED:
        a.on_cycle()
    assert a.goodbye_unplaceable and not a.said_goodbye


def test_breakin_prompt_teardown_control_is_never_an_ack():
    a, io = owing_a_goodbye()
    a.on_rx_packet(1, b"RMS", 0, True, breakin=True,
                   protocol=spec.Protocol.PACTOR3)
    assert a._breakin_armed
    a.on_cycle()
    assert io.events[-1] == ("cs", arq.CS_REQUEST)


def test_breakin_prompt_refusals_alone_do_not_end_a_link_the_peer_serves():
    """The round-22 budget, read at the seam it was written for.

    An idle 88 cycles is what it targets. A peer still putting CRC frames on the
    air is not that, and the refusals in those cycles are our own. Ending a link
    in ten cycles while the far end is still serving it is the worse failure, so
    the budget spends nothing while the displaced control keeps reaching the air
    -- and a seam that keys nothing at all still ends the link
    (`test_refused_breakin_ack`).
    """
    io = Seam()
    a = arq.PactorArq(io)
    a.role = arq.IRS
    a._enter_connected()
    a.on_host_data(b"; WS8EOC DE W9SSJ\r")
    a.on_host_breakin()
    for cycle in range(4 * a.cfg.max_retries):
        a.on_rx_packet(1, b"CMS", 0, True, breakin=True,
                       protocol=spec.Protocol.PACTOR3)
        a.on_cycle()
        assert a.state is arq.State.CONNECTED and not a._qrt_pending, cycle
        assert io.events[-1] == ("cs", arq.CS_ACK), cycle
        assert a._refused_cycles == 0 and bytes(a._outbuf) == b"; WS8EOC DE W9SSJ\r"


def test_breakin_prompt_no_zero_byte_breakin_before_the_client_has_a_reply():
    """Question one of round 23, answered at the seam that requested it.

    The greeting gate held. `_mail_app_turns` withholds the automatic break-in
    for as long as the client has emitted no text, and a banner fragment or a
    SID does not release it (`test_mail_wait_greeting`). What asked for the
    0-byte break-in on 0913-2050 was the teardown -- `_give_up` ->
    `on_host_disconnect` -- because a QRT rides a packet and has no payload to
    ride on. So the two changeovers are told apart here: the goodbye's is empty
    by construction, and the application's carries the reply's first bytes.
    """
    a, io = owing_a_goodbye()
    a.on_cycle()
    assert io.events[0] == ("packet", b"", True)

    io = Seam()
    a = arq.PactorArq(io)
    a.role = arq.IRS
    a._enter_connected()
    a.on_host_data(b";FW: W9SSJ\r")
    a.on_host_breakin()
    a.on_rx_packet(1, b"CMS", 0, True, breakin=True,
                   protocol=spec.Protocol.PACTOR3)
    a.on_cycle()
    assert io.events[0][0] == "packet" and io.events[0][2]
    assert io.events[0][1] and b";FW: W9SSJ\r".startswith(io.events[0][1])
