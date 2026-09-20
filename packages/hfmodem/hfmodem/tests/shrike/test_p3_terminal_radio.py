"""The terminal experiment uses the established DAC pulse and carrier cadence."""
import json

import numpy as np
import pytest

from hfmodem.shrike import arq, p3rx, placement, rx, rxfront, spec
from hfmodem.tests.shrike.test_p3_entry_iss_phase import entry_link
from hfmodem.tests.shrike.test_long_sl1 import linked


@pytest.mark.parametrize('grid_parity', [False, True])
def test_terminal_marker_preserves_entry_clock_and_carrier_alternation(tmp_path, grid_parity):
    session, tx, live, grid = entry_link(tmp_path, grid_parity=grid_parity)
    entry = tx.reply_clock.entry_phase
    observed = []
    for slot in (5, 6, 8):
        live.now = live.pos = grid.boundary(slot) - 10000
        tx.aim(grid, slot)
        assert tx.send_p3_terminal() != arq.REFUSED
        meta = json.loads((tmp_path/f'tx_{tx.n:02d}.json').read_text())
        phase = meta['audio_start'] + min(meta['pulse_offsets'])
        assert phase == entry + slot*60000
        swapped = meta['pulse_offsets'][0] > meta['pulse_offsets'][1]
        observed.append(swapped)
        audio = np.pad(live.dac_audio[-1], (4800, 4800))
        lead = min(meta['pulse_offsets']) + 4800
        z = {cn: rx._baseband(audio, cn, 48000, rx._pulse(480)) for cn in (5, 12)}
        header = p3rx.header_of(z, [lead+4320], placement.SPEED_PATHS[1])
        assert header is not None and header.vh == 1
        assert header.swapped == swapped
    assert observed[0] != observed[1] == observed[2]


def test_terminal_route_refuses_a_lower_protocol():
    host, tx = linked()
    host.protocol = spec.Protocol.PACTOR1
    assert host.send_p3_terminal() == arq.REFUSED
    assert not tx.frames


@pytest.mark.parametrize('role', [arq.IRS, arq.ISS])
def test_host_to_radio_waits_for_opposite_physical_ack_after_marker(role):
    host, tx = linked(role=role)
    host.p3_qrt_confirm = True
    host.arq.on_host_disconnect()
    host.tick()
    qrt_parity = host.arq.tx_seq & 1
    terminal_parity = qrt_parity ^ 1
    assert host.arq.said_goodbye

    def receive(cs):
        raw = placement.control_signal(cs)
        audio = np.pad(raw, (4800-placement.pulse_lead(raw), 4800))
        event = rxfront.SyncedRx().control_signal_at(audio, 4800, details=True)
        assert event is not None and event.cs == cs
        host.on_rx_event(event)

    receive(qrt_parity)
    assert host.arq.state == arq.State.DISCONNECTING
    assert host.arq.terminal_confirm_pending and not host.arq.terminal_confirm_emitted
    assert len(tx.frames) == 1
    host.tick()
    assert host.arq.terminal_confirm_emitted
    assert tx.frames[-1][0] == f'P3 TERMINAL marker VH{terminal_parity} (await CS{terminal_parity+1})'
    receive(qrt_parity)  # Repeating the preceding QRT ACK cannot close us.
    assert host.arq.state == arq.State.DISCONNECTING and not host.arq.goodbye_acked
    receive(terminal_parity)
    assert host.arq.state == arq.State.DISCONNECTED and host.arq.goodbye_acked
