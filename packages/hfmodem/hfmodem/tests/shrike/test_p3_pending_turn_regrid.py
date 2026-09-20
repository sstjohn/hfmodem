"""An unanswered CS3 still needs bounded packet/control reads in recovered slots.

WS8EOC2300 spent three recovered slots per hold after its first listening turn:
unconfirmed ISS selected the generic scan/flush path. About1.1s of aggregate
scan work then accompanied each5s window, and the packet clock aged out.
"""
from types import SimpleNamespace

import numpy as np

from hfmodem.shrike import arq, onair, rxfront, placement, spec
from hfmodem.tests.shrike.test_p3_breakin_timing import duplex, _Sessrx
from hfmodem.tests.shrike.test_p3_mail_changeover_phase import recorded_grid
from hfmodem.shrike.p3trial import ReplyClock


class Receiver(_Sessrx):
    def __init__(self, host, stream, grid, *, ack=False):
        self.host, self.stream, self.grid = host, stream, grid
        self._p3_row0 = 4128685 + 4320
        self._p3_cycle_n = 60000
        self._p3_span = rxfront._frame_span(placement.SPEED_PATHS[1])
        self._p3_delivered_at = self._p3_row0
        self.calls = []
        self.ack = ack

    def control_signal_in(self, audio, origin, grid):
        self.calls.append('control')
        self.stream.spend(48)  #1ms bounded control probe.
        return (0, origin) if self.ack else (None, None)

    def deep_scan(self, audio):
        tracked = getattr(self, '_tracked_only', False)
        self.calls.append('tracked' if tracked else 'blind')
        # Measured2300 aggregate spans hundreds of ms per blind recovered read;
        # the narrow CRC costs single digits. Charge on the sample clock.
        self.stream.spend(round((.006 if tracked else .230) * 48000))

    def flush(self):
        self.calls.append('flush')
        self.stream.spend(round(.040 * 48000))


def recover(tmp_path, *, pending=True, ack=False):
    clock = ReplyClock(entry_phase=1324493)
    grid = recorded_grid(clock, pulse_records=True)
    grid.reverse(to_iss=True)
    tx = duplex(grid, tmp_path)
    tx.reply_clock = clock
    tx.defer_p3_cs = True
    tx.breakin_due = True
    tx.breakin_cost_n = 96
    host = SimpleNamespace(protocol=spec.Protocol.PACTOR3,
        arq=SimpleNamespace(role=arq.ISS, state=arq.State.CONNECTED,
                            entry_pending=False, unconfirmed_breakin=pending,
                            cycle_request=None, cycle_command_emitted=False,
                            cycle_long=False), peer=tx)
    tx.host = host
    receiver = Receiver(host, tx.live, grid, ack=ack)
    tx.sessrx = receiver
    tx.aim(grid, 69)
    tx.live.now = tx.live.pos = tx.key_instant(grid, 69) + 480 - tx.live.key_notice
    slot, audio, origin = onair._regrid(tx.live, grid, tx, host, receiver, 69,
                                       np.zeros(0, np.float32), tx.live.pos, 1920)
    return slot, tx, grid, receiver


def test_pending_turn_recovered_slot_prioritizes_control_and_bounds_packet_work(tmp_path):
    slot, tx, grid, receiver = recover(tmp_path)
    assert slot == 70
    assert receiver.calls == ['control', 'tracked']
    assert tx.live.clamp_late(tx.key_instant(grid, slot)) == 0
    assert not tx.live.emissions  # Reading does not grant a turn or key itself.


def test_previous_iss_selection_reproduces_nonconverging_regrid(tmp_path, capsys):
    slot, tx, grid, receiver = recover(tmp_path, pending=False)
    assert slot == 69 + onair.REGRID_TRIES
    assert receiver.calls == ['blind', 'flush'] * onair.REGRID_TRIES
    assert tx.live.clamp_late(tx.key_instant(grid, slot)) > 240
    assert 'REGRID GAVE UP' in capsys.readouterr().out


def test_bare_ack_does_not_spend_a_packet_search_after_control_delivery(tmp_path):
    slot, tx, grid, receiver = recover(tmp_path, ack=True)
    assert slot == 70
    assert receiver.calls == ['control']
    assert tx.live.clamp_late(tx.key_instant(grid, slot)) == 0
