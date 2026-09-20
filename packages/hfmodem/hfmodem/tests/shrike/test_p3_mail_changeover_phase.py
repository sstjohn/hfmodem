"""The WS8EOC 2227 receive clock must survive a real outgoing changeover."""
import pytest
import numpy as np

from hfmodem.shrike import arq, onair, placement, rxfront, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_breakin_timing import duplex
from hfmodem.tests.shrike.test_p3_timing_trial import setup, receive


def recorded_grid(clock, *, pulse_records):
    # captures/onair-0919-2227: TX59 audio4112591, pulse4113443;
    # following CRC role request phase4128685, width39120, slot69.
    grid = onair._MasterGrid(32591, 60000, 0, packet_n=38880,
                            cs_n=10080, d_max_n=6240)
    grid.protocol, grid.sending = spec.Protocol.PACTOR3, False
    grid.acquired = grid.corroborated = True
    grid.d_n, grid.d_ref_n = 4800, 38880
    grid.reply_clock = clock
    if clock is not None:
        clock.target(0)  # This recording already emitted dozens of IRS ACKs.
    grid.note_p3_packet(4068685, 39120, 60000)
    grid.note_p3_control(4112591 + (852 if pulse_records else 0))
    grid.note_p3_packet(4128685, 39120, 60000)
    grid._p3_raster_origin = 1368925
    return grid


def test_actual_control_pulse_supplies_mail_turn_measurement(tmp_path):
    session, tx, stream, grid = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry)
    receive(session, tx, stream, grid, entry + 44400)
    tx.aim(grid, 1)
    onair._p3_place_reply(grid, tx, 1)
    tx.emit_pending_cs()
    phase = tx.tx_audio_start + min(tx.tx_pulse_offsets)
    assert not tx.refused
    assert grid._p3_controls[-1][0] == phase
    assert (phase - entry) % 60000 == 28950


def test_recorded_turn_replaces_ack_pulse_and_keeps_exact_mail_clock(tmp_path):
    clock = ReplyClock(entry_phase=1324493)
    grid = recorded_grid(clock, pulse_records=True)
    tx = duplex(grid, tmp_path)
    tx.reply_clock = clock
    tx.aim(grid, 69)
    tx.breakin_due = True
    tx._flip = lambda: False
    tx.live.now = tx.live.pos = 4160000
    expected = 4173443
    assert tx._breakin_reference(grid, 69) == expected
    end = grid.peer_packet_end(69)
    assert end.at_slot + grid.peer_read_gap == expected
    assert tx._clamp_refusal(end) is None
    key = tx.key_instant(grid, 69)
    tx.send_packet(1, b'abc', 0, breakin=True)
    assert not tx.refused
    phase = tx.tx_audio_start + min(tx.tx_pulse_offsets)
    assert phase == expected
    # The scheduler reserves the leading waveform skirt before its phase.
    assert key <= tx.tx_audio_start
    grid.reverse(to_iss=True)
    assert clock.target(0) % 60000 == expected % 60000
    grid.turn_accepted = True
    grid.reverse(to_iss=False)
    assert clock.target(0) % 60000 == (expected + 28800) % 60000


def test_legacy_audio_reference_reproduces_early_changeover_and_clock_reset(tmp_path):
    grid = recorded_grid(None, pulse_records=False)
    tx = duplex(grid, tmp_path)
    tx.aim(grid, 69)
    # Keeping B's audio boundary alone would place CS3's pulse17.75ms early.
    assert tx._breakin_key(grid, 69) == 4172591
    assert 4173443 - tx._breakin_key(grid, 69) == 852
    # Dropping B altogether moves the normal reply even farther backwards and
    # reproduces the refused turn in the actual recording.
    grid.p3_reply_shift(69)
    tx.aim(grid, 69)
    assert '20 ms' in tx._clamp_refusal(grid.peer_packet_end(69))


@pytest.mark.parametrize('first_swapped', [False, True])
def test_mail_cs3_and_next_two_ordinary_fields_share_emitted_phase(tmp_path, first_swapped):
    clock = ReplyClock(entry_phase=1324493)
    grid = recorded_grid(clock, pulse_records=True)
    tx = duplex(grid, tmp_path)
    tx.reply_clock = clock
    tx.aim(grid, 69)
    tx.breakin_due = True
    tx._flip = lambda: first_swapped
    tx.live.now = tx.live.pos = 4160000
    emitted = []
    send = tx._tx

    def measure(audio, label, **kwargs):
        # Measure the rendered waveform, independent of the requested lead
        # and the sidecar's claimed offsets.
        lead = placement.pulse_lead(audio)
        before = len(tx.live.emissions)
        send(audio, label, **kwargs)
        if len(tx.live.emissions) != before:
            emitted.append(tx.tx_audio_start + lead)

    tx._tx = measure
    tx.send_packet(1, b'abc', 0, breakin=True)
    assert not tx.refused
    grid.reverse(to_iss=True)
    tx.host.arq.role = arq.ISS
    for seq, swapped in ((1, not first_swapped), (2, first_swapped)):
        # The receiver's control moves600ms from its previous packet clock.
        # These ACK coordinates are a synthetic reactive peer, not capture
        # evidence that WS8EOC accepted our new outgoing packet.
        grid.note_peer_codeword(4217485 + (seq - 1) * 60000, 10080,
                                'CS1/ack' if seq == 1 else 'CS2/ack',
                                'WS8EOC', protocol=spec.Protocol.PACTOR3)
        tx.aim(grid, 69 + seq)
        tx._flip = lambda: swapped
        tx.send_packet(1, b'HELLO', seq)
        assert not tx.refused
    assert emitted == [4173443, 4233443, 4293443]


def test_unconfirmed_cs3_retry_and_return_keep_prior_receive_clock(tmp_path):
    clock = ReplyClock(entry_phase=1324493)
    grid = recorded_grid(clock, pulse_records=True)
    tx = duplex(grid, tmp_path)
    tx.reply_clock = clock
    tx._flip = lambda: False
    tx.live.now = tx.live.pos = 4160000
    for slot in (69, 70):
        tx.aim(grid, slot)
        tx.breakin_due = True
        tx.send_packet(1, b'abc', 0, breakin=True)
        assert not tx.refused
        assert tx.tx_audio_start + min(tx.tx_pulse_offsets) == 4173443 + (slot - 69) * 60000
        if slot == 69:
            grid.reverse(to_iss=True)
            tx.host.arq.role = arq.ISS
            grid.turn_accepted = False
    before = clock.target(0)
    grid.reverse(to_iss=False)
    assert clock.target(0) == before


@pytest.mark.parametrize('mail, under_own_carrier, keys', [
    (True, False, True), (False, False, False), (True, True, False)])
def test_new_ack_geometry_precedes_synchronous_next_packet_guard(
        tmp_path, mail, under_own_carrier, keys):
    clock = ReplyClock(entry_phase=1324493) if mail else None
    grid = recorded_grid(clock, pulse_records=mail)
    grid.reverse(to_iss=True)
    tx = duplex(grid, tmp_path)
    tx.reply_clock = clock
    session = _Session(role=arq.ISS)
    session.host.peer = tx
    tx.attach(session.host)
    tx.aim(grid, 70)
    tx._flip = lambda: False
    phase = 4217485
    tx.live.now = tx.live.pos = phase + 12000
    if under_own_carrier:
        tx.keyings.append((phase - 100, phase + 100))

    # ARQ._on_ack emits synchronously with buffered mail. Keep this test focused
    # on the receiver/guard ordering; the recorded mail-turn test runs real ARQ.
    session.host.on_rx_event = lambda ev: tx.send_packet(1, b'HELLO', 1)
    session.rx._on(rxfront.Event(phase / 48000, 'cs', 'ACK',
                                 protocol=spec.Protocol.PACTOR3, cs=0),
                    anchored=True)
    assert bool(tx.live.emissions) == keys
    assert tx.refused != keys


def test_inherited_rx_anchor_still_reads_first_bare_ack_after_mail_cs3(tmp_path):
    clock = ReplyClock(entry_phase=1324493)
    grid = recorded_grid(clock, pulse_records=True)
    # Last2227 IRS log: CS due4194414. Becoming ISS moves the receive
    # reference600ms; its inherited P1 prediction is119.35ms after this peer's
    # synthetic rotated control clock. Physical emitted-window admission must
    # still let the real P3 reader find the actual ACK.
    grid.d_ref_n = 4194414 - grid.boundary(69) - grid.d
    tx = duplex(grid, tmp_path)
    tx.reply_clock = clock
    tx._flip = lambda: False
    tx.live.now = tx.live.pos = 4160000
    session = _Session(role=arq.IRS)
    session.host.peer = tx
    tx.attach(session.host)
    tx.aim(grid, 69)
    tx.breakin_due = True
    tx.send_packet(1, b'abc', 0, breakin=True)
    assert not tx.refused
    grid.reverse(to_iss=True)
    session.host.arq.role = arq.ISS
    session.rx.new_cycle()
    phase = 4217485
    origin, end = tx.tx_end, 4229000
    audio = np.zeros(end - origin, np.float32)
    raw = placement.control_burst(0, tail=False, swapped=None)
    start = phase - int(np.argmax(placement.protocol_config().pulse())) - origin
    audio[start:start + len(raw)] = raw
    tx.aim(grid, 70)
    tx.live.now = tx.live.pos = end + 660
    assert grid.rx_due(69) - phase == 5729
    heard, _ = session.rx.control_signal_in(audio, origin, grid)
    assert heard == 0
    event = session.rx.cs_log[-1]
    assert event.protocol == spec.Protocol.PACTOR3
    assert abs(round(event.t * 48000) - phase) < 480
