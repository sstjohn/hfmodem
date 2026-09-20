# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Real recorded packets through pending-CS6 receive windows, no RF or sleeps."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, placement, rxfront, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_cs6_rx import FIXTURES, ROWS, recorded

FS = 48000
requires_recorded = pytest.mark.skipif(
    len(ROWS) < 2 or not all((FIXTURES / name).exists() for name in (
        'reference-long.wav', 'reference-short-a.wav')),
    reason='CS6 recorded PCM fixtures are absent from this checkout')


class PCMClock:
    key_notice = 1536  # 32 ms, the morning duplex scheduling requirement.
    holdback = 0

    def __init__(self, audio):
        self.audio = audio
        self.pos = self.samples = self.end = 0

    def wait_until(self, until):
        self.end = max(self.pos, min(int(until), len(self.audio)))
        return 0

    def read_ready(self):
        x = self.audio[self.pos:self.end]
        self.pos = self.samples = self.end
        return x

    def take_until(self, until):
        self.wait_until(until)
        return self.read_ready()

    def clamp_late(self, at):
        return max(0, self.samples + self.key_notice - at)

    def sample_now(self):
        return self.samples


def _arm(tmp_path, audio, *, old_long=False, sl=1, expected_seq=1):
    s = _Session(role=arq.IRS)
    host = s.host
    host.arq._cycle_long = old_long
    host.arq._cycle_request = not old_long
    host.arq._cycle_command_emitted = True
    host.arq._expected_seq = expected_seq
    host.arq.speed_level = sl
    # The WS8EOC recording is SL1, where the shipping P3_LADDER carries no long
    # frame, so an IRS declines the grant before any of this file's bookkeeping
    # runs. pactor3.md:640 publishes a 36-byte long SL1 payload; give the
    # fixture that rung so the cycle command exists to be tested. The refusal
    # itself lives in test_longcycle.py and test_cs6_arq.py.
    host.arq._ladder = replace(
        arq.P3_LADDER,
        payloads=((arq.P3_LADDER.payloads[0][0], 36),) + arq.P3_LADDER.payloads[1:])
    tx = onair.RadioTx(None, transmit=False, out_dev=None, outdir=tmp_path,
                      settle=.040)
    host.peer = tx
    tx.attach(host)
    tx.defer_p3_cs = True
    clock = PCMClock(np.pad(audio, (0, max(0, 4*FS-len(audio)))))
    tx.live = clock
    g = onair._MasterGrid(-14400, 60000, 8880, packet_n=46080,
                         cs_n=5760, d_max_n=6240)
    g.protocol = spec.Protocol.PACTOR3
    g.sending = False
    tx.aim(g, 1)  # peer phase at2880; leading CS pulse at45600:890ms later.
    s.rx._p3_row0 = 7200 - (180000 if old_long else 60000)
    s.rx._p3_cycle_n = 180000 if old_long else 60000
    path = (placement.LONG_PATHS if old_long else placement.SPEED_PATHS)[sl]
    s.rx._p3_span = rxfront._frame_span(path)
    g._p3_command_slot = 0
    g._p3_command_row0 = 7200
    return s, tx, g, clock


def _window(s, tx, g, clock, slot=1):
    tx.aim(g, slot)
    key = tx.key_instant(g, slot)
    until = onair._p3_frame_ready(s.rx, onair._p3_decode_deadline(clock, key, 1920))
    audio, origin, _ = onair._collect(clock, s.rx, np.zeros(0), 0, until)
    onair._scan_frame(s.rx, audio, origin, tracked_only=True)
    return onair._p3_transition_window(clock, g, tx, s.host, s.rx, slot,
                                       audio, origin, 1920)


@requires_recorded
@pytest.mark.parametrize('row', ROWS[:2] or [None],
                         ids=lambda r: r['file'] if r else 'no recording')
def test_short_retry_keeps_its_first_response_slot(tmp_path, row):
    s, tx, g, clock = _arm(tmp_path, recorded(row))
    slot, _, _ = _window(s, tx, g, clock)
    assert slot == 1
    assert s.packets[-1].packet[2] == b' Trim'
    assert s.packets[-1].carrier_swapped == row['carrier_swapped']
    assert not s.host.arq.cycle_long
    assert not s.host.arq.cycle_command_emitted
    assert tx._pending_p3_cs == arq.CS_CYCLE_TOG
    assert not clock.clamp_late(tx.key_instant(g, slot))


@requires_recorded
def test_long_answer_is_retained_until_crc_and_no_early_key(tmp_path):
    _, audio = wavfile.read(FIXTURES / 'reference-long.wav')
    s, tx, g, clock = _arm(tmp_path, audio.astype(float)/32768, sl=3, expected_seq=0)
    slot, _, _ = _window(s, tx, g, clock)
    assert slot == 3
    assert len(s.packets) == 1 and len(s.packets[0].packet[2]) == 276
    assert s.packets[0].cycle_long is True
    assert s.host.arq.cycle_long and g.ticks == 3
    assert not tx.keyed  # helper only listened; caller owns the later control.


@requires_recorded
def test_short_answer_to_downshift_is_read_at_first_short_slot(tmp_path):
    _, audio = wavfile.read(FIXTURES / 'reference-short-a.wav')
    s, tx, g, clock = _arm(tmp_path, audio.astype(float)/32768,
                          old_long=True, sl=3, expected_seq=2)
    slot, _, _ = _window(s, tx, g, clock)
    assert slot == 1
    assert s.packets[0].packet[1] == 0x1a
    assert not s.host.arq.cycle_long and g.ticks == 1


def test_silence_never_confirms_short_or_keys_inside_possible_long_frame(tmp_path):
    s, tx, g, clock = _arm(tmp_path, np.zeros(4*FS))
    slot, _, _ = _window(s, tx, g, clock)
    assert slot == 3 and not s.packets and not tx.keyed
    assert s.host.arq.cycle_request is True
    assert s.host.arq.cycle_command_emitted  # preserve possible-long protection
    assert not s.host.arq.cycle_long  # silence did not confirm a new geometry


@requires_recorded
def test_control_order_follows_physical_retries_and_first_pulse_hits_grid(tmp_path, monkeypatch):
    s, tx, g, clock = _arm(tmp_path, recorded(ROWS[0]))
    # Carrier order is only observable on the staggered SCS control; the
    # shipping default emits both tones on one clock.
    tx.p3_control_waveform, tx.p3_control_placement = 'current', 'pulse-center'
    slot, _, _ = _window(s, tx, g, clock)
    emitted=[]
    def capture(audio, what, **kwargs):
        emitted.append(kwargs)
    monkeypatch.setattr(tx, '_tx', capture)
    tx._send_p3_control(arq.CS_CYCLE_TOG)
    a = emitted[-1]
    # onair.py:1017-1021: a caller's control replies oppose the peer
    # permutation (PIII_Complete_1); this fixture is a caller arm.
    assert a['pulse_offsets'][1]-a['pulse_offsets'][0] == 240
    assert min(a['pulse_offsets']) == a['lead_n']
    # Skip one short cycle without inventing a fresh receive event: physical
    # order flips even though packet status/sequence have not changed.
    tx.aim(g, slot+1)
    tx._send_p3_control(arq.CS_CYCLE_TOG)
    b = emitted[-1]
    assert b['pulse_offsets'][0]-b['pulse_offsets'][1] == 240
    assert min(b['pulse_offsets']) == b['lead_n']


@pytest.mark.parametrize("notice", [1536, 2400])
def test_decode_deadline_reserves_notice_even_above_settle(notice):
    clock = PCMClock(np.zeros(1))
    clock.key_notice = notice
    deadline = onair._p3_decode_deadline(clock, 10000, 1920)
    assert deadline + notice + round(.006 * FS) <= 10000


@requires_recorded
def test_missing_first_probe_does_not_move_possible_long_origin(tmp_path):
    _, audio = wavfile.read(FIXTURES / 'reference-long.wav')
    s, tx, g, clock = _arm(tmp_path, audio.astype(float)/32768, sl=3, expected_seq=0)
    slot, _, _ = _window(s, tx, g, clock, slot=2)
    assert slot == 3
    assert len(s.packets) == 1 and s.packets[0].cycle_long
    assert s.host.arq.cycle_long


def test_missing_long_endpoint_hands_back_one_slot_and_keys_the_next(tmp_path):
    """An unresolved CS6 costs the slot it overran, and no slot after it.

    This case used to demand slot 6 for a search starting at 3, and a refusal
    from the transmitter at 4: our own command slot masked every boundary that
    was not three away from it, on the theory that the other two might lie
    inside a long packet the peer had yet to send. WS8EOC on 2026-09-13 sent no
    such packet -- eleven CS6 answered by eleven frames on the unmoved 1.25 s
    raster -- and the mask turned each overrun into a three-slot step: 80, 83,
    86, 89, then `REGRID GAVE UP`, twelve slots for one codeword. The comb is
    the peer's, and `ticks` moves it only where a CRC-valid long frame has been
    read (`test_crc_before_transition_helper_regears_recovery_past_long_body`).
    """
    s, tx, g, clock = _arm(tmp_path, np.zeros(8*FS))
    clock.pos = clock.samples = g.boundary(3) + 1
    assert g._p3_command_slot == 0 and not s.host.arq.cycle_long and g.ticks == 1
    assert onair._keyable_slot(clock, g, 3, 1920, listen=False) == 4
    tx.aim(g, 4)
    tx._send_p3_control(arq.CS_REQUEST)
    assert not tx.refused and tx.slots_used == [4]
    assert len(list(tmp_path.glob('tx_*.wav'))) == 1


@requires_recorded
def test_rolling_copy_cannot_cancel_new_command_or_redeliver_payload(tmp_path):
    s, tx, g, clock = _arm(tmp_path, recorded(ROWS[0]))
    _window(s, tx, g, clock)
    event = s.packets[0]
    s.host.arq._cycle_command_emitted = True
    g._p3_command_slot = 1
    g._p3_command_row0 = 67200
    s.rx.new_cycle()
    s.rx._on(event)
    assert len(s.packets) == 1
    assert s.host.arq.cycle_command_emitted
    assert g._p3_command_slot == 1


@requires_recorded
def test_command_epoch_is_only_recorded_after_actual_emission(tmp_path):
    s, tx, g, clock = _arm(tmp_path, recorded(ROWS[0]))
    # The slot-3 backstop arithmetic below is the pulse-centre geometry; under
    # the shipping audio-start default the emission fits in slot 1.
    tx.p3_control_waveform, tx.p3_control_placement = 'current', 'pulse-center'
    slot, _, _ = _window(s, tx, g, clock)
    tx.listening = True
    tx.emit_pending_cs()
    assert not s.host.arq.cycle_command_emitted and g._p3_command_slot is None
    tx.listening = False
    tx.send_cs(arq.CS_CYCLE_TOG)
    tx.emit_pending_cs()  # dry PCM output models emission without hardware
    assert s.host.arq.cycle_command_emitted
    # This emission uses the non-duplex dry path, which requires the full
    # 40 ms settle. The 32 ms notice PCM clock leaves only 38.1 ms here,
    # so its backstop must move to the next same-shift slot. Record the
    # actual command epoch, not the abandoned first probe.
    assert slot == 1
    assert g._p3_command_slot == tx.slot == 3
    assert abs(g._p3_command_row0 - 187200) <= 480
    assert len(tx.slots_used) == 1
    assert list(tmp_path.glob('tx_*.wav'))


@pytest.mark.parametrize('missed', [0, 2])
@requires_recorded
def test_recorded_reply_reaches_one_duplex_emission_with_pulse_timing(tmp_path, missed):
    s, tx, g, clock = _arm(tmp_path, np.pad(recorded(ROWS[0]), (missed*60000, 0)))
    slot, _, _ = _window(s, tx, g, clock)
    assert slot == missed + 1
    assert tx._pending_p3_cs == arq.CS_CYCLE_TOG
    # Charge measured full Session processing conservatively at6ms. Read and
    # DAC notice share the same clock; this is not a wall-time benchmark.
    clock.samples += round(.006 * FS)
    assert not clock.clamp_late(tx.key_instant(g, slot))
    calls = []
    def transmit(audio, *, at, settle, key, max_key):
        assert not clock.clamp_late(at)
        calls.append((at, len(audio)))
        clock.keyed_at = at - round(settle*FS)
        clock.keyed_s = settle + len(audio)/FS
        return at, at + len(audio)
    clock.transmit = transmit
    def flush_to(end):
        clock.pos = clock.samples = clock.end = end
    clock.flush_to = flush_to
    tx.rig = SimpleNamespace(ptt=lambda state: None, key_failure=lambda: None)
    tx.transmit = True  # Only the injected in-memory duplex above can run.
    s.host.tick(elapsed_ticks=missed+1, cycle_ticks=g.ticks)
    assert tx._pending_p3_cs == arq.CS_CYCLE_TOG
    tx.emit_pending_cs()
    assert len(calls) == 1 and len(tx.keyed) == 1
    assert tx.slots_used == [slot]
    assert s.host.arq.cycle_command_emitted
    assert g._p3_command_slot == slot
    meta = json.loads(next(tmp_path.glob('tx_*.json')).read_text())
    assert meta['control'] == 'CS6 CYCLE-TOG'
    # Audio-start placement: the audio opens on the boundary and the historical
    # control puts both tones on one clock, pulse centre 852 samples in.
    assert meta['audio_start'] == g.boundary(slot)
    assert meta['pulse_offsets'][0] == meta['pulse_offsets'][1]
    assert meta['audio_end'] - meta['audio_start'] == calls[0][1]


@pytest.mark.parametrize('exit_kind', ['reversal', 'protocol'])
def test_old_command_epoch_cannot_constrain_a_later_receiving_turn(tmp_path, exit_kind):
    s, tx, g, clock = _arm(tmp_path, np.zeros(4*FS))
    if exit_kind == 'reversal':
        g.reverse(to_iss=True)
        g.reverse(to_iss=False)
    else:
        g.keyed_slot = 0
        g.keying(spec.Protocol.PACTOR1)
        g.keying(spec.Protocol.PACTOR3)
    assert g._p3_command_slot is None and g._p3_command_row0 is None
    g.regear(True)
    assert onair._keyable_slot(clock, g, 4, 1920, listen=False) == 4


@requires_recorded
def test_recovered_endpoint_decodes_the_next_complete_long_frame(tmp_path, monkeypatch):
    _, audio = wavfile.read(FIXTURES / 'reference-long.wav')
    s, tx, g, clock = _arm(tmp_path, np.pad(audio.astype(float)/32768, (180000, 0)),
                          sl=3, expected_seq=0)
    tx.aim(g, 3)
    clock.pos = clock.samples = g.boundary(3) + 1
    def listen(live, n, host, sessrx, limit, **kwargs):
        return live.take_until(live.pos + max(0, n))
    monkeypatch.setattr(onair, '_listen_until_answer', listen)
    slot, _, _ = onair._regrid(clock, g, tx, s.host, s.rx, 3,
                               np.zeros(0), clock.pos, 1920)
    assert slot == 6
    assert len(s.packets) == 1 and s.packets[0].cycle_long
    assert s.host.arq.cycle_long and not s.host.arq.cycle_command_emitted
    assert not clock.clamp_late(tx.key_instant(g, slot))


@requires_recorded
def test_unseen_precommand_frame_cannot_resolve_a_later_cs6(tmp_path):
    s, tx, g, clock = _arm(tmp_path, recorded(ROWS[0]))
    _window(s, tx, g, clock)
    event = s.packets[0]
    # A late transmitter keyed CS6 two slots beyond its last decoded frame.
    # The rolling receiver now discovers an intervening, previously unread retry.
    s.host.arq._cycle_command_emitted = True
    g._p3_command_slot = 3
    g._p3_command_row0 = 187200
    s.rx.new_cycle()
    s.rx._on(replace(event, t=event.t + 1.25))
    assert len(s.packets) == 1
    assert s.host.arq.cycle_command_emitted and g._p3_command_slot == 3


@requires_recorded
def test_crc_before_transition_helper_regears_recovery_past_long_body(tmp_path):
    _, audio = wavfile.read(FIXTURES / 'reference-long.wav')
    s, tx, g, clock = _arm(tmp_path, audio.astype(float)/32768, sl=3, expected_seq=0)
    tx.aim(g, 3)
    deadline = onair._p3_decode_deadline(clock, tx.key_instant(g, 3), 1920)
    captured, origin, _ = onair._collect(clock, s.rx, np.zeros(0), 0, deadline)
    # Decode the retained physical header before entering the helper, as a
    # caller's initial scan may do. Resolving ARQ clears the command guard.
    onair._scan_frame(s.rx, captured, origin, tracked_only=True, p3_row0=7200)
    assert len(s.packets) == 1 and s.packets[0].cycle_long
    assert s.host.arq.cycle_long and not s.host.arq.cycle_command_emitted
    assert g._p3_command_slot is None and g.ticks == 1
    slot, _, _ = onair._p3_transition_window(
        clock, g, tx, s.host, s.rx, 3, captured, origin, 1920)
    assert slot == tx.slot == 3 and g.ticks == 3
    # Even after the current ACK deadline passes, the next placeable reply is
    # slot6. Slot4 would lie inside the now-confirmed next long packet.
    clock.pos = clock.samples = g.boundary(3) + 1
    assert onair._keyable_slot(clock, g, 3, 1920, listen=False) == 6
