# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Recorded CS6 replies under elapsed-time faults, without audio devices."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair
from hfmodem.tests.shrike.test_cs6_driver import _arm, _window, requires_recorded
from hfmodem.tests.shrike.test_cs6_rx import FIXTURES, ROWS, recorded

FS = 48000


class ElapsedPCM:
    """A buffered reader cannot rewind the converter clock after decoder work.

    Only this in-memory object implements transmission. Capture position and
    elapsed samples deliberately diverge during a stall, as they do in duplex.
    """
    holdback = 0

    def __init__(self, audio, notice):
        self.audio = audio
        self.key_notice = notice
        self.pos = self.samples = self.end = 0
        self.emissions = []

    def wait_until(self, until):
        self.end = max(self.pos, min(int(until), len(self.audio)))
        self.samples = max(self.samples, self.end)
        return 0

    def read_ready(self):
        self.end = max(self.pos, min(self.samples, len(self.audio)))
        audio = self.audio[self.pos:self.end]
        self.pos = self.end
        return audio

    def take_until(self, until):
        self.wait_until(until)
        # Unlike read_ready, this read stops at the requested sample even if
        # the decoder has allowed later capture to accumulate in the queue.
        audio = self.audio[self.pos:self.end]
        self.pos = self.end
        return audio

    def sample_now(self):
        return self.samples

    def clamp_late(self, at):
        return max(0, self.samples + self.key_notice - at)

    def transmit(self, audio, *, at, settle, key, max_key):
        assert not self.clamp_late(at), "production guard allowed late RF"
        assert len(audio) / FS <= max_key
        self.keyed_at = max(self.samples, at - round(settle * FS))
        self.keyed_s = (at + len(audio) - self.keyed_at) / FS
        self.emissions.append((at, at + len(audio), self.keyed_at))
        self.samples = at + len(audio)
        return at, self.samples

    def flush_to(self, end):
        self.pos = self.end = end
        self.samples = max(self.samples, end)


def _duplex(tx, s, audio, notice):
    clock = ElapsedPCM(audio, notice)
    tx.live = clock
    tx.sessrx = s.rx
    tx.rig = SimpleNamespace(ptt=lambda state: None, key_failure=lambda: None)
    tx.transmit = True  # All output is intercepted by ElapsedPCM above.
    return clock


def _buffered_listen(live, n, host, sessrx, limit, **kwargs):
    return live.take_until(live.pos + max(0, n))


@requires_recorded
@pytest.mark.parametrize('long_reply,notice,stall_ms', [
    (False, 1536, 6), (False, 2400, 35),
    (False, 1536, 1400), (True, 1536, 5000),
])
def test_decode_stall_recovers_on_peer_cycle_without_late_emission(
        tmp_path, monkeypatch, long_reply, notice, stall_ms):
    if long_reply:
        _, pcm = wavfile.read(FIXTURES / 'reference-long.wav')
        pcm = pcm.astype(float) / 32768
    else:
        pcm = recorded(ROWS[0])
    period = 180000 if long_reply else 60000
    tape = np.zeros(45 * FS)
    for start in range(0, len(tape) - len(pcm), period):
        tape[start:start + len(pcm)] = pcm
    s, tx, g, _ = _arm(tmp_path, tape, sl=3 if long_reply else 1,
                        expected_seq=0 if long_reply else 1)
    clock = _duplex(tx, s, tape, notice)
    slot, audio, origin = _window(s, tx, g, clock)
    assert bool(s.host.arq.cycle_long) == long_reply
    clock.samples += round(stall_ms * FS / 1000)
    stalled_at = clock.samples
    monkeypatch.setattr(onair, '_listen_until_answer', _buffered_listen)
    slot, _, _ = onair._regrid(clock, g, tx, s.host, s.rx, slot,
                               audio, origin, 1920)
    assert clock.samples >= stalled_at
    assert not clock.clamp_late(tx.key_instant(g, slot))
    if long_reply:
        assert slot % 3 == 0
    tx.emit_pending_cs()
    assert len(clock.emissions) == len(tx.keyed) == 1
    assert tx.slots_used == [slot]
    meta = json.loads(next(tmp_path.glob('tx_*.json')).read_text())
    # Audio-start placement is the shipping default: the audio opens on the
    # boundary, and the historical control puts both tones on one clock with
    # the pulse centre inside the burst. The pulse-centre law these lines used
    # to assert is now a selectable experiment (test_control_profiles.py).
    assert meta['audio_start'] == g.boundary(slot)
    assert meta['pulse_offsets'][0] == meta['pulse_offsets'][1]


@requires_recorded
def test_cs6_then_retry_acks_keep_monotonic_clock_and_command_epoch(
        tmp_path, monkeypatch):
    tape = np.zeros(30 * FS)
    # First reply resolves CS6 as short. Lose the next two packets, then let
    # the real old-cycle retry recover at the protected long endpoint.
    for cycle in range(20):
        if cycle in (1, 2):
            continue
        pcm = recorded(ROWS[cycle % 2])
        start = cycle * 60000
        tape[start:start + len(pcm)] = pcm
    s, tx, g, _ = _arm(tmp_path, tape)
    clock = _duplex(tx, s, tape, 1536)
    monkeypatch.setattr(onair, '_listen_until_answer', _buffered_listen)
    previous_samples = 0
    next_slot = 1
    for turn in range(4):
        s.rx.new_cycle()
        slot, audio, origin = _window(s, tx, g, clock, next_slot)
        if turn == 1:
            assert slot == 4  # no key in the two missing/possible-long bodies
        clock.samples += round((.035 if turn == 2 else .006) * FS)
        slot, _, _ = onair._regrid(clock, g, tx, s.host, s.rx, slot,
                                   audio, origin, 1920)
        s.host.tick(elapsed_ticks=slot - next_slot + 1, cycle_ticks=g.ticks)
        tx.emit_pending_cs()
        assert clock.samples > previous_samples
        previous_samples = clock.samples
        # The first packet can grant CS6. Its old-cycle retries are ACKed,
        # retiring the completed command's receive window, not renewing it.
        assert s.host.arq.cycle_command_emitted is (turn == 0)
        assert g._p3_command_slot == (tx.slot if turn == 0 else None)
        assert not s.host.arq.cycle_long
        next_slot = tx.slot + 1
    assert len(clock.emissions) == len(tx.keyed) == 4
    assert len(set(tx.slots_used)) == 4
    # Retries are recorded as events but the ARQ delivers their bytes once.
    assert s.host.rcvd_total == len(b' Trim')


def test_stall_in_final_buffer_drain_refuses_cs6_without_recording_emission(
        tmp_path, monkeypatch):
    tape = np.zeros(8 * FS)
    s, tx, g, _ = _arm(tmp_path, tape)
    clock = _duplex(tx, s, tape, 1536)
    tx.aim(g, 3)
    bridge = s.rx.bridge

    def stalled_bridge(audio):
        bridge(audio)
        clock.samples += round(.035 * FS)

    monkeypatch.setattr(s.rx, 'bridge', stalled_bridge)
    tx.send_cs(arq.CS_CYCLE_TOG)
    tx.emit_pending_cs()
    assert tx.refused
    assert not clock.emissions and not tx.keyed and not tx.slots_used
    assert g._p3_command_slot == 0  # retain the previous actual CS6 epoch
    assert not list(tmp_path.glob('tx_*.json'))


@requires_recorded
def test_sustained_decode_overrun_bounds_recovery_and_preserves_grid(
        tmp_path, monkeypatch):
    pcm = recorded(ROWS[0])
    tape = np.zeros(45 * FS)
    for start in range(0, len(tape) - len(pcm), 60000):
        tape[start:start + len(pcm)] = pcm
    s, tx, g, _ = _arm(tmp_path, tape)
    clock = _duplex(tx, s, tape, 1536)
    slot, audio, origin = _window(s, tx, g, clock)
    clock.samples += round(1.4 * FS)
    scan = onair._scan_frame
    calls = []

    def stalled_scan(*args, **kwargs):
        scan(*args, **kwargs)
        calls.append(clock.samples)
        clock.samples += round(1.4 * FS)

    monkeypatch.setattr(onair, '_scan_frame', stalled_scan)
    monkeypatch.setattr(onair, '_listen_until_answer', _buffered_listen)
    slot, _, _ = onair._regrid(clock, g, tx, s.host, s.rx, slot,
                               audio, origin, 1920)
    assert len(calls) == onair.REGRID_TRIES
    assert clock.clamp_late(tx.key_instant(g, slot))
    # The bounded recovery handed the still-late burst to the final backstop.
    # That backstop may move the rendered control to a matching later slot,
    # but it may not move its physical leading pulse off the peer's grid.
    tx.emit_pending_cs()
    assert len(clock.emissions) == 1 and tx.slot > slot
    meta = json.loads(next(tmp_path.glob('tx_*.json')).read_text())
    assert meta['audio_start'] == g.boundary(tx.slot)
    assert meta['pulse_offsets'][0] == meta['pulse_offsets'][1]
    assert s.host.rcvd_total == len(b' Trim')


@pytest.mark.parametrize('slot', [3, 6, 9, 12])
def test_final_drain_reserves_a_delivered_capture_block(tmp_path, slot):
    tape = np.zeros(20 * FS)
    s, tx, g, _ = _arm(tmp_path, tape)
    clock = _duplex(tx, s, tape, 2400)
    clock._blk = 128
    take = clock.take_until

    def delivered_block(until):
        result = take(until)
        clock.samples = max(clock.samples, ((clock.pos + 127) // 128) * 128)
        return result

    clock.take_until = delivered_block
    tx.aim(g, slot)
    tx.send_cs(arq.CS_CYCLE_TOG)
    tx.emit_pending_cs()
    assert len(clock.emissions) == 1 and not tx.refused
    assert tx.slots_used == [slot]
    meta = json.loads(next(tmp_path.glob('tx_*.json')).read_text())
    assert meta['audio_start'] == g.boundary(slot)
    assert meta['pulse_offsets'][0] == meta['pulse_offsets'][1]


def _enqueue_clock(samples):
    # Exercise the production admission boundary without starting its callback,
    # watchdog or an audio device. Reaching callback arming is a test failure.
    native = object.__new__(onair._LiveInput)
    native._duplex, native.fs, native._blk, native._lat = True, FS, 128, 1152
    native.tx_latency_n = 0
    native.samples, native._tx = samples, None
    native._tx_done = SimpleNamespace(
        clear=lambda: pytest.fail('a stale explicit-at burst reached callback arming'))
    return native


def test_expired_explicit_dac_slot_is_refused_before_callback_arming():
    native = _enqueue_clock(10000)
    keyed = []
    at = native.samples + native.key_notice - 1
    with pytest.raises(onair._MissedTxSlot):
        native.transmit(np.ones(100), at=at, settle=.04, key=keyed.append)
    assert native._tx is None and not keyed


def test_stall_after_final_guard_leaves_no_emission_records(tmp_path):
    tape = np.zeros(8 * FS)
    s, tx, g, _ = _arm(tmp_path, tape)
    clock = _duplex(tx, s, tape, 1536)
    tx.aim(g, 3)
    old_slot, old_end = g.keyed_slot, tx.tx_end

    def stalled_enqueue(audio, **kwargs):
        native = _enqueue_clock(clock.samples + round(.035 * FS))
        return native.transmit(audio, **kwargs)

    clock.transmit = stalled_enqueue
    tx.send_cs(arq.CS_CYCLE_TOG)
    tx.emit_pending_cs()
    assert tx.refused and not tx.keyed and not tx.slots_used
    assert g.keyed_slot == old_slot and tx.tx_end == old_end
    assert g._p3_command_slot == 0 and not clock.emissions
    assert not list(tmp_path.glob('tx_*.json'))
