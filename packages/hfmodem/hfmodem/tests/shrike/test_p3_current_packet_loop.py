# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Current-counter ACKs through the real loop using WS8EOC's recorded PCM.

The tape does not respond to our emissions. This checks local decoding and
scheduling, never the probability that a remote modem accepts an ACK. The real
hold loop determines every pre-key input buffer; no final saved window is
substituted at an earlier deadline.

The deterministic clock models 532 samples of input latency, 660 samples of
holdback, 128-frame callbacks delivered in 384-sample groups, 6 ms per deep_scan,
and 2 ms of final render work. These costs cover the proposed bounded reader;
they are not a reconstruction of arbitrary interpreter pauses or every CPU
stage in the live run. Cold CHANGEOVER acquisition may use the existing late
clamp. From the first ordinary packet onward the DAC pulse must be exact.

This regression uses the local uncommitted continuous recording. It explicitly
skips when that recording is absent rather than claiming capture coverage.
"""
import json
import sys
from collections import deque

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import onair, p3acquire, p3rx, placement, rx, rxfront
from hfmodem.tests import evidence
from hfmodem.tests.shrike import test_late_entry_loop as loop

CAP = evidence.CAPTURES / 'onair-0919-2029'
FS, CYCLE, BASE = 48000, 60000, 511822
NFRAMES = 147


@pytest.fixture(scope='module')
def captured_turn():
    if not (CAP / 'stream.wav').exists():
        pytest.skip('local WS8EOC September 19 20:29 capture unavailable')
    fs, raw = wavfile.read(CAP / 'stream.wav')
    assert fs == FS and raw.ndim == 1
    raw = raw.astype(np.float32) / 32768
    rows = []
    for cycle in range(NFRAMES):
        phase = BASE + cycle*CYCLE
        x = raw[phase-4800:phase+45000]
        if cycle < 2:
            g = p3acquire.control_signal(x, offsets=(0,))
            assert g and g.event.packet
            status, payload = g.event.packet[1:3]
            swapped = g.event.carrier_swapped
        else:
            # Only cycle 127 needs a nearby 4 Hz hypothesis in the independent
            # full-capture audit. This reference is not the live receiver.
            x = p3acquire.compensate(x, 4 if cycle == 127 else 1.8)
            z = {cn: rx._baseband(x, cn, FS, rx._pulse(480)) for cn in (5, 12)}
            header = p3rx.header_of(z, range(8640, 9601, 30),
                                    placement.SPEED_PATHS[1], fs=FS)
            assert header is not None
            g = p3rx.decode_at(x, header.at+4320, 1, fs=FS, Z=z, header=header)
            assert g is not None
            status, payload, swapped = g.status, g.payload, g.carrier_swapped
        rows.append(dict(cycle=cycle, seq=status & 3, payload=payload,
                         swapped=swapped))
    assert {r['swapped'] for r in rows} == {False, True}
    assert [r['seq'] for r in rows[:18]] == [0,0]+[1]*8+[2]*3+[3]*3+[0]*2
    return raw[BASE-4800:BASE+(NFRAMES-1)*CYCLE+45000], rows


def _run_captured_turn(monkeypatch, tmp_path, captured_turn, batch_phase, nframes=18,
                       *, expect_current=True):
    tape, rows = captured_turn
    tape = tape[:4800+(nframes-1)*CYCLE+45000]

    class CapturedPeer(loop.LatePeer):
        def __init__(self, *args):
            super().__init__(*args)
            self.audio = np.zeros(280*FS, np.float32)
            self.limit = len(self.audio)
            self.holdback, self.lat_in = 660, 532
            # Keep the actual 128-frame codec contract. Delivery comes in
            # three-callback groups, as measured in the September 19 capture.
            self._callback_timing = deque(
                (batch_phase+k*128, 128, (k//3)*.008+(k%3)*.00001,
                 (batch_phase+k*128)/FS)
                for k in range(24))

        @property
        def samples(self):
            return max(0, (self.now-self.lat_in-batch_phase)//384*384+batch_phase)

        def read(self, count):
            wanted = max(self.pos, self.floor)+max(0, int(count))
            if wanted > self.samples:
                self._advance(((wanted-batch_phase+383)//384)*384+batch_phase+self.lat_in)
            return super().read(count)

        def start_p3(self, entry_at):
            # The recorded peer head follows the emitted entry pulse by
            # 44,252 samples; entry_at is the trimmed DAC start (pulse +666).
            self.p3_phase = entry_at+666+44252
            at = self.p3_phase-4800
            self.audio[at:] = 0
            self.audio[at:at+len(tape)] = tape
            self.responses.append((at, at+len(tape)))

    class ChargedTx(onair.RadioTx):
        def cycle_cost_n(self):
            # Recorded final tick/render estimates were 1.3--2.0 ms. Give the
            # scheduler the same 2 ms history and spend it before the last
            # guard, where waveform rendering actually consumes that time.
            return max(super().cycle_cost_n(), 96)

        def _tx(self, audio, *args, **kwargs):
            if self.live is not None:
                self.live.spend(96)
            return super()._tx(audio, *args, **kwargs)

    monkeypatch.setattr(onair, 'RadioTx', ChargedTx)
    monkeypatch.setattr(loop, 'LatePeer', CapturedPeer)
    main = onair.main

    def timing_main():
        # Unbounded scoring allows the counter wrap. The finite tape ends on
        # ordinary link loss, so no real device or operator is needed.
        sys.argv += ['--p3-timing-trial', 'B', '--p3-timing-unbounded',
                     '--p3-timing-reply-delay', '3.125', '--p3-wideband-prekey']
        return main()

    monkeypatch.setattr(onair, 'main', timing_main)
    got = loop.run_late(monkeypatch, tmp_path, 1, False)
    trial = json.loads((tmp_path/'p3-timing-trial.json').read_text())
    assert trial['replies'], got['log']
    # LatePeer independently decodes the actual fake-DAC control waveform.
    emitted = {start: (ci+1, errors) for start, _, ci, errors in got['peer'].controls}
    scored = []
    for reply in trial['replies']:
        pulse = reply['audio_start']+min(reply['pulse_offsets'])
        cycle = (pulse-got['peer'].p3_phase)//CYCLE
        if 0 <= cycle < nframes:
            assert emitted[reply['audio_start']] == (reply['cs'], 0), reply
            if expect_current:
                # Equal counters on a repeat cannot hide reading an older
                # physical copy: this ACK must carry the current CRC clock.
                decoded_cycle = round((reply['peer_phase']-got['peer'].p3_phase)/CYCLE)
                assert decoded_cycle == cycle, (cycle, decoded_cycle, reply)
            row = rows[cycle]
            scored.append((cycle, reply['cs'], 1+(row['seq'] & 1)))
            delay = (pulse-trial['entry_phase']-28950) % CYCLE
            if cycle >= 2:
                assert delay == 0, reply
            else:
                # Initial CHANGEOVER acquisition can consume the unchanged
                # late-clamp allowance; ordinary packets may never do so.
                assert 0 <= delay <= round(.005*FS), reply
    assert scored
    return scored


@pytest.mark.parametrize('batch_phase', [0, 96, 192, 288])
def test_current_recorded_packet_acked_in_its_reply_slot(
        monkeypatch, tmp_path, captured_turn, batch_phase):
    scored = _run_captured_turn(monkeypatch, tmp_path, captured_turn, batch_phase)
    assert all(cs == want for _, cs, want in scored), scored
    # Once the first CRC establishes the receiver, every complete peer frame
    # gets a same-cycle emission: skips cannot disappear behind old-counter
    # equality during repeated payloads.
    assert [cycle for cycle, _, _ in scored] == list(range(scored[0][0], 18)), scored


def test_legacy_wideband_first_path_replies_with_previous_counter(
        monkeypatch, tmp_path, captured_turn):
    # Return the new bounded reader's ordinary miss. The existing production
    # wideband-first path then runs unchanged on the actual pre-key buffer.
    # This reproduces the previously observed wrong ACK immediately after the
    # first advance; it does not manufacture an old packet or pending CS.
    monkeypatch.setattr(rxfront.SyncedRx, 'sl1_packet_at', lambda *args: None)
    scored = _run_captured_turn(monkeypatch, tmp_path, captured_turn, 0,
                                expect_current=False)
    assert (2, 1, 2) in scored, scored


def test_full_recorded_turn_counters_wrap_without_stale_replies(
        monkeypatch, tmp_path, captured_turn):
    scored = _run_captured_turn(monkeypatch, tmp_path, captured_turn, 0, NFRAMES)
    assert all(cs == want for _, cs, want in scored), scored
    assert [cycle for cycle, _, _ in scored] == list(range(scored[0][0], NFRAMES)), scored
