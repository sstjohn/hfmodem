# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A pre-key read closed on what its reader needs reads what a whole frame does.

The window in front of an IRS reply closes on the DELIVERY of the packet and
its last row's filter support (`rxfront.FILTER_REACH_N`), and a first
changeover's on its last field row's (`_SessionRx.CHANGEOVER_READ_N`). Every
recorded packet here is read from that close and from a window running a whole
frame on, and has to come out the same: field, status, row and arrangement.

The reads behind the close cost less and return the same: the carrier shift
without `hilbert`'s complex transform pair, and a changeover body searched on
both arrangements at each alignment.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import onair, p3acquire, p3frame, p3trial, placement, rx, rxfront
from hfmodem.tests import evidence

FS, SPS, CYCLE = onair.FS, rxfront.SPS, 60000
LEAD = p3frame.DATA_OFFSET * SPS
FIXTURES = Path(__file__).with_name('fixtures')


def _pcm(path):
    fs, pcm = wavfile.read(path)
    assert fs == FS
    return pcm.astype(float) / 32768


def _rows():
    """(name, audio, row 0, long cycle, offset, speed level), per recorded packet."""
    rows = []
    meta = FIXTURES / 'prekey-read-0913' / 'metadata.json'
    if meta.is_file():
        m = json.loads(meta.read_text())
        audio = _pcm(meta.with_name(m['file']))
        for c in m['cycles']:
            rows.append((f"ws8eoc-0913-hold{c['hold']}", audio, c['row0'], False,
                         m['offset_hz'], c['speed_level']))
    meta = FIXTURES / 'gear-0920' / 'metadata.json'
    if meta.is_file():
        for r in json.loads(meta.read_text()):
            rows.append((r['file'], _pcm(meta.with_name(r['file'])),
                         r['row0'] - r['start'], False, r['offset_hz'], 2))
    meta = FIXTURES / 'p3-long-prekey-1107' / 'metadata.json'
    if meta.is_file():
        for r in json.loads(meta.read_text()):
            rows.append((r['file'], _pcm(meta.with_name(r['file'])),
                         r['row0'] - r['start'], True, r['offset_hz'], r['sl']))
    meta = FIXTURES / 'sl1-prekey-0920' / 'metadata.json'
    if meta.is_file():
        for r in json.loads(meta.read_text()):
            rows.append((r['file'], _pcm(meta.with_name(r['file'])),
                         r['row0_local'], r['cycle_long'], r['offset_hz'], 1))
    return rows


ROWS = _rows()


def _read(audio, row, long_cycle, hz, sl, end):
    lo = max(0, row - LEAD - 2400)
    window = p3acquire.compensate(audio[lo:end], hz)
    sync = rxfront.SyncedRx()
    read = sync.sl1_packet_at if sl == 1 else sync.sl2_packet_at
    ev = read(window, row - lo, long_cycle=long_cycle)
    return None if ev is None else (ev.packet, ev.start + lo, ev.carrier_swapped)


@pytest.mark.skipif(not ROWS, reason='no recorded PACTOR-3 packets in this distribution')
@pytest.mark.parametrize('row', ROWS, ids=lambda r: r[0])
def test_a_packet_reads_the_same_from_its_close_as_from_the_whole_frame(row):
    name, audio, r0, long_cycle, hz, sl = row
    paths = placement.LONG_PATHS if long_cycle else placement.SPEED_PATHS
    span = rxfront._frame_span(paths[sl])
    whole = _read(audio, r0, long_cycle, hz, sl, min(audio.size, r0 + span + 4 * SPS))
    assert whole is not None and whole[0][3], name
    close = r0 + rxfront._packet_span(span) + rxfront.FILTER_REACH_N
    assert close <= audio.size
    assert _read(audio, r0, long_cycle, hz, sl, close) == whole


CHANGEOVERS = FIXTURES / 'p3-first-changeover-1107' / 'metadata.json'


@pytest.mark.skipif(not CHANGEOVERS.is_file(),
                    reason='11-07 PACTOR-3 recordings are not included in this distribution')
@pytest.mark.parametrize('row', json.loads(CHANGEOVERS.read_text()) if CHANGEOVERS.is_file()
                         else [], ids=lambda r: r['file'])
def test_a_first_changeover_body_reads_the_same_from_its_close(row):
    audio = _pcm(CHANGEOVERS.with_name(row['file']))
    head = row['head'] - row['origin']
    lo = head - 8 * SPS

    def body(end):
        seg = p3acquire.compensate(audio[lo:end], row['offset_hz'])
        ev = rxfront._cs_event(seg, placement.BREAKIN_CS, 0, head - lo, 0.0, '')
        return ev.packet, ev.carrier_swapped
    whole = body(head + round(placement.PACKET_S * FS) + 8 * SPS)
    assert whole[0] is not None and whole[0][2] == row['payload'].encode()
    assert body(head + onair._SessionRx.CHANGEOVER_READ_N) == whole


def test_the_shift_is_the_analytic_signal_shift():
    """`compensate` against `scipy.signal.hilbert` on a pre-key window, a long
    one and a short one, at the offsets a session follows."""
    from scipy.fft import next_fast_len
    from scipy.signal import hilbert
    rng = np.random.default_rng(1107)
    for n in (41970, 180000, 1001):
        x = rng.normal(0, .1, n).astype(np.float32)
        for hz in (-5.4, 13.25, -100.0):
            ref = (hilbert(x.astype(float), next_fast_len(n))[:n]
                   * np.exp(-2j * np.pi * hz * np.arange(n) / FS)).real
            assert np.abs(p3acquire.compensate(x, hz) - ref).max() < 1e-12


@pytest.mark.skipif(not CHANGEOVERS.is_file(),
                    reason='11-07 PACTOR-3 recordings are not included in this distribution')
@pytest.mark.parametrize('row', json.loads(CHANGEOVERS.read_text()) if CHANGEOVERS.is_file()
                         else [], ids=lambda r: r['file'])
def test_a_swapped_changeover_body_is_read_at_its_first_alignment(row, monkeypatch):
    """Every 11-07 first changeover body arrived on the swapped arrangement: it
    reads at the head's own alignment and one neighbour, three trellis runs,
    where the home arrangement's seventeen failures came first."""
    runs = []
    decode = rx.decode_case0_softs
    monkeypatch.setattr(rx, 'decode_case0_softs',
                        lambda *a, **k: runs.append(1) or decode(*a, **k))
    audio = _pcm(CHANGEOVERS.with_name(row['file']))
    head = row['head'] - row['origin']
    lo = head - 8 * SPS
    seg = p3acquire.compensate(audio[lo:head + onair._SessionRx.CHANGEOVER_READ_N],
                               row['offset_hz'])
    ev = rxfront._cs_event(seg, placement.BREAKIN_CS, 0, head - lo, 0.0, '')
    assert ev.packet[2] == row['payload'].encode() and ev.carrier_swapped
    assert len(runs) == 3


def test_a_close_at_the_deadline_keeps_the_reserve_across_a_delivery_batch():
    """A read that waits for a sample's delivery returns with the rest of its
    callback batch: six blocks here, as a driver may hand them over. Closed on
    the deadline, the delivered count still leaves the decode its reserve in
    front of the DAC's notice, so an unarmed reply is not given away."""
    from types import SimpleNamespace
    from hfmodem.tests.shrike.test_p3_callback_budget import BatchedStream, callbacks

    class SixBlocks(BatchedStream):
        def __init__(self):
            super().__init__(0)
            self._callback_timing = callbacks(batch=6)

        def _advance(self, at):
            self.now = max(self.now, at)
            self.samples = (self.now - self.holdback) // 768 * 768

        def read(self, count):
            end = self.pos + count
            if count and end > self.samples:
                self._advance(-(-end // 768) * 768 + self.holdback)
            self.pos = end
            return np.zeros(count, np.float32)

    live = SixBlocks()
    assert onair._callback_delivery_n(live) == 768
    key = live.pos + 60000
    deadline = onair._p3_decode_deadline(live, key, round(.04 * FS))
    span = 35760
    row = deadline - rxfront._packet_span(span) - 100   # its reach runs past it
    rx = SimpleNamespace(_p3_row0=row, _p3_span=span, _p3_cycle_n=CYCLE)
    assert onair._p3_frame_ready(rx, deadline) == deadline
    live.take_until(deadline)
    assert live.clamp_late(key - round(onair.P3_DECODE_RESERVE_S * FS)) == 0


CAP = evidence.CAPTURES / 'onair-0919-2227'


@pytest.mark.skipif(not (CAP / 'stream.wav').exists(),
                    reason='local WS8EOC September 19 22:27 recording unavailable')
@pytest.mark.parametrize('answer_ms', [890.0, 895.0])
def test_every_short_packet_is_read_in_front_of_its_answer(tmp_path, answer_ms):
    """WS8EOC's 09-19 SL1 packets through the hold loop's own arithmetic: the
    window closes where `_p3_window_close` puts it, holds only what was
    delivered by then, and the delivered count stands at the end of the block
    that carried it. At 899 ms 23 of these read in front of their key with the
    frame span and the holdback in the close, and at 890 none did."""
    from hfmodem.tests.shrike.test_p3_timing_trial import receive, setup
    from hfmodem.winlink import B2FSession, MailClient
    fs, raw = wavfile.read(CAP / 'stream.wav')
    raw = raw.astype(np.float32) / 32768
    report = json.loads((CAP / 'p3-timing-trial.json').read_text())
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    live.audio = np.zeros(180 * FS, np.float32)
    live.limit = len(live.audio)
    entry = tx.timing_trial.entry_phase
    s.host.arq.cfg.traffic_sl = 1
    s.host.arq._sl = 1
    tx.timing_trial = None
    tx.reply_clock = p3trial.ReplyClock()
    tx.reply_clock.entry(entry)
    tx.p3_answer_n = round(answer_ms / 1e3 * FS)
    mail = MailClient(B2FSession('W9SSJ', target='WS8EOC', password='offline-only'),
                      s.host.arq.on_host_data)
    s.host.app = mail
    mail.link_up()
    first = entry + (report['opened'] - report['entry_phase'])
    shift = first - report['opened']
    receive(s, tx, live, grid, first)
    onair._p3_place_reply(grid, tx, 1)
    tx.emit_pending_cs()
    s.rx.p3_receive_offset_hz = -5.4
    read, previous = [], report['opened']
    for packet in report['packets'][1:]:
        phase = packet['phase']
        grid.cycles += max(1, round((phase - previous) / CYCLE))
        previous = phase
        head = phase + shift
        live.now = live.pos = head + round(.84 * FS)
        slot = round((head + round(.91 * FS) - grid.anchor) / CYCLE)
        tx.aim(grid, slot)
        s.rx.new_cycle()
        close = onair._p3_window_close(live, s.rx, s.host, tx, grid,
                                       tx.key_instant(grid, slot),
                                       round(tx.settle * FS), True)
        before = len(s.packets)
        if close is not None:
            live.now, live.pos = -(-close // live._blk) * live._blk, close
            onair._scan_frame(s.rx, raw[phase - 4800:close - shift].copy(), head - 4800,
                              tracked_only=True, p3_row0=head + LEAD)
            read.append(len(s.packets) > before and s.packets[-1].packet[1:3] == (
                packet['status'], bytes.fromhex(packet['payload_hex'])))
        if len(s.packets) == before:
            onair._scan_frame(s.rx, raw[phase - 4800:phase + 45000].copy(), head - 4800,
                              tracked_only=True, p3_row0=head + LEAD)
        onair._mail_app_turns(s.host, mail, wait_greeting=True)
        onair._p3_place_reply(grid, tx, slot)
        onair._reverse_before_key(grid, s.host, tx, slot)
        s.host.tick()
        onair._grid_reversal(grid, s.host)
        tx.emit_pending_cs()
    assert len(read) >= 30 and all(read), read
