# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The RTTY QSO session on a bench card whose output loops into its own capture.

The loop stands in for the FTDX10's USB feed, which carries our own monitored
transmit audio: the sample armed for capture index i comes back at i. A second
station is added to the capture where a test puts it. Everything runs on the
card's sample clock, block by block, so nothing here depends on wall time.
"""
from __future__ import annotations

import io
import json
import threading
import time

import numpy as np
import pytest

from hfmodem.core import audio as card
from hfmodem.core.audio import ReplayAudio, StationAudio
from hfmodem.core.ptt import Attempt, Keyer as RealKeyer
from hfmodem.lanner import qso, rtty
from hfmodem.station.qsolog import QsoLog
from hfmodem.tests import fakeclaim

FS = card.CARD_RATE_HZ
LAT = 1152
TX_LATENCY_N = 336


class _Times:
    def __init__(self, adc, dac):
        self.inputBufferAdcTime, self.outputBufferDacTime = adc, dac


class _Status:
    input_overflow = output_underflow = priming_output = False

    def __bool__(self):
        return False


class Loopback:
    """The card's callback, driven by hand, with the DAC fed back into the ADC."""

    def __init__(self, seconds):
        self.far = np.zeros(round(seconds * FS), np.float32)
        self.echo = np.zeros(len(self.far) + FS, np.float32)
        self.sa = StationAudio(input_device='bench', output_device='bench',
                               tx_latency_n=TX_LATENCY_N)
        self.sa.sample_now = lambda: float(self.sa.samples)
        self.t = 0.0
        for _ in range(card._LAT_MEDIAN_N):
            self.block()

    def block(self):
        n, n0 = self.sa.blocksize, self.sa.samples
        indata = np.zeros((n, 1), np.float32)
        indata[:, 0] = self.far[n0:n0 + n] + self.echo[n0:n0 + n]
        outdata = np.zeros((n, 1), np.float32)
        self.sa._callback(indata, outdata, n, _Times(self.t, self.t + LAT / FS), _Status())
        start = n0 + self.sa.loop_lat
        self.echo[start:start + n] += outdata[:, 0]
        self.t += n / FS

    def place(self, at_s, text, *, mark=2295.0, space=2125.0, level=0.3):
        sent = level * rtty.modulate(text, mark_hz=mark, space_hz=space)
        at = round(at_s * FS)
        self.far[at:at + len(sent)] += sent[:len(self.far) - at]


class Keyer:
    def __init__(self, sa):
        self.sa, self.events, self.up, self.drops = sa, [], False, True

    def key(self, on, note=''):
        self.events.append((on, self.sa.samples))
        self.up = on or not self.drops
        return on or self.drops


class Line:
    """A keying line that reads back what it was last told."""
    device, place, peer, paced, reads_back = 'bench', 'line', 'the keying line', False, True

    def __init__(self):
        self.up = False

    def drive(self, on, timeout=None, *, confirm_silence=True):
        self.up = on
        return Attempt('confirmed', state=on, state_read=True)

    def arm(self):
        pass

    def sense(self):
        return self.up

    def release(self):
        return not self.up


def armed(keyer):
    keyer.arm()
    return keyer


class Stalled(io.StringIO):
    """A stdout whose reader has stopped."""

    def __init__(self):
        super().__init__()
        self.gate = threading.Event()

    def write(self, text):
        self.gate.wait(10)
        return super().write(text)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    def make(seconds=12, out=None, contest=None, **kw):
        loop = Loopback(seconds)
        monkeypatch.setattr(card.time, 'monotonic', lambda: loop.t)
        loop.block()
        keyer = Keyer(loop.sa)
        out = out or io.StringIO()
        session = qso.Session(loop.sa, keyer, QsoLog(tmp_path / 'log.adi', 'W9SSJ', contest=contest),
                              mycall='W9SSJ', dial_hz=14_080_000, out=out,
                              directory=tmp_path, shout=lambda line: None, **kw)
        return loop, keyer, session, out
    return make


def run(loop, session, seconds, *, until=None, step=True):
    for _ in range(round(seconds * FS / loop.sa.blocksize)):
        loop.block()
        if step:
            session.step()
        if until is not None and until():
            return True
    return False


def said(session, out):
    session.flush()
    return out.getvalue()


def text(session, kind):
    return ''.join(t for k, t, _ in session.pieces if k == kind)


def emitted(loop, until=None, rate=rtty.AMATEUR):
    return rtty.demodulate(loop.echo[:until], baud=rate.baud, mark_hz=2295, space_hz=2125)


def test_our_over_is_not_read_back_and_the_answer_is(bench):
    loop, keyer, session, _ = bench(16)
    for ch in 'CQ CQ DE W9SSJ K':
        session.type(ch)
    session.turn_over()
    assert run(loop, session, 6, until=lambda: session.state == 'rx')
    assert [on for on, _ in keyer.events] == [True, False]
    assert text(session, 'tx') == 'CQ CQ DE W9SSJ K\n'
    assert emitted(loop) == 'CQ CQ DE W9SSJ K\n'
    loop.place(loop.sa.samples / FS + 0.5, 'W9SSJ DE K1ABC K1ABC 599 K\n')
    run(loop, session, 8)
    assert 'CQ' not in text(session, 'rx')
    assert text(session, 'rx').endswith('W9SSJ DE K1ABC K1ABC 599 K\n')
    assert session.set_call() == 'K1ABC'


def test_typeahead_goes_out_in_order_and_backspace_works_until_it_has(bench):
    loop, keyer, session, _ = bench()
    for ch in 'ABX':
        session.type(ch)
    session.backspace()
    session.key()
    run(loop, session, 0.5)
    for ch in 'CD 12':
        session.type(ch)
    assert not session.type('@')
    session.turn_over()
    assert run(loop, session, 6, until=lambda: session.state == 'rx')
    assert text(session, 'tx') == 'ABCD 12\n'
    assert emitted(loop) == 'ABCD 12\n'


def test_stopping_drops_the_carrier_and_keeps_what_was_not_sent(bench):
    loop, keyer, session, _ = bench()
    for ch in 'THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG':
        session.type(ch)
    session.key()
    run(loop, session, 1.0)
    session.abort()
    assert session.state == 'rx' and not keyer.up
    sent = text(session, 'tx')
    assert 0 < len(sent) < 20
    assert emitted(loop, until=keyer.events[-1][1]).startswith(sent)
    assert ''.join(session.typeahead) == 'THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG'[len(sent):]
    stopped = loop.sa.samples
    run(loop, session, 0.5)
    assert not np.any(loop.echo[stopped + loop.sa.loop_lat + loop.sa.blocksize:])


def test_the_watchdog_unkeys_a_stream_nobody_feeds(bench, monkeypatch):
    loop, keyer, session, out = bench()
    session.key()
    monkeypatch.setattr(session, '_pump', lambda: None)
    fired = run(loop, session, 3, until=lambda: (session.watchdog(), not keyer.up)[1])
    assert fired
    starved = session._tx_stream.starved_at
    assert loop.sa.samples - starved >= qso.STARVED_S * FS
    assert loop.sa.samples - starved < qso.STARVED_S * FS + 2 * loop.sa.blocksize
    assert 'WATCHDOG' in said(session, out)
    session.step()
    assert session.state == 'rx'


def test_the_watchdog_drops_ptt_when_the_card_stops_calling_back(bench):
    loop, keyer, session, out = bench()
    for ch in 'THE QUICK BROWN FOX':
        session.type(ch)
    session.key()
    run(loop, session, 0.5)
    assert keyer.up
    for _ in range(12):
        loop.t += 0.1
        session.step()
        session.watchdog()
    assert not keyer.up
    assert 'WATCHDOG -- the audio callbacks stopped' in said(session, out)


def test_the_watchdog_drops_ptt_when_the_transmit_thread_stalls_after_the_last_sample(bench):
    loop, keyer, session, out = bench()
    session.type('E')
    session.turn_over()
    run(loop, session, 3, until=lambda: session._tx_stream.closed)
    run(loop, session, 3, step=False, until=lambda: (session.watchdog(), not keyer.up)[1])
    assert not keyer.up
    assert 'WATCHDOG -- the transmit thread stopped' in said(session, out)


def test_a_transmit_thread_that_fails_drops_ptt_on_its_way_out(bench, monkeypatch):
    loop, keyer, session, out = bench()
    session.type('E')
    session.key()
    monkeypatch.setattr(session.audio, 'breathe', lambda: None)

    def fail():
        raise RuntimeError('decoder fell over')

    monkeypatch.setattr(session, 'step', fail)
    session._running = True
    with pytest.raises(RuntimeError):
        session._work()
    assert not keyer.up
    assert 'thread stopped while keyed' in said(session, out)


def test_keying_on_a_dead_card_says_so_and_transmits_nothing(bench):
    loop, keyer, session, out = bench()
    loop.t += 2
    session.type('E')
    session.key()
    assert keyer.events == [] and session.state == 'rx'
    assert 'audio callbacks stopped' in text(session, 'note')


def test_ctrl_c_unkeys_while_another_thread_holds_the_session(bench):
    import threading
    loop, keyer, session, out = bench()
    session.type('E')
    session.key()
    held, release = threading.Event(), threading.Event()

    def hold():
        with session._lock:
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    stopper = threading.Thread(target=session.abort)
    stopper.start()
    stopper.join(0.5)
    try:
        assert not keyer.up
    finally:
        release.set()
        holder.join()
        stopper.join()
    assert session.state == 'rx'


def test_the_ptt_leads_the_audio_by_the_settle_and_unkeys_after_the_last_sample(bench):
    loop, keyer, session, _ = bench(settle_s=0.05)
    session.type('E')
    session.turn_over()
    (on, keyed), = keyer.events
    stream = session._tx_stream
    assert stream.at - keyed == round(0.05 * FS)
    done = []
    run(loop, session, 4, until=lambda: (done or loop.sa.burst_done() and done.append(loop.sa.samples),
                                         not keyer.up)[1])
    off = keyer.events[-1][1]
    assert off >= done[0] + TX_LATENCY_N
    assert off >= stream.at + stream.written
    assert stream.at <= int(np.flatnonzero(loop.echo)[0]) <= stream.at + 1


def test_macros_fill_from_the_station_and_the_qso_in_progress(bench, tmp_path):
    loop, keyer, session, _ = bench(grid='EN63bc')
    session.macro(1)
    assert ''.join(session.typeahead) == '\n DE W9SSJ\n'
    session.typeahead.clear()
    session._heard = 'CQ CQ DE N0XYZ N0XYZ K\n'
    session.set_call()
    session.macro(2)
    assert ''.join(session.typeahead) == '\nN0XYZ DE W9SSJ UR 599 599 QTH EN63BC EN63BC BK\n'
    session.typeahead.clear()
    session.macro(0)
    assert session.state == 'tx'
    assert session.commit().call == 'N0XYZ'
    session.flush()
    assert 'N0XYZ' in (tmp_path / 'log.adi').read_text()


def test_a_station_with_no_grid_sends_an_empty_grid_not_a_borrowed_one(bench):
    loop, keyer, session, _ = bench()
    session._heard = 'CQ CQ DE N0XYZ N0XYZ K\n'
    session.set_call()
    session.macro(2)
    assert ''.join(session.typeahead) == '\nN0XYZ DE W9SSJ UR 599 599 QTH   BK\n'


def test_the_receiver_acquires_an_off_frequency_station_from_a_recording(tmp_path):
    sent = 'CQ CQ DE K1ABC K1ABC K\n' * 3
    samples = np.r_[np.zeros(FS), 0.3 * rtty.modulate(sent, mark_hz=2330, space_hz=2160),
                    np.zeros(FS)]
    replay = ReplayAudio(samples)
    session = qso.Session(replay, None, QsoLog(tmp_path / 'log.adi', 'W9SSJ'),
                          mycall='W9SSJ', out=io.StringIO())
    while not replay.exhausted():
        replay.pump(8)
        session.step()
    assert session.rx_centre == pytest.approx(2245, abs=5)
    assert session.tx_centre == qso.CENTRE_HZ
    assert 'K1ABC K1ABC K\n' in text(session, 'rx')


def test_reverse_and_nudge_move_the_receive_pair_only(bench):
    loop, keyer, session, _ = bench()
    session.toggle_reverse()
    assert (session.rx_mark, session.rx_space) == (2125, 2295)
    session.nudge(-1)
    assert session.rx_centre == qso.CENTRE_HZ - qso.NUDGE_HZ
    assert (session.tx_mark, session.tx_space) == (2295, 2125)
    assert session.mark_rf_hz == 14_082_295
    session.toggle_usos()
    assert session.stream.unshift_on_space is False


def test_the_session_records_a_transcript_and_both_sides_of_the_capture(bench, tmp_path):
    import wave
    loop, keyer, session, out = bench()
    for ch in 'TEST DE W9SSJ':
        session.type(ch)
    session.turn_over()
    run(loop, session, 6, until=lambda: session.state == 'rx')
    run(loop, session, 0.5)
    session.close()
    session.finish()
    transcript = (tmp_path / 'transcript.txt').read_text()
    assert 'TX TEST DE W9SSJ' in transcript
    assert 'mark RF 14082295 Hz on dial 14080000' in transcript and 'drive 0.500' in transcript
    assert 'tx_event ' in out.getvalue()
    with wave.open(str(tmp_path / 'stream.wav')) as tape:
        recorded = np.frombuffer(tape.readframes(tape.getnframes()), '<i2') / 32767
    assert rtty.demodulate(recorded, mark_hz=2295, space_hz=2125) == 'TEST DE W9SSJ\n'
    assert session.id_at is not None


def test_what_the_console_recorders_say_reaches_the_screen(bench, tmp_path):
    loop, keyer, session, _ = bench()
    session.read_news()
    with (tmp_path / 'recorders.log').open('a') as log:
        log.write('witness placed, pid 7\nNO TA')
    session.read_news()
    with (tmp_path / 'recorders.log').open('a') as log:
        log.write('P: tap.log says why\n')
    session.read_news()
    notes = [t for k, t, _ in session.pieces if k == 'note']
    assert notes[-2:] == ['witness placed, pid 7', 'NO TAP: tap.log says why']


def test_a_key_up_that_will_not_drop_is_an_alarm_and_never_rx(bench):
    from hfmodem.lanner import terminal
    loop, keyer, session, out = bench()
    keyer.drops = False
    session.type('E')
    session.turn_over()
    run(loop, session, 4, until=lambda: session.state == 'rx')
    assert keyer.up and session.ptt_unconfirmed
    assert 'unkeyed at card' not in said(session, out)
    assert 'PTT DOWN NOT CONFIRMED' in text(session, 'alarm')
    assert terminal.status(session).startswith('PTT UNCONFIRMED  !! PTT DOWN NOT CONFIRMED')
    keyer.drops = True
    session.abort()
    assert not keyer.up and not session.ptt_unconfirmed
    assert terminal.status(session).startswith('RX  45.45/170')


def test_a_watchdog_drop_that_does_not_read_down_says_so(bench, monkeypatch):
    loop, keyer, session, out = bench()
    session.key()
    keyer.drops = False
    monkeypatch.setattr(session, '_pump', lambda: None)
    run(loop, session, 3, until=lambda: (session.watchdog(), session._halted is not None)[1])
    assert 'PTT DOWN NOT CONFIRMED -- unkey the rig by hand' in text(session, 'alarm')


def test_a_stalled_log_disk_holds_neither_the_over_nor_ctrl_c(bench, monkeypatch):
    import threading
    from hfmodem.station import qsolog
    loop, keyer, session, out = bench()
    gate = threading.Event()
    monkeypatch.setattr(qsolog.os, 'fsync', lambda fd: gate.wait(5))
    session._heard = 'CQ DE N0XYZ K'
    session.set_call()
    for ch in 'N0XYZ 599 001':
        session.type(ch)
    session.key()
    logging = threading.Thread(target=session.commit)
    logging.start()
    logging.join(0.5)
    try:
        assert session._lock.acquire(timeout=0.5)
        session._lock.release()
        run(loop, session, 0.5)
        assert session._tx_stream.starved_at is None
        session.abort()
        assert not keyer.up
    finally:
        gate.set()
        logging.join()
    session.flush()
    assert 'logged N0XYZ serial 001' in text(session, 'note')


def test_a_transcript_that_cannot_be_written_warns_and_the_session_goes_on(bench):
    loop, keyer, session, out = bench()

    class Full:
        def write(self, text):
            raise OSError(28, 'No space left on device')

        def close(self):
            pass

    session._transcript = Full()
    session.type('E')
    session.turn_over()
    assert run(loop, session, 4, until=lambda: session.state == 'rx')
    session.flush()
    assert 'No space left on device' in text(session, 'note')
    assert said(session, out).count('tx: E') == 1


def test_a_log_write_that_fails_keeps_the_record_on_screen(bench, monkeypatch):
    loop, keyer, session, out = bench()

    def refuse(record):
        raise PermissionError(13, 'Permission denied')

    monkeypatch.setattr(session.log, 'write', refuse)
    session._heard = 'CQ DE N0XYZ K'
    session.set_call()
    assert session.commit().call == 'N0XYZ'
    session.flush()
    assert 'NOT LOGGED N0XYZ' in text(session, 'note') and '<CALL:5>N0XYZ' in text(session, 'note')


def test_after_a_recovered_underrun_the_unkey_waits_for_the_last_sample_out(bench):
    loop, keyer, session, out = bench()
    for ch in 'RYRYRY':
        session.type(ch)
    session.key()
    run(loop, session, 0.5)
    run(loop, session, 0.3, step=False)
    assert session._tx_stream.starved_at is not None
    session.turn_over()
    assert run(loop, session, 6, until=lambda: session.state == 'rx')
    last_out = int(np.flatnonzero(loop.echo)[-1]) + 1
    off = keyer.events[-1][1]
    assert off >= last_out
    event = next(line for line in said(session, out).splitlines() if 'tx_event ' in line)
    assert json.loads(event.split('tx_event ', 1)[1])['end_index'] == last_out


def test_ctrl_c_keeps_every_character_the_radio_did_not_finish(bench):
    loop, keyer, session, out = bench()
    for ch in 'ABCD':
        session.type(ch)
    session.key()
    run(loop, session, 0.35)
    session.abort()
    assert text(session, 'tx') == '' and ''.join(session.typeahead) == 'ABCD'
    for ch in 'EFGH':
        session.type(ch)
    session.key()
    run(loop, session, 1.0)
    session.abort()
    sent = text(session, 'tx')
    assert sent and emitted(loop, until=keyer.events[-1][1]).startswith(sent)
    assert sent + ''.join(session.typeahead) == 'ABCDEFGH'


@pytest.mark.parametrize('rate', [rtty.AMATEUR, rtty.Rate.parse('75/170')], ids=str)
def test_a_character_shows_as_sent_only_once_it_is_on_the_air(bench, rate):
    loop, keyer, session, out = bench(rate=rate)
    for ch in 'ABCD':
        session.type(ch)
    session.key()
    assert run(loop, session, 2, until=lambda: text(session, 'tx') == 'A')
    session.abort()
    assert emitted(loop, until=keyer.events[-1][1], rate=rate).startswith('A')
    assert text(session, 'tx') + ''.join(session.typeahead) == 'ABCD'


def test_the_id_clock_starts_when_the_callsign_has_gone_out(bench):
    loop, keyer, session, out = bench()
    for ch in 'DE W9SSJ':
        session.type(ch)
    session.key()
    run(loop, session, 0.3)
    assert session.id_at is None
    assert run(loop, session, 2.5, until=lambda: session.id_at is not None)
    assert emitted(loop) == 'DE W9SSJ'


def test_a_capture_resync_with_no_index_gap_revokes_the_shift(bench):
    loop, keyer, session, out = bench(14)
    loop.place(0.5, 'THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG ' * 3)
    run(loop, session, 4)
    assert session.stream.shift_known
    seen = len(session.pieces)
    session.lane.resync(loop.sa.samples)
    run(loop, session, 2)
    dims = [dim for k, _, dim in list(session.pieces)[seen:] if k == 'rx']
    assert True in dims and all(dims[dims.index(True):]) and len(dims) > 10
    assert not session.stream.shift_known


def test_the_dial_follows_the_vfo_while_unkeyed_and_is_never_asked_while_keyed(bench):
    loop, keyer, session, out = bench()
    dials, asked = iter([14_081_000, 14_085_000]), []
    session._dial_reader = lambda admit: asked.append(session.state) or next(dials)
    session.refresh_dial()
    assert session.mark_rf_hz == 14_083_295
    session._heard = 'CQ DE N0XYZ K'
    session.set_call()
    session.type('E')
    session.key()
    session.refresh_dial()
    assert asked == ['rx'] and session.dial_hz == 14_081_000
    session.abort()
    session.refresh_dial()
    assert session.dial_hz == 14_085_000
    session.commit()
    session.flush()
    assert '<FREQ:9>14.087295' in session.log.path.read_text()
    lines = said(session, out)
    assert 'mark RF 14083295 Hz, drive' in lines
    assert '"mark_rf_hz":14083295' in lines


def test_the_dial_is_read_from_the_launchers_rigctld():
    import socket
    import threading
    server = socket.create_server(('127.0.0.1', 0))
    port = server.getsockname()[1]

    def answer():
        conn, _ = server.accept()
        with conn:
            assert conn.recv(16) == b'f\n'
            conn.sendall(b'14081500\n')

    threading.Thread(target=answer, daemon=True).start()
    try:
        assert qso.rigctld_dial(port=port) == 14_081_500
    finally:
        server.close()
    assert qso.rigctld_dial(port=port) is None


def test_a_dial_read_waiting_to_send_when_the_operator_keys_is_never_sent(bench):
    loop, keyer, session, out = bench()
    connected, go, sent = threading.Event(), threading.Event(), []

    def reader(admit=lambda: True):
        connected.set()
        go.wait(5)
        if admit():
            sent.append(session.state)
            return 14_085_000
        return None

    session._dial_reader = reader
    asking = threading.Thread(target=session.refresh_dial)
    asking.start()
    assert connected.wait(5)
    session.type('E')
    session.key()
    go.set()
    asking.join(5)
    assert sent == [] and session.dial_hz == 14_080_000


def _held_key_up(session):
    line, gate, stuck = Line(), threading.Event(), threading.Event()

    def log(said):
        if said.startswith('PTT ON'):
            stuck.set()
            gate.wait(5)

    session.keyer = armed(RealKeyer(line, log))
    session.type('E')
    keying = threading.Thread(target=session.key)
    keying.start()
    assert stuck.wait(5) and line.up
    return line, gate, keying


def test_the_watchdog_drops_a_key_up_that_never_finished(bench):
    loop, keyer, session, out = bench()
    line, gate, keying = _held_key_up(session)
    try:
        session.watchdog()
        assert line.up
        loop.t += qso.STARVED_S + 0.1
        session.watchdog()
        assert not line.up
    finally:
        gate.set()
        keying.join(5)
    assert not line.up and session.state == 'rx' and session._tx_stream is None
    assert 'key-up still unfinished' in text(session, 'alarm')
    assert ''.join(session.typeahead) == 'E'


def test_ctrl_c_drops_a_key_up_that_never_finished(bench):
    loop, keyer, session, out = bench()
    line, gate, keying = _held_key_up(session)
    stopping = threading.Thread(target=session.abort)
    try:
        stopping.start()
        stopping.join(0.5)
        assert not line.up
    finally:
        gate.set()
        keying.join(5)
        stopping.join(5)
    assert not line.up and session.state == 'rx'


def test_a_key_up_that_cannot_arm_the_card_drops_ptt_and_says_so(bench, monkeypatch):
    loop, keyer, session, out = bench()

    def refuse(*args):
        raise card.AudioError('the card went away')

    monkeypatch.setattr(loop.sa, 'arm_stream', refuse)
    session.type('E')
    session.key()
    assert not keyer.up and session.state == 'rx' and ''.join(session.typeahead) == 'E'
    assert 'key-up failed (the card went away); PTT dropped' in text(session, 'alarm')
    session.close()
    assert not keyer.up


def test_a_stalled_stdout_holds_neither_keying_nor_the_recording(bench, tmp_path):
    import wave
    out = Stalled()
    loop, keyer, session, _ = bench(out=out)
    line = Line()
    session.keyer = armed(RealKeyer(line, session.say, session.alarm))
    try:
        for ch in 'TEST DE W9SSJ':
            session.type(ch)
        session.turn_over()
        assert run(loop, session, 6, until=lambda: session.state == 'rx')
        assert not line.up
        session._disk.flush(5)
        with wave.open(str(tmp_path / 'stream.wav')) as tape:
            assert tape.getnframes() > 0
    finally:
        out.gate.set()
        session.close()
        session.finish()


def test_a_stalled_disk_drops_recording_with_a_note_and_never_grows(bench, monkeypatch):
    monkeypatch.setattr(qso, 'DISK_JOBS', 20)
    loop, keyer, session, out = bench()
    gate = threading.Event()
    record = session._record
    monkeypatch.setattr(session, '_record', lambda samples: (gate.wait(10), record(samples)))
    try:
        run(loop, session, 2)
        assert session._disk._jobs.qsize() <= 20
        session.type('E')
        session.turn_over()
        assert run(loop, session, 4, until=lambda: session.state == 'rx')
    finally:
        gate.set()
    session._disk.flush(5)
    run(loop, session, 0.1)
    assert 'recording gap: ' in text(session, 'note') and 'of stream.wav from card' in text(session, 'note')


def test_a_contact_is_logged_while_stdout_is_stalled(bench, tmp_path):
    out = Stalled()
    loop, keyer, session, _ = bench(out=out)
    try:
        session._heard = 'CQ DE N0XYZ K'
        session.set_call()
        session.commit()
        session.flush(1)
        assert '<CALL:5>N0XYZ' in (tmp_path / 'log.adi').read_text()
    finally:
        out.gate.set()


def test_a_contact_that_cannot_be_written_is_held_set_aside_and_keeps_its_serial(bench, tmp_path,
                                                                                  monkeypatch):
    loop, keyer, session, out = bench(contest='TEST')
    monkeypatch.setattr(session.log, 'write',
                        lambda record: (_ for _ in ()).throw(PermissionError(13, 'Permission denied')))
    session._heard = 'CQ DE N0XYZ K'
    session.set_call()
    assert session.commit().serial == 1
    session.flush(5)
    assert 'NOT LOGGED N0XYZ serial 001 yet' in text(session, 'note')
    session.close()
    said = session.finish()
    assert any('NOT LOGGED' in line and '<CALL:5>N0XYZ' in line for line in said)
    path = tmp_path / 'log.adi'
    assert QsoLog(path, 'W9SSJ', contest='TEST').serial == 2
    later = qso.Session(loop.sa, keyer, QsoLog(path, 'W9SSJ', contest='TEST'), mycall='W9SSJ',
                        out=io.StringIO(), shout=lambda line: None)
    later.flush(5)
    assert '<STX:1>1' in path.read_text() and not later.log.held.exists()
    assert later.log.serial == 2
    later.close()
    assert later.finish() == []


def test_an_alarm_from_handing_the_line_back_reaches_the_operator(bench):
    loop, keyer, session, out = bench()
    shouted = []
    session._shout = shouted.append
    session.close()
    session.alarm('*** keying line handed back NOT CONFIRMED LOW ***')
    assert any('handed back NOT CONFIRMED' in line for line in shouted)
    assert 'handed back NOT CONFIRMED' in said(session, out)
    session.finish()
    session.alarm('*** later still ***')
    assert any('later still' in line for line in shouted)


def test_the_launchers_warning_stays_on_the_status_line(bench):
    from hfmodem.lanner import terminal
    loop, keyer, session, out = bench(warnings=['rig mode LSB, not PKTUSB'])
    assert 'rig mode LSB, not PKTUSB' in text(session, 'alarm')
    session.type('E')
    session.turn_over()
    run(loop, session, 4, until=lambda: session.state == 'rx')
    assert terminal.status(session).startswith('RX  !! rig mode LSB, not PKTUSB')


def test_a_signal_that_lands_during_key_up_leaves_ptt_down_and_rx(bench):
    loop, keyer, session, out = bench()
    line = Line()

    class Interrupted(RealKeyer):
        def key(self, on, note=''):
            up = super().key(on, note)
            if on:
                raise SystemExit(143)
            return up

    session.keyer = armed(Interrupted(line, session.say))
    session.type('E')
    with pytest.raises(SystemExit):
        session.key()
    assert not line.up and session.state == 'rx' and ''.join(session.typeahead) == 'E'
    assert 'key-up failed (interrupted); PTT dropped' in text(session, 'alarm')
    session.close()
    assert not line.up


def test_a_session_thread_that_dies_in_rx_says_so_and_keying_is_declined(bench, monkeypatch):
    loop, keyer, session, out = bench()
    monkeypatch.setattr(session.audio, 'breathe', lambda: None)
    monkeypatch.setattr(session, 'step', lambda: 1 / 0)
    session._running = True
    with pytest.raises(ZeroDivisionError):
        session._work()
    assert 'the session thread failed (ZeroDivisionError' in text(session, 'alarm')
    session.type('E')
    session.key()
    assert keyer.events == [] and 'session thread failed' in text(session, 'note')
    assert 'session thread failed' in session.alarm_text


def test_a_callsign_split_across_two_overs_is_not_an_id(bench):
    loop, keyer, session, out = bench()
    session._sent_tail = 'DE W9S'
    for ch in 'SJ':
        session.type(ch)
    session.turn_over()
    assert run(loop, session, 4, until=lambda: session.state == 'rx')
    assert text(session, 'tx') == 'SJ\n' and session.id_at is None


def test_each_set_aside_contact_leaves_the_set_aside_file_as_it_is_written(bench, tmp_path,
                                                                          monkeypatch):
    loop, keyer, session, out = bench()
    path = tmp_path / 'log.adi'
    first = '<CALL:4>K1AA<CONTEST_ID:4>TEST<STX:1>1<EOR>\n'
    second = '<CALL:4>K1BB<CONTEST_ID:4>TEST<STX:1>2<EOR>\n'
    log = QsoLog(path, 'W9SSJ', contest='TEST')
    assert log.keep([first, second])
    write = log.write
    monkeypatch.setattr(log, 'write', lambda record: write(record) if record == first
                        else (_ for _ in ()).throw(OSError(5, 'Input/output error')))
    later = qso.Session(loop.sa, keyer, log, mycall='W9SSJ', out=io.StringIO(),
                        shout=lambda line: None)
    later.flush(5)
    assert log.kept() == [second] and 'K1AA' in path.read_text()
    later.close()
    later.finish()


class Flaky(Line):
    """A line that rises without reading back, then will not read down for a while."""

    def __init__(self, stuck):
        super().__init__()
        self.stuck = stuck

    def drive(self, on, timeout=None, *, confirm_silence=True):
        if on:
            self.up = True
            return Attempt('failed', state=None)
        if self.stuck:
            self.stuck -= 1
            return Attempt('failed', state=True, state_read=True)
        return super().drive(False)


def test_a_key_up_whose_rollback_will_not_read_down_stays_unconfirmed_until_it_does(bench):
    loop, keyer, session, out = bench()
    line = Flaky(stuck=2)
    session.keyer = armed(RealKeyer(line, session.say, session.alarm))
    session.type('E')
    session.key()
    assert line.up and session.state == 'rx' and session.ptt_unconfirmed
    assert 'PTT DOWN NOT CONFIRMED' in text(session, 'alarm')
    session.abort()
    assert not line.up and not session.ptt_unconfirmed
    session.key()
    assert not line.up and 'restart the terminal' in text(session, 'note')


def test_close_retries_a_line_that_did_not_read_down(bench):
    loop, keyer, session, out = bench()
    line = Flaky(stuck=2)
    session.keyer = armed(RealKeyer(line, session.say, session.alarm))
    session.type('E')
    session.key()
    assert line.up
    session.close()
    assert not line.up


def _halted_before_a_is_out(bench):
    loop, keyer, session, out = bench()
    for ch in 'ABCD':
        session.type(ch)
    session.key()
    queued = lambda: (session._flight and session._flight[0][1] is not None
                      and session._flight[0][1] <= session._tx_stream.written)
    assert run(loop, session, 2, until=queued)
    assert text(session, 'tx') == ''
    return loop, keyer, session


def test_a_second_halt_never_moves_the_cut_past_the_first(bench):
    loop, keyer, session = _halted_before_a_is_out(bench)
    session.halt('the watchdog')
    first = keyer.events[-1][1]
    run(loop, session, 0.5, step=False)
    session.halt('a signal')
    session.close()
    assert not emitted(loop, until=first).startswith('A')
    assert text(session, 'tx') == '' and ''.join(session.typeahead) == 'ABCD'


def test_a_halt_during_the_pump_still_settles_at_the_drop(bench, monkeypatch):
    loop, keyer, session = _halted_before_a_is_out(bench)
    pump = session._pump

    def halted_mid_pump():
        session.halt('the watchdog')
        for _ in range(200):
            loop.block()
        pump()

    monkeypatch.setattr(session, '_pump', halted_mid_pump)
    session.step()
    monkeypatch.setattr(session, '_pump', pump)
    session.step()
    assert text(session, 'tx') == '' and ''.join(session.typeahead) == 'ABCD'


def test_a_record_whose_sync_failed_is_not_appended_again(bench, tmp_path, monkeypatch):
    from hfmodem.station import qsolog
    loop, keyer, session, out = bench()
    real, failed, synced = qsolog.os.fsync, [], []

    def sync_once_fails(fd):
        if not failed:
            failed.append(fd)
            raise OSError(5, 'Input/output error')
        synced.append(fd)
        real(fd)

    monkeypatch.setattr(qsolog.os, 'fsync', sync_once_fails)
    session._heard = 'CQ DE N0XYZ K'
    session.set_call()
    session.commit()
    session.flush(5)
    assert 'NOT LOGGED N0XYZ' in text(session, 'note')
    session.flush(5)
    assert (tmp_path / 'log.adi').read_text().count('<CALL:5>N0XYZ') == 1
    assert synced and 'logged N0XYZ' in text(session, 'note')


def test_a_set_aside_record_already_in_the_log_is_not_written_twice(bench, tmp_path, monkeypatch):
    loop, keyer, session, out = bench()
    path = tmp_path / 'log.adi'
    record = '<CALL:4>K1AA<CONTEST_ID:4>TEST<STX:1>1<EOR>\n'
    log = QsoLog(path, 'W9SSJ', contest='TEST')
    log.write(record)
    assert log.keep([record])
    monkeypatch.setattr(log, 'keep', lambda records: False)
    later = qso.Session(loop.sa, keyer, log, mycall='W9SSJ', out=io.StringIO(),
                        shout=lambda line: None)
    later.flush(5)
    assert path.read_text().count('K1AA') == 1
    assert f'could not update {log.held}' in text(later, 'note')


def test_setting_contacts_aside_keeps_to_the_shutdown_deadline(bench, monkeypatch):
    loop, keyer, session, out = bench()
    monkeypatch.undo()
    gate = threading.Event()
    monkeypatch.setattr(session.log, 'write', lambda record: (_ for _ in ()).throw(OSError(28, 'full')))
    monkeypatch.setattr(session.log, 'keep', lambda records: gate.wait(10))
    session._heard = 'CQ DE N0XYZ K'
    session.set_call()
    session.commit()
    session.flush(5)
    session.close()
    began = time.monotonic()
    try:
        said = session.finish(timeout=0.5)
        assert time.monotonic() - began < 3
    finally:
        gate.set()
    assert any('NOT LOGGED' in line and 'N0XYZ' in line for line in said)
    assert any('NOT in' in line for line in said)


def test_a_console_that_stops_reading_halts_the_over(bench):
    class Gone(io.StringIO):
        def write(self, text):
            raise BrokenPipeError(32, 'Broken pipe')

    loop, keyer, session, out = bench(out=Gone())
    session.type('E')
    session.key()
    session._pipe.flush(5)
    assert not keyer.up
    assert 'the console stopped reading the event log' in text(session, 'alarm')


def test_a_log_that_cannot_be_read_on_retry_keeps_the_record_held(bench, tmp_path, monkeypatch):
    from hfmodem.station import qsolog
    loop, keyer, session, out = bench()
    monkeypatch.setattr(qsolog.os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError(5, 'EIO')))
    session._heard = 'CQ DE N0XYZ K'
    session.set_call()
    session.commit()
    session.flush(5)
    read = qsolog.Path.read_text

    def unreadable(path, *args, **kw):
        if path == session.log.path:
            raise PermissionError(13, 'Permission denied')
        return read(path, *args, **kw)

    monkeypatch.setattr(qsolog.Path, 'read_text', unreadable)
    session.flush(5)
    monkeypatch.setattr(qsolog.Path, 'read_text', read)
    assert (tmp_path / 'log.adi').read_text().count('<CALL:5>N0XYZ') == 1
    assert len(session._contacts._held) == 1 and 'logged N0XYZ' not in text(session, 'note')


def test_a_present_record_whose_sync_still_fails_stays_held(bench, tmp_path, monkeypatch):
    from hfmodem.station import qsolog
    loop, keyer, session, out = bench()
    monkeypatch.setattr(qsolog.os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError(5, 'EIO')))
    session._heard = 'CQ DE N0XYZ K'
    session.set_call()
    session.commit()
    session.flush(5)
    session.flush(5)
    assert len(session._contacts._held) == 1 and 'logged N0XYZ' not in text(session, 'note')


def test_a_console_that_vanishes_during_idle_diddle_halts_the_over_and_declines_keying(bench):
    import os
    heard, owner = os.pipe()
    loop, keyer, session, out = bench(owner=heard)
    session.key()
    run(loop, session, 0.2, until=lambda: (session.watchdog(), not keyer.up)[1])
    assert keyer.up
    os.close(owner)
    run(loop, session, 0.3, until=lambda: (session.watchdog(), not keyer.up)[1])
    assert not keyer.up
    assert 'the console that owns this session is gone' in text(session, 'alarm')
    run(loop, session, 0.1)
    session.key()
    assert not keyer.up and session.state == 'rx'


def test_a_console_that_exited_unreaped_is_gone(bench):
    import os
    import subprocess
    heard, owner = os.pipe()
    console = subprocess.Popen(['true'], pass_fds=(owner,))
    os.close(owner)
    loop, keyer, session, out = bench(owner=heard)
    time.sleep(0.5)
    session.type('E')
    session.key()
    assert keyer.events == [] and 'the console that owns this session is gone' in text(session, 'note')
    console.wait()


def test_an_orphaned_session_declines_keying(bench):
    import os
    loop, keyer, session, out = bench(parent=os.getppid() + 1)
    session.watchdog()
    session.type('E')
    session.key()
    assert keyer.events == [] and 'the launcher that started this session is gone' in text(session, 'alarm')


def test_a_screen_failure_halts_first_and_words_its_report_from_the_verdict(bench):
    loop, keyer, session, out = bench()
    session.key()
    assert session.screen_failed(RuntimeError('get_wch failed')) == 1
    assert not keyer.up and session.alarm_text.endswith('unkeyed and left')


def test_a_screen_failure_whose_unkey_fails_says_so(bench):
    loop, keyer, session, out = bench()
    session.key()
    keyer.drops = False
    session.screen_failed(RuntimeError('get_wch failed'))
    assert keyer.up and session.alarm_text.endswith('PTT DOWN NOT CONFIRMED -- unkey the rig by hand')
    assert 'unkeyed and left' not in text(session, 'alarm')

@pytest.fixture
def whole(tmp_path, monkeypatch):
    """`qso.main` in this process on the fake card and line, with the screen replaced
    by `run`; returns everything said and done, in order."""
    import importlib
    import signal
    import sys
    from hfmodem.lanner import terminal
    monkeypatch.setenv('FAKE_LOG', str(tmp_path / 'fake.log'))
    monkeypatch.delenv('HFMODEM_STATION', raising=False)
    monkeypatch.delenv(qso.OWNER_ENV, raising=False)
    from hfmodem.core import devices, ptt
    for owner, name in ((ptt, 'LineKeyer'), (devices, 'find_device'), (qso, 'rigctld_dial'),
                        (terminal, 'draw')):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.setitem(sys.modules, 'sounddevice', sys.modules.get('sounddevice'))
    fakes = importlib.reload(importlib.import_module('hfmodem.tests.lanner.qso_on_fakes'))
    order = []

    class Line(fakes.Keyer):
        def key(self, on, note=''):
            order.append('key up' if on else 'key down')
            return super().key(on, note)

        def hand_back(self):
            order.append('hand back')

    class Said(io.StringIO):
        def write(self, text):
            if text.strip():
                order.append(f'said {text.strip()}')
            return len(text)

    monkeypatch.setattr(fakes, 'Keyer', Line)
    fakeclaim.install(monkeypatch, order)
    monkeypatch.setattr(terminal, 'claim_tty', io.StringIO)
    monkeypatch.setattr(sys, 'stderr', Said())
    monkeypatch.setattr(sys, '__stdout__', Said())
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}

    def go(screen, *extra):
        monkeypatch.setattr(terminal, 'run', screen)
        try:
            code = fakes.main(['--audio-in', 'in', '--audio-out', 'out', '--ptt-device', 'line',
                               '--rig', 'ftdx10', '--adif', str(tmp_path / 'log.adi'), *extra])
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
        return code, order
    return go


def test_main_halts_on_a_screen_failure_and_speaks_only_after_the_hand_back(whole):
    def broken(session):
        session.key()
        raise RuntimeError('get_wch failed')

    code, order = whole(broken)
    assert code == 1
    assert order[0] == 'inherited' and order.index('opened') < order.index('key up')
    assert order[-1] == 'failed'
    assert order.index('key down') < order.index('hand back')
    spoken = [i for i, done in enumerate(order) if done.startswith('said')]
    assert spoken and min(spoken) > order.index('hand back')
    assert any('the screen failed (RuntimeError: get_wch failed); unkeyed and left' in done
               for done in order)


def test_a_launcher_lost_while_the_card_opens_is_caught_before_any_key(whole):
    import os
    tried = []

    def keying(session):
        session.type('E')
        session.key()
        tried.append(session.state)
        return 0

    code, order = whole(keying, '--launcher', str(os.getppid() + 7))
    assert tried == ['rx'] and 'key up' not in order


@pytest.mark.parametrize('keyed', [False, True])
def test_sigint_unkeys_and_exits_in_rx_or_tx(whole, keyed):
    import signal
    order_seen = []

    def interrupted(session):
        if keyed:
            session.key()
        order_seen.append(session.keyer)
        signal.raise_signal(signal.SIGINT)
        pytest.fail('SIGINT returned to the session')

    with pytest.raises(SystemExit) as stopped:
        whole(interrupted)
    assert stopped.value.code == 130
    assert not order_seen[0].up


@pytest.mark.parametrize('keyed', [False, True])
def test_repeated_sigint_during_cleanup_still_hands_back_once(whole, monkeypatch, keyed):
    import signal
    events = []
    real_signal = signal.signal
    repeated = False

    def install(sig, handler):
        nonlocal repeated
        previous = real_signal(sig, handler)
        # A group INT also reaches the launcher's forwarding trap. Deliver its
        # second INT while cleanup changes TERM, before it changes INT.
        if sig == signal.SIGTERM and getattr(handler, '__name__', '') == 'halt_only' and not repeated:
            repeated = True
            signal.raise_signal(signal.SIGINT)
        return previous

    monkeypatch.setattr(signal, 'signal', install)

    def interrupted(session):
        key, hand_back = session.keyer.key, session.keyer.hand_back

        def recorded_key(on, note=''):
            events.append('key up' if on else 'key down')
            return key(on, note)

        def recorded_hand_back():
            events.append('hand back')
            return hand_back()

        monkeypatch.setattr(session.keyer, 'key', recorded_key)
        monkeypatch.setattr(session.keyer, 'hand_back', recorded_hand_back)
        if keyed:
            session.key()
        signal.raise_signal(signal.SIGINT)
        pytest.fail('SIGINT returned to the session')

    with pytest.raises(SystemExit) as stopped:
        whole(interrupted)
    assert stopped.value.code == 130
    assert repeated
    assert events.count('key down') >= 2
    assert events.count('hand back') == 1
    assert events.index('key down') < events.index('hand back')
