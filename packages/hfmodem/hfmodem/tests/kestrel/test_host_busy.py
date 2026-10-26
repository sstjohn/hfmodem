# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The host's BUSY against stock's: when it rises, how long it holds, what does not count."""
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.kestrel.host.busy_gate import BandGate
from hfmodem.kestrel.host.vara_modem import BUSY_ON_S, VaraHostModem
from hfmodem.kestrel.monitor import FS, SEG_FRAME, SEG_HANG
from hfmodem.kestrel.rx.cq import build_frame
from hfmodem.kestrel.tx.beacon import synth_beacon
from hfmodem.kestrel.tx.cq import synth_frame
from hfmodem.tests.evidence import RECORD, WORKING, require_input
from .test_compression import IO

FIXTURES = Path(__file__).with_name('fixtures')

#: AudioHostRuntime.step's cadence, which is how often the gate's state is reported.
STEP = .05
#: How long our gate stays open after a signal ends.
GATE_TAIL_S = SEG_HANG * SEG_FRAME / FS
SESSIONS = RECORD / 'operator-data' / 'working' / 'vara-chat' / 'sessions'
TWENTY = '20261023T155947666857Z'
#: The end of the audited 20 m session; the operator went on in its directory after it.
AUDITED = '2026-10-23T16:21:20'


def replay(raw, end):
    """The BUSY edges our host reports for gate `raw` [(t, open)], read every STEP until `end`."""
    now = [raw[0][0]]
    modem = VaraHostModem(IO(), bandwidth='500', clock=lambda: now[0])
    edges = []
    modem.start(SimpleNamespace(modem_busy=lambda on: edges.append((now[0], on))))
    i, active, k = 0, False, 0
    while now[0] <= end:
        while i < len(raw) and raw[i][0] <= now[0]:
            active = raw[i][1]
            i += 1
        modem.report_channel_activity(active)
        k += 1
        now[0] = raw[0][0] + k * STEP
    return edges


def session(name, kind):
    path = require_input(SESSIONS / name / kind)
    return [json.loads(line) for line in path.read_text().splitlines()]


def utc(record):
    return datetime.fromisoformat(record['utc']).timestamp()


def gate(name):
    """The 10-23 host reported its gate raw, so its BUSY lines are the gate's edges."""
    return [(utc(r), r['raw'] == 'BUSY ON') for r in session(name, 'host.jsonl')
            if r.get('raw') in ('BUSY ON', 'BUSY OFF') and r['utc'] < AUDITED]


def stock_cqs():
    """Stock A's CQFRAMEs and idle stock B's BUSY around each, from the chat-p0 bench."""
    for path in sorted(require_input(WORKING / 'vara' / 'oracle' / 'chat-p0' / 'item4').glob('*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rx = [(r['t'], r['who'], r['text']) for r in rows if r['kind'] == 'cmd_rx' and r['t'] > 5]
        keyed = None
        for t, who, text in rx:
            if who == 'A' and text == 'PTT ON':
                keyed = t
            elif who == 'A' and text == 'PTT OFF' and keyed is not None:
                on = next(s for s, w, x in rx if w == 'B' and x == 'BUSY ON' and s >= keyed)
                off = next(s for s, w, x in rx if w == 'B' and x == 'BUSY OFF' and s > t)
                yield keyed, t, on, off
                keyed = None


def test_busy_rises_and_clears_where_stocks_does_around_a_cq():
    pairs = list(stock_cqs())
    assert len(pairs) == 14
    for keyed, unkeyed, on, off in pairs:
        # Stock: ON 0.02-0.21 s after the far end keys, OFF 1.48-1.75 s after it unkeys.
        assert 0 < on - keyed < .21 and 1.48 <= off - unkeyed < 1.76
        edges = replay([(keyed, True), (unkeyed + GATE_TAIL_S, False)], unkeyed + 3)
        (rose, up), (fell, down) = edges[-2:]
        assert up and not down
        assert rose - keyed <= BUSY_ON_S + STEP and 1.48 <= fell - unkeyed < 1.76


def test_a_static_crash_is_not_busy_and_a_frame_is():
    # 0.16 s is the median vetoing burst on 20 m; 0.45 s is VARA's shortest frame.
    assert [on for _, on in replay([(0, True), (.16, False)], 5)] == [False]
    assert [on for _, on in replay([(0, True), (.45, False)], 5)] == [True, False]


def test_the_retry_flicker_on_20_m_reads_as_one_busy_spell():
    # 16:01:32-52Z: the gate toggled every 0.2-0.7 s and the client re-sent the CQ 12 times.
    raw = [(t, on) for t, on in gate(TWENTY)]
    start = datetime.fromisoformat('2026-10-23T16:01:32.425+00:00').timestamp()
    end = start + 20
    edges = replay(raw, end)
    assert [on for t, on in edges if t < start][-1]
    assert not [on for t, on in edges if start <= t <= end]


def test_blips_no_longer_veto_the_20_m_sniffs_or_flap():
    raw = gate(TWENTY)
    edges = replay(raw, raw[-1][0])
    sniffs, opened = [], None
    for r in session(TWENTY, 'events.jsonl'):
        if r.get('kind') == 'state' and r['utc'] < AUDITED:
            if opened is not None:
                sniffs.append((opened, utc(r)))
                opened = None
            if r['state'] == 'sniffing-slot':
                opened = utc(r)
    vetoed = lambda reports: sum(any(a < t < b and on for t, on in reports) for a, b in sniffs)
    assert len(sniffs) == 10 and vetoed(raw) == 6
    assert vetoed(edges) == 2
    assert sum(on for _, on in raw) == 304
    assert sum(on for _, on in edges) == 91


def test_a_cq_is_dropped_on_the_reported_busy_not_on_a_crash():
    from .test_cq import session_mock
    session, commands, modem = session_mock()
    modem._transport.receiving = True
    session._dispatch('CQFRAME W9SSJ 500')
    modem.poll()
    assert commands[:3] == ['OK', 'PENDING', 'PTT ON']
    session, commands, modem = session_mock()
    modem._channel_busy = True
    session._dispatch('CQFRAME W9SSJ 500')
    modem.poll()
    assert commands == ['OK']


@pytest.mark.parametrize('read,named', [('', None), ('UNKNOWN', None), ('KA8CPF', 'KA8CPF')])
def test_an_attempt_that_ends_unlinked_keeps_the_caller_it_read(read, named):
    from .test_cq import session_mock
    session, commands, modem = session_mock()
    modem.station.io.pending()
    modem.station.caller = read
    modem.abort()
    modem.poll()
    assert commands[-1] == 'CANCELPENDING' and modem.attempt_caller == named


def test_each_busy_edge_logs_what_the_gate_read():
    io = IO()
    io.lines, io.log, io.busy_detail = [], lambda m: io.lines.append(m), lambda: 'floor -40.0 dBFS'
    now = [0.0]
    modem = VaraHostModem(io, bandwidth='500', clock=lambda: now[0])
    modem.start(SimpleNamespace(modem_busy=lambda on: None))
    for t, active in ((0, False), (1, True), (1.5, True), (2, False), (4, False)):
        now[0] = t
        modem.report_channel_activity(active)
    assert io.lines == ['BUSY OFF: floor -40.0 dBFS', 'BUSY ON: floor -40.0 dBFS',
                        'BUSY OFF: floor -40.0 dBFS']


# -- the BW500 band: what stock's BUSY hears (bench 3, 10-23) -------------------------

def receiver(rng, seconds, rms=.01):
    """Band noise as the FTDX10's 3 kHz DATA filter delivers it."""
    n = int(seconds * FS)
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / FS)
    spec[(f < 100) | (f > 3000)] = 0
    x = np.fft.irfft(spec, n)
    return x * rms / x.std()


class Reported:
    """The BUSY a modem reports for the band gate fed in 50 ms chunks, on the audio's clock."""

    def __init__(self):
        self.gate, self.now, self.busy = BandGate(), [0.0], []
        self.modem = VaraHostModem(IO(), bandwidth='500', clock=lambda: self.now[0])
        self.modem.start(SimpleNamespace(modem_busy=self.busy.append))

    def feed(self, x):
        for i in range(0, len(x), 2400):
            active = self.gate.push(x[i:i + 2400])
            self.now[0] += len(x[i:i + 2400]) / FS
            self.modem.report_channel_activity(active)
            yield bool(self.modem._channel_busy)

    def skip(self, seconds):
        self.gate.restart()
        self.now[0] += seconds


def opens(stimulus, *, step_db=0.0):
    """Whether BUSY rises on `stimulus` over 10 s of the receiver's floor, the floor itself
    raised `step_db` for the stimulus's length."""
    rng = np.random.default_rng(23)
    host = Reported()
    list(host.feed(receiver(rng, 10)))
    x = receiver(rng, len(stimulus) / FS) * 10 ** (step_db / 20) + stimulus
    return any(host.feed(x))


def offset(x, hz):
    return np.real(hilbert(x) * np.exp(2j * np.pi * hz * np.arange(len(x)) / FS))


FRAMES = {'cq': synth_frame(build_frame('W9SSJ-9', 500), amplitude=.1),
          'beacon': synth_beacon('20OEN63BCW9SSJ', amplitude=.1)}


# Stock's own rows; the beacon 750 Hz up is left out because stock rose on it once, for
# its 1.46 s minimum, 4.2 s into the frame.
@pytest.mark.parametrize('kind,hz,busy', [(kind, hz, abs(hz) < 750) for kind in FRAMES
                                          for hz in (0, 400, -400, 750, -750) if (kind, hz) != ('beacon', 750)])
def test_a_frame_in_the_passband_is_busy_and_one_in_a_slot_is_not(kind, hz, busy):
    assert opens(offset(FRAMES[kind], hz)) == busy


def test_band_noise_steps_and_a_carrier_off_the_passband_are_not_busy():
    for db in (3, 6, 10):
        assert not opens(np.zeros(8 * FS), step_db=db)
    t = np.arange(8 * FS) / FS
    assert opens(.05 * np.sin(2 * np.pi * 1500 * t))
    assert not opens(.05 * np.sin(2 * np.pi * 2250 * t))


def test_the_40_m_session_is_busy_where_stock_was_busy_on_it():
    # The live host's broadband gate was BUSY for 39 % of this session, stock for 22 % of
    # the same audio. Our CQ is on the tape but never reached the host's gate.
    stock = json.loads((FIXTURES / 'stock-busy-live40-1023.json').read_text())
    from hfmodem.core.wav import read
    rate = 48000
    x = read(require_input(RECORD / stock['source']), target_fs=rate)
    end = stock['seconds'][1]
    x = x[:end * rate]
    tx = [int(s * rate) for s in stock['our_cq_s']]
    host, ours = Reported(), np.zeros(end * 10, bool)
    for i in range(0, len(x), 2400):
        if tx[0] <= i < tx[1] + 4800:
            host.skip(2400 / rate)
        else:
            ours[i * 10 // rate:(i + 2400) * 10 // rate] = next(host.feed(x[i:i + 2400]))
    theirs = np.zeros(end * 10, bool)
    for a, b in stock['busy']:
        theirs[int(a * 10):int(b * 10)] = True
    keep = np.ones(end * 10, bool)
    keep[int(stock['our_cq_s'][0] * 10):int(stock['our_cq_s'][1] * 10) + 10] = False
    assert ours[keep].mean() < .25 and np.mean(ours[keep] == theirs[keep]) > .9
