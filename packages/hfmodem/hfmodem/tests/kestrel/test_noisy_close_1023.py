# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The BW500 close under noise: bench 3 b899, 2026-10-23.

Kestrel CQ'd, stock answered with 0.55 and 0.9 RMS of band noise on kestrel's
input (-9.4 and -11.8 dB in 2.9 kHz), the overs went both ways at levels 2-4,
and the close failed: stock's frame-1 acknowledgement of kestrel's last over is an
11-symbol two-tone burst, and at those SNRs its pairs are never all exact. The
recordings are kestrel's input cable, cut around the contact (rf-corpus
varac-noisy-close-1023; the fixture's provenance.json has the offsets).
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.core.wav import source
from hfmodem.tests import evidence
from hfmodem.kestrel.monitor import Segmenter
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M

HERE = Path(__file__).with_name('fixtures') / 'varac-noisy-close-1023'
CUT = evidence.CORPUS / 'varac-noisy-close-1023'
pytestmark = pytest.mark.skipif(not CUT.is_dir(),
                                reason=f'recording not present ({evidence.CORPUS})')
STOCK, KESTREL = 'WI9YZA', 'W9SSJ'
BLOCK = 1024
CUT_START = 30.0                   # wav second the cuts begin at


class Clock:
    t = 1000.

    @classmethod
    def monotonic(cls):
        return cls.t


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(A, 'time', Clock)


class IO(A.VaraIO):
    def __init__(self):
        self.sent, self.host, self.logs = [], [], []
        self.receiving = False

    def tx(self, samples):
        self.sent.append((Clock.t, np.asarray(samples, float)))

    def data(self, payload):
        self.host.append(bytes(payload))

    def log(self, msg):
        self.logs.append((Clock.t, msg))


def audio(name):
    prov = json.loads((HERE / 'provenance.json').read_text())[name]
    rate, x = wavfile.read(source(CUT / prov['file']))
    assert rate == M.FS and x.dtype == np.int16
    return x.astype(float) / 32768


def band_noise(n, rms, seed=20260821, cutoff=2900.0):
    """The bench injector's noise: flat Gaussian to ``cutoff``, clipped at 0.98."""
    rng = np.random.default_rng(seed)
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    spec[np.fft.rfftfreq(n, 1.0 / M.FS) > cutoff] = 0.0
    x = np.fft.irfft(spec, n)
    return np.clip(x / np.sqrt(np.mean(x * x)) * rms, -0.98, 0.98)


def opening(hs):
    hs.send(b'de W9SSJ <R-12>\r\r')
    hs.send(b'<NAME:Saul>\r<QTH:Madison>\r<LOC:EN63bc>\r\r\r\r')


def replay(hs, x, until, *, connect_at=None, host=None, guard=.1, ticks=True):
    """The transport's order, nothing under our own keying or ``guard`` behind
    it, the driver's ticks between blocks, and ``host`` driving the session as
    gate5's client did. Wav time is the clock."""
    io, seg, deaf, tick_ok, linked = hs.io, Segmenter(), 0., 0., False
    for at in range(0, min(len(x), int((until - CUT_START) * M.FS)), BLOCK):
        Clock.t = CUT_START + (at + BLOCK) / M.FS
        if connect_at is not None and Clock.t >= connect_at:
            connect_at = None
            assert hs.originate(STOCK, KESTREL)
        if Clock.t < deaf:
            seg = Segmenter()
            continue
        chunk = x[at:at + BLOCK]
        sent = len(io.sent)
        io.receiving = bool(seg.inb)
        hs.on_rx_stream(chunk)
        for _, burst in seg.push(chunk):
            hs.on_rx_audio(np.r_[burst, np.zeros(M.FS // 4)])
        io.receiving = bool(seg.inb)
        if host is not None:
            host(hs)
        if ticks and hs.state == A.VaraState.CONNECTED and not io.receiving and Clock.t >= tick_ok:
            query, retry, poll = (hs.intermediate_query_due_in(), hs.data_retry_due_in(),
                                  hs.idle_poll_due_in())
            if query is not None and query <= 0:
                hs.query_intermediate_answer()
            elif retry is not None and retry <= 0:
                hs.retry_data_over()
            elif poll is not None and poll <= 0:
                hs.idle_keepalive()
        if len(io.sent) > sent:
            deaf = Clock.t + len(io.sent[-1][1]) / M.FS + guard
            tick_ok = deaf + 1.0
        linked |= hs.state == A.VaraState.CONNECTED
        if linked and hs.state == A.VaraState.DISCONNECTED:
            break


class Answerer:
    """gate5's K client: the opening at CONNECTED, its text 2 s on, and 73<DISC>
    the moment stock's text is read; then DISCONNECT once the drain empties."""

    def __init__(self):
        self.connected_at = self.drained_at = None
        self.text = self.farewell = False

    def __call__(self, hs):
        if hs.state == A.VaraState.CONNECTED and self.connected_at is None:
            self.connected_at = Clock.t
            opening(hs)
        if self.connected_at and not self.text and Clock.t >= self.connected_at + 2.0:
            self.text = True
            hs.send(b'text from K on the slot\r')
        if not self.farewell and any(b'text from S' in h for h in hs.io.host):
            self.farewell = True
            hs.send(b'73<DISC>\r\r')
        if self.farewell and self.drained_at is None and hs.payload_ledger.pending == 0:
            self.drained_at = Clock.t
        if self.drained_at is not None and hs.state in (A.VaraState.CONNECTED,
                                                        A.VaraState.DISCONNECTING):
            hs.disconnect(acquire=True)


def answerer():
    io = IO()
    hs = A.VaraStationHandshake([KESTREL], io, bw='500', high_speed=True,
                                compression='TEXT', chat=True)
    hs.listen(True)
    return hs, io


def our_over(io):
    return next(t for t, burst in io.sent if len(burst) > 3 * M.FS)


# -- kestrel closes: stock's ack of the last over ----------------------------

@pytest.mark.parametrize('name', ['noise09-answerer', 'noise055-answerer'])
def test_stocks_ack_of_our_last_over_is_read_and_the_close_goes_on(name):
    """Stock's frame-1 ack follows our 4.12 s over by 0.3 s (its PTT 74.33-74.82
    at 0.9, transcript time). b8996e08b read none of its eleven repeats; the
    farewell's drain timed out at 30 s and stock closed on its own 90 s later."""
    hs, io = answerer()
    client = Answerer()
    replay(hs, audio(name), 90.0, host=client)
    assert client.farewell, io.logs
    over = our_over(io)
    assert client.drained_at is not None and client.drained_at - (over + 4.12) < 1.2, (over, io.logs[-8:])
    assert any('read on their weaker carriers' in m for _, m in io.logs)
    assert any('asked WI9YZA for the turn to close in' in m for _, m in io.logs), io.logs[-8:]


def test_the_callers_turn_idle_settles_an_over_whose_acks_went_unread(monkeypatch):
    """With the soft route out of the way, b8996e08b's close: eleven acks unread.
    Stock then keys its 32-symbol turn-idle every 3.1 s (99.5-100.9 s, then
    102.6-104.0), which a caller keys only with nothing outstanding from us; the
    first one read whole behind its head settles the over. The driver's own
    queries are held off here: at 1.8 s against stock's 3.1 s they key across
    the frames in this replay, where the recorded stock cannot answer them."""
    monkeypatch.setattr(A, '_SOFT_PAIR_MATCH', 99.0)
    hs, io = answerer()
    client = Answerer()
    replay(hs, audio('noise09-answerer'), 105.0, host=client, ticks=False)
    over = our_over(io)
    idle = [t for t, m in io.logs if 'holds the turn idle, so our last over was taken' in m]
    assert idle and 100.8 < idle[0] < 104.3, io.logs[-8:]
    assert client.drained_at is not None and client.drained_at > over + 20, io.logs[-8:]
    assert not any('read on their weaker carriers' in m for _, m in io.logs)


@pytest.mark.parametrize('name', ['noise09-answerer', 'noise055-answerer'])
def test_nothing_else_on_the_cable_reads_as_a_control_burst(name):
    """Every 2 s window of the cut away from the ack repeats — noise, stock's
    request, link-setup, overs, release, turn-idles and CW ident — against every
    frame of the link's family."""
    x = audio(name)
    family = A._control_family(STOCK, KESTREL, '500')
    band = M.band_for('500')
    hits = []
    for start in np.arange(0, len(x) - 2 * M.FS, M.FS):
        t = CUT_START + start / M.FS
        if 72.5 < t < 99.0:
            continue
        power = A._band_power(x[int(start):int(start) + 2 * M.FS], band)
        for form in family:
            if A._soft_pairs(power, form, family, band) is not None:
                hits.append((t, form))
    assert not hits


def test_the_acks_themselves_read_at_every_repeat():
    """Ten of the eleven acks at -11.8 dB and all eleven at -9.4 pass the floors
    on their own; the one that does not (97.6 s at 0.9) is the last repeat."""
    family = A._control_family(STOCK, KESTREL, '500')
    f1 = tuple(map(tuple, F.control_bursts(STOCK, '500', 1)[0]))
    band = M.band_for('500')
    read = {}
    for name, first, misses in (('noise09-answerer', 75.18, 1), ('noise055-answerer', 75.01, 0)):
        x = audio(name)
        found = 0
        for k in range(11):
            t = first + 2.24 * k - CUT_START
            seg = x[int((t - 0.3) * M.FS):int((t + 1.0) * M.FS)]
            found += A._soft_pairs(A._band_power(seg, band), f1, family, band) is not None
        read[name] = found
        assert found >= 11 - misses, read


# -- kestrel calls: stock's connected-ack ------------------------------------

def caller():
    io = IO()
    hs = A.VaraStationHandshake([KESTREL], io, bw='500', high_speed=True,
                                compression='TEXT', chat=True)
    return hs, io


class Caller:
    """gate5's K answerer calling stock's CQ: opening and text at CONNECTED."""

    def __init__(self):
        self.connected_at = None

    def __call__(self, hs):
        if hs.state == A.VaraState.CONNECTED and self.connected_at is None:
            self.connected_at = Clock.t
            opening(hs)
            hs.send(b'text from K on the slot\r')


def test_the_clean_call_closes_on_stocks_receiver_disconnect():
    hs, io = caller()
    replay(hs, audio('clean-caller'), 70.0, connect_at=35.85, host=Caller())
    assert hs.state == A.VaraState.DISCONNECTED
    assert any('rx receiver-disconnect' in m for _, m in io.logs), io.logs[-5:]
    assert b'73<DISC>' in b''.join(io.host)


@pytest.mark.parametrize('rms', [0.51, 0.68])
def test_stocks_connected_ack_is_read_under_noise(rms):
    """Stock's connected-ack is its frame-1 control burst (43.2-43.7 s on the
    cut). Read by its preamble alone it needs four pairs exact; under the bench's
    noise it is read on its weaker carriers, 0.1 s past its last symbol."""
    hs, io = caller()
    x = audio('clean-caller')
    x = np.clip(x + band_noise(len(x), rms), -1, 1)
    client = Caller()
    replay(hs, x, 52.0, connect_at=35.85, host=client)
    assert client.connected_at is not None and 43.7 <= client.connected_at <= 44.2, io.logs[-6:]
    assert any('connected-ack on the receive stream, read on its weaker carriers' in m
               for _, m in io.logs)
    # Stock's frame-1 poll behind our confirm (46.2-46.7) is the same kind of
    # frame; its grant answers the recorded kestrel's request, not ours.
    polls = [t for t, m in io.logs if 'rx control-burst poll from WI9YZA' in m]
    assert polls and 46.7 <= polls[0] <= 47.3, io.logs[-6:]


# -- af547a830 on bench 3: the frames the first fix left ------------------------

@pytest.mark.parametrize('name, ask', [('noise055-caller', 3), ('noise09-caller', 2)])
def test_stocks_short_connected_ack_to_a_low_ask_is_read(name, ask):
    """Our greeting ask was graded to 3 and 2 under the noise, and stock's
    answerer acknowledged the link-setup with its 17-symbol frame (its 1079
    stream, frame 31), not the 11-symbol control burst; af547a830 waited for the
    control burst through six repeats and never connected."""
    hs, io = caller()
    client = Caller()
    replay(hs, audio(name), 48.0, connect_at=36.05, host=client)
    assert any(f'asking its greeting at level {ask}' in m for _, m in io.logs), io.logs[-6:]
    assert client.connected_at is not None and 43.9 <= client.connected_at <= 44.4, io.logs[-6:]
    assert any('rx session-connected-ack-short on the receive stream' in m for _, m in io.logs)


def test_the_callers_turn_request_read_short_of_whole_is_granted():
    """Stock asked for the turn with its 32-symbol request six times at -11.8 dB
    (44.98 s on, transcript), read 29-31 of 32 each time with one to three
    tones under the clearance; af547a830 wanted every tone behind the head
    exact and clear and granted none of them for 30 s."""
    hs, io = answerer()
    client = Answerer()
    replay(hs, audio('noise09-answerer-b'), 54.0, host=client)
    grant = M.synth_tone_pairs(F.turn_grant(KESTREL, '500'))
    granted = [t for t, burst in io.sent if np.array_equal(burst, grant)]
    assert any('rx session-keepalive-b' in m for _, m in io.logs), io.logs[-6:]
    assert granted and 51.5 <= granted[0] <= 53.3, (granted, io.logs[-6:])
