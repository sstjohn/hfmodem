# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Wide link-setup reception through the station's capture and energy gate."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.kestrel.host.station_runtime import CHUNK, StationTransport
from hfmodem.kestrel.monitor import Segmenter
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M, vara_ofdm as OF

CALLER, CALLED = 'K7ABC', 'W9SSJ'
FS = M.FS


class Transport(StationTransport):
    def __init__(self):
        self.sent, self.connections, self.quality = [], [], []
        station = SimpleNamespace(audio=SimpleNamespace(samples=0), arbiter=None)
        super().__init__(SimpleNamespace(station=station, emit=self.emit))

    def emit(self, samples, bandwidth, **kwargs):
        self.sent.append(np.asarray(samples))
        self.link.station.audio.samples += len(samples)
        return True

    def connected(self, *args):
        self.connections.append(args)

    def admit_contact(self):
        return True

    def snr(self, sn):
        self.quality.append(sn)

    def report_ptt(self, on):
        pass


def responder(bw):
    io = Transport()
    hs = A.VaraStationHandshake([CALLED], io, bw=bw, high_speed=True)
    io.hs = hs
    hs.listen(True)
    request = M.synth_burst(CALLED, F.connect_request(bw))
    hs.on_rx_audio(np.r_[np.zeros(4800), request, np.zeros(4800)])
    assert hs._awaiting_link_setup() and len(io.sent) == 1
    return hs, io


def in_noise(burst, snr, seed=1042):
    """SNR2500 uses keyed signal power and noise in the 300–2800 Hz passband."""
    lead = 6 * FS
    signal = np.r_[np.zeros(lead), burst, np.zeros(3 * FS)]
    noise = np.random.default_rng(seed).normal(size=len(signal))
    z = np.fft.rfft(noise)
    f = np.fft.rfftfreq(len(noise), 1 / FS)
    z[(f < 300) | (f >= 2800)] = 0
    noise = np.fft.irfft(z, len(noise))
    keyed = noise[lead:lead + len(burst)]
    z = np.fft.rfft(keyed)
    f = np.fft.rfftfreq(len(keyed), 1 / FS)
    noise_power = 2 * np.sum(abs(z[(f >= 300) & (f < 2800)]) ** 2) / len(keyed) ** 2
    noise *= np.sqrt(np.mean(burst ** 2) / noise_power) * 10 ** (-snr / 20)
    return signal + noise


def receive(hs, io, samples):
    brackets = []
    for start in range(0, len(samples), CHUNK):
        chunk = samples[start:start + CHUNK]
        at = io.link.station.audio.samples
        io.link.station.audio.samples += len(chunk)
        io.feed(at, chunk)
        burst = io.next_rx_burst(.001, hs=hs)
        if burst is not None:
            brackets.append(burst)
            hs.on_rx_audio(burst)
    assert not io.fault.is_set()
    return brackets


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_zero_db_setup_connects_without_an_energy_bracket(bw):
    hs, io = responder(bw)
    samples = in_noise(OF.link_setup_tx(CALLER, bw=bw), 0)
    gate = Segmenter()
    assert not [b for at in range(0, len(samples), CHUNK)
                for _, b in gate.push(samples[at:at + CHUNK])]
    assert not receive(hs, io, samples)
    assert hs.state == A.VaraState.CONNECTED and hs.caller == CALLER, list(io.messages)
    assert io.connections == [(CALLER, CALLED, bw)]
    assert len(io.sent) == 2 and io.quality


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_minus_five_db_setup_connects_despite_a_keying_noise_tail(bw):
    hs, io = responder(bw)
    samples = in_noise(OF.link_setup_tx(CALLER, bw=bw), -5, seed=0)
    lv, _, _, mag = hs._index_alignment(samples)
    r = A._RX2300.RECORDS[lv]
    assert lv == A._RX2300.BASE_LEVELS[bw] and mag is not None
    ref = A._lit(mag[:r.ncols] ** 2)
    now = A._lit(mag[r.ncols:r.ncols + A._KEYING_COLS] ** 2)
    assert now >= A._KEYING_FRAC * ref
    receive(hs, io, samples)
    assert hs.state == A.VaraState.CONNECTED, list(io.messages)
    assert io.connections == [(CALLER, CALLED, bw)]
    assert len(io.sent) == 2


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_setup_does_not_wait_for_the_middle_zone_probe(bw):
    hs, io = responder(bw)
    burst = OF.link_setup_tx(CALLER, bw=bw)
    r = A._RX2300.RECORDS[A._RX2300.BASE_LEVELS[bw]]
    samples = in_noise(burst, -5, seed=0)[:6 * FS + len(burst) + 8 * r.dw50]
    _, _, _, mag = hs._index_alignment(samples)
    assert A._KEYING_COLS <= len(mag) - r.ncols < A._PROBE_COLS
    receive(hs, io, samples)
    assert hs.state == A.VaraState.CONNECTED, list(io.messages)
    assert io.connections == [(CALLER, CALLED, bw)] and len(io.sent) == 2


def test_a_setup_recoverable_by_refinement_connects_without_a_nak():
    bw = '2300'
    hs, io = responder(bw)
    receive(hs, io, in_noise(OF.link_setup_tx(CALLER, bw=bw), -7, seed=9))
    assert hs.state == A.VaraState.CONNECTED, list(io.messages)
    assert io.connections == [(CALLER, CALLED, bw)]
    assert len(io.sent) == 2
    nak = M.synth_burst(CALLED, F.for_bw(F.SESSION_LINK_SETUP_NAK, bw))
    assert not any(np.array_equal(sent, nak) for sent in io.sent)


@pytest.mark.parametrize('bw,seed', [('2300', 4), ('2750', 11)])
def test_a_clean_data_frame_is_not_rescanned_into_a_setup_nak(bw, seed):
    hs, io = responder(bw)
    burst = OF.data_over_tx(bytes(range(90)), bw=bw, level=A._RX2300.BASE_LEVELS[bw])
    receive(hs, io, in_noise(burst, -7, seed=seed))
    assert hs._awaiting_link_setup() and not io.connections
    assert len(io.sent) == 1, list(io.messages)


@pytest.mark.parametrize('data_first', [True, False])
def test_setup_refinement_does_not_take_a_neighbouring_data_frame(data_first):
    from hfmodem.kestrel.tx import varahf2300_tx as tx
    bw = '2300'
    lv = A._RX2300.BASE_LEVELS[bw]
    data = OF.data_over_tx(bytes(range(90)), bw=bw, level=lv)
    data = in_noise(data, -5, seed=0)[6 * FS:6 * FS + len(data)]
    frame = bytearray(F.link_setup_frame(CALLER, bw=bw))
    frame[-1] ^= 0xff
    bad = tx.synth_frame(bytes(frame), level=lv)
    first, second = (data, bad) if data_first else (bad, data)
    hs, io = responder(bw)
    hs.on_rx_stream(np.r_[np.zeros(4800), first, np.zeros(4800),
                          second, np.zeros(4800)])
    hs.on_rx_stream(np.zeros(6 * FS))
    assert hs._awaiting_link_setup() and not io.connections
    assert len(io.sent) == 2, list(io.messages)
    assert np.array_equal(io.sent[-1], M.synth_burst(
        CALLED, F.for_bw(F.SESSION_LINK_SETUP_NAK, bw)))

    hs, io = responder(bw)
    hs.on_rx_stream(np.r_[np.zeros(4800), data, np.zeros(4800)])
    hs.on_rx_stream(np.zeros(6 * FS))
    assert hs._awaiting_link_setup() and not io.connections
    assert len(io.sent) == 1, list(io.messages)


@pytest.mark.parametrize('bw,seed', [('2300', 0), ('2750', 2)])
def test_window_state_distinguishes_weak_noise_from_a_second_frame(bw, seed):
    lv = A._RX2300.BASE_LEVELS[bw]
    r = A._RX2300.RECORDS[lv]
    frame = OF.link_setup_tx(CALLER, bw=bw)[-r.ncols * r.dw50:]

    def magnitudes(snr, continuation):
        burst = np.r_[frame, frame[:A._PROBE_COLS * r.dw50]] if continuation else frame
        samples = in_noise(burst, snr, seed=seed)[6 * FS:]
        return A._RX2300._band_mag(samples, lv, r.ncols + A._PROBE_COLS)

    mag = magnitudes(-5, False)
    ref = A._lit(mag[:r.ncols] ** 2)
    now = A._lit(mag[r.ncols:r.ncols + A._KEYING_COLS] ** 2)
    assert A._KEYING_FRAC * ref <= now < 10.0
    assert A._window_state(mag[:r.ncols + A._KEYING_COLS], r) == A._UNKNOWN
    assert A._window_state(mag, r) == A._QUIET
    continuation = magnitudes(-5, True)
    assert A._next_frame_here(continuation[r.ncols:], r) == A._PROBE_REFS
    assert A._window_state(continuation, r) == A._KEYING
    strong = magnitudes(25, True)[:r.ncols + A._KEYING_COLS]
    assert A._lit(strong[r.ncols:] ** 2) >= max(
        A._KEYING_FRAC * A._lit(strong[:r.ncols] ** 2), 10.0)
    assert A._window_state(strong, r) == A._KEYING


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_a_weak_data_over_is_answered_without_a_keying_hold(bw):
    from hfmodem.tests.kestrel.test_data_over_gate import _connected
    hs, io = _connected(bw)
    payload = b'A' * 89
    body = A._phy.vara_body(payload, hs.caller, tail=0x81)
    assert A._phy.vara_payload(body, caller=hs.caller) == payload
    burst = OF.data_over_tx(body, bw=bw, level=A._RX2300.BASE_LEVELS[bw])
    samples = in_noise(burst, -5, seed=0)
    for start in range(0, len(samples), CHUNK):
        hs.on_rx_stream(samples[start:start + CHUNK])
        if io.host:
            assert io.keys == 1 and hs._held_answer is None, io.msgs
    assert io.host == [payload], io.msgs
    assert io.keys == 1, io.msgs
    assert hs._held_answer is None
    assert not any('still keying' in m for m in io.msgs), io.msgs


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_high_snr_setup_keys_one_connected_ack(bw):
    hs, io = responder(bw)
    burst = OF.link_setup_tx(CALLER, bw=bw)
    receive(hs, io, in_noise(burst, 25))
    assert hs.state == A.VaraState.CONNECTED, list(io.messages)
    assert hs._stream_link_setup_at is not None
    hs.on_rx_audio(np.r_[np.zeros(4800), burst, np.zeros(4800)])
    assert io.connections == [(CALLER, CALLED, bw)]
    assert len(io.sent) == 2, list(io.messages)
    acked_at, progress = hs._acked_at, hs.progress
    receive(hs, io, in_noise(burst, 25))
    assert len(io.sent) == 2, list(io.messages)
    assert hs._acked_at == acked_at and hs.progress == progress
    assert io.connections == [(CALLER, CALLED, bw)]


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_a_refused_stream_ack_waits_for_a_new_setup(bw):
    hs, io = responder(bw)
    samples = in_noise(OF.link_setup_tx(CALLER, bw=bw), 25)
    io.link.emit = lambda *args, **kwargs: False
    receive(hs, io, samples)
    assert hs._awaiting_link_setup() and hs._stream_link_setup_at is not None
    assert not io.connections and len(io.sent) == 1
    io.link.emit = io.emit
    hs.on_rx_audio(samples)
    assert hs._awaiting_link_setup() and len(io.sent) == 1
    receive(hs, io, samples)
    assert io.connections == [(CALLER, CALLED, bw)] and len(io.sent) == 2


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_a_refused_stream_ack_keeps_the_bracket_retry(bw):
    hs, io = responder(bw)
    burst = OF.link_setup_tx(CALLER, bw=bw)
    io.link.emit = lambda *args, **kwargs: False
    receive(hs, io, in_noise(burst, 25))
    assert hs._awaiting_link_setup()
    assert not io.connections and len(io.sent) == 1

    level = A._RX2300.BASE_LEVELS[bw]
    record = A._RX2300.RECORDS[level]
    references = np.flatnonzero(A._RX2300._BASE_TABLES[level][1])
    retry = burst.copy()
    for col in references[:12]:
        start = (record.lead + col) * record.dw50
        retry[start:start + record.dw50] = 0
    assert OF.link_setup_read(retry, bw=bw) == F.link_setup_frame(CALLER, bw=bw)
    samples = in_noise(retry, 25)
    probe, probe_io = responder(bw)
    for start in range(0, len(samples), CHUNK):
        probe.on_rx_stream(samples[start:start + CHUNK])
    assert probe._awaiting_link_setup() and len(probe_io.sent) == 1

    io.link.emit = io.emit
    assert receive(hs, io, samples)
    assert hs.state == A.VaraState.CONNECTED, list(io.messages)
    assert io.connections == [(CALLER, CALLED, bw)] and len(io.sent) == 2


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_a_new_contact_keeps_the_bracket_fallback(bw):
    hs, io = responder(bw)
    burst = OF.link_setup_tx(CALLER, bw=bw)
    receive(hs, io, in_noise(burst, 0))
    assert hs._stream_link_setup_at is not None
    hs.state = A.VaraState.DISCONNECTED
    hs.listen(True)
    hs.on_rx_audio(M.synth_burst(CALLED, F.connect_request(bw)))
    assert hs._awaiting_link_setup()
    hs.on_rx_stream(np.zeros(CHUNK))
    hs.on_rx_audio(np.r_[np.zeros(4800), burst, np.zeros(4800)])
    assert io.connections == [(CALLER, CALLED, bw)] * 2
    assert len(io.sent) == 4


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('boundary', ['rx_discontinuity', 'retuned', 'flushed', 'key', 'state'])
def test_setup_search_does_not_splice_across_a_boundary(bw, boundary):
    hs, io = responder(bw)
    samples = in_noise(OF.link_setup_tx(CALLER, bw=bw), 0)
    split = 8 * FS
    receive(hs, io, samples[:split])
    assert len(hs._ov_buf) > FS
    if boundary == 'flushed':
        io.flushed(io.link.station.audio.samples)
        io.before_step()
    elif boundary == 'key':
        hs._key(True)
        hs._key(False)
    elif boundary == 'state':
        hs.state = A.VaraState.DISCONNECTED
        hs.listen(True)
        receive(hs, io, samples[:CHUNK])
    else:
        getattr(hs, boundary)()
    assert not len(hs._ov_buf)
    receive(hs, io, samples[split:])
    assert not io.connections
    if boundary != 'state':
        receive(hs, io, samples)
        assert io.connections == [(CALLER, CALLED, bw)]


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('kind', ['noise', 'data', 'bad-crc'])
def test_only_a_clean_link_setup_completes_the_connect(bw, kind):
    hs, io = responder(bw)
    if kind == 'bad-crc':
        from hfmodem.kestrel.tx import varahf2300_tx as tx
        frame = bytearray(F.link_setup_frame(CALLER, bw=bw))
        frame[-1] ^= 0xff
        burst = tx.synth_frame(bytes(frame), level=A._RX2300.BASE_LEVELS[bw])
    elif kind == 'data':
        burst = OF.data_over_tx(bytes(range(90)), bw=bw, level=A._RX2300.BASE_LEVELS[bw])
    samples = (np.random.default_rng(1042).normal(0, .05, 12 * FS) if kind == 'noise'
               else in_noise(burst, 0))
    receive(hs, io, samples)
    assert hs._awaiting_link_setup() and not io.connections
    assert len(io.sent) == (2 if kind == 'bad-crc' else 1)
    if kind == 'bad-crc':
        assert np.array_equal(io.sent[-1], M.synth_burst(
            CALLED, F.for_bw(F.SESSION_LINK_SETUP_NAK, bw)))
        receive(hs, io, in_noise(OF.link_setup_tx(CALLER, bw=bw), 0))
        assert hs.state == A.VaraState.CONNECTED, list(io.messages)
        assert io.connections == [(CALLER, CALLED, bw)] and len(io.sent) == 3


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_a_bad_setup_is_nacked_once_across_rescans_and_its_energy_bracket(bw, monkeypatch):
    from hfmodem.kestrel.tx import varahf2300_tx as tx
    hs, io = responder(bw)
    frame = bytearray(F.link_setup_frame(CALLER, bw=bw))
    frame[-1] ^= 0xff
    burst = tx.synth_frame(bytes(frame), level=A._RX2300.BASE_LEVELS[bw])
    samples = in_noise(burst, 25)
    scans = []
    align = hs._index_alignment

    def alignment(x):
        scans.append(len(x))
        return align(x)

    monkeypatch.setattr(hs, '_index_alignment', alignment)
    receive(hs, io, samples)
    count = len(scans)
    receive(hs, io, np.zeros(6 * FS))
    assert len(scans) > count + 1
    assert hs._stream_link_setup_at is not None
    monkeypatch.setattr(hs, '_rx_link_setup', lambda x: pytest.fail(
        'the energy bracket must not answer the stream-owned setup'))
    hs.on_rx_audio(np.r_[samples, np.zeros(6 * FS)])
    assert hs._awaiting_link_setup() and not io.connections
    assert len(io.sent) == 2, list(io.messages)
    assert np.array_equal(io.sent[-1], M.synth_burst(
        CALLED, F.for_bw(F.SESSION_LINK_SETUP_NAK, bw)))


class CallerIO:
    def __init__(self):
        self.msgs, self.sent = [], []

    def key(self, on): ...
    def tx(self, samples): self.sent.append(np.asarray(samples, float))
    def tx_went_out(self): return True
    def pending(self): ...
    def connected(self, *a): ...
    def data(self, payload): ...
    def log(self, msg): self.msgs.append(msg)


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_our_caller_connects_to_us_on_the_resend_a_level_below_our_nak(bw):
    """Our caller resends a NAKed link-setup a level down, so our responder,
    having NAKed it, reads the resend there and connects on it."""
    from hfmodem.kestrel.tx import varahf2300_tx as tx
    hs, io = responder(bw)
    frame = bytearray(F.link_setup_frame(CALLER, bw=bw))
    frame[-1] ^= 0xff
    receive(hs, io, in_noise(tx.synth_frame(bytes(frame),
                                            level=A._RX2300.BASE_LEVELS[bw]), 25))
    nak = io.sent[-1]
    assert np.array_equal(nak, M.synth_burst(CALLED, F.for_bw(F.SESSION_LINK_SETUP_NAK, bw)))

    cio = CallerIO()
    caller = A.VaraStationHandshake([CALLER], cio, bw=bw)
    caller.role, caller.called, caller.caller = 'initiator', CALLED, CALLER
    caller.state, caller.step = A.VaraState.CONNECTING, A._I_LINKSETUP_SENT
    caller._linksetup_tx = 1
    assert caller._link_setup_nacked(np.r_[np.zeros(int(.3 * FS)), nak,
                                           np.zeros(int(.12 * FS))])
    assert caller._setup_level == 3 and len(cio.sent) == 1, cio.msgs

    receive(hs, io, in_noise(cio.sent[0], 25))
    assert hs.state == A.VaraState.CONNECTED and hs.caller == CALLER, list(io.messages)
    assert io.connections == [(CALLER, CALLED, bw)] and len(io.sent) == 3


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_a_nak_that_did_not_go_out_reads_no_lower(bw):
    from hfmodem.kestrel.tx import varahf2300_tx as tx
    hs, io = responder(bw)
    frame = bytearray(F.link_setup_frame(CALLER, bw=bw))
    frame[-1] ^= 0xff
    bad = in_noise(tx.synth_frame(bytes(frame), level=A._RX2300.BASE_LEVELS[bw]), 25)
    io.link.emit = lambda *args, **kwargs: False
    receive(hs, io, bad)
    assert hs._setup_read == 4 and len(io.sent) == 1
    io.link.emit = io.emit
    receive(hs, io, bad)
    assert hs._setup_read == 3 and len(io.sent) == 2


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_a_lost_resend_still_lets_the_next_retry_connect_two_levels_down(bw):
    """Behind our NAK the caller's level-3 resend is lost; its silent retry
    comes at level 2, below the level our one NAK asked for, and connects."""
    from hfmodem.kestrel.tx import varahf2300_tx as tx
    hs, io = responder(bw)
    frame = bytearray(F.link_setup_frame(CALLER, bw=bw))
    frame[-1] ^= 0xff
    receive(hs, io, in_noise(tx.synth_frame(bytes(frame),
                                            level=A._RX2300.BASE_LEVELS[bw]), 25))
    nak = io.sent[-1]
    assert hs._setup_read == 3

    cio = CallerIO()
    caller = A.VaraStationHandshake([CALLER], cio, bw=bw)
    caller.role, caller.called, caller.caller = 'initiator', CALLED, CALLER
    caller.state, caller.step = A.VaraState.CONNECTING, A._I_LINKSETUP_SENT
    caller._linksetup_tx = 1
    assert caller._link_setup_nacked(np.r_[np.zeros(int(.3 * FS)), nak,
                                           np.zeros(int(.12 * FS))])
    assert caller.retry_connect() and caller._setup_level == 2 and len(cio.sent) == 2

    receive(hs, io, in_noise(cio.sent[1], 25))
    assert hs.state == A.VaraState.CONNECTED and hs.caller == CALLER, list(io.messages)
    assert io.connections == [(CALLER, CALLED, bw)] and len(io.sent) == 3
