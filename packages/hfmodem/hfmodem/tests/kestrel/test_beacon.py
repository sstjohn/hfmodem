# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VarAC's advanced beacon: the recorded frames, stock's own emission, our synthesis,
and the host's KISS port both ways."""
import json
from pathlib import Path
import socket
import threading

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import hilbert

from hfmodem.ax25 import kiss
from hfmodem.core.wav import source
from hfmodem.kestrel.host.kiss import KissPort
from hfmodem.kestrel.host.server import HostSession, VaraServer
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.rx import beacon as B
from hfmodem.kestrel.rx.cq import build_frame
from hfmodem.kestrel.rx.cq_stream import CQDetector
from hfmodem.kestrel.tx.beacon import synth_beacon
from hfmodem.kestrel.tx.cq import synth_cq, synth_frame
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from hfmodem.tests.evidence import CORPUS, require_input
from .test_compression import IO
from .test_cq import MockSocket, replay

FIXTURES = Path(__file__).with_name('fixtures') / 'varac-beacon-1023'
PROVENANCE = json.loads((FIXTURES / 'provenance.json').read_text())
#: KG9B's record-1 payload off 40 m (session 20261023T042123057106Z, 34943.25 s) and the
#: one stock keyed on bench 3 for the KISS frame `C0 02 "20OEN63BCW9SSJ      " C0`.
KG9B = bytes.fromhex('2164c27c59cd5a46992e3b984810204081020f14440000')
STOCK = bytes.fromhex('2164c27c59cd99c2875dcd3a7290204081020f14000000')


def recording(name):
    prov = PROVENANCE[name]
    rate, x = wavfile.read(source(require_input(CORPUS / 'varac-beacon-1023' / f'{name}.flac')))
    assert rate == prov['rate'] and len(x) == prov['samples'][1] - prov['samples'][0]
    return rate, x / 32768


def test_payloads_are_byte_exact_against_the_air_and_the_bench():
    assert B.build_payload('20OEN54FLKG9B', 0x44) == KG9B[:22]
    assert B.build_payload('20OEN63BCW9SSJ') == STOCK[:22]
    assert B.parse_payload(KG9B).text == '20OEN54FLKG9B'
    assert B.parse_payload(STOCK).text == '20OEN63BCW9SSJ'
    assert B.unpack7(KG9B) == '20OEN54FLKG9B       '
    with pytest.raises(ValueError):
        B.pack7('twenty-one characters')
    assert B.parse_payload(bytes([0x50]) + KG9B[1:]) is None
    assert B.parse_payload(bytes(22)) is None


def shifted(x, rate, hz):
    return np.real(hilbert(x) * np.exp(2j * np.pi * hz * np.arange(len(x)) / rate))


@pytest.mark.parametrize('name', list(PROVENANCE))
def test_recorded_beacons_decode_in_the_stream(name):
    rate, x = recording(name)
    got = replay(CQDetector(rate), shifted(x, rate, -PROVENANCE[name].get('offset_hz', 0)))
    assert [r.text for r in got] == [PROVENANCE[name]['text']]
    if PROVENANCE[name]['sn_live'] is not None:
        assert got[0].sn > 0


def test_an_off_centre_beacon_is_read_where_stock_reads_it():
    # Stock B, bench 3: KC3QLX's 200 Hz-low beacon not as heard, nor shifted +50 Hz; read
    # shifted +75 to +300 Hz. The synthesised frames: read 140 Hz off the centre, not 150.
    rate, x = recording('kc3qlx-fn20aq-200hz-low')
    for hz in (0, 75, 100, 200, 300):
        assert bool(replay(CQDetector(rate), shifted(x, rate, hz))) == (hz != 0), hz
    rng = np.random.default_rng(3)
    for frame in (synth_beacon('20OEN63BCW9SSJ', amplitude=.1), synth_cq('W9SSJ-9', 500, amplitude=.1)):
        for hz in (-140, 140, -200, 200):
            y = np.r_[np.zeros(24000), shifted(frame, 48000, hz), np.zeros(48000)]
            got = replay(CQDetector(), y + rng.normal(0, .003, len(y)))
            assert bool(got) == (abs(hz) == 140), hz


def test_our_beacon_round_trips_in_noise_and_beside_a_cq():
    rng = np.random.default_rng(7)
    air = synth_beacon('20OEN63BCW9SSJ')
    assert len(air) == 32 * 2048 + 1536 + (B.RECORD.lead + B.RECORD.ncols) * B.RECORD.dw
    for sigma in (0, .2, .35):
        x = np.r_[np.zeros(24000), air, np.zeros(48000)] + rng.normal(0, sigma, len(air) + 72000)
        got = replay(CQDetector(), x)
        assert [r.text for r in got] == ['20OEN63BCW9SSJ']
    x = np.r_[np.zeros(24000), synth_cq('W9SSJ-3', 500), np.zeros(24000), air, np.zeros(48000)]
    got = replay(CQDetector(), x)
    assert [getattr(r, 'call', None) or r.text for r in got] == ['W9SSJ-3', '20OEN63BCW9SSJ']


def test_nothing_else_is_a_beacon():
    rng = np.random.default_rng(84)
    header = M.synth_tones(B.HEADER)
    corrupt = bytearray(B.build_payload('20OEN63BCW9SSJ'))
    corrupt[3] ^= 0x55
    from hfmodem.kestrel.tx.varahf500_tx import build_l3_frame, synth_l3_burst
    bad_body = synth_l3_burst(build_l3_frame(bytes(corrupt), 0, level=2), level=2, training=B.TRAINING) / 60
    x = np.r_[rng.normal(0, .01, 48000 * 10),
              M.synth_burst('W9SSJ', F.connect_request('500')), np.zeros(48000),
              synth_frame(build_frame('W9SSJ', 500)), np.zeros(48000),
              header, np.zeros(48000 * 6),
              header[:32 * 2048], bad_body, np.zeros(48000)]
    d = CQDetector()
    got = replay(d, x)
    assert [type(r).__name__ for r in got] == ['CQResult']


# -- the host -------------------------------------------------------------------------

class Kiss:
    def __init__(self): self.delivered = []
    def deliver(self, text): self.delivered.append(text)


def host(chat=True):
    modem = VaraHostModem(IO(), bandwidth='500')
    port = Kiss()
    session = HostSession(MockSocket(), MockSocket(), modem, kiss=port)
    commands = []
    session._send_cmd = commands.append
    modem.start(session)
    modem.poll()
    for command in (['MYCALL W9SSJ', 'CHAT ON'] if chat else ['MYCALL W9SSJ']):
        session._dispatch(command)
    modem.poll()
    commands.clear()
    return session, modem, commands, port


def test_a_kiss_frame_keys_one_beacon_as_a_cq_is_keyed():
    session, modem, commands, port = host()
    tx = []
    modem._transport.tx = tx.append
    modem.send_beacon('20OEN63BCW9SSJ')
    with pytest.raises(ValueError):
        modem.send_beacon('20OEN63BCW9SSJ')
    assert not tx
    modem.poll()
    assert commands == ['PENDING', 'PTT ON', 'BITRATE (2)  41 bps TX', 'PTT OFF', 'CANCELPENDING']
    assert len(tx) == 1
    got = replay(CQDetector(), np.r_[tx[0], np.zeros(24000)])
    assert [r.text for r in got] == ['20OEN63BCW9SSJ']
    with pytest.raises(ValueError):
        modem.send_beacon('x' * 21)


def test_a_busy_channel_or_a_link_holds_the_beacon():
    session, modem, commands, port = host()
    tx = []
    modem._transport.tx = tx.append
    modem._channel_busy = True
    modem.send_beacon('20OEN63BCW9SSJ')
    modem.poll()
    assert not tx and commands == []
    modem.station.state = A.VaraState.CONNECTED
    with pytest.raises(ValueError):
        modem.send_beacon('20OEN63BCW9SSJ')


@pytest.mark.parametrize('chat', [True, False])
def test_a_heard_beacon_reaches_the_kiss_port_as_stock_hands_it_over(chat):
    session, modem, commands, port = host(chat)
    rate, x = recording('stock-kiss2-w9ssj')
    station = modem.station
    rng = np.random.default_rng(411)
    i = 0
    while i < len(x):
        n = int(rng.integers(128, 4097))
        station.on_rx_stream(x[i:i + n]); i += n
    assert port.delivered == ['20OEN63BCW9SSJ      ']
    assert [c[:2] for c in commands] == (['SN'] if chat else [])
    assert station.state in (A.VaraState.DISCONNECTED, A.VaraState.LISTENING) and not modem.connected


def test_the_kiss_port_speaks_command_two_both_ways():
    texts = []
    port = KissPort(texts.append, port=0)
    port.start()
    try:
        client = socket.create_connection(('127.0.0.1', port.port), timeout=5)
        client.sendall(kiss.encode(b'20OEN63BCW9SSJ      ', 2) + kiss.encode(b'ignored', 1)
                       + kiss.encode(b'\xc0\x01ax25', 0) + kiss.encode(b'\xff\xfe', 2))
        deadline = threading.Event()
        for _ in range(50):
            if texts:
                break
            deadline.wait(.05)
        assert texts == ['20OEN63BCW9SSJ      ']
        for _ in range(50):
            if port._clients:
                break
            deadline.wait(.05)
        port.deliver('20OEN54FLKG9B')
        parser = kiss.Parser()
        got = []
        while not got:
            got = parser.feed(client.recv(4096))
        assert got == [(0, 2, b'20OEN54FLKG9B')]
        client.close()
    finally:
        assert port.stop()


def test_the_server_owns_the_port_and_drops_frames_with_no_application():
    srv = VaraServer(cmd_port=0, data_port=0, kiss_port=0)
    srv.start_background()
    try:
        assert srv.kiss_port and srv._session is None
        client = socket.create_connection(('127.0.0.1', srv.kiss_port), timeout=5)
        client.sendall(kiss.encode(b'20OEN63BCW9SSJ', 2))
        client.settimeout(.3)
        with pytest.raises(socket.timeout):
            client.recv(10)
        client.close()
    finally:
        srv.stop()
    assert VaraServer(cmd_port=0, data_port=0).kiss_port is None


def test_a_beacon_the_modem_refuses_does_not_stop_the_kiss_port():
    texts = []

    def refuse(text):
        texts.append(text)
        if len(texts) == 1:
            raise ConnectionAbortedError('attachment retiring')

    port = KissPort(refuse, port=0)
    port.start()
    try:
        client = socket.create_connection(('127.0.0.1', port.port), timeout=5)
        client.sendall(kiss.encode(b'20OEN63BCW9SSJ', 2))
        wait = threading.Event()
        for _ in range(50):
            if texts:
                break
            wait.wait(.05)
        client.sendall(kiss.encode(b'20OEN54FLKG9B', 2))
        for _ in range(50):
            if len(texts) == 2:
                break
            wait.wait(.05)
        assert texts == ['20OEN63BCW9SSJ', '20OEN54FLKG9B']
        client.close()
    finally:
        assert port.stop()
