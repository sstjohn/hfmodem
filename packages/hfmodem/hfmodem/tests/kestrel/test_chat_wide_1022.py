# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VarAC chat at BW2300 and BW2750 does what stock does: bench 3, 2026-10-22.

The reference pair consists of two stock 4.9.0s in CHAT ON (`working/varac-on-air/bench3/b1022/s2s-2300`
and `s2s-2750`); the faults are the pinned kestrel runs beside them."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.core.wav import source
from hfmodem.tests import evidence
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.host.server import HostSession
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_varac_answer_1020 import live

HERE = Path(__file__).with_name('fixtures')
WIDE = evidence.CORPUS / 'varac-chat-wide-1022'
pytestmark = pytest.mark.skipif(not WIDE.is_dir(), reason=f'recordings not present ({evidence.CORPUS})')
STOCK, KESTREL = 'WI9YZA', 'W9SSJ'
BWS = ['2300', '2750']


class Clock:
    t = 1000.

    @classmethod
    def monotonic(cls):
        return cls.t


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    Clock.t = 1000.
    monkeypatch.setattr(A, 'time', Clock)


class IO(A.VaraIO):
    def __init__(self):
        self.sent, self.host, self.logs = [], [], []

    def tx(self, samples):
        self.sent.append(np.asarray(samples, float))

    def data(self, payload):
        self.host.append(bytes(payload))

    def log(self, msg):
        self.logs.append(msg)


def audio(name):
    prov = json.loads((HERE / WIDE.name / 'provenance.json').read_text())[name]
    rate, x = wavfile.read(source(WIDE / prov.get('file', f'{name}.flac')))
    assert rate == M.FS and x.dtype == np.int16
    return x.astype(float) / 32767


def feed(hs, x):
    """The live order: every sample to the stream, then the bracket."""
    x = np.r_[x, np.zeros(M.FS // 4)]
    for at in range(0, len(x), 1024):
        chunk = x[at:at + 1024]
        Clock.t += len(chunk) / M.FS
        hs.on_rx_stream(chunk)
    hs.on_rx_audio(x)


def keyed_pairs(io, want):
    return np.array_equal(io.sent[-1], M.synth_tone_pairs(want))


def keyed_burst(io, call, kind, bw):
    return np.array_equal(io.sent[-1], M.synth_burst(call, F.for_bw(kind, bw)))


def carries(io, bw, level, text):
    """Our last DATA over opens with ``text``, ahead of its trailer and padding."""
    read = phy.decode_level(io.sent[-1], bw, level)
    assert read is not None, io.logs[-3:]
    return phy.vara_payload(read.frames[0], caller=KESTREL,
                            body_len=len(read.frames[0])).startswith(text)


def answerer(bw):
    io = IO()
    hs = A.VaraStationHandshake([KESTREL], io, bw=bw, high_speed=True, compression='TEXT', chat=True)
    hs.listen(True)
    return hs, io


def connected(bw):
    hs, io = answerer(bw)
    feed(hs, audio(f's{bw}-request'))
    feed(hs, audio(f's{bw}-link-setup'))
    assert hs.state == A.VaraState.CONNECTED, io.logs
    return hs, io


def holding(bw):
    """The caller's f20 behind our ack: an empty turn, held and polled."""
    hs, io = connected(bw)
    feed(hs, audio(f's{bw}-f20'))
    assert hs.turn == A._TURN_OURS and hs.idle_poll_due_in() is not None, io.logs
    return hs, io


def poll(hs, io, bw):
    Clock.t += hs.idle_poll_due_in()
    hs.idle_keepalive()
    assert keyed_pairs(io, F.control_bursts(STOCK, bw, 1)[1]), io.logs[-3:]


def delivered(bw):
    """Our L5 reply on the air behind the caller's drained frame."""
    hs, io = holding(bw)
    poll(hs, io, bw)
    feed(hs, audio(f's{bw}-keepalive-b'))
    Clock.t += 2
    hs.send(b'reply from the answerer\r')
    assert keyed_pairs(io, F.control_bursts(STOCK, bw, 0)[1]), io.logs[-3:]
    feed(hs, audio(f's{bw}-drained'))
    assert carries(io, bw, 5, b'reply from the answerer\r'), io.logs[-3:]
    return hs, io


# -- the answerer: stock calls kestrel ---------------------------------------

@pytest.mark.parametrize('bw', BWS)
def test_the_chat_answerer_answers_short_and_acks_with_frame_1(bw):
    hs, io = answerer(bw)
    feed(hs, audio(f's{bw}-request'))
    assert keyed_burst(io, KESTREL, F.chat_setup_response(bw), bw), io.logs
    feed(hs, audio(f's{bw}-link-setup'))
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert keyed_pairs(io, F.control_bursts(STOCK, bw, 1)[1]), io.logs


@pytest.mark.parametrize('bw', BWS)
def test_the_callers_f20_is_held_and_polled_on_stocks_cadence(bw):
    hs, io = connected(bw)
    sent = len(io.sent)
    feed(hs, audio(f's{bw}-f20'))
    assert len(io.sent) == sent and hs.turn == A._TURN_OURS, io.logs
    assert hs.idle_poll_due_in() == pytest.approx(A._HOLD_FIRST_POLL_S, abs=1.0)
    polls = []
    for _ in range(4):
        poll(hs, io, bw)
        polls.append(Clock.t)
        sent = len(io.sent)
        feed(hs, audio(f's{bw}-keepalive-b'))
        assert len(io.sent) == sent, io.logs[-3:]
    assert np.allclose(np.diff(polls), A._IDLE_POLL_S, atol=.6)
    assert hs.state == A.VaraState.CONNECTED and hs._since_progress == 0


@pytest.mark.parametrize('bw', BWS)
def test_a_reply_written_while_holding_asks_with_frame_0_and_goes_out_on_stocks_drained(bw):
    hs, io = delivered(bw)
    assert hs._tx_pending is not None and hs._tx_pending[2] == phy.level_key(bw, 5)


@pytest.mark.parametrize('bw', BWS)
def test_the_callers_frame_1_draws_the_bandwidths_idle_handback_and_its_poll_the_idle_response(bw):
    hs, io = delivered(bw)
    feed(hs, audio(f's{bw}-caller-f1'))
    assert keyed_pairs(io, F.responder_post_delivery_idle(KESTREL, bw)), io.logs[-3:]
    assert hs.turn == A._TURN_PEER and hs._tx_pending is None
    feed(hs, audio(f's{bw}-caller-poll'))
    assert keyed_burst(io, KESTREL, F.SESSION_IDLE_RESPONSE, bw), io.logs[-3:]
    feed(hs, audio(f's{bw}-caller-f0'))
    assert keyed_burst(io, KESTREL, F.SESSION_DRAINED_RESPONDER, bw), io.logs[-3:]
    feed(hs, audio(f's{bw}-data-2'))
    assert b''.join(io.host) == b'second line from the caller\r', io.logs[-3:]
    assert keyed_pairs(io, F.control_bursts(STOCK, bw, 1)[1]), io.logs[-3:]
    feed(hs, audio(f's{bw}-caller-idle-11'))
    assert hs.turn == A._TURN_OURS and hs.idle_poll_due_in() is not None, io.logs[-3:]


@pytest.mark.parametrize('bw', BWS)
def test_the_callers_close_and_ident_end_the_link(bw):
    hs, io = holding(bw)
    poll(hs, io, bw)
    feed(hs, audio(f's{bw}-close'))
    assert hs.state == A.VaraState.DISCONNECTED, io.logs[-3:]


def test_a_caller_that_gives_up_idents_and_the_link_is_closed():
    hs, io = holding('2300')
    feed(hs, audio('x2300-ident'))
    assert hs.state == A.VaraState.DISCONNECTED, io.logs[-3:]


@pytest.mark.parametrize('bw', BWS)
def test_a_write_right_after_connect_waits_for_the_callers_handover(bw):
    """cq2300-kestrel-cqs: the opening keyed a 32-symbol request 0.02 s behind
    the caller's confirm and the link never carried a byte. Behind the f20 the
    ask waits out the request the caller keys 0.3 s later when its own host
    wrote too (fix2b53 cq2300 44.397), and goes as frame 0 when none comes."""
    hs, io = connected(bw)
    sent = len(io.sent)
    hs.send(b'de W9SSJ <R+28>\r')
    assert len(io.sent) == sent, io.logs[-3:]
    feed(hs, audio(f's{bw}-f20'))
    assert len(io.sent) == sent and 0 < hs.idle_poll_due_in() <= A._F20_REASK_S, io.logs[-3:]
    Clock.t += hs.idle_poll_due_in()
    hs.idle_keepalive()
    assert keyed_pairs(io, F.control_bursts(STOCK, bw, 0)[1]), io.logs[-3:]
    feed(hs, audio(f's{bw}-drained'))
    assert carries(io, bw, 5, b'de W9SSJ <R+28>\r'), io.logs[-3:]


@pytest.mark.parametrize('bw', BWS)
def test_the_callers_request_behind_its_f20_is_granted_and_our_turn_follows_its_release(bw):
    hs, io = connected(bw)
    hs.send(b'de W9SSJ <R+28>\r')
    feed(hs, audio(f's{bw}-f20'))
    sent = len(io.sent)
    feed(hs, M.synth_burst(STOCK, F.for_bw(F.SESSION_TURN_REQUEST_SHORT, bw)))
    assert len(io.sent) == sent + 1 and keyed_pairs(io, F.turn_grant(KESTREL, bw)), io.logs[-3:]
    assert hs.turn == A._TURN_PEER and hs.idle_poll_due_in() is None


def test_the_cq_answerer_at_2300_grants_the_callers_request_and_delivers_behind_its_release():
    """fix2b53 cq2300-kestrel-cqs, both openings queued at CONNECTED: stock's
    f20 43.373-44.099 and its 17-symbol request 44.397-45.123; our frame-0 ask
    went out at 44.32, 0.39 s under the request, and stock asked again with 32
    symbols at 48.23. Granted at once, its over follows and our opening goes
    behind its f13 release (here 55.293-56.019)."""
    hs, io = answerer('2300')
    io.receiving = False
    keyed = live(hs, audio('fix2b53-cq2300-answer'), 35.3, lambda hs: hs.send(b'de W9SSJ <R+28>\r'),
                 clock=Clock)
    assert not [t for t in keyed if 44.1 < t < 45.12], (keyed, io.logs)
    grant = M.synth_tone_pairs(F.turn_grant(KESTREL, '2300'))
    granted = [t for t, burst in zip(keyed, io.sent) if np.array_equal(burst, grant)]
    assert granted and 0 <= granted[0] - 45.123 <= .4, (keyed, io.logs)
    assert len(b''.join(io.host)) == 85, io.logs
    over = [t for t, burst in zip(keyed, io.sent) if len(burst) > 3 * M.FS]
    assert over and 0 <= over[0] - 56.019 <= .3, (over, io.logs[-6:])
    assert carries(io, '2300', phy.host_level('2300', hs._tx_pending[2]), b'de W9SSJ <R+28>\r')


@pytest.mark.parametrize('bw', BWS)
def test_the_answerers_close_asks_with_the_32_symbol_request_and_closes_on_the_callers_frame_1(bw):
    hs, io = delivered(bw)
    feed(hs, audio(f's{bw}-caller-f1'))
    feed(hs, audio(f's{bw}-caller-poll'))
    assert hs.turn == A._TURN_PEER
    hs.disconnect(acquire=True)
    assert hs.state == A.VaraState.CONNECTED
    assert keyed_burst(io, STOCK, F.SESSION_TURN_REQUEST_RESPONDER, bw), io.logs[-3:]
    feed(hs, audio(f's{bw}-f1-go-ahead'))
    assert hs.state == A.VaraState.DISCONNECTED, io.logs[-3:]
    assert keyed_burst(io, KESTREL, F.SESSION_DISCONNECT_REQ_RESPONDER, bw), io.logs[-3:]


# -- the host sees the close and answers the next call -----------------------

class Transport(A.VaraIO):
    def __init__(self):
        self.sent = []

    def tx(self, samples):
        self.sent.append(np.asarray(samples, float))


class Wire:
    def __init__(self):
        self.data = bytearray()

    def sendall(self, blob):
        self.data.extend(blob)


def host(bw):
    cmd, data, transport = Wire(), Wire(), Transport()
    modem = VaraHostModem(transport)
    session = HostSession(cmd, data, modem)
    modem.start(session)
    modem.poll()
    for line in ('PUBLIC ON', 'CWID ON', 'COMPRESSION TEXT', f'MYCALL {KESTREL}', 'CHAT ON', f'BW{bw}'):
        session._dispatch(line)
        modem.poll()
    return session, modem, cmd, transport


def test_the_callers_close_reaches_the_host_and_the_next_call_is_answered():
    session, modem, cmd, transport = host('2300')
    for name in ('s2300-request', 's2300-link-setup', 's2300-f20'):
        feed(modem.station, audio(name))
        modem.poll()
    assert b'CONNECTED WI9YZA W9SSJ 2300' in cmd.data
    feed(modem.station, audio('s2300-close'))
    modem.poll()
    assert b'DISCONNECTED' in cmd.data
    sent = len(transport.sent)
    feed(modem.station, audio('s2300-request-2'))
    modem.poll()
    assert modem.station.state == A.VaraState.CONNECTING and len(transport.sent) == sent + 1


# -- the caller: kestrel calls stock -----------------------------------------

def caller(bw='2300'):
    io = IO()
    hs = A.VaraStationHandshake([KESTREL], io, bw=bw, high_speed=True, compression='TEXT', chat=True)
    assert hs.originate(STOCK, KESTREL)
    feed(hs, audio('cq2300-chat-answer'))
    feed(hs, audio('cq2300-ack-f1'))
    assert hs.state == A.VaraState.CONNECTED, io.logs
    return hs, io


def test_the_callers_poll_draws_stocks_receiver_close_and_ident_at_2300():
    """cq2300-stock-cqs: stock's 8-symbol close behind our poll went unread
    and the link was closed 73.9 s late, across stock's next call."""
    hs, io = caller()
    feed(hs, audio('cq2300-data'))
    assert b'text from S on the slot\n' in b''.join(io.host), io.logs[-3:]
    feed(hs, audio('cq2300-grant-f13-2'))
    assert hs.turn == A._TURN_OURS and hs.idle_poll_due_in() is not None, io.logs[-3:]
    Clock.t += hs.idle_poll_due_in()
    hs.idle_keepalive()
    assert keyed_pairs(io, F.control_bursts(KESTREL, '2300', 1)[0]), io.logs[-3:]
    feed(hs, audio('cq2300-close-behind-poll'))
    assert hs.state == A.VaraState.DISCONNECTED, io.logs[-3:]


def holding_caller():
    """Stock's L5 reply read, its f13 hand-over taken, and our poll keyed."""
    hs, io = caller()
    feed(hs, audio('cq2300-data'))
    feed(hs, audio('cq2300-grant-f13-2'))
    Clock.t += hs.idle_poll_due_in()
    hs.idle_keepalive()
    assert keyed_pairs(io, F.control_bursts(KESTREL, '2300', 1)[0]), io.logs[-3:]
    return hs, io


def stream(hs, x):
    for at in range(0, len(x), 1024):
        Clock.t += len(x[at:at + 1024]) / M.FS
        hs.on_rx_stream(x[at:at + 1024])


def test_the_callers_close_ask_waits_out_the_answer_to_its_poll():
    """Stock keys nothing across its peer: the ask that opens a close from a
    held turn goes behind the answer our poll drew."""
    hs, io = holding_caller()
    answer = M.synth_burst(STOCK, F.for_bw(F.SESSION_IDLE_RESPONSE, '2300'))
    half = len(answer) // 2
    stream(hs, np.r_[np.zeros(int(.1 * M.FS)), answer[:half]])
    sent = len(io.sent)
    hs.disconnect(acquire=True)
    assert len(io.sent) == sent, io.logs[-3:]
    stream(hs, answer[half:])
    assert len(io.sent) == sent, io.logs[-3:]
    stream(hs, np.zeros(M.FS // 2))
    assert len(io.sent) == sent + 1, io.logs[-3:]
    assert keyed_pairs(io, F.control_bursts(KESTREL, '2300', 0)[0]), io.logs[-3:]


def test_a_write_behind_our_final_answer_waits_for_stocks_hand_over():
    """fix2b53 cq2300-kestrel-cqs 118.4, the QSY'd contact: stock's over
    answered, the opening queued behind the answer, and our 17-symbol ask keyed
    0.06 s behind it, under stock's 11-symbol hand-over 0.11 s behind it; 10.9 s
    to the re-ask. The hand-over carries the turn, and the opening goes out in it."""
    hs, io = caller()
    feed(hs, audio('fix2b53-cq2300-qsy-over'))
    assert b''.join(io.host).endswith(b'<QSYR>'), io.logs[-3:]
    assert keyed_pairs(io, F.control_bursts(KESTREL, '2300', 0)[0]), io.logs[-3:]
    sent = len(io.sent)
    hs.send(b'de W9SSJ <R+59>\r')
    assert len(io.sent) == sent, io.logs[-3:]
    feed(hs, audio('fix2b53-cq2300-qsy-idle'))
    keyed = io.sent[sent:]
    assert keyed and len(keyed[0]) > 3 * M.FS, io.logs[-3:]
    reads = (phy.decode_level(keyed[0], '2300', level) for level in range(1, 10))
    read = next((r for r in reads if r is not None), None)
    assert read is not None and phy.vara_payload(
        read.frames[0], caller=KESTREL, body_len=len(read.frames[0])).startswith(b'de W9SSJ <R+59>\r')


def test_a_wide_caller_asks_behind_stocks_poll_not_behind_a_bw500_frame():
    """Holding the turn empty at BW2300 and BW2750 stock's answerer polls with
    frame 1 alone (s2s-2300, s2s-2750): 1078/63 is BW500's."""
    hs, io = holding_caller()
    feed(hs, audio('cq2300-turn-request-responder'))
    hs.send(b'a line for the answerer\r')
    asked = len(io.sent)
    feed(hs, M.synth_burst(KESTREL, F.for_bw(F.SESSION_TURN_IDLE_RESPONDER, '2300')))
    assert len(io.sent) == asked, io.logs[-3:]


def recorded_tones(x, tones, bw):
    """The recorded stock burst carries exactly these tones."""
    return any(list(M.demod_tones(x[at:], len(tones), M.band_for(bw))) == list(tones)
               for at in range(0, 9600, 64))


def recorded_pairs(x, want, bw):
    want = [tuple(sorted(p)) for p in want]
    return any([tuple(sorted(p)) for p in M.demod_tone_pairs(x[at:], len(want), M.band_for(bw))]
               == want for at in range(0, 9600, 64))


@pytest.mark.parametrize('bw', BWS)
def test_a_caller_holding_the_turn_asks_with_frame_0_and_closes_in_the_drained_grant(bw):
    """s2s-2300 132.89 and s2s-2750 132.83: stock's caller, holding an empty
    turn, keyed frame 0 on DISCONNECT, the answerer granted with its drained
    frame, and the 32-symbol close went 0.09 s behind it."""
    io = IO()
    hs = A.VaraStationHandshake([STOCK], io, bw=bw, high_speed=True, compression='TEXT', chat=True)
    hs.role, hs.caller, hs.called = 'initiator', STOCK, KESTREL
    hs.state, hs.step, hs.turn = A.VaraState.CONNECTED, A._I_CONNECTED, A._TURN_OURS
    hs._hold_idle()
    hs.disconnect(acquire=True)
    ask = F.control_bursts(STOCK, bw, 0)[0]
    assert recorded_pairs(audio(f's{bw}-caller-close-ask'), ask, bw)
    assert keyed_pairs(io, ask) and hs.state == A.VaraState.CONNECTED, io.logs[-3:]
    sent = len(io.sent)
    grant = audio(f's{bw}-close-grant')
    stream(hs, grant[:-int(.2 * M.FS)])
    assert len(io.sent) == sent, io.logs[-3:]
    stream(hs, np.r_[grant[-int(.2 * M.FS):], np.zeros(M.FS // 2)])
    close = F.handshake_tones(KESTREL, F.for_bw(F.SESSION_DISCONNECT_REQ, bw))
    assert recorded_tones(audio(f's{bw}-close'), close, bw)
    assert keyed_burst(io, KESTREL, F.SESSION_DISCONNECT_REQ, bw), io.logs[-3:]
    assert hs.state == A.VaraState.DISCONNECTED


def test_a_caller_holding_the_turn_does_not_poll_across_its_close_ask():
    io = IO()
    hs = A.VaraStationHandshake([STOCK], io, bw='2300', high_speed=True, compression='TEXT', chat=True)
    hs.role, hs.caller, hs.called = 'initiator', STOCK, KESTREL
    hs.state, hs.step, hs.turn = A.VaraState.CONNECTED, A._I_CONNECTED, A._TURN_OURS
    hs._hold_idle()
    hs.disconnect(acquire=True)
    sent = len(io.sent)
    for _ in range(30):
        Clock.t += .5
        hs.idle_keepalive()
    assert len(io.sent) == sent, io.logs[-3:]
