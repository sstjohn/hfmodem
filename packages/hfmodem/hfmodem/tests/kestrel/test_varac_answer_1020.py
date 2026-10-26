# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock VARA calls kestrel at BW500 with chat on: bench 3 gate 1, 2026-10-20."""
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
from hfmodem.kestrel.vara import compression as C, vara_arq as A, vara_frames as F, vara_mfsk as M

HERE = Path(__file__).with_name('fixtures')
GATE = evidence.CORPUS / 'varac-answer-1020'
CHAT = evidence.CORPUS / 'bw500-chat-answer-1018'
pytestmark = pytest.mark.skipif(not (GATE.is_dir() and CHAT.is_dir()),
                                reason=f'recordings not present ({evidence.CORPUS})')
STOCK, KESTREL = 'WI9YZA', 'W9SSJ'


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


def audio(directory, name):
    prov = json.loads((HERE / directory.name / 'provenance.json').read_text())[name]
    # The sidecar ships; these full bench waveforms are optional external corpus.
    path = evidence.require_input(directory / prov.get('file', f'{name}.flac'))
    rate, x = wavfile.read(source(path))
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


def recorded(x, tones):
    """The recorded stock burst carries exactly these tones."""
    band = M.band_for('500')
    return any(list(M.demod_tones(x[at:], len(tones), band)) == list(tones)
               for at in range(0, 9600, 64))


def keyed_pairs(io, want):
    return np.array_equal(io.sent[-1], M.synth_tone_pairs(want))


def keyed_burst(io, call, kind):
    return np.array_equal(io.sent[-1], M.synth_burst(call, F.for_bw(kind, '500')))


def answerer(mycall, chat):
    io = IO()
    hs = A.VaraStationHandshake([mycall], io, bw='500', high_speed=True,
                                compression='TEXT', chat=chat)
    hs.listen(True)
    return hs, io


PLAIN = ['a2-request', 'a2-link-setup', 'a2-confirm', 'a2-confirm-32', 'a2-turn-request',
         'a2-data-1', 'a2-data-2', 'a2-release']


def recorded_plain_session():
    """Stock's own keyings, up to the caller handing over its turn."""
    hs, io = answerer(KESTREL, chat=False)
    for name in PLAIN:
        feed(hs, audio(GATE, name))
    return hs, io


def wire_text(text):
    return C.encode_text(text)


# -- failure 1: BW500 sent after CHAT ON ------------------------------------

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


def host():
    cmd, data, transport = Wire(), Wire(), Transport()
    modem = VaraHostModem(transport)
    session = HostSession(cmd, data, modem)
    modem.start(session)
    modem.poll()
    return session, modem, cmd, transport


def command(session, modem, *lines):
    for line in lines:
        session._dispatch(line)
        modem.poll()


def test_bw500_after_chat_on_moves_the_listener_to_500():
    session, modem, _, transport = host()
    command(session, modem, 'PUBLIC ON', 'CWID ON', 'COMPRESSION TEXT', f'MYCALL {KESTREL}',
            'CHAT ON', 'BW500')
    assert modem.station.bw == '500' and modem.station.state == A.VaraState.LISTENING
    feed(modem.station, audio(GATE, 'a1-request'))
    assert modem.station.state == A.VaraState.CONNECTING, modem.station.io
    assert len(transport.sent) == 1


# -- failure 2: the answer profile a CHAT ON stock answerer keys ------------

def recorded_chat_session():
    hs, io = answerer(STOCK, chat=True)
    feed(hs, audio(CHAT, 'chat-w9-on-callerclose-00-a2b'))
    return hs, io


def test_chat_on_answers_the_stock_caller_with_stocks_chat_profile():
    hs, io = recorded_chat_session()
    assert keyed_burst(io, STOCK, F.SESSION_CHAT_SETUP_RESPONSE), io.logs
    assert recorded(audio(CHAT, 'chat-w9-on-callerclose-01-b2a'),
                    F.handshake_tones(STOCK, F.for_bw(F.SESSION_CHAT_SETUP_RESPONSE, '500')))

    feed(hs, audio(CHAT, 'chat-w9-on-callerclose-02-a2b'))
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert keyed_pairs(io, F.control_bursts(KESTREL, '500', 1)[1])

    sent = len(io.sent)
    feed(hs, audio(CHAT, 'chat-w9-on-callerclose-04-a2b'))
    assert len(io.sent) == sent and hs.turn == A._TURN_OURS, io.logs
    feed(hs, audio(CHAT, 'chat-w9-on-callerclose-05-a2b'))
    assert keyed_pairs(io, F.turn_grant(STOCK, '500')), io.logs

    feed(hs, audio(CHAT, 'chat-w9-on-callerclose-07-a2b'))
    assert b''.join(io.host) == b'hi from A\r'
    assert keyed_pairs(io, F.control_bursts(KESTREL, '500', 1)[1]), io.logs

    feed(hs, audio(CHAT, 'chat-w9-on-callerclose-04-a2b'))
    hs.send(b'hi from B\r')
    assert keyed_pairs(io, F.control_bursts(KESTREL, '500', 0)[1]), io.logs
    feed(hs, audio(CHAT, 'chat-w9-on-callerclose-11-a2b'))
    read = phy.decode_level(io.sent[-1], '500', 5)
    assert read is not None, io.logs
    assert phy.vara_payload(read.frames[0], caller=KESTREL,
                            body_len=len(read.frames[0])) == b'hi from B\r'

    feed(hs, audio(CHAT, 'chat-w9-on-callerclose-13-a2b'))
    assert keyed_burst(io, STOCK, F.SESSION_TURN_IDLE_RESPONDER_SHORT), io.logs
    assert hs.turn == A._TURN_PEER and hs.payload_ledger.pending == 0


def test_chat_on_holds_the_empty_turn_on_stocks_recorded_poll_cadence():
    hs, io = recorded_chat_session()
    for name in ('02', '04'):
        feed(hs, audio(CHAT, f'chat-w9-on-callerclose-{name}-a2b'))
    reply = audio(CHAT, 'idle12-a2b-006')
    polls = []
    for _ in range(60):
        Clock.t += hs.idle_poll_due_in()
        hs.idle_keepalive()
        polls.append(Clock.t)
        assert keyed_pairs(io, F.control_bursts(KESTREL, '500', 1)[1]), io.logs
        feed(hs, reply)
    assert hs.state == A.VaraState.CONNECTED and len(polls) == 60
    assert np.allclose(np.diff(polls), A._IDLE_POLL_S, atol=.6)


def test_the_doubled_line_was_on_the_wire_and_reaches_the_host_once():
    hs, io = recorded_plain_session()
    line = b'hello from stock, gate 1 session 1\r'
    assert io.host == [line + line[:31], b'n 1\r']
    assert wire_text(line + line[:31]) == phy.vara_payload(
        phy.decode_level(audio(GATE, 'a2-data-1'), '500', 4).frames[0], caller=STOCK,
        body_len=44)


# -- failure 3: the caller's handover, the held turn and a dead link --------

def test_recorded_plain_handover_draws_control_one_and_a_held_turn():
    hs, io = recorded_plain_session()
    assert keyed_pairs(io, F.control_bursts(STOCK, '500', 1)[1]), io.logs
    assert hs.turn == A._TURN_OURS and hs.idle_poll_due_in() is not None
    hs.send(b'reply from kestrel, gate 1 session 1\r')
    assert keyed_pairs(io, F.control_bursts(STOCK, '500', 0)[1]), io.logs
    assert hs._tx_pending is None
    feed(hs, audio(GATE, 'a2-release'))
    assert hs._tx_pending is not None and hs._tx_pending[2] == phy.level_key('500', 6)


def test_unanswered_over_is_queried_then_the_dead_link_closes():
    hs, io = recorded_plain_session()
    hs.send(b'reply from kestrel, gate 1 session 1\r')
    feed(hs, audio(GATE, 'a2-release'))
    queries = 0
    while hs.state == A.VaraState.CONNECTED and Clock.t < 1200:
        due = hs.intermediate_query_due_in()
        if due is not None and due <= 0:
            hs.query_intermediate_answer()
            queries += keyed_pairs(io, F.answer_query(KESTREL, '500'))
        else:
            hs.idle_keepalive()
        Clock.t += .25
    assert queries >= 1 and hs.state == A.VaraState.DISCONNECTED, io.logs[-5:]
    assert Clock.t - 1000 < 60 + A._MAX_WITHOUT_PROGRESS * A._IDLE_POLL_S


@pytest.mark.parametrize('chat', [False, True])
def test_held_turn_with_a_silent_caller_closes(chat):
    hs, io = recorded_plain_session()
    hs.chat = chat
    while hs.state == A.VaraState.CONNECTED and Clock.t < 1300:
        Clock.t += 1
        hs.idle_keepalive()
    assert hs.state == A.VaraState.DISCONNECTED, io.logs[-5:]
    assert Clock.t - 1000 < 40 + A._MAX_WITHOUT_PROGRESS * A._IDLE_POLL_S


def test_dead_link_reaches_the_host_and_the_next_call_is_answered():
    session, modem, cmd, transport = host()
    command(session, modem, f'MYCALL {KESTREL}', 'COMPRESSION TEXT', 'BW500', 'LISTEN ON')
    for name in PLAIN:
        feed(modem.station, audio(GATE, name))
        modem.poll()
    assert b'CONNECTED WI9YZA W9SSJ 500' in cmd.data
    modem.transmit(b'reply from kestrel, gate 1 session 1\r')
    modem.poll()
    command(session, modem, 'DISCONNECT')
    while b'DISCONNECTED' not in cmd.data and Clock.t < 1300:
        Clock.t += 1
        modem.station.idle_keepalive()
        modem.poll()
    assert b'DISCONNECTED' in cmd.data
    sent = len(transport.sent)
    feed(modem.station, audio(GATE, 'a2-request-2'))
    modem.poll()
    assert modem.station.state == A.VaraState.CONNECTING and len(transport.sent) == sent + 1


# -- bench 2: the connect answer lands where stock's own does ---------------

@pytest.mark.parametrize('name,kind', [('sk-request', F.connect_request('500')),
                                       ('sk-retry', F.connect_request_retry('500'))])
@pytest.mark.parametrize('chunk,phase', [(1024, 0), (4096, 0), (4096, 1024), (4096, 2048),
                                         (4096, 3072)])
def test_recorded_request_is_answered_in_the_block_holding_its_last_symbol(name, kind, chunk, phase):
    """Stock retries 1.29 s after its request; kestrel's 0.98 s answer has to
    start where stock's does (+0.10-0.13), so the request is taken as soon as
    its last symbol is in, not a symbol later."""
    hs, io = answerer(STOCK, chat=False)
    x = audio(GATE, name)
    kind = F.for_bw(kind, '500')
    tones = np.asarray(F.handshake_tones(STOCK, kind))
    _heard, at = A._payload_fit(x, kind, tones, None, M.band_for('500'))
    end = at + A._span(kind)
    edges = [phase] + list(range(phase + chunk, len(x), chunk))
    taken = None
    for lo, hi in zip([0] + edges, edges):
        hs.on_rx_stream(x[lo:hi])
        if io.sent:
            taken = (hi - end) / M.FS
            break
    assert taken is not None, io.logs
    assert 0 <= taken <= chunk / M.FS + .001, taken


# -- the answerer's own close, and its release forms behind a level-6 over ----

def recorded_pairs(x, want):
    """The recorded stock two-tone burst carries exactly these pairs."""
    band, want = M.band_for('500'), [tuple(sorted(p)) for p in want]
    return any([tuple(sorted(p)) for p in M.demod_tone_pairs(x[at:], len(want), band)] == want
               for at in range(0, 9600, 64))


def chat_session_after_our_release():
    """CHAT ON, stock's caller-close run up to our f20 release behind L5."""
    hs, io = recorded_chat_session()
    for name in ('02', '04', '05', '07', '04'):
        feed(hs, audio(CHAT, f'chat-w9-on-callerclose-{name}-a2b'))
    hs.send(b'hi from B\r')
    feed(hs, audio(CHAT, 'chat-w9-on-callerclose-11-a2b'))
    feed(hs, audio(CHAT, 'chat-w9-on-callerclose-13-a2b'))
    assert keyed_burst(io, STOCK, F.SESSION_TURN_IDLE_RESPONDER_SHORT), io.logs
    return hs, io


def test_answerer_close_behind_its_17_symbol_release_asks_with_1079_then_closes():
    """P0 item7 chat-called-disc: 1079 at 48.760, the caller's grant at 49.630,
    the close at 50.175."""
    hs, io = chat_session_after_our_release()
    hs.disconnect(acquire=True)
    assert hs.state == A.VaraState.CONNECTED
    assert keyed_burst(io, KESTREL, F.SESSION_TURN_REQUEST_RESPONDER_SHORT), io.logs
    assert recorded(audio(CHAT, 'item-chat-called-disc-15-b2a'), F.handshake_tones(
        KESTREL, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER_SHORT, '500')))
    feed(hs, audio(CHAT, 'item-chat-called-disc-16-a2b'))
    assert hs.state == A.VaraState.DISCONNECTED, io.logs
    assert keyed_burst(io, STOCK, F.SESSION_DISCONNECT_REQ_RESPONDER)
    assert recorded(audio(CHAT, 'item-chat-called-disc-17-b2a'), F.handshake_tones(
        STOCK, F.for_bw(F.SESSION_DISCONNECT_REQ_RESPONDER, '500')))


def plain_link(*keyed):
    """P0 item7 plain-called-disc, CHAT OFF, through the caller's ``keyed``."""
    io = IO()
    hs = A.VaraStationHandshake([STOCK], io, bw='500', high_speed=True, compression='TEXT')
    hs.role, hs.caller, hs.called = 'responder', KESTREL, STOCK
    hs.state, hs.turn = A.VaraState.CONNECTED, A._TURN_PEER
    hs._link_high_speed, hs._awaiting_caller_confirm = True, True
    hs._lead = A.VB6Rnd(F.lead_state(KESTREL, True, '500'))
    hs._start_compression('TEXT')
    hs._progressed()
    for n in keyed:
        feed(hs, audio(CHAT, f'item-plain-called-disc-{n}-a2b'))
    return hs, io


def plain_answerer():
    hs, io = plain_link('04', '06', '08', '10', '12', '14')
    assert b''.join(io.host) == b'hi from A\r', io.logs
    hs.send(b'hi from B\r')
    feed(hs, audio(CHAT, 'item-plain-called-disc-16-a2b'))
    assert hs._tx_pending is not None and hs._tx_pending[2] == phy.level_key('500', 6), io.logs
    return hs, io


def test_release_behind_a_level_6_over_is_stocks_11_symbol_form_and_the_close_polls():
    """P0 item7 plain-called-disc: our L6 'hi from B', the caller's frame 0
    draws the grant (40.455), its frame 1 the pair-136 idle form, and the
    close is asked for with control frame 0 (49.055) and keyed on the
    caller's grant (49.645)."""
    hs, io = plain_answerer()
    feed(hs, audio(CHAT, 'item-plain-called-disc-18-a2b'))
    assert keyed_pairs(io, F.turn_grant(STOCK, '500')), io.logs
    assert recorded_pairs(audio(CHAT, 'item-plain-called-disc-19-b2a'), F.turn_grant(STOCK, '500'))
    feed(hs, M.synth_tone_pairs(F.control_bursts(KESTREL, '500', 1)[0]))
    assert keyed_pairs(io, F.responder_post_delivery_idle(STOCK, '500')), io.logs
    assert recorded_pairs(audio(CHAT, 'chat-clean-40-b2a'), F.responder_post_delivery_idle(STOCK, '500'))
    hs.disconnect(acquire=True)
    assert keyed_pairs(io, F.control_bursts(KESTREL, '500', 0)[1]), io.logs
    assert recorded_pairs(audio(CHAT, 'item-plain-called-disc-22-b2a'), F.control_bursts(KESTREL, '500', 0)[1])
    feed(hs, audio(CHAT, 'item-plain-called-disc-23-a2b'))
    assert hs.state == A.VaraState.DISCONNECTED, io.logs
    assert keyed_burst(io, STOCK, F.SESSION_DISCONNECT_REQ_RESPONDER)


def test_a_plain_answerer_asks_straight_behind_the_f20():
    """CHAT OFF, the Pat and Winlink answerer: no caller has been heard asking
    the turn back behind its f20 there, and the queued reply asks at once."""
    hs, io = plain_link('04', '06', '08', '10', '12')
    hs.send(b'hi from B\r')
    sent = len(io.sent)
    feed(hs, audio(CHAT, 'item-plain-called-disc-14-a2b'))
    assert len(io.sent) == sent + 1, io.logs[-3:]
    assert keyed_pairs(io, F.control_bursts(KESTREL, '500', 0)[1]), io.logs[-3:]


# -- bench 3 rerun r3: a host write while we hold the empty turn --------------

# Stock's bursts in the cut, b2a seconds: CR, link-setup, f20, 851, DATA, f20.
R3_BURSTS = [(3.065, 4.820), (5.700, 9.985), (10.905, 11.635), (13.105, 13.835),
             (14.625, 18.910), (19.835, 20.565)]


def r3_holding():
    """r3 gate 1, CHAT ON: stock's line delivered, its f20, our first poll
    (30.88) answered by its keepalive-b."""
    hs, io = answerer(KESTREL, chat=True)
    x = audio(GATE, 'r3-opening')
    for a, b in R3_BURSTS:
        feed(hs, x[int((a - .1 - 2.6) * M.FS):int((b + .1 - 2.6) * M.FS)])
    assert io.host == [b'hello from stock, gate 1 session 1\r'], io.logs
    assert hs.turn == A._TURN_OURS and hs.idle_poll_due_in() is not None, io.logs
    Clock.t += hs.idle_poll_due_in()
    hs.idle_keepalive()
    assert keyed_pairs(io, F.control_bursts(STOCK, '500', 1)[1]), io.logs
    feed(hs, audio(GATE, 'r3-poll-answer'))
    return hs, io


def test_a_reply_written_while_holding_asks_with_frame_0_and_goes_out_on_stocks_drained():
    """43.02: our frame 0, then stock's 32-symbol session-drained (60/807) at
    43.55, the go-ahead a stock answerer sends its DATA on (item2, 773.09 and
    773.67). Kestrel took it for noise, asked six more times and closed with
    the 37 bytes queued."""
    hs, io = r3_holding()
    reply = b'reply from kestrel, gate 1 session 1\r'
    hs.send(reply)
    assert keyed_pairs(io, F.control_bursts(STOCK, '500', 0)[1]), io.logs
    assert any(f'asking {STOCK} for the turn' in line for line in io.logs), io.logs
    feed(hs, audio(GATE, 'r3-ask-answer'))
    assert hs._tx_pending is not None, io.logs
    read = phy.decode_level(io.sent[-1], '500', phy.host_level('500', hs._tx_pending[2]))
    assert read is not None and phy.vara_payload(
        read.frames[0], caller=STOCK, body_len=len(read.frames[0])) == reply


# -- bench 2, 3c gate 1: the held send's latency and the answerer's close ----

def b2(t):
    return audio(GATE, f'b2-{t:07.2f}')


def b2_session(opening):
    hs, io = answerer(KESTREL, chat=True)
    for t in opening:
        feed(hs, b2(t))
    return hs, io


def poll(hs, io):
    Clock.t += hs.idle_poll_due_in()
    hs.idle_keepalive()
    assert keyed_pairs(io, F.control_bursts(STOCK, '500', 1)[1]), io.logs


def test_a_write_just_after_our_poll_asks_behind_the_polls_answer():
    """42.69 our poll, 42.9 the host's write, 43.22 our frame 0 on top of
    stock's keepalive-b answer (43.235, read as a 26-symbol burst), and the
    line waited for the next poll: 19 s. Stock asks behind the answer
    (item2: answer ends 772.745, frame 0 at 773.09)."""
    hs, io = b2_session((3.05, 5.70, 10.89, 13.07, 14.44, 19.62))
    assert io.host == [b'hello from stock, gate 1 session 1\r'], io.logs
    poll(hs, io)
    feed(hs, b2(31.13))
    poll(hs, io)
    sent = len(io.sent)
    Clock.t += .2
    hs.send(b'reply from kestrel, gate 1 session 1\r')
    assert len(io.sent) == sent, io.logs
    feed(hs, b2(43.23))
    assert len(io.sent) == sent + 1 and keyed_pairs(io, F.control_bursts(STOCK, '500', 0)[1]), io.logs
    feed(hs, b2(55.84))
    assert hs._tx_pending is not None, io.logs


def test_the_answerers_close_ask_goes_into_the_callers_poll():
    """Session 2: our close's frame 0 (974.62) landed on stock's poll
    (974.51); stock polled again at 976.81 and we answered with an idle
    response, asked 2.3 s after the first, and closed at 980.10. The caller
    listens behind its poll, so the ask goes there."""
    hs, io = b2_session((904.75, 907.31, 912.50, 914.67, 916.09, 921.25))
    poll(hs, io)
    feed(hs, b2(932.75))
    poll(hs, io)
    hs.send(b'kestrel closes next\r')
    feed(hs, b2(944.85))
    feed(hs, b2(957.45))
    assert hs._tx_pending is not None, io.logs
    feed(hs, b2(963.23))
    assert keyed_pairs(io, F.responder_post_delivery_idle(KESTREL, '500')), io.logs
    Clock.t += 11.
    hs.disconnect(acquire=True)
    assert keyed_pairs(io, F.control_bursts(STOCK, '500', 0)[1]), io.logs
    Clock.t += 1.8
    feed(hs, b2(976.81))
    assert keyed_pairs(io, F.control_bursts(STOCK, '500', 0)[1]), io.logs
    assert hs.state == A.VaraState.CONNECTED
    feed(hs, b2(979.46))
    assert hs.state == A.VaraState.DISCONNECTED, io.logs
    assert keyed_burst(io, KESTREL, F.SESSION_DISCONNECT_REQ_RESPONDER)


# -- bench 3 r3c gate 3: the end of a high-level delivery ---------------------

def high_level_answerer(expect, compression='TEXT'):
    io = IO()
    hs = A.VaraStationHandshake([KESTREL], io, bw='500', high_speed=True,
                                compression=compression, chat=True)
    hs.role, hs.caller, hs.called = 'responder', STOCK, KESTREL
    hs.state, hs.turn = A.VaraState.CONNECTED, A._TURN_PEER
    hs._chat_profile, hs._link_high_speed = True, True
    hs._lead = A.VB6Rnd(F.lead_state(STOCK, True, '500'))
    hs._start_compression(compression)
    hs._expect_level, hs._peer_over, hs._peer_delivery_open = expect, 5, True
    hs._progressed()
    return hs, io


def stream_from(hs, x, t0, guard=None):
    """The stream as the transport hands it over; with ``guard``, nothing
    under our own keying or for ``guard`` seconds behind it."""
    keyed, deaf = [], 0.
    for at in range(0, len(x), 1024):
        Clock.t = t0 + (at + 1024) / M.FS
        if guard is not None and Clock.t < deaf:
            continue
        sent = len(hs.io.sent)
        hs.on_rx_stream(x[at:at + 1024])
        if len(hs.io.sent) > sent:
            keyed.append(Clock.t)
            deaf = Clock.t + len(hs.io.sent[-1]) / M.FS + (guard or 0.)
    return keyed


def test_a_single_frame_at_the_top_level_is_answered_when_the_burst_ends():
    """35.935-39.755: stock's last over, one L13 frame. Kestrel waited for a
    second frame and answered at 41.015; stock's own poll at 41.395 covered it."""
    hs, io = high_level_answerer(13)
    Clock.t = 35.84
    hs._key(False)
    hs._lv_armed = hs._lv_owed = hs._lv_prompt = True
    keyed = stream_from(hs, audio(GATE, 'r3c-l13-close'), 35.84)
    assert keyed and keyed[0] - 39.755 < .3, (keyed, io.logs)
    assert [len(h) for h in io.host] == [673]
    assert keyed_pairs(io, F.control_bursts(STOCK, '500', 1)[1])


def test_stocks_queries_behind_an_unheard_last_answer_draw_it_again():
    """41.395 the caller's 8-symbol over-poll, 43.51 its 16-symbol query, then
    the 32-symbol query every 3.1 s for 56 s, none answered."""
    hs, io = high_level_answerer(13)
    Clock.t = 35.84
    hs._key(False)
    hs._lv_armed = hs._lv_owed = hs._lv_prompt = True
    stream_from(hs, audio(GATE, 'r3c-l13-close'), 35.84)
    sent = len(io.sent)
    keyed = stream_from(hs, audio(GATE, 'r3c-queries'), 41.30)
    assert len(keyed) >= 3, (keyed, io.logs)
    assert all(np.array_equal(x, M.synth_tone_pairs(F.control_bursts(STOCK, '500', 1)[1]))
               for x in io.sent[sent:])


def test_stocks_11_symbol_handover_behind_a_high_level_delivery_holds_the_turn():
    """chat-clean, 2026-10-18: the caller's L12 last over, our frame 1, and
    its 11-symbol hand-over (57.075, pairs 137-143 of its answer stream) in
    place of the f20."""
    hs, io = high_level_answerer(12, 'OFF')
    hs.caller, hs.called = KESTREL, STOCK
    hs.mycalls = [STOCK]
    hs._lead = A.VB6Rnd(F.lead_state(KESTREL, True, '500'))
    hs._key(False)
    hs._lv_armed = hs._lv_owed = hs._lv_prompt = True
    feed(hs, audio(CHAT, 'chat-clean-21-a2b'))
    assert io.host, io.logs
    assert keyed_pairs(io, F.control_bursts(KESTREL, '500', 1)[1]), io.logs
    assert recorded_pairs(audio(CHAT, 'chat-clean-22-b2a'), F.control_bursts(KESTREL, '500', 1)[1])
    feed(hs, audio(CHAT, 'chat-clean-23-a2b'))
    assert hs.turn == A._TURN_OURS and hs.idle_poll_due_in() is not None, io.logs
    hs.send(b'hi from B\r')
    assert keyed_pairs(io, F.control_bursts(KESTREL, '500', 0)[1]), io.logs


def test_clear_on_a_disconnected_modem_with_bytes_queued_fails():
    """r3c gate 3: 2720 bytes written after the link closed, CLEANTXBUFFER at
    240.2; kestrel said OK, stock answers FAILED there."""
    session, modem, cmd, _ = host()
    command(session, modem, f'MYCALL {KESTREL}', 'BW500', 'CHAT ON')
    modem.transmit(b'line 0000 the quick brown fox\r' * 90)
    modem.poll()
    cmd.data.clear()
    command(session, modem, 'CLEANTXBUFFER')
    assert cmd.data.startswith(b'CLEANTXBUFFER FAILED \r'), cmd.data


# -- bench 3 r4 gate 3: the hand-over behind an L13 close, and the timeout ----

def test_stocks_hand_over_behind_an_l13_close_holds_the_turn_under_the_echo_guard():
    """453.515-457.335 stock's L13 last over, our frame 1, and 0.09 s behind it
    its 11-symbol hand-over at pair phase 1 of frame 19 (458.205), which went
    unread for 60 s; chat-clean's L12 close used phase 4."""
    hs, io = high_level_answerer(13)
    Clock.t = 453.40
    hs._key(False)
    hs._lv_armed = hs._lv_owed = hs._lv_prompt = True
    keyed = stream_from(hs, audio(GATE, 'r4-l13-handover'), 453.40, guard=.1)
    assert keyed and keyed[0] - 457.335 < .3, (keyed, io.logs)
    assert hs.turn == A._TURN_OURS and hs.idle_poll_due_in() is not None, io.logs


def test_the_callers_timeout_ident_closes_the_link():
    """518.205: stock gave up and keyed only its CW ident; kestrel read
    nothing and closed on its own clock 15 s later."""
    hs, io = high_level_answerer(13)
    hs.turn = A._TURN_OURS
    feed(hs, audio(GATE, 'r4-timeout-ident'))
    assert hs.state == A.VaraState.DISCONNECTED, io.logs
    assert not io.sent


def test_a_caller_holding_the_turn_asks_with_frame_0_and_closes_in_the_grant():
    """P0 item7 caller-disc, 48.77-49.91: stock's caller, holding an empty turn
    behind the callee's 11-symbol hand-over, keyed frame 0 0.07 s after its
    DISCONNECT, the callee granted with frame 13, and the close went 0.12 s
    behind it."""
    io = IO()
    hs = A.VaraStationHandshake([KESTREL], io, bw='500', high_speed=True,
                                compression='TEXT', chat=True)
    hs.role, hs.caller, hs.called = 'initiator', KESTREL, STOCK
    hs.state, hs.step, hs.turn = A.VaraState.CONNECTED, A._I_CONNECTED, A._TURN_OURS
    hs._hold_idle()
    hs.disconnect(acquire=True)
    ask = F.control_bursts(KESTREL, '500', 0)[0]
    assert recorded_pairs(audio(CHAT, 'item-chat-caller-disc-ask'), ask)
    assert keyed_pairs(io, ask) and hs.state == A.VaraState.CONNECTED, io.logs[-3:]
    feed(hs, audio(CHAT, 'item-chat-caller-disc-grant'))
    assert recorded(audio(CHAT, 'item-chat-caller-disc-close'),
                    F.handshake_tones(STOCK, F.for_bw(F.SESSION_DISCONNECT_REQ, '500')))
    assert keyed_burst(io, STOCK, F.SESSION_DISCONNECT_REQ), io.logs[-3:]
    assert hs.state == A.VaraState.DISCONNECTED


def test_nothing_is_read_as_an_over_behind_the_callers_last_one():
    """453.515-457.335 stock's L13 last over and our frame 1. A CHAT ON caller
    hands the turn back behind its last over, so nothing of its that follows,
    its hand-over unread here and its control bursts in noise, is an over of
    its: no NAK goes out."""
    hs, io = high_level_answerer(13)
    Clock.t = 453.40
    hs._key(False)
    hs._lv_armed = hs._lv_owed = hs._lv_prompt = True
    x = audio(GATE, 'r4-l13-handover')
    x[int(4.7 * M.FS):] = 0
    poll = 0.3 * M.synth_tone_pairs(F.control_bursts(STOCK, '500', 1)[0])
    x = np.r_[x, np.zeros(35 * M.FS)]
    for t in (4.8, 16.8, 28.8):
        x[int(t * M.FS):int(t * M.FS) + len(poll)] += poll
    x += 0.02 * np.random.default_rng(3).standard_normal(len(x))
    keyed = stream_from(hs, x, 453.40, guard=.1)
    assert len(keyed) == 1 and keyed[0] - 457.335 < .3, (keyed, io.logs)
    assert not any('NAK' in m for m in io.logs), io.logs


# -- bench 3 af39 CQ 500, kestrel answers: the release read 16/17 ----------

def live(hs, x, t0, on_connected, guard=.1, clock=Clock):
    """The transport's order: every block to the stream, the segmenter's
    brackets to on_rx_audio, nothing under our own keying or for ``guard``
    behind it, and the driver's tick between blocks. Returns the keying times."""
    from hfmodem.kestrel.monitor import Segmenter
    Clock = clock
    io, keyed, deaf, seg, idle_at = hs.io, [], 0., Segmenter(), None
    connected = False
    for at in range(0, len(x), 1024):
        Clock.t = t0 + (at + 1024) / M.FS
        if Clock.t < deaf:
            seg = Segmenter()
            continue
        chunk = x[at:at + 1024]
        sent = len(io.sent)
        io.receiving = bool(seg.inb)
        hs.on_rx_stream(chunk)
        for _, burst in seg.push(chunk):
            hs.on_rx_audio(np.r_[burst, np.zeros(M.FS // 4)])
        io.receiving = bool(seg.inb)
        if hs.state == A.VaraState.CONNECTED and not connected:
            connected = True
            on_connected(hs)
        if hs.state == A.VaraState.CONNECTED and not io.receiving:
            idle_at = Clock.t + 10 if idle_at is None else idle_at
            poll = hs.idle_poll_due_in()
            if poll is not None and poll <= 0 or Clock.t >= idle_at:
                hs.idle_keepalive()
                idle_at = Clock.t + 10
        if len(io.sent) > sent:
            keyed.append(Clock.t)
            deaf = Clock.t + len(io.sent[-1]) / M.FS + guard
    return keyed


def test_a_release_read_16_of_17_draws_the_over_in_the_turnaround_it_opens():
    """af39 cq500-kestrel-cqs (bench 3, 2026-10-22): the opening queued at
    CONNECTED, stock's two overs delivered, then its release 63.057-63.788
    read 16/17 with its first tone one bin high. Held for the preamble, the
    stream took the release 0.83 s late and the over went out 0.93 s behind
    it, past stock's window: ten queries unanswered, no text across, stock
    timed out at +112 s. Stock's window is the release's turnaround: r3c's
    answerer keyed its over 0.09 s behind the release and was read."""
    hs, io = answerer(KESTREL, chat=True)
    io.receiving = False
    x = audio(GATE, 'af39-cq500-answer')
    keyed = live(hs, x, 35.4, opening)
    assert [len(h) for h in io.host] == [44, 41], io.logs
    assert f'rx caller handover — {STOCK} released the turn' in io.logs, io.logs[-6:]
    over = [t for t, burst in zip(keyed, io.sent) if len(burst) > 3 * M.FS]
    assert over and 0 <= over[0] - 63.788 <= .15, (over, io.logs[-6:])
    read = phy.decode_level(io.sent[-1], '500', phy.host_level('500', hs._tx_pending[2]))
    assert read is not None and phy.vara_payload(
        read.frames[0], caller=STOCK, body_len=len(read.frames[0])).startswith(b'de W9SSJ')


def opening(hs):
    hs.send(b'de W9SSJ <R+21>\r\r')
    hs.send(b'<NAME:Saul>\r<QTH:Madison>\r<LOC:EN63bc>\r\r\r\r')


def test_the_callers_request_behind_its_f20_is_granted_not_asked_across():
    """The same cable: stock's f20 43.441-44.171 and, its client's opening
    queued at CONNECTED too, its 17-symbol request 44.465-45.195. Our frame-0
    ask went out at 44.47, 0.40 s under the request; stock asked again with 32
    symbols at 51.03 and was granted 5.8 s late. Two stocks collide the same
    way (bench 3 s2s-connect, 2026-10-22) and the caller's request wins: the
    answerer grants it and asks for its own turn behind the caller's release."""
    hs, io = answerer(KESTREL, chat=True)
    io.receiving = False
    keyed = live(hs, audio(GATE, 'af39-cq500-answer'), 35.4, opening)
    # Keyings land on block ends, and the block holding the request's last 3 ms
    # is the one the stream reads its last symbol in.
    block = 1024 / M.FS
    assert not [t for t in keyed if 44.17 < t < 45.195 - block], (keyed, io.logs)
    grant = M.synth_tone_pairs(F.turn_grant(KESTREL, '500'))
    granted = [t for t, burst in zip(keyed, io.sent) if np.array_equal(burst, grant)]
    assert granted and -block <= granted[0] - 45.195 <= .4, (keyed, io.logs)
