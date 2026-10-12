# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""BW500 as a caller, 2026-09-23/24: KB0WZI off 40 m and stock 4.9.0 on the cables.

KB0WZI acked our level-1 link-setup with a connected-ack whose opening symbols
did not reach us; the frame read whole is the ack. The rest of the session's
short frames went unread because BW500 had no stream route for them.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.tests.evidence import require_input
from hfmodem.kestrel.rx import varahf500 as RX
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK, vara_ofdm as OF
from hfmodem.tests.kestrel.test_bw500_recovery import _station

FS = MK.FS
FIXTURES = Path(__file__).with_name('fixtures') / 'bw500-oracle-0923'
#: KB0WZI's carriers as the handshake measured them off its connect-response.
KB0WZI_HZ = 4.5


class IO:
    def __init__(self):
        self.msgs, self.sent = [], []

    def key(self, on): ...
    def tx(self, samples): self.sent.append(len(samples))
    def tx_went_out(self): return True
    def pending(self): ...
    def connected(self, *a): ...
    def data(self, payload): ...
    def log(self, msg): self.msgs.append(msg)


def clip(name):
    path = FIXTURES / f'{name}.wav'
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == FS and x.dtype == np.int16
    return x.astype(float) / 32768


def awaiting_ack():
    io = IO()
    hs = VA.VaraStationHandshake(['W9SSJ'], io, bw='500')
    hs.role, hs.called, hs.caller = 'initiator', 'KB0WZI', 'W9SSJ'
    hs.state, hs.step = VA.VaraState.CONNECTING, VA._I_LINKSETUP_SENT
    hs._linksetup_tx = 1
    hs._peer_offset = KB0WZI_HZ / (FS / MK.NFFT)
    return hs, io


def test_provenance_matches_the_clips():
    prov = json.loads(require_input(FIXTURES / 'provenance.json').read_text())
    for name, facts in prov.items():
        fs, x = wavfile.read(require_input(FIXTURES / name))
        assert (fs, len(x)) == (facts['sample_rate'], facts['samples'])


def test_weak_headed_ack_connects_read_whole():
    x = clip('kb0wzi-connected-ack')
    hs, io = awaiting_ack()
    assert VA._ack_plateau(x, hs._peer_shift, band=hs._band,
                           bin_offset=hs._grid) < VA._ACK_PLATEAU
    hs.on_rx_audio(x)
    assert hs.state is VA.VaraState.CONNECTED
    assert any('reads whole' in m for m in io.msgs)


def test_the_peers_turn_request_is_not_read_as_the_ack():
    hs, _ = awaiting_ack()
    assert not hs._ack_whole_500(clip('kb0wzi-turn-request'))


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_whole_ack_is_the_narrow_readers_alone(bw):
    hs, _ = awaiting_ack()
    hs.bw = bw
    assert not hs._ack_whole_500(clip('kb0wzi-connected-ack'))


def _connected(turn):
    hs, io = _station('KB0WZI')
    hs._peer_offset = KB0WZI_HZ / (FS / MK.NFFT)
    hs.turn, hs._txq = turn, [b'x' * 43, b'y' * 10]
    return hs, io


def _stream(hs, x, poll=960):
    x = np.concatenate([x, np.zeros(FS // 2)])
    for i in range(0, len(x), poll):
        hs.on_rx_stream(x[i:i + poll])


@pytest.mark.parametrize('name,turn', [
    ('kb0wzi-drained-responder', VA._TURN_ASKED),
    ('kb0wzi-turn-release-responder', VA._TURN_PEER)])
def test_the_narrow_stream_takes_the_turn_the_bracket_missed(name, turn):
    hs, io = _connected(turn)
    _stream(hs, clip(name))
    assert hs.turn == VA._TURN_OURS
    assert any(m.startswith('tx DATA over #1') for m in io.msgs)


def _sending(called='W1AW'):
    hs, io = _station(called)
    hs.turn = VA._TURN_OURS
    io.tx_went_out = lambda: True
    return hs, io


def _frames(samples):
    x = np.concatenate([np.zeros(FS // 2), samples / np.abs(samples).max() / 2,
                        np.zeros(FS // 2)])
    return [(f.level, f.crc_ok, len(f.payload)) for f in RX.decode_stream(x).frames]


def test_a_two_frame_over_renders_and_reads_back():
    bodies = [bytes(range(43)) + b'\x95', bytes(range(50, 93)) + b'\x81']
    x = OF.data_over_tx(bodies, bw='500', level=4)
    assert _frames(x) == [(4, True, 43), (4, True, 43)]


def test_the_seventh_over_of_a_delivery_goes_in_two_frames():
    """Stock A keyed nothing at our seventh single over, twice; a stock sender
    keys two frames there and the receiver continues them at frame 0."""
    hs, io = _sending()
    hs.send(bytes(525))
    for _ in range(VA._SINGLES_BEFORE_PAIR):
        hs._continue_frame = (hs._tx_pending, 0)
        hs._took_over_continue()
    assert [m for m in io.msgs if m.startswith('tx DATA')][-1].startswith(
        f'tx DATA over #{VA._SINGLES_BEFORE_PAIR + 1} in two frames')
    assert _frames(io.sent[-1]) == [(4, True, 43), (4, True, 43)]


def test_frame_ten_puts_the_next_over_one_record_down_and_six_brings_it_back():
    hs, io = _sending()
    hs.send(bytes(300))
    hs._continue_frame = (hs._tx_pending, VA._NARROW_DOWN)
    hs._took_over_continue()
    assert io.msgs[-1].startswith('tx DATA over #2 at record 2 (34 payload bytes')
    hs._continue_frame = (hs._tx_pending, VA._NARROW_UP)
    hs._took_over_continue()
    assert io.msgs[-1].startswith('tx DATA over #3 (43 payload bytes')


def test_kb0wzi_over_nak_after_our_query_resends_lower():
    """KB0WZI answered each query about our first over with 288/1117 and we
    read none of them; read, the over goes again at the lowest record."""
    hs, io = _sending('KB0WZI')
    hs._peer_offset = KB0WZI_HZ / (FS / MK.NFFT)
    hs.probe_intermediate_query = True
    hs.send(b'\rFC EM RD3RVCBHOEQD 702 509 0\rF> 6A\r' + bytes(43))
    hs._probe_intermediate_answer()
    _stream(hs, clip('kb0wzi-over-nak-responder'))
    assert any(m.startswith('rx responder NAK from KB0WZI') for m in io.msgs)
    assert 'record 0, 9 pending bytes' in io.msgs[-1]


def test_frame_ten_behind_a_pair_resends_its_second_frame():
    """A stock pair's 665-byte delivery is 43 bytes short unless the record-2
    over behind frame 10 carries the pair's second frame."""
    hs, io = _sending()
    data = bytes(i % 251 for i in range(525))
    hs.send(data)
    carried = []
    for frame in [0] * VA._SINGLES_BEFORE_PAIR + [VA._NARROW_DOWN] + [0] * 8:
        if hs._tx_pending is None:
            break
        pending = hs._tx_pending
        bodies = hs._over_bodies(pending[0])
        if frame == VA._NARROW_DOWN:
            bodies = bodies[:1]
        carried += [VA._phy.vara_payload(b, caller='W9SSJ', body_len=len(b))
                    for b in bodies]
        if VA._phy.over_is_last(pending[0], 'W9SSJ'):
            break
        hs._continue_frame = (pending, frame)
        hs._took_over_continue()
    assert b''.join(carried) == data


@pytest.mark.parametrize('name,frame,record', [
    ('stock-record0-answer-frame0', 0, 0),
    ('stock-record0-answer-frame6', VA._NARROW_UP, 1)])
def test_a_low_record_over_is_answered_on_the_288_stream(name, frame, record):
    """Stock answers a level-1 over with 288/(1 + 62*frame) and no eight-symbol
    continue; the caller takes it as that continue."""
    hs, io = _sending()
    hs._tx_pending = (bytes(9) + b'\x89', 3, 0)
    hs._txq, hs._tx_recovery_levels = [bytes(9), b'z'], [0, 0]
    _stream(hs, clip(name))
    assert any(f'record-0 over, frame {frame}' in m for m in io.msgs)
    assert io.msgs[-1].startswith(f'tx DATA over #1 at record {record}')


def test_a_record_zero_close_keeps_its_short_end_out_of_the_mail():
    """Stock's record-0 close of 2026-09-23: eight payload bytes, then 04 0b.
    The 0x04 reached the host as the head of the next message."""
    body = b'0\rF> 56\r\x04\x0b'
    assert VA._phy.vara_payload(body, caller='W9SSJ', body_len=10) == b'0\rF> 56\r'
    assert VA._phy.over_is_last(body, 'W9SSJ')


def test_a_pairs_second_frame_reads_on_its_own_cells():
    """Frame 2 of a burst reads the used-map, the grid and the placement one
    on. Stock's second frames put 13 cells where frame 0's map would not, and
    stock A lost the second frame of both pairs we keyed on frame 0's."""
    frames = [bytes(range(43)) + b'\x8d', bytes(range(60, 103)) + b'\x81']
    x = OF.data_over_tx(frames, bw='500', level=4)
    (start, stop), = RX.burst_spans(np.concatenate([np.zeros(FS), x, np.zeros(FS)]))
    fr = RX.decode_burst_frames(np.concatenate([np.zeros(FS), x, np.zeros(FS)]),
                                start, stop)
    assert all(f.crc_ok and f.self_consistency > 0.99 for f in fr) and len(fr) == 2
    assert len(set(RX.frame_cells(1)) - set(RX.frame_cells(0))) > 0


def test_a_sender_left_low_stays_there_until_told():
    """Stock A read our record-0 overs after its over-NAK and none of the
    record-1 overs we climbed to on our own, 14 of 14 at noise 0.6; only its
    continue at frame 6 moves the record up."""
    hs, io = _sending()
    hs._relevel(0, 'test')
    hs.send(bytes(200))
    for _ in range(8):
        hs._continue_frame = (hs._tx_pending, 0)
        hs._took_over_continue()
    assert hs._tx_pending[2] == 0
    hs._continue_frame = (hs._tx_pending, VA._NARROW_UP)
    hs._took_over_continue()
    assert hs._tx_pending[2] == 1


@pytest.mark.parametrize('level_byte,record', [(4, 4), (3, 2), (2, 1), (1, 0)])
def test_the_next_delivery_starts_where_the_peers_close_says(level_byte, record):
    """Stock closes ``<level> <field>``; its peer's next delivery comes at that
    level, and a delivery at any other is not read."""
    hs, io = _station('W1AW')
    hs._relevel(4, 'test')
    close = bytearray(VA._phy.vara_body(b'CMS via W1AW >\r', 'W9SSJ', body_len=44))
    close[-2] = level_byte
    burst = OF.data_over_tx(bytes(close), bw='500', level=4)
    hs._answer_data_over(np.concatenate([np.zeros(FS // 2), burst / np.abs(burst).max() / 2,
                                         np.zeros(FS // 2)]))
    hs.turn = VA._TURN_OURS
    io.tx_went_out = lambda: True
    hs.send(bytes(100))
    assert hs._tx_pending[2] == record


def test_an_unanswered_close_is_queried_and_goes_again_lower():
    """A close at record 1 that stock did not hear drew no final query at BW500
    and the link closed with it outstanding; the query's over-NAK now sends
    it again two records down."""
    hs, io = _sending('KB0WZI')
    hs._peer_offset = KB0WZI_HZ / (FS / MK.NFFT)
    hs.probe_intermediate_query = True
    hs._relevel(1, 'test')
    hs.send(bytes(7))
    assert VA._phy.over_is_last(hs._tx_pending[0], 'W9SSJ')
    hs._probe_intermediate_answer()
    _stream(hs, clip('kb0wzi-over-nak-responder'))
    assert 'over #1, record 0,' in io.msgs[-1]


def test_k5fit_record0_resend_behind_our_nak_is_read():
    """K5FIT resent its first over at record 0 behind each NAK of ours, 2-6 dB
    over the band, and the span reader never saw the burst end; decoded where
    it has to start, it is over #1."""
    hs, io = _station('K5FIT')
    hs._answer_owed, hs._owed_block, hs._owed_recovery = VA._OWED_OVER, True, True
    hs._resend_500 = True
    _stream(hs, clip('k5fit-record0-resend'))
    assert any(m.startswith('delivered 9 payload bytes') for m in io.msgs), io.msgs


def test_aa5af_unread_over_draws_a_nak_not_an_ack():
    """AA5AF's second greeting over faded past decoding and we acknowledged the
    idle behind it, which AA5AF took for that over; 43 bytes were lost. Its grid
    still locks, so the idle now draws the NAK that asks for it again."""
    hs, io = _station('AA5AF')
    hs._peer_offset = -6.5 / (FS / MK.NFFT)
    hs._peer_over, hs._answer_owed, hs._reacks = 1, VA._OWED_OVER, 1
    _stream(hs, clip('aa5af-unread-over-and-idle'))
    assert any('nothing read it' in m for m in io.msgs), io.msgs
    assert any(m.startswith('tx NAK (1/4)') for m in io.msgs)


@pytest.mark.parametrize('name', ['kb0wzi-turn-release-responder', 'kb0wzi-over-nak-responder'])
def test_short_frames_alone_do_not_lock_as_an_over(name):
    hs, _ = _station('KB0WZI')
    x = np.concatenate([clip(name), np.zeros(5 * FS)])
    assert RX.lock_score(x, 0) < VA._UNREAD_LOCK


@pytest.mark.parametrize('start,frame,record', [
    (4, 5, 2), (2, 5, 1), (1, 6, 2), (2, 6, 4), (0, 7, 4), (1, 4, 4), (4, 0, 4)])
def test_a_continue_frame_steps_the_level(start, frame, record):
    """Eight stock-pair sessions: 5 one level down, 6 one up, 0 the same. Stock
    answered our record-0 over with 7 and read the next at level 4 and not at 3,
    and our record-1 over with 4."""
    hs, io = _sending()
    hs._relevel(start, 'test')
    hs.send(bytes(300))
    hs._continue_frame = (hs._tx_pending, frame)
    hs._took_over_continue()
    assert hs._tx_pending[2] == record


@pytest.mark.parametrize('n', [6, 7, 8])
def test_a_tight_close_keeps_its_short_end(n):
    """Our 7-byte record-0 close wrote its trailer over the level byte and stock
    faulted four times of four behind it; stock's own 8-byte close is payload
    then ``04 0b``, and read ours as ``04 8b`` on the cables."""
    body = VA._phy.vara_body(bytes(range(1, n + 1)), 'W9SSJ', tail=0x89, body_len=10)
    assert body[-2:] == bytes([4, 0x8b if n == 8 else 0x82]) and body[:n] == bytes(range(1, n + 1))
    assert VA._phy.vara_payload(body, caller='W9SSJ', body_len=10) == bytes(range(1, n + 1))
    assert body[n:-2] == bytes([0x14, 0xF6])[:8 - n]
    assert VA._phy.over_is_last(body, 'W9SSJ')


def _wide(called='K5FIT'):
    io = IO()
    io.tx_went_out = lambda: True
    hs = VA.VaraStationHandshake(['W9SSJ'], io, bw='2300')
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', called
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_PEER
    return hs, io


def test_k5fit_2300_close_names_level_2_for_our_delivery():
    """K5FIT closed its greeting ``02 42`` on 40 m and read none of our level-4
    overs; a stock pair at BW2300 does the same with ``02 02``."""
    hs, io = _wide()
    _stream(hs, clip('k5fit-2300-close-02'))
    hs.turn = VA._TURN_OURS
    hs.send(bytes(79))
    assert hs._tx_pending[2] == 1


def test_a_close_resent_at_a_lower_record_is_not_delivered_twice():
    """Stock A re-sent its ``FS +`` a record down after missing our answer on the
    BW2750 cables, and the second copy reached the host and ended the exchange."""
    hs, io = _wide('W1AW')
    got = []
    io.data = got.append
    for record in (103, 101):
        body = VA._phy.vara_body(b'FS +\r', 'W9SSJ', body_len=VA._phy.body_size('2750', record))
        hs._deliver([body])
    assert got == [b'FS +\r']


def _live(hs, io, x, poll=960):
    """Feed as the live cursor does: what we key is not heard back."""
    io.tx_went_out = lambda: True
    keyed = []
    skip = [0]
    def tx(samples):
        keyed.append(i)
        skip[0] += len(samples) + FS // 20
    io.tx = tx
    i = 0
    while i < len(x):
        if skip[0]:
            i, skip[0] = i + skip[0], 0
            continue
        hs.on_rx_stream(x[i:i + poll])
        i += poll
    return keyed


def test_k5fit_unread_over_draws_a_nak_and_the_read_one_an_answer_on_time():
    """K5FIT 40 m, 2026-09-24 05:30Z: we answered a level-4 over 1.9 s after it
    ended, then acknowledged it again in the idle behind the next over, which
    nothing had read; K5FIT took that for the unread over and its 43 bytes were
    lost."""
    hs, io = _station('K5FIT')
    keyed = _live(hs, io, clip('k5fit-0530-over-unread-over-idle'))
    end = int((84.46 - 79.6) * FS)
    assert keyed and 0 <= keyed[0] - end <= FS // 2, keyed
    assert any('nothing read it' in m or 'will not decode' in m for m in io.msgs), io.msgs
    assert not any(m.startswith('re-acknowledging') for m in io.msgs), io.msgs
    assert any(m.startswith('tx session-over-nak') for m in io.msgs), io.msgs


def test_aa5af_record0_resend_behind_an_idle_nak_is_read():
    """AA5AF 40 m, 06:47Z: the over-NAK our idle answer keyed drew a record-0
    resend, CRC-clean offline, that nothing was looking for."""
    hs, io = _station('AA5AF')
    hs._peer_offset = -5.0 / (FS / MK.NFFT)
    _live(hs, io, clip('aa5af-0647-idle-and-record0-resend'))
    assert any(m.startswith('delivered 9 payload bytes') for m in io.msgs), io.msgs


def test_we_ask_one_level_up_after_two_clean_reads_and_never_two():
    """We asked K5FIT for two levels behind its level-2 over, 05:30Z, and lost
    the level-4 over it sent next. A stock receiver steps one at a time."""
    hs, io = _station('K5FIT')
    frames = {}
    for i, lv in enumerate((2, 2, 2)):
        last = hs._peer_over_state
        hs._read_streak = hs._read_streak + 1 if last and last[0] == lv else 1
        hs._peer_over_state = (lv, 0x99)
        tail = hs._continue_tail()
        frames[i] = next(f for f in (0, 4, 5, 6, 7) if tail == VF.over_continue_frames(
            'K5FIT', '500', (f,))[f][0])
    assert frames == {0: 0, 1: 6, 2: 0}


def test_our_close_names_the_level_we_read_the_peer_at():
    hs, io = _sending('K5FIT')
    hs._peer_over_state = (2, 0x99)
    hs.send(b'FS +\r')
    assert hs._tx_pending[0][-2] == 2


def _live_ticking(hs, io, x, tick_s, poll=960):
    """:func:`_live` with the idle tick fired at ``tick_s`` into the clip, where
    it fired live."""
    io.tx_went_out = lambda: True
    keyed, skip = [], [0]
    def tx(samples):
        keyed.append(i)
        skip[0] += len(samples) + FS // 20
    io.tx = tx
    i, ticked = 0, False
    while i < len(x):
        if skip[0]:
            i, skip[0] = i + skip[0], 0
            continue
        if not ticked and i >= tick_s * FS:
            ticked = True
            hs.idle_keepalive()
        hs.on_rx_stream(x[i:i + poll])
        i += poll
    return keyed


def test_a_pair_at_noise_is_answered_once_whole_and_nothing_keys_into_it():
    """Stock A on the cables at noise 1.1, 2026-09-24: six level-1 singles,
    then a pair (6.5-17.2 s here) that an idle-tick re-ack keyed into. The
    idle at the end is the live peer's, behind the answer it never had."""
    hs, io = _station('W1AW')
    hs._peer_over_state, hs._peer_run_500, hs._overs_hint = (1, 0x99), 5, 4
    hs._peer_over, hs._next_500 = 5, 1
    hs._key(False)
    keyed = _live_ticking(hs, io, clip('stock-noise11-single-then-pair'), 10.0)
    assert keyed[0] < 7 * FS and 17.1 * FS <= keyed[1] <= 17.6 * FS, (keyed, io.msgs)
    assert sum('two frames' in m for m in io.msgs) == 1, io.msgs
    assert sum(m.startswith('delivered 9 payload') for m in io.msgs) == 3, io.msgs


def test_the_first_message_over_behind_our_release_is_not_keyed_over():
    """Stock A at noise 0.9, 2026-09-24: its record-2 message began 0.05 s
    behind our release, and a second release 3.4 s later keyed into it."""
    hs, io = _station('W1AW')
    hs._peer_over_state, hs._next_500, hs._released = (2, 0x82), 2, True
    hs._key(False)
    keyed = _live_ticking(hs, io, clip('stock-noise09-message-behind-release'), 3.4)
    assert keyed and keyed[0] >= int(5.0 * FS), (keyed, io.msgs)
    assert any('where it was due' in m for m in io.msgs), io.msgs


@pytest.mark.parametrize('name,call,offered', [
    ('aa5af-0519-before-link-setup', 'AA5AF', 3),
    ('k5fit-0546-before-link-setup', 'K5FIT', 1)])
def test_the_link_setup_asks_the_greeting_at_the_level_we_heard(name, call, offered):
    """AA5AF 05:19Z and K5FIT 80 m 05:46Z, 2026-09-24: our link-setup asked
    level 4 whatever the connect-response sounded like, both gateways sent a
    level-4 greeting, and we read none of it."""
    io = IO()
    hs = VA.VaraStationHandshake(['W9SSJ'], io, bw='500')
    hs.role, hs.called, hs.caller, hs._setup_level = 'initiator', call, 'W9SSJ', offered
    hs._rx_recent = clip(name)
    ask, grade = hs._greeting_ask()
    body = VF.link_setup_frame('W9SSJ', '500', offered, ask, grade)
    assert (ask, body[-4]) == (1, 1), io.msgs
