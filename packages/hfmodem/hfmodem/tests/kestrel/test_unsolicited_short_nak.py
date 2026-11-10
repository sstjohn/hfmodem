# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An unsolicited, call-keyed short NAK repeats only the pending geometry."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_data_over_gate import _connected
from .test_idle_poll import clock, _Clock  # noqa: F401
from .test_limited_responder_query_recovery import on_air
from .test_responder_control_stream import feed
from .test_wide_answer_query import pending, timed_transmit


def sender(role, bw, final=False):
    h, io = _connected(role=role, bw=bw)
    h.turn = A._TURN_OURS
    h._start_compression('OFF')
    h.send(b'last' if final else bytes(i % 251 for i in range(393)))
    # The recording sink is instantaneous; DATA must finish before its answer.
    _Clock.t += len(io.sent[-1]) / M.FS + .1
    return h, io


def short(h):
    return F.for_bw(F.SESSION_OVER_NAK_SHORT if h.role == 'responder'
                    else F.SESSION_OVER_NAK_RESPONDER_SHORT, h.bw)


def hear(h, audio, route):
    if route == 'stream':
        feed(h, on_air(audio))
    else:
        # A bracket callback observes the completed NAK, not its key-down.
        _Clock.t += len(audio) / M.FS
        h.on_rx_audio(audio)


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('route', ['stream', 'burst'])
@pytest.mark.parametrize('final', [False, True], ids=['full', 'last'])
def test_unsolicited_short_nak_repeats_pending_geometry(role, bw, route, final, clock):  # noqa: F811
    h, io = sender(role, bw, final)
    pending, pair, queued = h._tx_pending, h._tx_pair, list(h._txq)
    confirmed = h.payload_ledger.confirmed
    phase, delivery = h._query_retry_phase, h._delivery_keyed
    keyed = []
    io.tx_metadata = lambda **metadata: keyed.append(metadata)
    before = len(io.sent)
    hear(h, M.synth_burst(h.called, short(h)), route)
    assert len(io.sent) == before + 1, io.msgs
    assert h._tx_pending == pending and h._tx_pair == pair and h._txq == queued
    assert h._tx_retries == 1 and h._delivery_keyed == delivery + 1
    assert keyed[-1]['retry'] and keyed[-1]['over_id'] == pending[1]
    assert keyed[-1]['speed_level'] == pending[2]
    assert h.payload_ledger.confirmed == confirmed
    assert h._query_retry_phase == (phase if final else phase ^ 1)
    assert h._answer_query_for is h._intermediate_query_for is h._final_query_for is None
    assert h._pending_answer_at == h._over_keyed_at
    assert not h._intermediate_query_attempts
    # Stream ownership prevents a second retry from the same bracket.
    if route == 'stream':
        h.on_rx_audio(on_air(M.synth_burst(h.called, short(h))))
        assert h._tx_retries == 1 and len(io.sent) == before + 1


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'burst'])
def test_stray_short_nak_without_pending_is_ignored(role, route, clock):  # noqa: F811
    h, io = _connected(role=role)
    h.turn = A._TURN_OURS
    before = h.state, h.turn, h.progress, h._since_progress, h._tx_retries
    hear(h, M.synth_burst(h.called, short(h)), route)
    assert not io.sent and not io.host
    assert (h.state, h.turn, h.progress, h._since_progress, h._tx_retries) == before


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_bracket_short_nak_after_stream_owned_setup_still_names_pending(role, clock):  # noqa: F811
    h, io = sender(role, '2300')
    h._stream_owns_responder_control = h._stream_owns_answer = True
    pending, before = h._tx_pending, len(io.sent)
    hear(h, M.synth_burst(h.called, short(h)), 'burst')
    assert h._tx_pending == pending and h._tx_retries == 1, io.msgs
    assert len(io.sent) == before + 1


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'burst'])
@pytest.mark.parametrize('other', ['wrong-call', 'confirm'])
def test_other_sixteen_symbol_frame_keeps_its_handling(role, route, other, clock):  # noqa: F811
    h, io = sender(role, '2300')
    pending, before = h._tx_pending, len(io.sent)
    kind = short(h) if other == 'wrong-call' else F.SESSION_CONFIRM
    call = 'W1AW' if other == 'wrong-call' else h.called
    hear(h, M.synth_burst(call, kind), route)
    assert h._tx_pending == pending and h._tx_retries == 0
    # A genuine caller confirm still draws the responder's control reply.
    assert len(io.sent) == before + (role == 'responder' and other == 'confirm'), io.msgs


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('refused', [False, True], ids=['spent', 'refused'])
def test_short_nak_uses_existing_retry_budget(role, refused, clock):  # noqa: F811
    h, io = sender(role, '2300')
    pending = h._tx_pending
    h._tx_retries = 1 if refused else A._OVER_RETRY_MAX
    if refused:
        io.tx_went_out = lambda: False
    hear(h, M.synth_burst(h.called, short(h)), 'burst')
    assert h._tx_pending == pending
    if refused:
        assert h.state == A.VaraState.CONNECTED and h._tx_retries == 1
    else:
        assert h.state == A.VaraState.DISCONNECTED and h._tx_retries == A._OVER_RETRY_MAX
    assert not any('retry DATA' in m for m in io.msgs)


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('route', ['stream', 'burst'])
@pytest.mark.parametrize('age', ['wall', 'samples'])
def test_one_nak_per_aired_keying_then_later_nak_retries_again(role, route, age, clock):  # noqa: F811
    h, io = sender(role, '2300')
    pending, before = h._tx_pending, len(io.sent)
    audio = M.synth_burst(h.called, short(h))
    # Model actual playback: the retry has not unkeyed until all DATA aired.
    transmit = io.tx
    keyed = []

    def timed_transmit(samples):
        keyed.append(_Clock.t)
        transmit(samples)
        _Clock.t += len(samples) / M.FS

    io.tx = timed_transmit
    hear(h, audio, route)
    retry_end = _Clock.t
    assert h._tx_retries == 1 and len(io.sent) == before + 1
    # This is the same emission, observed before the retry finished airing.
    _Clock.t = retry_end - len(io.sent[-1]) / M.FS
    hear(h, audio, route)
    assert h._tx_retries == 1 and len(io.sent) == before + 1
    # A new NAK can finish only after the retry and its own full keying.
    _Clock.t = retry_end
    h._heard = 0
    if age == 'wall':
        _Clock.t += len(audio) / M.FS + .1
    else:
        h._heard = len(audio) + M.FS // 10
    if age == 'samples' and route == 'burst':
        # Keep wall time fixed: only the receive sample clock admits this NAK.
        h.on_rx_audio(audio)
    else:
        hear(h, audio, route)
    assert h._tx_retries == 2 and len(io.sent) == before + 2, io.msgs
    assert h._tx_pending == pending and h.payload_ledger.confirmed == 0
    if age == 'samples' and route == 'burst':
        assert keyed[-1] == retry_end  # No wall-time admission through hear().


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_refused_retry_does_not_let_same_keying_spend_another_nak(role, clock):  # noqa: F811
    h, io = sender(role, '2300')
    io.tx_went_out = lambda: False
    audio = M.synth_burst(h.called, short(h))
    hear(h, audio, 'burst')
    before = len(io.sent)
    _Clock.t += 10
    hear(h, audio, 'burst')
    assert len(io.sent) == before and h._tx_retries == 0


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('route', ['stream', 'burst'])
@pytest.mark.parametrize('final', [False, True], ids=['full', 'last'])
@pytest.mark.parametrize('premature', [True, False], ids=['premature', 'after-end'])
def test_original_keying_requires_nak_onset_after_rendered_end(
        role, bw, route, final, premature, clock):  # noqa: F811
    h, io = sender(role, bw, final)
    pending, pair, queued = h._tx_pending, h._tx_pair, list(h._txq)
    before, progress = len(io.sent), h.progress
    delivery, phase = h._delivery_keyed, h._query_retry_phase
    keyed_at = h._over_keyed_at  # The original sink has not advanced time.
    end = keyed_at + len(io.sent[-1]) / M.FS
    audio = M.synth_burst(h.called, short(h))
    # This emission begins during the original DATA, even if observed later.
    _Clock.t = end - .2 if premature else end + .1
    hear(h, audio, route)
    assert h._tx_pending is pending and h._tx_pair == pair and h._txq == queued
    assert h.payload_ledger.confirmed == 0 and not io.host
    if premature:
        assert h._tx_retries == 0 and len(io.sent) == before, io.msgs
        assert h.progress == progress and h._delivery_keyed == delivery
        assert h._query_retry_phase == phase and h._short_nak_keying is None
    else:
        # The identical NAK waveform answers DATA only on the later timeline.
        assert h._tx_retries == 1 and len(io.sent) == before + 1, io.msgs
        assert h._delivery_keyed == delivery + 1


@pytest.mark.parametrize('route,expired', [('stream', False), ('stream', True),
                                         ('burst', True)],
                         ids=['fresh-stream', 'expired-stream', 'expired-burst'])
def test_queried_short_nak_keeps_solicited_identity_after_expiry(route, expired, clock):  # noqa: F811
    h, io = pending('2300')  # Intermediate record-3 over, as in Astra's probe.
    original, queued = h._tx_pending, list(h._txq)
    assert original[2] == 3
    # Let the original DATA finish even though the fixture's sink is instant.
    _Clock.t += len(io.sent[-1]) / M.FS
    timed_transmit(io)
    assert h._query_answerer_over()
    assert h._answer_query_for is original
    if expired:
        _Clock.t += 3
    assert h._caller_over_nak_fresh() is (not expired)
    sent = len(io.sent)
    hear(h, M.synth_burst(h.called, short(h)), route)
    assert not any('unsolicited' in msg for msg in io.msgs), io.msgs
    assert h.payload_ledger.confirmed == 0 and not io.host
    if expired:
        assert h._answer_query_for is original
        assert h._tx_pending is original and h._txq == queued
        assert h._tx_retries == 0 and len(io.sent) == sent
    else:
        assert h._tx_retries == 1 and len(io.sent) == sent + 1
        assert h._tx_pending[1] == original[1] and h._tx_pending[2] == 1


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('overhead', [.100, .200], ids=['return-E100', 'return-E200'])
@pytest.mark.parametrize('onset', [-.050, .075, .126], ids=['before-E', 'E75', 'E126'])
def test_original_nak_floor_excludes_playback_completion_overhead(role, overhead, onset, clock):  # noqa: F811
    h, io = _connected(role=role, bw='2300')
    h.turn = A._TURN_OURS
    h._start_compression('OFF')
    transmit = io.tx
    ends = []

    def playback(samples):
        transmit(samples)
        ends.append(_Clock.t + len(samples) / M.FS)
        _Clock.t = ends[-1] + overhead

    io.tx = playback
    h.send(bytes(i % 251 for i in range(393)))
    end, pending, sent = ends[-1], h._tx_pending, len(io.sent)
    audio = M.synth_burst(h.called, short(h))
    # The bracket is delivered on completion, including answers captured while
    # tx() was draining/unkeying. Its onset is independent of tx() return.
    _Clock.t = end + onset
    hear(h, audio, 'burst')
    expected = int(onset >= 0)
    assert h._tx_retries == expected and len(io.sent) == sent + expected, io.msgs
    assert h._tx_pending is pending and h.payload_ledger.confirmed == 0


@pytest.mark.parametrize('final', [False, True], ids=['full', 'last'])
def test_bw500_consecutive_premature_then_later_streamed_nak(final, clock):  # noqa: F811
    h, io = sender('initiator', '500', final)
    pending, sent = h._tx_pending, len(io.sent)
    end = h._over_keyed_at + len(io.sent[-1]) / M.FS
    audio = on_air(M.synth_burst(h.called, short(h)))
    _Clock.t = end - .2
    feed(h, audio)
    assert h._tx_retries == 0 and len(io.sent) == sent
    _Clock.t = max(_Clock.t, end) + .1
    feed(h, audio)
    assert not any('receive stream reset' in msg for msg in io.msgs), io.msgs
    assert h._tx_retries == 1 and len(io.sent) == sent + 1, io.msgs
    assert h._tx_pending is pending and h.payload_ledger.confirmed == 0


@pytest.mark.parametrize('final', [False, True], ids=['full', 'last'])
def test_bw500_nonkeying_idle_reset_retains_control_history(final, clock):  # noqa: F811
    h, io = sender('initiator', '500', final)
    pending, sent = h._tx_pending, len(io.sent)
    idle = on_air(M.synth_burst(h.called, F.for_bw(F.SESSION_RESPONDER_IDLE, h.bw)))
    feed(h, idle)
    assert any('not progress' in msg for msg in io.msgs), io.msgs
    assert h._ans_skip and len(io.sent) == sent
    # The idle reader reset without a keying/generation change. Let its skip
    # expire, then hear a genuine NAK for the still-pending DATA.
    feed(h, np.zeros(h._ans_skip + M.FS // 10))
    feed(h, on_air(M.synth_burst(h.called, short(h))))
    assert not any('receive stream reset' in msg for msg in io.msgs), io.msgs
    assert h._tx_retries == 1 and len(io.sent) == sent + 1, io.msgs
    assert h._tx_pending is pending and h.payload_ledger.confirmed == 0


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('retry', [False, True], ids=['original', 'retry'])
@pytest.mark.parametrize('route', ['stream', 'burst'])
@pytest.mark.parametrize('onset', [-.2, .075], ids=['premature', 'E75'])
def test_nak_floor_starts_after_delayed_key_down(role, retry, route, onset, clock):  # noqa: F811
    h, io = _connected(role=role, bw='2300')
    h.turn = A._TURN_OURS
    h._start_compression('OFF')
    key, transmit = io.key, io.tx
    ends = []

    def delayed_key(on):
        key(on)
        if on:
            _Clock.t += .3

    def playback(samples):
        transmit(samples)
        ends.append(_Clock.t + len(samples) / M.FS)
        # Queue playback without moving the clock; receive callbacks below
        # advance monotonically toward the independently recorded sample end.

    io.key, io.tx = delayed_key, playback
    h.send(bytes(i % 251 for i in range(393)))
    audio = M.synth_burst(h.called, short(h))

    def hear_at(onset_at):
        # hear() adds 4800 leading quiet samples on the stream route. Place
        # the NAK itself, rather than that quiet, at the requested onset.
        start = onset_at - (4800 / M.FS if route == 'stream' else 0)
        assert start >= _Clock.t
        _Clock.t = start
        hear(h, audio, route)

    if retry:
        hear_at(ends[-1] + .075)  # Reach retry through the real NAK route.
        assert h._tx_retries == 1, io.msgs
    end = ends[-1]
    pending, pair, queued = h._tx_pending, h._tx_pair, list(h._txq)
    before, progress = len(io.sent), h.progress
    delivery, phase = h._delivery_keyed, h._query_retry_phase
    consumed = h._short_nak_keying
    hear_at(end + onset)
    admitted = int(onset > 0)
    assert h._tx_retries == int(retry) + admitted, io.msgs
    assert len(io.sent) == before + admitted
    assert h._delivery_keyed == delivery + admitted
    assert h._tx_pending is pending and h._tx_pair == pair and h._txq == queued
    assert h.payload_ledger.confirmed == 0 and not io.host
    if not admitted:
        assert h.progress == progress and h._query_retry_phase == phase
        assert h._short_nak_keying == consumed


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('retry', [False, True], ids=['original', 'retry'])
@pytest.mark.parametrize('route', ['stream', 'burst'])
@pytest.mark.parametrize('onset', [-.2, .075], ids=['premature', 'E75'])
def test_station_transport_nak_floor_uses_playback_evidence(
        role, retry, route, onset, clock):  # noqa: F811
    from hfmodem.kestrel.host.station_runtime import StationTransport
    from hfmodem.station.link import KestrelLink
    from .test_station_runtime import station

    st, _ = station()
    link = KestrelLink(st)
    io = StationTransport(link)
    sent, ends, timestamps, host = [], [], {}, []
    io.report_ptt = lambda on: None
    io.data = host.append
    # Exercise real key(), tx(), emit() and transmit(); only the arbiter's
    # physical emitter is simulated. Key-down happens INSIDE tx(), not key().
    def emit(req):
        _Clock.t += .3
        sent.append(req.audio)
        ends.append(_Clock.t + len(req.audio) / M.FS)
        st.audio.samples += len(req.audio)
        timestamps[st.audio.samples] = ends[-1]
        return st.audio.samples

    st.arbiter.submit = emit
    st.audio.loop_lat = 0
    st.audio.dac_time = timestamps.__getitem__
    h, _ = _connected(role=role, bw='2300', io=io)
    io.hs = h
    h.turn = A._TURN_OURS
    h._start_compression('OFF')
    try:
        h.send(bytes(i % 251 for i in range(393)))
        audio = M.synth_burst(h.called, short(h))

        def hear_at(at):
            start = at - (4800 / M.FS if route == 'stream' else 0)
            assert start >= _Clock.t
            _Clock.t = start
            hear(h, audio, route)

        if retry:
            hear_at(ends[-1] + .075)
            assert h._tx_retries == 1, io.messages
        end = ends[-1]
        pending, pair, queued = h._tx_pending, h._tx_pair, list(h._txq)
        before, progress = len(sent), h.progress
        delivery, phase = h._delivery_keyed, h._query_retry_phase
        consumed = h._short_nak_keying
        hear_at(end + onset)
        admitted = int(onset > 0)
        assert h._tx_retries == int(retry) + admitted, io.messages
        assert len(sent) == before + admitted
        assert h._delivery_keyed == delivery + admitted
        assert h._tx_pending is pending and h._tx_pair == pair and h._txq == queued
        assert h.payload_ledger.confirmed == 0 and not host
        if not admitted:
            assert h.progress == progress and h._query_retry_phase == phase
            assert h._short_nak_keying == consumed
    finally:
        link.stop()
