# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Pre-greeting NAKs are free; only the setup grant has a fixed deadline."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock, _Clock  # noqa: F401
from .test_limited_responder_query_recovery import on_air
from .test_responder_control_stream import feed
from .test_rxwindow_adapter import station, query, receive


def setup_grant(role):
    """Complete the real setup-idle route, without injecting receive events."""
    if role == 'initiator':
        from .test_setup_turn_idle import awaiting, idle

        h, io = awaiting()
        h.on_rx_audio(np.r_[np.zeros(960), idle(), np.zeros(9600)])
    else:
        from .test_setup_query_recovery import setup, query as setup_query

        h, io = setup(marked=True)
        h.on_rx_audio(np.r_[np.zeros(4800), setup_query(), np.zeros(24000)])
    assert h.state == A.VaraState.CONNECTED
    assert h.turn == A._TURN_PEER
    assert h._rx_window.before_data and h._rx_window.A.kind == 'nak'
    assert h._greeting_at == _Clock.t
    assert h._query_data_due is (role == 'initiator')
    assert not io.host
    return h, io


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('expiry', ['idle', 'query'])
def test_pregreeting_idles_cannot_extend_fixed_cap(role, expiry, clock):  # noqa: F811
    h, io = setup_grant(role)
    start = _Clock.t
    for elapsed in (30, 60, 85.7, 119.9):
        _Clock.t = start + elapsed
        assert query(h)
        assert h.state == A.VaraState.CONNECTED
        assert h._reacks == h._since_progress == 0 and not io.host
    # The driver's timer must allow the same bound in the responder role too.
    h.idle_keepalive()
    assert h.state == A.VaraState.CONNECTED
    _Clock.t = start + 120
    if expiry == 'idle':
        h.idle_keepalive()
    else:
        query(h)
    assert h.state == A.VaraState.DISCONNECTED
    assert h._receive_reports[-1].code == 'budget'
    assert not io.host


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_driver_ticks_between_greeting_idles_keep_the_fixed_deadline(role, clock):  # noqa: F811
    h, io = station(role)
    h._progressed()
    h._setup_grant_pending = True
    h._grant_peer_turn()
    start = _Clock.t
    for elapsed in range(10, 120, 10):
        _Clock.t = start + elapsed
        assert query(h, explicit=False)
        sent = len(io.sent)
        h.idle_keepalive()
        assert h.state == A.VaraState.CONNECTED
        assert len(io.sent) == sent
        assert h._reacks == h._since_progress == 0
    _Clock.t = start + 120
    h.idle_keepalive()
    assert h.state == A.VaraState.DISCONNECTED
    assert h._receive_reports[-1].code == 'budget' and not io.host


def request_completed_setup(chat, route, *, refused=False):
    from .test_setup_turn_idle import IO, feed as setup_feed

    io = IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw='500', chat=chat, high_speed=True)
    h.originate('K0EPI', 'W9SSJ')
    h.on_rx_tones(F.handshake_tones('K0EPI', F.connect_response('500', 1)),
                  F.connect_response('500'))
    assert h.state == A.VaraState.CONNECTING and h.step == A._I_LINKSETUP_SENT
    assert h._linksetup_tx == 1 and len(io.sent) == 2
    assert len(io.sent[-1]) > M.FS  # The generated link setup reached IO.tx.
    request = M.synth_burst(h.caller, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, h.bw))
    io.accept = not refused
    setup_feed(h, np.r_[np.zeros(960), request, np.zeros(9600)], route)
    if refused:
        assert h.state == A.VaraState.CONNECTED and h._release_owed
        assert h._rx_window.A is None and not io.host
        return h, io
    assert h.state == A.VaraState.CONNECTED and h.turn == A._TURN_PEER, io.logs
    assert len(io.connections) == 1 and len(io.sent) == 3
    assert h._rx_window.before_data and h._rx_window.A.kind == 'nak'
    assert h._query_data_due and not h._setup_grant_pending and not io.host
    return h, io


def request_setup_idle(h, route):
    from .test_setup_turn_idle import feed as setup_feed

    poll = M.synth_tone_pairs(F.control_bursts(h.called, h.bw, 1)[0])
    setup_feed(h, np.r_[np.zeros(960), poll, np.zeros(9600)], route)
    h.idle_keepalive()


@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('expiry', ['idle', 'query'])
def test_request_completed_setup_has_fixed_cap(chat, route, expiry, clock):  # noqa: F811
    h, io = request_completed_setup(chat, route)
    start = _Clock.t
    for elapsed in (30, 60, 85.7, 119.9):
        _Clock.t = start + elapsed
        if expiry == 'idle':
            request_setup_idle(h, route)
        else:
            assert query(h)
        assert h.state == A.VaraState.CONNECTED
        assert h._reacks == h._since_progress == 0 and not io.host
    _Clock.t = start + 120
    if expiry == 'idle':
        h.idle_keepalive()
    else:
        query(h)
    assert h.state == A.VaraState.DISCONNECTED
    assert h._receive_reports[-1].code == 'budget' and not io.host


@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('retry', ['timer', 'request'])
@pytest.mark.parametrize('expiry', ['idle', 'query'])
def test_refused_setup_release_keeps_cap_until_successful_grant(route, retry, expiry, clock):  # noqa: F811
    from .test_setup_turn_idle import feed as setup_feed

    h, io = request_completed_setup(False, route, refused=True)
    assert h._setup_grant_pending and h._setup_greeting_at is None
    _Clock.t += 5
    io.accept = True
    if retry == 'timer':
        h.idle_keepalive()
    else:
        request = M.synth_burst(h.caller, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, h.bw))
        setup_feed(h, np.r_[np.zeros(960), request, np.zeros(9600)], route)
    assert not h._release_owed and h._rx_window.before_data, io.logs
    assert h._rx_window.A.kind == 'nak' and h.turn == A._TURN_PEER
    assert_setup_retry_deadline(h, io, expiry)


@pytest.mark.parametrize('expiry', ['idle', 'query'])
def test_refused_setup_confirmation_then_request_keeps_cap(expiry, clock):  # noqa: F811
    from .test_setup_turn_idle import awaiting, feed as setup_feed

    h, io = awaiting()
    io.accept = False
    ack = M.synth_tone_pairs(F.control_bursts(h.called, h.bw, 0)[1])
    setup_feed(h, np.r_[np.zeros(960), ack, np.zeros(9600)], 'bracket')
    assert h.state == A.VaraState.CONNECTED and h._rx_window.A is None, io.logs
    assert h._setup_grant_pending and h._setup_greeting_at is None
    _Clock.t += 5
    io.accept = True
    request = M.synth_burst(h.caller, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, h.bw))
    setup_feed(h, np.r_[np.zeros(960), request, np.zeros(9600)], 'bracket')
    assert h._rx_window.before_data and h._rx_window.A.kind == 'nak', io.logs
    assert_setup_retry_deadline(h, io, expiry)


def assert_setup_retry_deadline(h, io, expiry):
    start = _Clock.t
    assert not h._setup_grant_pending and h._setup_greeting_at == start
    # A timer re-release retains its ordinary idle-keying count. Polling
    # must not spend any further receive-recovery budget before the cap.
    counters = h._reacks, h._since_progress
    for elapsed in (30, 60, 85.7, 119.9):
        _Clock.t = start + elapsed
        if expiry == 'idle':
            request_setup_idle(h, 'bracket')
        else:
            assert query(h)
        assert h.state == A.VaraState.CONNECTED, io.logs
        assert (h._reacks, h._since_progress) == counters and not io.host
    _Clock.t = start + 120
    if expiry == 'idle':
        h.idle_keepalive()
    else:
        query(h)
    assert h.state == A.VaraState.DISCONNECTED, io.logs
    assert h._receive_reports[-1].code == 'budget' and not io.host


@pytest.mark.parametrize('boundary', ['ours', A.VaraState.DISCONNECTED,
                                     A.VaraState.LISTENING, A.VaraState.CONNECTING])
def test_refused_setup_provenance_ends_at_our_turn_or_link_reset(boundary, clock):  # noqa: F811
    from .test_setup_turn_idle import feed as setup_feed

    h, io = request_completed_setup(False, 'bracket', refused=True)
    assert h._setup_grant_pending
    if boundary == 'ours':
        grant = M.synth_tone_pairs(F.turn_grant(h.called, h.bw))
        setup_feed(h, np.r_[np.zeros(960), grant, np.zeros(9600)], 'bracket')
        assert h.turn == A._TURN_OURS, io.logs
    else:
        h.state = boundary
    assert not h._setup_grant_pending and h._setup_greeting_at is None
    if boundary == 'ours':
        io.accept = True
        assert h._release_turn()
        start = _Clock.t
        for elapsed in (119.9, 120, 139):
            _Clock.t = start + elapsed
            assert query(h)
            assert h.state == A.VaraState.CONNECTED
            assert h._setup_greeting_at is None
    assert not io.host and not any(r.code == 'budget' for r in h._receive_reports)


@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
def test_request_setup_later_handover_never_arms_cap(chat, route, clock):  # noqa: F811
    from .test_setup_turn_idle import feed as setup_feed

    h, io = request_completed_setup(chat, route)
    # The peer grants, then requests another turn on this very same link.
    grant = M.synth_tone_pairs(F.turn_grant(h.called, h.bw))
    setup_feed(h, np.r_[np.zeros(960), grant, np.zeros(9600)], route)
    assert h.turn == A._TURN_OURS, io.logs
    request = M.synth_burst(h.caller, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, h.bw))
    setup_feed(h, np.r_[np.zeros(960), request, np.zeros(9600)], route)
    assert h.turn == A._TURN_PEER and h._rx_window.before_data, io.logs
    assert h._setup_greeting_at is None
    start = _Clock.t
    for elapsed in (30, 60, 119.9, *range(120, 140)):
        _Clock.t = start + elapsed
        assert query(h)
        assert h.state == A.VaraState.CONNECTED
        assert h._setup_greeting_at is None and not io.host
        assert h._reacks == h._since_progress == 0
        assert not any(report.code == 'budget' for report in h._receive_reports)


def test_repeated_drained_after_solicited_final_preserves_sent_nak(monkeypatch, clock):  # noqa: F811
    from .test_final_ack_recovery import pending

    h, io = pending()
    events = []
    event = h._rx_window.event

    def traced(name, **kw):
        events.append(name)
        return event(name, **kw)

    monkeypatch.setattr(h._rx_window, 'event', traced)
    _Clock.t += len(io.sent[-1]) / M.FS
    assert h._query_final_answer()
    request = on_air(M.synth_burst(
        h.caller, F.for_bw(F.SESSION_TURN_REQUEST_RESPONDER, h.bw)))
    feed(h, request)
    due = h.final_request_due_in()
    assert due is not None
    _Clock.t += due
    assert h.service_final_request()
    assert h._tx_pending is None and h._final_request_granted, io.msgs
    assert h.turn == A._TURN_PEER
    # The first query spends this grant's NAK. Repeating the already aired
    # drained handover must retain that exact obligation and its sent phase.
    assert query(h, phase=0)
    obligation = h._rx_window.A
    assert obligation.kind == 'nak' and obligation.phase == 0 and obligation.sent
    before = list(events)
    greeting, sent = h._greeting_at, len(io.sent)
    _Clock.t += 1
    feed(h, request)
    assert len(io.sent) == sent + 1, io.msgs
    drained = M.synth_burst(h.called, F.for_bw(F.SESSION_DRAINED, h.bw))
    assert (io.sent[-1] == drained).all()
    assert events == before
    assert h._rx_window.A is obligation
    assert (obligation.phase, obligation.sent) == (0, True)
    assert h._greeting_at == greeting and not io.host


@pytest.mark.parametrize('kind', [F.SESSION_TURN_REQUEST_SHORT, F.SESSION_TURN_REQUEST],
                         ids=['short', 'long'])
def test_repeated_request_after_unread_data_cannot_restore_greeting(kind, monkeypatch, clock):  # noqa: F811
    h, io = station(high=False)
    h._answer_owed, h._owed_block = A._OWED_OVER, True
    events = []
    event = h._rx_window.event

    def traced(name, **kw):
        events.append(name)
        return event(name, **kw)

    monkeypatch.setattr(h._rx_window, 'event', traced)
    h._grant_rx_window()  # G at 1000.
    assert h._greeting_at == 1000 and h._rx_window.before_data
    _Clock.t = 1010
    assert receive(monkeypatch, h, A._UNDECODED)  # X/E and an aired NAK.
    assert events == ['G', 'X', 'E']
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert not h._rx_window.before_data and h._reacks == h._since_progress == 1
    request = on_air(M.synth_burst(h.caller, F.for_bw(kind, h.bw)))
    _Clock.t = 1119.92 - len(request) / M.FS
    sent = len(io.sent)
    feed(h, request, block=len(request))
    assert _Clock.t == pytest.approx(1119.92)
    assert len(io.sent) == sent + 1  # The grant itself must still be repeated.
    after_request = h._rx_window.before_data, h._greeting_at, list(events)
    # Astra's ten phaseless recovery queries, through 1213, without a T.
    for at in range(1123, 1214, 10):
        _Clock.t = at
        if h.state == A.VaraState.CONNECTED:
            query(h)
    assert after_request == (False, 1000, ['G', 'X', 'E'])
    assert h.state == A.VaraState.DISCONNECTED
    assert h._receive_reports[-1].code == 'budget' and not io.host


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('short', [False, True], ids=['long', 'short'])
def test_initiator_repeated_request_does_not_regrant(bw, route, short, monkeypatch, clock):  # noqa: F811
    h, io = station('initiator', high=True, bw=bw)
    h._expect_level, h._rx_read = 4, (4, None)
    h._answer_owed, h._owed_block = A._OWED_OVER, True
    events = []
    event = h._rx_window.event

    def traced(name, **kw):
        events.append(name)
        return event(name, **kw)

    monkeypatch.setattr(h._rx_window, 'event', traced)
    h._grant_rx_window()
    _Clock.t = 1010
    assert receive(monkeypatch, h, A._UNDECODED)
    assert events == ['G', 'X', 'E']
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    budget = h._reacks, h._since_progress
    kind = (F.SESSION_TURN_REQUEST_RESPONDER_SHORT if short
            else F.SESSION_TURN_REQUEST_RESPONDER)
    audio = on_air(M.synth_burst(h.caller, F.for_bw(kind, bw)))
    _Clock.t = 1119.92 - len(audio) / M.FS
    sent = len(io.sent)
    if route == 'stream':
        feed(h, audio, block=len(audio))
    else:
        _Clock.t += len(audio) / M.FS
        h.on_rx_audio(audio)
    # The existing initiator recognizer accepts long requests only. Short
    # requests must remain harmless to the same G/exemption/deadline state.
    assert len(io.sent) == sent + (not short), io.msgs
    assert events == ['G', 'X', 'E']
    assert h._greeting_at == 1000 and not h._rx_window.before_data
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert (h._reacks, h._since_progress) == budget
    for at in range(1123, 1214, 10):
        _Clock.t = at
        if h.state == A.VaraState.CONNECTED:
            query(h)
    assert h.state == A.VaraState.DISCONNECTED
    assert h._receive_reports[-1].code == 'budget' and not io.host


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('turn', [A._TURN_OURS, A._TURN_ASKED])
@pytest.mark.parametrize('expiry', ['idle', 'query'])
def test_real_release_grants_after_turn_change_without_renewing_cap(
        role, turn, expiry, monkeypatch, clock):  # noqa: F811
    h, io = station(role, turn=turn)
    events = []
    event = h._rx_window.event

    def traced(name, **kw):
        events.append(name)
        return event(name, **kw)

    monkeypatch.setattr(h._rx_window, 'event', traced)
    assert h._rx_window.A is None and not h._rx_window.before_data
    assert h._release_turn()  # Real keyed release, never inject G.
    assert h.turn == A._TURN_PEER and len(io.sent) == 1
    assert events == ['T', 'G']
    obligation = h._rx_window.A
    assert obligation is not None and obligation.kind == 'nak'
    assert h._rx_window.before_data and h._greeting_at == 1000
    assert h._query_data_due
    grants = events.count('G')
    _Clock.t = 1119.9
    # Re-key the release of this same turn, then hear its first query.
    assert h._release_turn() and len(io.sent) == 2
    assert query(h)
    assert events == ['T', 'G', 'Q']
    assert events.count('G') == grants
    assert h._rx_window.A is obligation and h._rx_window.before_data
    assert h._greeting_at == 1000 and h.state == A.VaraState.CONNECTED
    assert h._reacks == h._since_progress == 0 and not io.host
    # Conductor 3b-2aa continued: this is a handover G, never a setup cap.
    # Preserve its obligation and timestamp while polling for at least 139 s.
    for at in range(1120, 1140):
        _Clock.t = at
        if expiry == 'idle':
            h.idle_keepalive()
            assert query(h, explicit=False)
        else:
            assert query(h)
            h.idle_keepalive()
        assert h.state == A.VaraState.CONNECTED
        assert not any(report.code == 'budget' for report in h._receive_reports)
        assert not io.host
    assert events.count('G') == grants and h._rx_window.A is obligation
    assert h._greeting_at == 1000 and h._reacks == h._since_progress == 0
