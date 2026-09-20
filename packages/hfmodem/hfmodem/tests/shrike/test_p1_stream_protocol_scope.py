# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Protocol scope, control preservation and same-call upgrade transitions."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import live, onair, p1rx, rxfront


@pytest.mark.parametrize('options,expected', [({}, True), ({'p3_packets': True}, True),
                                              ({'p3_packets': False}, False)])
def test_packet_flag_default_and_explicit(monkeypatch, options, expected):
    calls = []
    event = rxfront.Event(.1, 'packet', 'ordinary', protocol='PACTOR-3',
                          packet=(3, 0x21, b'field', True), start=4800)

    def ordinary(audio, *, envelope, not_before=None):
        calls.append(envelope)
        yield event

    monkeypatch.setattr(rxfront, 'p3_packet_events', ordinary)
    events = list(rxfront.decode_events(np.zeros(12000), p3_envelope=False, **options))
    assert calls == ([False] if expected else [])
    assert (event in events) is expected


@pytest.mark.parametrize('index,kind', [(3, 'cs'), (5, 'unassigned')])
def test_suppression_preserves_p1_control_dispatch(monkeypatch, index, kind):
    monkeypatch.setattr(rxfront, '_p1_cs_bursts', lambda audio: [(0., .12)])
    monkeypatch.setattr(p1rx, 'decode_control_signal',
                        lambda *args: p1rx.CS(index, 0, 1))
    monkeypatch.setattr(rxfront, 'p3_packet_events',
                        lambda *args, **kwargs: pytest.fail('ordinary P3 was not suppressed'))
    events = list(rxfront.decode_events(np.zeros(12000), p3_packets=False,
                                       p3_envelope=False))
    got = [ev for ev in events if ev.kind == kind]
    assert len(got) == 1
    assert got[0].protocol == 'PACTOR-1'
    assert got[0].sense == 1
    assert (got[0].cs if kind == 'cs' else got[0].spare) == index


@pytest.mark.parametrize('protocol,state,expected', [
    (onair.Protocol.PACTOR1, onair.State.CONNECTING, False),
    (onair.Protocol.PACTOR1, onair.State.CONNECTED, True),
    (onair.Protocol.PACTOR3, onair.State.CONNECTING, True),
    (onair.Protocol.PACTOR3, onair.State.CONNECTED, True),
])
def test_real_session_context_is_live(protocol, state, expected):
    host = onair.PtcHost(mycall='W9SSJ')
    host.arq.on_host_connect('W9SSJ', 'KB5LZK')
    receiver = onair._SessionRx(host, 'OFFLINE')
    receiver.host = SimpleNamespace(protocol=protocol, arq=SimpleNamespace(state=state))
    assert receiver.rx.p3_packets() is expected
    receiver.host = SimpleNamespace(protocol=onair.Protocol.PACTOR1,
                                    arq=SimpleNamespace(state=onair.State.CONNECTING))
    assert receiver.rx.p3_packets() is False
    receiver.host.arq.state = onair.State.CONNECTED
    assert receiver.rx.p3_packets() is True


@pytest.mark.parametrize('entry', ['push', 'flush'])
@pytest.mark.parametrize('initial,transition', [(False, False), (False, True), (True, False)])
def test_transition_reads_first_field_once(monkeypatch, entry, initial, transition):
    state = {'enabled': initial}
    ordinary_calls = []
    control_calls = []
    delivered = []
    control = rxfront.Event(.1, 'cs', 'native-shape control', protocol='PACTOR-1', cs=3)
    packet = rxfront.Event(.2, 'packet', 'ordinary', protocol='PACTOR-3',
                           packet=(3, 0x21, b'field', True), start=9600)

    def ordinary(audio, *, envelope, not_before=None):
        ordinary_calls.append(envelope)
        yield packet

    def decode(audio, *, p3_packets, **kwargs):
        if p3_packets:
            yield from ordinary(audio, envelope=False)
        control_calls.append(p3_packets)
        yield control

    def receive(event):
        delivered.append(event)
        if transition and event.kind == 'cs':
            state['enabled'] = True

    monkeypatch.setattr(rxfront, 'decode_events', decode)
    monkeypatch.setattr(rxfront, 'p3_packet_events', ordinary)
    receiver = live.RollingRx(receive, window_s=1., keep_s=.5,
                              p3_packets=lambda: state['enabled'])
    receiver.t0 = 2.
    audio = np.zeros(24000)
    if entry == 'push':
        receiver.push(audio)
    else:
        receiver.hold(audio)
        receiver.flush()
    expected = initial or transition
    assert ordinary_calls == ([False] if expected else [])
    assert control_calls == [initial]
    fields = [ev for ev in delivered if ev.kind == 'packet']
    assert len(fields) == int(expected)
    if expected:
        assert fields[0].t == pytest.approx(2.2)
        assert fields[0].start == 9600
    receiver._decode(audio)
    assert len([ev for ev in delivered if ev.kind == 'packet']) == int(expected)
    assert len([ev for ev in delivered if ev.kind == 'cs']) == 1


def test_rolling_default_keeps_ordinary_scan(monkeypatch):
    flags = []
    def decode(audio, **kwargs):
        flags.append(kwargs['p3_packets'])
        return iter(())
    monkeypatch.setattr(rxfront, 'decode_events', decode)
    live.RollingRx(lambda ev: None)._decode(np.zeros(24000))
    assert flags == [True]
