# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""An enabling control bounds ordinary P3 headers, not global receive events."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import live, onair, rxfront


def packet(header_start):
    start = round(header_start * rxfront.FS) + rxfront.p3frame.DATA_OFFSET * rxfront.SPS
    return SimpleNamespace(start=start, sl=1, status=0x03, payload=b'1w9ss',
                           long_cycle=False, carrier_swapped=False,
                           report=lambda: 'CRC-qualified ordinary test field')


@pytest.mark.parametrize('header_start,expected', [(.04, False), (1.15, False),
                                                  (1.2, True), (1.205, True),
                                                  (1.59, True)])
def test_nominal_header_floor_not_data_row_or_filter_history(monkeypatch, header_start, expected):
    field = packet(header_start)
    monkeypatch.setattr(rxfront.p3rx, 'decode_p3_packets',
                        lambda audio, **kwargs: SimpleNamespace(packets=[field]))
    events = list(rxfront.p3_packet_events(np.zeros(144000), envelope=False, not_before=1.2))
    assert len(events) == int(expected)
    if events:
        assert events[0].start == field.start
        assert events[0].t == field.start / rxfront.FS
        assert events[0].packet[-1] is True


def test_default_helper_preserves_prior_unbounded_behavior(monkeypatch):
    fields = [packet(.04), packet(1.59)]
    monkeypatch.setattr(rxfront.p3rx, 'decode_p3_packets',
                        lambda audio, **kwargs: SimpleNamespace(packets=fields))
    assert len(list(rxfront.p3_packet_events(np.zeros(144000), envelope=False))) == 2


@pytest.mark.parametrize('external', [False, True])
@pytest.mark.parametrize('header_start,accepted', [(.04, False), (1.15, False),
                                                  (1.59, True)])
def test_actual_host_transition_retains_floor_across_flush(monkeypatch, external, header_start, accepted):
    host = onair.PtcHost(mycall='W9SSJ')
    host.arq.on_host_connect('W9SSJ', 'KB5LZK')
    host.no_p3_fallback = True
    host.p1_grant_only = True
    receiver = onair._SessionRx(host, 'OFFLINE')
    origin = 26.1419375
    control = rxfront.Event(1.2, 'cs', 'native-qualified CS4 shape',
                           protocol='PACTOR-1', cs=3, sense=1)
    field = packet(header_start)
    monkeypatch.setattr(rxfront.p3rx, 'decode_p3_packets',
                        lambda audio, **kwargs: SimpleNamespace(packets=[field]))

    def primary(audio, *, p3_packets, p3_packets_after, **kwargs):
        if p3_packets:
            yield from rxfront.p3_packet_events(audio, envelope=False, not_before=p3_packets_after)
        if not external:
            yield control

    monkeypatch.setattr(rxfront, 'decode_events', primary)
    receiver.rx.t0 = origin
    receiver.rx.buf = np.zeros(144000)
    if external:
        receiver._on(rxfront.Event(origin + control.t, control.kind, control.text,
                                    protocol=control.protocol, cs=control.cs, sense=control.sense))
    receiver.rx._decode(receiver.rx.buf)
    assert host.arq.state == onair.State.CONNECTED
    assert receiver.rx._p3_packets_after == pytest.approx(origin + control.t)
    assert bool(host.arq._peer_heard) is accepted
    assert receiver.count == (2 if accepted else 1)
    receiver.rx.flush()
    assert bool(host.arq._peer_heard) is accepted
    assert receiver.count == (2 if accepted else 1)


def test_default_rolling_has_no_epoch_floor(monkeypatch):
    flags = []
    def primary(audio, **kwargs):
        flags.append((kwargs['p3_packets'], kwargs['p3_packets_after']))
        return iter(())
    monkeypatch.setattr(rxfront, 'decode_events', primary)
    receiver = live.RollingRx(lambda event: None)
    receiver._decode(np.zeros(24000))
    assert flags == [(True, None)]
