# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from hfmodem.kestrel.arq import rxwindow as RW


def key_answer(hs, last=False, owes_release=False):
    hs._rx_window.A = RW.Answer(0, hs._rx_window.alt, 'final' if last else 'continue')
    hs._perform_rx_actions(hs._rx_window._key(basis='completed fixture', explicit=False))


def audio_clock(monkeypatch, station_class, fs):
    now = [100.0]
    monkeypatch.setattr('hfmodem.kestrel.vara.vara_arq.time.monotonic', lambda: now[0])
    for name in ('on_rx_audio', 'on_rx_stream'):
        original = getattr(station_class, name)
        def receive(hs, samples, *args, original=original, **kwargs):
            now[0] += len(samples) / fs
            return original(hs, samples, *args, **kwargs)
        monkeypatch.setattr(station_class, name, receive)


def arm_replay(hs):
    anchor = hs._rx_window.K
    assert anchor is not None
    assert hs._rx_window.A.kind in ('continue', 'frame10')
    assert hs._rx_window.A.sent
    assert hs._rx_query(query_start=hs._unkeyed_at + 60)
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert hs._rx_window.T is not None
    assert (hs._rx_window.T.start, hs._rx_window.T.frontier) == (anchor.start, hs._rx_window.S)
