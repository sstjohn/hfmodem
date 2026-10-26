# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Production ARDOP owner and station audio seam, with no devices opened."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.besra.arq import session as S
from hfmodem.besra.frame import callsign
from hfmodem.besra.host.protocol import ArdopState
from hfmodem.besra.phy import modulator
from hfmodem.besra.station_runtime import StationArdopModem
from hfmodem.core.audio import Block
from hfmodem.core.resample import to_card
from hfmodem.station.link import BesraLink
from hfmodem.station.arbiter import TxArbiter


class Observer:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        return lambda *args: self.events.append((name, args, threading.current_thread().name))


def wait_for(predicate):
    until = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < until
        time.sleep(.01)


def station():
    sent = []
    st = SimpleNamespace(audio=SimpleNamespace(samples=0),
                         rig=SimpleNamespace(dial=lambda: 7100000),
                         cfg=SimpleNamespace(rig=SimpleNamespace(
                             power_w=10, ptt=SimpleNamespace(settle_s=0))),
                         arbiter=None)
    st.arbiter = TxArbiter(st.audio, st.rig)
    st.arbiter.submit = lambda req: sent.append(req) or 48000
    return st, sent


def test_owner_applies_controls_and_rejects_codec_stop_without_retirement():
    modem = StationArdopModem(lambda samples: None)
    observer = Observer()
    modem.start(observer)
    try:
        modem.set_mycall('W9SSJ')
        modem.set_listen(True)
        modem.transmit(b'pending')
        generation = modem.generation
        with pytest.raises(ValueError, match='idle lane'):
            modem.set_codec(False)
        assert modem.generation == generation
        assert modem.queued == 7
        assert modem.state == ArdopState.DISC
        assert modem.core._session.discarded_total == 0
        assert modem.core._session.mycall == 'W9SSJ'
        assert modem.core._session.listen
        assert {thread for _, _, thread in observer.events} == {'station-ardop-owner'}
    finally:
        modem.stop()
    assert not modem._thread.is_alive()
    assert modem.core._session is None


def test_old_contact_frames_never_reach_replacement_owner():
    modem = StationArdopModem(lambda samples: None)
    modem.start(Observer())
    received = []
    modem.core.receive_frames = lambda frames: received.extend(frames)
    try:
        old = modem.generation
        modem.abort()
        modem.deliver_frames(['old'], generation=old)
        modem.deliver_frames(['current'])
        wait_for(lambda: received)
        assert received == ['current']
    finally:
        modem.stop()


def test_failed_submission_unkeys_and_preserves_original_bytes(monkeypatch):
    monkeypatch.setattr(modulator, 'render_frame',
                        lambda *a, **kw: np.array([-1000, 0, 1000], dtype='<i2'))
    def refuse(samples):
        raise OSError('station arbiter refused')
    modem = StationArdopModem(refuse)
    observer = Observer()
    modem.start(observer)
    try:
        modem.set_mycall('W9SSJ')
        modem.transmit(b'original')
        with pytest.raises(OSError, match='arbiter refused'):
            modem.connect('K7ABC')
        assert modem.state == ArdopState.DISC
        assert modem.core._session.last_unconfirmed == b'original'
        assert modem.core._session.confirmed_total == 0
        ptt = [args[0] for name, args, _ in observer.events if name == 'modem_ptt']
        assert ptt == [True, False]
    finally:
        modem.stop()


def test_station_transmit_uses_arbiter_native_rate_and_effective_drive(monkeypatch):
    st, sent = station()
    link = BesraLink(st)
    monkeypatch.setattr(modulator, 'render_frame',
                        lambda *a, **kw: np.array([-10000, 0, 10000], dtype='<i2'))
    link.modem.start(Observer())
    try:
        link.modem.set_mycall('W9SSJ')
        link.modem.set_drivelevel(25)
        link.modem.connect('K7ABC')
        assert len(sent) == 1
        assert sent[0].rate == 12000
        assert sent[0].protocol == 'ardop'
        assert sent[0].audio.dtype == np.dtype('<i2')
        assert sent[0].audio.tolist() == [-2500, 0, 2500]
        assert link.active
        link.modem.abort()
        assert not link.active
    finally:
        link.stop()


def conreq_audio():
    payload = callsign.pack_callsign('K7ABC') + callsign.pack_callsign('W9SSJ')
    native = modulator.render_frame(S.CONREQ_MAX[500], payload=payload, session_id=255)
    return np.concatenate([np.zeros(4800, np.float32), to_card(native, 12000),
                           np.zeros(48000, np.float32)])


def feed(link, samples):
    for at in range(0, len(samples), 4800):
        block = samples[at:at+4800]
        link.station.audio.samples = at + len(block)
        link.lane.push(Block(0, at, block))
        link.poll()


def test_rolling_card_receiver_decodes_once_and_posts_to_owner():
    st, _ = station()
    link = BesraLink(st)
    received = []
    link.modem.start(Observer())
    link.modem.core.receive_frames = lambda frames: received.extend(
        (frame, threading.current_thread().name) for frame in frames)
    try:
        feed(link, conreq_audio())
        wait_for(lambda: received)
        assert len(received) == 1
        frame, thread = received[0]
        assert frame.ok and frame.caller == 'K7ABC' and frame.target == 'W9SSJ'
        assert thread == 'station-ardop-owner'
        assert np.max(np.abs(link._decoder._buf)) < 1
    finally:
        link.stop()


def test_contact_birth_discards_audio_already_captured_before_new_identity():
    st, _ = station()
    link = BesraLink(st)
    received = []
    link.modem.start(Observer())
    link.modem.core.receive_frames = lambda frames: received.extend(frames)
    try:
        audio = conreq_audio()
        st.audio.samples = len(audio)
        link.lane.push(Block(0, 0, audio))
        link.modem.set_mycall('W9SSJ')
        link.poll()
        assert not len(link._decoder._buf)
        assert link._next == len(audio)
        assert not received
    finally:
        link.stop()


def test_station_host_uses_the_same_modem_and_reports_effective_listen():
    from hfmodem.station.hosts import _build_one
    st, _ = station()
    link = BesraLink(st)
    st.lanes = {'ardop': link}
    served = _build_one('ardop', SimpleNamespace(host='ardop', cmd_port=0, data_port=0), st)
    assert served.server.modem is link.modem
    assert served.server._cfg['LISTEN'] == 'FALSE'
    assert served.keys
    link.stop()


def test_birth_floor_survives_successive_polls_of_capture_backlog():
    st, _ = station()
    link = BesraLink(st)
    link.modem.start(Observer())
    try:
        st.audio.samples = 14400
        link.modem.set_mycall('W9SSJ')
        generation = link.modem.generation
        for at in (0, 4800, 9600):
            link.lane.push(Block(0, at, np.ones(4800, np.float32)*.1))
            link.poll()
            assert not len(link._decoder._buf)
            assert link._next == 14400
        assert link.modem.generation == generation
    finally:
        link.stop()


def test_own_transmit_flush_keeps_the_live_contact_and_real_loss_retires_it(monkeypatch):
    st, _ = station()
    link = BesraLink(st)
    monkeypatch.setattr(modulator, 'render_frame',
                        lambda *a, **kw: np.array([-1000, 0, 1000], dtype='<i2'))
    link.modem.start(Observer())
    try:
        link.modem.set_mycall('W9SSJ')
        link.modem.transmit(b'original')
        link.modem.connect('K7ABC')
        generation = link.modem.generation
        st.audio.samples = 10000
        link.lane.flush_to(10000)
        link.lane.push(Block(0, 10000, np.zeros(4800, np.float32)))
        link.poll()
        assert not link.modem._fault.is_set()
        assert link.modem.generation == generation
        assert link.active
        st.audio.samples = 16000
        link.lane.fault(16000, 'timestamp loss')
        link.lane.push(Block(0, 16000, np.zeros(4800, np.float32)))
        link.poll()
        wait_for(lambda: not link.active)
        assert link.modem.core._session.last_unconfirmed == b'original'
        assert link.modem.core._session.confirmed_total == 0
    finally:
        link.stop()


def test_station_refusal_does_not_publish_logical_ptt(monkeypatch):
    st, _ = station()
    st.arbiter = None
    link = BesraLink(st)
    observer = Observer()
    monkeypatch.setattr(modulator, 'render_frame',
                        lambda *a, **kw: np.array([-1000, 0, 1000], dtype='<i2'))
    link.modem.start(observer)
    try:
        link.modem.set_mycall('W9SSJ')
        with pytest.raises(OSError, match='cannot transmit'):
            link.modem.connect('K7ABC')
        assert not [event for event in observer.events if event[0] == 'modem_ptt']
        assert not link.active
    finally:
        link.stop()
