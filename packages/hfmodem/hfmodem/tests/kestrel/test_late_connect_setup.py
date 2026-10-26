# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KQ4QWH 2026-10-25: setup crossed the request deadline; its reply was lost."""
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_frames as F
from hfmodem.kestrel.vara.vara_arq import VaraState
from hfmodem.tests.kestrel.test_session_retry import runtime


def late_setup(*, refused=False):
    owner, modem, audio, clock = runtime()
    owner.step()
    hs = modem.station
    hs.called = 'KQ4QWH'
    request_end = owner._attempt_end

    def response(timeout, hs=None):
        audio.next_rx_burst = lambda **kw: None
        clock.now = request_end - 1
        audio.refused = refused
        hs.on_rx_tones(F.handshake_tones('KQ4QWH', F.connect_response('500', 4)),
                       F.connect_response('500'))
        return None

    audio.next_rx_burst = response
    owner.step()
    return owner, modem, audio, clock, request_end


def test_recorded_reply_completes_setup_after_original_deadline():
    owner, modem, audio, clock, request_end = late_setup()
    assert clock.now > request_end
    assert modem.station.connect_setup_pending
    assert modem.observer.closes == 0
    assert owner._attempt_end == pytest.approx(modem.last_unkey_at + 30)
    path = Path(__file__).with_name('fixtures') / 'kq4qwh-connect-1025' / 'reply.wav'
    with wave.open(str(path), 'rb') as w:
        reply = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(float)/32768
    audio.next_rx_burst = lambda **kw: reply
    clock.now += 5
    owner.step()
    assert modem.connected
    assert modem.station.state == VaraState.CONNECTED
    assert modem.observer.closes == 0


def test_setup_retries_do_not_replenish_stage_window():
    owner, modem, audio, clock, _ = late_setup()
    deadline = owner._attempt_end
    clock.now = owner._retry_at
    owner.step()
    assert len(audio.bursts) == 3
    assert owner._attempt_end == deadline
    audio.receiving = True  # Busy reception cannot prolong this window either.
    clock.now = deadline
    owner.step()
    assert modem.station.state != VaraState.CONNECTING
    assert modem.observer.closes == 1


def test_refused_setup_does_not_buy_more_time():
    owner, modem, audio, clock, request_end = late_setup(refused=True)
    assert not modem.station.connect_setup_pending
    assert owner._attempt_end == request_end
    clock.now = request_end
    owner.step()
    assert modem.station.state != VaraState.CONNECTING
    assert modem.observer.closes == 1


def test_no_answer_still_expires_at_request_deadline():
    owner, modem, audio, clock = runtime()
    owner.step()
    deadline = owner._attempt_end
    audio.receiving = True
    clock.now = deadline
    owner.step()
    assert modem.station.state != VaraState.CONNECTING
    assert modem.observer.closes == 1


def test_operator_abort_during_setup_is_immediate():
    owner, modem, audio, clock, _ = late_setup()
    modem.abort()
    owner.step()
    assert modem.station.state != VaraState.CONNECTING
    assert not modem.connected


def test_new_contact_gets_its_own_setup_window():
    owner, modem, audio, clock, _ = late_setup()
    modem.abort()
    owner.step()
    modem.connect('W9SSJ', 'KQ4QWH')
    owner.step()
    assert not owner._setup_window_started
    deadline = owner._attempt_end
    clock.now = deadline - 1
    hs = modem.station
    hs.on_rx_tones(F.handshake_tones('KQ4QWH', F.connect_response('500', 4)),
                   F.connect_response('500'))
    owner.step()
    assert modem.station.connect_setup_pending
    assert owner._attempt_end > deadline


def test_early_setup_does_not_shorten_existing_attempt():
    owner, modem, audio, clock = runtime()
    owner.step()
    deadline = owner._attempt_end
    hs = modem.station
    hs.on_rx_tones(F.handshake_tones('WI9YZA', F.connect_response('500', 4)),
                   F.connect_response('500'))
    owner.step()
    assert owner._attempt_end == deadline
    assert owner._setup_window_started
