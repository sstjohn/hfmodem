# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair
from hfmodem.tests.shrike.onair_support import _Bench as _Bench
from hfmodem.tests.shrike.p3_cycle_support import NativeCycleClock, _ClockDone as _ClockDone


FS = onair.FS
AT = 48000
HELD = np.full(128, 1, np.float32)
FRESH = np.full(128, 2, np.float32)


@pytest.fixture
def live(monkeypatch):
    native = onair._LiveInput.__new__(onair._LiveInput)
    native.fs, native._blk, native._lat = FS, 128, 0
    native._duplex = True
    native.geometry_n = native.tx_latency_n = 0
    native.samples = 0
    native._dac = (0.0, 0)
    native._armed = native._tx = None
    native.submitted = []
    native._tx_end_time = 0.0
    monkeypatch.setattr(onair.time, 'monotonic', lambda: native.samples / FS)
    # The tests drive the keyer synchronously so no worker outlives the fixture.
    monkeypatch.setattr(onair.threading, 'Thread',
                        lambda **kwargs: SimpleNamespace(start=lambda: None))

    def spend(n):
        native.samples += n

    def drain(timeout):
        assert native._tx is not None
        native.submitted.append((native._tx_at, native._tx.copy()))
        native.output_enqueue_sample = native._tx_at
        native.output_end_sample = native._tx_at + native._tx.size
        native._tx_end_time = native._dac_time(native.output_end_sample)
        native._tx = None
        return True

    native.spend = spend
    native._sleep_until = lambda t: spend(max(0, round(t * FS) - native.samples))
    native.sample_now = lambda: native.samples
    native._tx_done = SimpleNamespace(clear=lambda: None, wait=drain)
    return native


def _arm(live, *, at=AT, key=None):
    return live._queue(HELD, at=at, settle=.04, key=key, max_key=40)


@pytest.mark.parametrize('delta', [-1, 0, 1])
@pytest.mark.parametrize('latency,correction,geometry', [(0, 0, 0), (1152, 240, 96)])
def test_cnew_exact_delivered_sample(live, delta, latency, correction, geometry):
    live._lat, live.tx_latency_n, live.geometry_n = latency, correction, geometry
    start = AT + geometry - latency - correction
    live.spend(start - 384 + delta)
    assert live.samples > start - 1920
    if delta > 0:
        with pytest.raises(onair._MissedTxSlot):
            _arm(live)
        assert live._armed is None and live._tx is None
    else:
        burst = _arm(live)
        assert (burst.at, burst.start, burst.first) == (AT, start, AT + geometry)
        assert not burst.keying and not burst.done.is_set()
        np.testing.assert_array_equal(live._tx, HELD)


@pytest.mark.parametrize('delta', [-1, 0, 1])
@pytest.mark.parametrize('keying', [False, True])
def test_cword_exact_delivered_sample(live, delta, keying):
    burst = _arm(live)
    burst.keying = keying
    live.spend(47616 + delta)
    live._propose(burst, FRESH, AT)
    assert burst.missed == max(0, delta)
    np.testing.assert_array_equal(live._tx, HELD if delta > 0 else FRESH)
    assert burst.start == AT


@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_cwd_exact_keyer_transition_sample(live, delta):
    results = []

    def key(up):
        if up:
            live.spend(delta)
            results.append((live.samples, live.disarm()))

    burst = _arm(live, key=key)
    if delta < 0:
        live.spend(46080 + delta)
        # A stopped keyer wakes before disarm joins it, as the live worker does.
        wait = burst.done.wait

        def stopped(timeout):
            live._key(burst)
            return wait(0)

        burst.done.wait = stopped
        assert live.disarm()
        assert burst.stop.is_set() and burst.done.is_set()
        assert not burst.keying and live._armed is None
        assert not results and not live.submitted
    else:
        live._key(burst)
        assert results == [(46080 + delta, False)]
        assert burst.keying and burst.done.is_set() and burst.played
        assert not burst.stop.is_set()
        assert live.submitted[0][0] == AT
        np.testing.assert_array_equal(live.submitted[0][1], HELD)


@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_cword_post_ptt_earlier_instant_is_refused(live, delta):
    results = []

    def key(up):
        if up:
            assert not live.disarm()
            live._propose(burst, FRESH, AT + delta)
            results.append(burst.missed)

    burst = _arm(live, key=key)
    live._key(burst)
    assert results == [1 if delta < 0 else 0]
    assert live.submitted[0][0] == AT + max(0, delta)
    np.testing.assert_array_equal(live.submitted[0][1], HELD if delta < 0 else FRESH)


@pytest.mark.parametrize('state', ['enqueued', 'done'])
def test_cword_refused_after_enqueue_or_completion(live, state):
    burst = _arm(live)
    live.spend(47201)
    if state == 'enqueued':
        live.output_enqueue_sample = AT
    else:
        burst.done.set()
    live._propose(burst, FRESH, AT)
    assert burst.missed == 1
    np.testing.assert_array_equal(live._tx, HELD)


@pytest.mark.parametrize('decision', [47201, 47616, 47617])
def test_post_ptt_word_replacement_is_submitted_at_armed_instant(live, decision):
    outcomes = []

    def key(up):
        if up:
            live.spend(decision - live.samples)
            outcomes.append(live.disarm())
            live._propose(burst, FRESH, AT)

    burst = _arm(live, key=key)
    live._key(burst)
    assert outcomes == [False]
    assert burst.missed == (1 if decision == 47617 else 0)
    assert [(at, int(audio[0])) for at, audio in live.submitted] == [
        (48000, 1 if decision == 47617 else 2)]


def test_native_cycle_clock_preserves_a_committed_slot(monkeypatch):
    clock = NativeCycleClock(monkeypatch)
    clock.arm(HELD, at=AT, settle=.04)
    clock.spend(47201)
    assert not clock.disarm()
    assert clock.transmit(FRESH, at=AT, settle=.04) == (48000, 48128)
    assert clock.emissions == [(48000, 48128)]
    assert [(at, int(audio[0])) for at, audio in clock.submitted] == [(48000, 2)]
    assert not clock.missed and clock.armed_at is None


def test_native_cycle_clock_radio_reply(tmp_path, monkeypatch):
    from hfmodem.shrike import arq
    from hfmodem.tests.shrike.p3_reply_support import mail_link

    s, tx, _, grid, _, index = mail_link(tmp_path)
    clock = NativeCycleClock(monkeypatch)
    tx.live = clock
    tx.aim(grid, index + 1)
    onair._p3_place_reply(grid, tx, index + 1)
    tx.arm_p3_reply()
    at = clock.armed_at
    assert at is not None
    clock.spend(at - 600)
    assert not clock.disarm()
    tx._pending_p3_cs = arq.CS_ACK
    tx.emit_pending_cs()
    assert not tx.refused
    assert tx.slots_used[-1:] == [index + 1]
    assert tx.tx_audio_start == at
    assert len(clock.emissions) == len(clock.submitted) == 1
    assert clock.submitted[0][0] == at
    assert tx._p3_emitted_cs == arq.CS_ACK
