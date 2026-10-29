# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The owner's idle tick never fires inside the reply window of our own TX."""
from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
from hfmodem.kestrel.vara.vara_arq import VaraState


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Station:
    state = VaraState.CONNECTED
    progress = 0
    idle_keyed = 0

    def __init__(self, clock):
        self.clock = clock
        self.ticks = []

    def intermediate_query_due_in(self):
        return None

    data_retry_due_in = idle_poll_due_in = intermediate_query_due_in

    def idle_keepalive(self):
        self.ticks.append(self.clock())


class Modem:
    observer = object()
    bandwidth = '2300'
    unkey_sequence = 0
    last_unkey_at = None

    def __init__(self, station):
        self.station = station

    def poll(self):
        pass

    def report_channel_activity(self, active):
        pass


class Transport:
    receiving = False

    def next_rx_burst(self, timeout, hs=None):
        return None


def owner():
    clock = Clock()
    runtime = AudioHostRuntime(Transport(), clock=clock)
    station = Station(clock)
    runtime.modem = Modem(station)
    return runtime, station, clock


def test_turnaround_ask_reanchors_the_idle_tick_to_its_unkey():
    runtime, station, clock = owner()
    runtime.step()
    assert runtime._idle_at == 10
    # The turnaround ask is keyed from receive processing: neither progress
    # nor idle_keyed moves, only the key.
    clock.now = 9.3
    runtime.modem.last_unkey_at, runtime.modem.unkey_sequence = 9.3, 1
    runtime.step()
    clock.now = 10.0
    runtime.step()
    clock.now = 19.29
    runtime.step()
    assert station.ticks == []
    clock.now = 19.3
    runtime.step()
    assert station.ticks == [19.3]
    assert runtime._idle_at == 29.3


def test_unkey_never_pulls_the_idle_tick_earlier():
    runtime, station, clock = owner()
    runtime.modem.last_unkey_at, runtime.modem.unkey_sequence = -5.0, 1
    runtime.step()
    assert runtime._idle_at == 10
