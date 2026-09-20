# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Opt-in host timing, with all audio/DSP/rig work replaced by tiny mocks."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import corpora


@pytest.fixture
def transport(monkeypatch):
    kc = corpora.harness("kestrel_connect")
    clock = SimpleNamespace(now=100.0)

    def advance(seconds):
        clock.now += seconds

    monkeypatch.setattr(kc.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(kc.time, "sleep", advance)
    monkeypatch.setattr(kc, "_BracketSegmenter", lambda: SimpleNamespace(restart=lambda: None))
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(
        query_devices=lambda _dev: {"max_output_channels": 2},
        InputStream=lambda **kw: SimpleNamespace()))

    def drive(samples, _amp):
        advance(.002)
        return samples

    monkeypatch.setattr(kc.levels, "at_drive", drive)
    made = []

    def make(trace=False, *, armed=True, refuse=False, retired=False, rig=True,
             monitored=False, drain_late_s=4 / kc.FS):
        calls = []

        def key(on):
            calls.append(on)
            advance(.010)
            return not on or (armed and not refuse)

        # `drain_late_s` is the capture the monitor took between our last sample
        # and `play_drained` returning — the lag `tx` reads the keyed hold off.
        late = int(drain_late_s * kc.FS)
        radio = SimpleNamespace(key=key, armed=armed, retired=retired) if rig else None
        io = kc.AudioVaraIO("mock" if monitored else "mock-out", "mock",
                            rig=radio, tx_tail=.050, trace_timing=trace)
        io.samples = 1000
        played = []

        def play(block, _fs, _device):
            advance(.500)
            played.append(block.copy())
            io.samples += 4 + late
            io._buf.append(np.zeros(4 + late))

        def capture_end(audio, expected):
            assert len(audio) == 4 + late and expected == 4
            advance(.004)
            return expected

        monkeypatch.setattr(kc, "play_drained", play)
        monkeypatch.setattr(kc, "_tx_end_in_capture", capture_end)
        made.append((io, calls, played))
        return io, calls, played

    return kc, clock, make


def records(capsys):
    return [dict(item.split("=", 1) for item in line.split()[1:])
            for line in capsys.readouterr().out.splitlines() if line.startswith("timing ")]


def test_trace_separates_software_key_prep_play_drain_cursor_and_tail(transport, capsys):
    _, _, make = transport
    io, calls, played = make(True)
    io.trace_timing_event("occupancy_ready", forced=0, window_s=8.0, clear_windows=1)
    io.key(True)
    io.tx(np.zeros(4))
    assert capsys.readouterr().out == "", "formatting/output must wait until key-down returns"
    io.key(False)
    trace = records(capsys)
    assert [r["event"] for r in trace] == [
        "occupancy_ready", "key_request", "key_return", "tx_entry", "play_start",
        "play_return", "capture_cursor", "tail_start", "tail_end", "tx_return",
        "key_request", "key_return"]
    stamps = [float(r["mono_s"]) for r in trace]
    assert stamps == sorted(stamps)
    by_event = {r["event"]: r for r in trace[3:10]}
    duration = lambda a, b: float(by_event[b]["mono_s"]) - float(by_event[a]["mono_s"])
    assert stamps[2] - stamps[1] == pytest.approx(.010)
    assert duration("tx_entry", "play_start") == pytest.approx(.002)
    assert duration("play_start", "play_return") == pytest.approx(.500)
    assert duration("play_return", "capture_cursor") == pytest.approx(.004)
    assert duration("tail_start", "tail_end") == pytest.approx(.050)
    assert by_event["play_return"]["capture_samples"] == "1008"
    cursor = by_event["capture_cursor"]
    assert (cursor["buffer_samples"], cursor["tx_end_in_buffer"], cursor["cursor"],
            cursor["guard_samples"]) == ("8", "4", "4", "0")
    assert calls == [True, False] and len(played) == 1
    assert len({r["key_seq"] for r in trace[1:]}) == 1
    assert all(r["tx_seq"] == "1" for r in trace[3:])


@pytest.mark.parametrize("late,sleep", [(.09, 0.0), (.02, .03), (0.0, .05)])
def test_the_keyed_hold_is_spent_from_our_last_sample_not_from_the_drains_return(
        transport, capsys, late, sleep):
    """The idle hold is what is owed past our audio, and the drain has already paid
    some of it.

    `play_drained` opens and closes a stream per burst, and `Pa_StopStream` returns
    when the device has finished rather than when our last sample went out: 0.14 to
    0.20 s past it on the cable and ~0.09 s on the rig, where the 2026-09-11 KC9GHZ
    recordings hold PTT up 0.14-0.15 s past our last sample for a 0.05 hold. That
    is 0.09 s of unmodulated carrier on a shared band and 0.09 s of a peer's answer
    the receiver is muted through, and nothing asked for either.

    The monitor measures it — our last sample and the drain's return are both
    places in the capture, so the input latency cancels — and the sleep is the
    remainder. It cannot clip a tail: the shortest it goes is zero, which is the
    unkey happening where it happened before there was a hold at all.
    """
    _, _, make = transport
    io, calls, _ = make(True, monitored=True, drain_late_s=late)
    io.key(True)
    io.tx(np.zeros(4))
    io.key(False)
    trace = records(capsys)
    by_event = {r["event"]: r for r in trace}
    assert float(by_event["tail_start"]["drain_late_s"]) == pytest.approx(late)
    assert float(by_event["tail_start"]["sleep_s"]) == pytest.approx(sleep)
    assert (float(by_event["tail_end"]["mono_s"])
            - float(by_event["tail_start"]["mono_s"])) == pytest.approx(sleep)
    assert late + sleep >= io.tx_tail, "the key came up owing the peer a symbol"
    assert calls == [True, False]


def test_default_trace_is_quiet_and_reads_no_monotonic_clock(transport, monkeypatch, capsys):
    kc, _, make = transport
    io, _, played = make()
    monkeypatch.setattr(kc.time, "monotonic", lambda: pytest.fail("disabled trace read a clock"))
    io.key(True)
    io.tx(np.zeros(4))
    io.key(False)
    assert capsys.readouterr().out == ""
    assert len(played) == 1


@pytest.mark.parametrize("mode,outcome", [("refused", "key_refused"), ("retired", "retired")])
def test_trace_preserves_suppression_without_touching_waveform(transport, capsys, mode, outcome):
    _, _, make = transport
    io, calls, played = make(True, refuse=mode == "refused", retired=mode == "retired")
    io.key(True)
    io.tx(None)  # Guards must still precede waveform processing.
    io.key(False)
    trace = records(capsys)
    assert not played and calls == [True, False]
    assert not any(r["event"] == "play_start" for r in trace)
    assert next(r for r in trace if r["event"] == "tx_return")["outcome"] == outcome


def test_disarmed_and_no_rig_bench_paths_still_play(transport, capsys):
    _, _, make = transport
    for opts in ({"armed": False}, {"rig": False}):
        io, _, played = make(True, **opts)
        io.key(True)
        io.tx(np.zeros(4))
        io.key(False)
        assert len(played) == 1
        trace = records(capsys)
        assert any(r["event"] == "play_return" for r in trace)


def test_playback_exception_is_traced_and_propagated(transport, monkeypatch, capsys):
    kc, _, make = transport
    io, calls, _ = make(True)

    def fail(*_args):
        raise RuntimeError("sample payload must never be logged")

    monkeypatch.setattr(kc, "play_drained", fail)
    io.key(True)
    with pytest.raises(RuntimeError):
        try:
            io.tx(np.zeros(4))
        finally:
            io.key(False)
    output = capsys.readouterr().out
    assert "event=tx_error" in output and "error=RuntimeError" in output
    assert "sample payload" not in output and "event=tail_start" not in output
    assert calls == [True, False]


def test_ready_occupancy_decision_shares_first_key_clock(transport, monkeypatch, capsys):
    kc, clock, make = transport
    io, _, _ = make(True)

    def sense(seconds, _bw):
        clock.now += seconds
        return False, -2.0

    monkeypatch.setattr(io, "channel_busy", sense)
    monkeypatch.setattr(io, "rx_level_range", lambda _seconds: (-30.0, -40.0))
    monkeypatch.setattr(io, "prime_floor", lambda: None)
    assert kc.wait_clear_channel(io, "2750", 8, False)
    clock.now += .007
    io.key(True)
    io.tx(np.zeros(4))
    io.key(False)
    trace = records(capsys)
    assert trace[0]["event"] == "occupancy_ready"
    assert float(trace[1]["mono_s"]) - float(trace[0]["mono_s"]) == pytest.approx(.007)


@pytest.mark.parametrize("trace", [False, True])
def test_cli_routes_timing_flag_without_opening_devices(monkeypatch, trace):
    kc = corpora.harness("kestrel_connect")
    options = []
    stopped = []

    def io(*args, **kwargs):
        options.append(kwargs)
        return SimpleNamespace(stop=lambda: stopped.append(True))

    def refused(*args, **kwargs):
        raise kc.ChannelBusy

    monkeypatch.setattr(kc, "AudioVaraIO", io)
    monkeypatch.setattr(kc, "VaraStationHandshake", lambda *a, **kw: object())
    monkeypatch.setattr(kc, "attach_mail", lambda *a, **kw: (None, None))
    monkeypatch.setattr(kc, "connect", refused)
    monkeypatch.setattr(sys, "argv", ["kestrel_connect", "--gateway", "W1AAA",
                                      "--mycall", "W9SSJ", "--no-record"]
                        + (["--trace-timing"] if trace else []))
    assert kc.main() == kc.CHANNEL_BUSY
    assert options[0]["trace_timing"] is trace
    assert stopped == [True]


def test_key_exception_is_reported_without_claiming_return(transport, capsys):
    _, _, make = transport
    io, _, played = make(True)

    def failed_key(_on):
        raise OSError("private device details")

    io.rig.key = failed_key
    with pytest.raises(OSError):
        io.key(True)
    assert capsys.readouterr().out == ""
    io.rig.key = lambda on: True
    io.key(False)
    trace = records(capsys)
    assert [r["event"] for r in trace] == ["key_request", "key_error", "key_request", "key_return"]
    assert trace[1]["error"] == "OSError"
    assert not played


def test_refused_region_does_not_suppress_next_accepted_key(transport, capsys):
    _, _, make = transport
    io, _, played = make(True, refuse=True)
    io.key(True)
    io.tx(None)
    io.key(False)
    io.rig.key = lambda on: True
    io.key(True)
    io.tx(np.zeros(4))
    io.key(False)
    trace = records(capsys)
    assert len(played) == 1
    entries = [r for r in trace if r["event"] == "tx_return"]
    assert [(r["key_seq"], r["tx_seq"], r["outcome"]) for r in entries] == [
        ("1", "1", "key_refused"), ("2", "2", "returned")]


def test_failed_unkey_keeps_output_buffered_until_confirmed(transport, capsys):
    _, _, make = transport
    io, _, _ = make(True)
    io.key(True)
    io.tx(np.zeros(4))
    io.rig.key = lambda on: False
    io.key(False)
    assert capsys.readouterr().out == ""
    io.rig.key = lambda on: True
    io.key(False)
    trace = records(capsys)
    assert [r["ok"] for r in trace if r["event"] == "key_return"] == ["1", "0", "1"]


def test_default_trace_does_not_require_constructor_only_fields(monkeypatch, capsys):
    kc = corpora.harness("kestrel_connect")
    io = object.__new__(kc.AudioVaraIO)
    io.rig = None
    io._refused = True
    io.key(True)
    io.tx(None)
    io.key(False)
    assert "timing " not in capsys.readouterr().out
