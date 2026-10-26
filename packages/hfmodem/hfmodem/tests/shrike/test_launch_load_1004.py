# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Injected startup advisory controls; no actual load query or launcher."""
import ast
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from hfmodem.core import launch_load


@pytest.mark.parametrize("loads", [(1.2, 0.8, 0.6), (1000000.0, 2000000.0, 3000000.0)])
def test_each_field_is_sampled_once_and_returns_no_admission(loads):
    calls = []
    sink = io.StringIO()
    result = launch_load.emit_load_advisory(
        load_sampler=lambda: calls.append("load") or loads,
        cpu_sampler=lambda: calls.append("cpu") or 8, stream=sink)
    assert result is None
    assert calls == ["load", "cpu"]
    expected = "/".join(f"{value:.2f}" for value in loads)
    assert sink.getvalue() == (f"LOAD ADVISORY: load average (1/5/15 min)={expected}; "
                               "logical CPUs=8; advisory only, no timing guarantee.\n")


@pytest.mark.parametrize("value", [None, (), (1, 2), (1, 2, 3, 4), (True, 0, 0),
                                   (-1, 0, 0), (float("inf"), 0, 0),
                                   (float("nan"), 0, 0), ("1", 0, 0), object()])
def test_invalid_load_retains_independent_cpu_context(value):
    line = launch_load.advisory_line(load_sampler=lambda: value, cpu_sampler=lambda: 4)
    assert line == ("LOAD ADVISORY: load average (1/5/15 min)=unavailable; "
                    "logical CPUs=4; advisory only, no timing guarantee.")


@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, "4", object()])
def test_invalid_cpu_retains_load(value):
    line = launch_load.advisory_line(load_sampler=lambda: (0, 0, 0), cpu_sampler=lambda: value)
    assert "=0.00/0.00/0.00; logical CPUs=unknown;" in line


def test_missing_and_failing_queries_are_closed_independently(monkeypatch):
    monkeypatch.delattr(launch_load.os, "getloadavg", raising=False)
    assert "=unavailable; logical CPUs=3;" in launch_load.advisory_line(cpu_sampler=lambda: 3)
    def failed():
        raise OSError("private diagnostic material")
    assert "=0.00/0.00/0.00; logical CPUs=unknown;" in launch_load.advisory_line(
        load_sampler=lambda: (0, 0, 0), cpu_sampler=failed)
    assert "=unavailable; logical CPUs=3;" in launch_load.advisory_line(
        load_sampler=failed, cpu_sampler=lambda: 3)


@pytest.mark.parametrize("where", ["write", "flush"])
def test_output_oserror_is_not_a_new_refusal_or_duplicate(where):
    events = []
    class Sink:
        def write(self, line):
            events.append("write")
            if where == "write":
                raise OSError("private diagnostic material")
        def flush(self):
            events.append("flush")
            if where == "flush":
                raise OSError("private diagnostic material")
    launch_load.emit_load_advisory(load_sampler=lambda: (0, 0, 0), cpu_sampler=lambda: 1, stream=Sink())
    assert events == (["write"] if where == "write" else ["write", "flush"])


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("field", ["load", "cpu", "write", "flush"])
def test_cancellation_is_not_swallowed(exc, field):
    def cancelled():
        raise exc()
    class Sink:
        def write(self, line):
            if field == "write":
                cancelled()
        def flush(self):
            if field == "flush":
                cancelled()
    with pytest.raises(exc):
        launch_load.emit_load_advisory(
            load_sampler=cancelled if field == "load" else lambda: (0, 0, 0),
            cpu_sampler=cancelled if field == "cpu" else lambda: 1, stream=Sink())


def test_numeric_subclass_cannot_inject_output():
    class Load(float):
        def __format__(self, spec):
            raise AssertionError("should format closed primitive")
    class Cpu(int):
        def __str__(self):
            raise AssertionError("should format closed primitive")
    assert "=1.00/2.00/3.00; logical CPUs=4;" in launch_load.advisory_line(
        load_sampler=lambda: (Load(1), Load(2), Load(3)), cpu_sampler=lambda: Cpu(4))


class BoundaryReached(Exception):
    pass


def _arguments(monkeypatch, tmp_path):
    from hfmodem.shrike import onair
    parsed = []
    with monkeypatch.context() as parse:
        parse.setattr(onair, "run", lambda args: parsed.append(args) or 0)
        parse.setattr(onair.config, "mail_password", lambda *args: None)
        parse.setattr(onair.sys, "argv", ["onair", "--mycall", "TEST", "--dxcall", "PEER"])
        assert onair.main() == 0
    args = parsed[0]
    args.outdir = str(tmp_path / "startup")
    args.transmit = True
    args.listen_only = args.preflight = args.tune = None
    args.reply_wav = None
    args.answer = False
    args.rig = next(name for name, row in onair.RIGS.items()
                    if row.get("settle", 0.1) + 0.01 <= onair.ANSWER_D_S)
    args.serial = "offline-injected"
    args.ptt_port = "offline-injected"
    args.dial = 7100000
    return onair, args


def _boundaries(monkeypatch, onair, timeline):
    def stop(*args, **kwargs):
        timeline.append("boundary")
        raise BoundaryReached()
    monkeypatch.setattr(onair.ota, "Rig", stop)
    monkeypatch.setattr(onair, "find_device", stop)
    monkeypatch.setattr(onair, "RadioTx", stop)


@pytest.mark.parametrize("answer", [False, True])
@pytest.mark.parametrize("loads", [(0, 0, 0), (1000000, 1000000, 1000000), None])
def test_outgoing_and_answer_start_once_at_same_boundary(monkeypatch, tmp_path, answer, loads):
    onair, args = _arguments(monkeypatch, tmp_path)
    args.answer = answer
    if answer:
        args.dxcall = None
    timeline = []
    _boundaries(monkeypatch, onair, timeline)
    monkeypatch.setattr(launch_load.os, "getloadavg", lambda: timeline.append("load") or loads)
    monkeypatch.setattr(launch_load.os, "cpu_count", lambda: timeline.append("cpu") or 4)
    sink = io.StringIO()
    with monkeypatch.context() as capture:
        capture.setattr(onair.sys, "stdout", sink)
        with pytest.raises(BoundaryReached):
            onair.run(args)
    assert timeline == ["load", "cpu", "boundary"]
    assert sink.getvalue().count("LOAD ADVISORY:") == 1
    assert args.transmit is True
    assert args.answer is answer
    assert args.dial == 7100000


@pytest.mark.parametrize("mode", ["passive", "listen", "preflight", "tune", "file"])
def test_ineligible_modes_do_not_sample(monkeypatch, tmp_path, mode):
    onair, args = _arguments(monkeypatch, tmp_path)
    if mode == "passive":
        args.transmit = False
    elif mode == "listen":
        args.listen_only = 1
    elif mode == "preflight":
        args.preflight = True
    elif mode == "tune":
        args.tune = 1
    else:
        args.reply_wav = "offline-injected.wav"
    timeline = []
    _boundaries(monkeypatch, onair, timeline)
    def unexpected():
        raise AssertionError("ineligible startup must not sample")
    monkeypatch.setattr(launch_load.os, "getloadavg", unexpected)
    monkeypatch.setattr(launch_load.os, "cpu_count", unexpected)
    with pytest.raises(BoundaryReached):
        onair.run(args)
    assert timeline == ["boundary"]


@pytest.mark.parametrize("refusal", ["argument", "budget", "missing-peer"])
def test_existing_pre_anchor_refusals_sample_nothing(monkeypatch, tmp_path, refusal):
    onair, args = _arguments(monkeypatch, tmp_path)
    if refusal == "argument":
        args.p1_drive = 0
    elif refusal == "budget":
        monkeypatch.setattr(onair, "_budget", lambda *args: (False, "closed budget refusal"))
    else:
        args.dxcall = None
    def unexpected():
        raise AssertionError("refused before anchor must not sample")
    monkeypatch.setattr(launch_load.os, "getloadavg", unexpected)
    monkeypatch.setattr(launch_load.os, "cpu_count", unexpected)
    with pytest.raises(SystemExit):
        onair.run(args)


@pytest.mark.parametrize("loads", [(0, 0, 0), (1000000, 1000000, 1000000), None])
def test_advisory_does_not_bypass_existing_serial_refusal(monkeypatch, tmp_path, loads):
    onair, args = _arguments(monkeypatch, tmp_path)
    args.serial = None
    calls = []
    monkeypatch.setattr(launch_load.os, "getloadavg", lambda: calls.append("load") or loads)
    monkeypatch.setattr(launch_load.os, "cpu_count", lambda: calls.append("cpu") or 4)
    with pytest.raises(SystemExit):
        onair.run(args)
    assert calls == ["load", "cpu"]
    assert args.serial is None
    assert args.transmit is True


def test_view_visibility_and_actual_pump_sink_order(monkeypatch,tools):
    from consolelib import arm
    from consolelib.view import View
    line = launch_load.advisory_line(load_sampler=lambda: (0, 0, 0), cpu_sampler=lambda: 4) + "\n"
    assert tuple(View().feed(line)) == (line,)
    events = []
    view = View()
    class Sink:
        def write(self, value):
            events.append("sink")
            assert value == line
    class Screen:
        def feed(self, value):
            assert events == ["sink"]
            events.append("view")
            return view.feed(value)
    launcher = SimpleNamespace(stdout=iter([line]), wait=lambda: 7)
    stdout = io.StringIO()
    monkeypatch.setattr(arm.sys, "stdout", stdout)
    assert arm.pump(launcher, Sink(), Screen()) == 7
    assert events == ["sink", "view"]
    assert stdout.getvalue() == line


def test_callsite_is_single_and_before_unchanged_rig_boundary():
    from hfmodem.shrike import onair
    tree = ast.parse(Path(onair.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run")
    calls = [n for n in ast.walk(run) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "emit_load_advisory"]
    rig = next(n for n in run.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "rig" for t in n.targets))
    assert len(calls) == 1
    assert calls[0].lineno < rig.lineno
