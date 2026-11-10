# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The M1 acceptance gate: creance against the real modems, no radio, no audio.

Spawns the two targets that are driveable today — the kestrel LoopbackModem
echo server and the sabir virtual pair — on ephemeral ports, writes throwaway
site configs pointing at them, and then drives the CLI's own subcommands:

  1. conform --golden kestrel-loopback   single-modem catalog vs the table
  2. conform --golden sabir-hostapi-pair       single + paired catalog vs the table
  3. run unidir / echo over the pair     real CWP sessions against a live
                                         responder, gated on byte-exactness
  4. run unidir at the loopback          echo_peer self-test: framer, parser
                                         and integrity path in one process
  5. campaign --all over sabir-A       autonomous matrix + aggregate report
  6. report                              the results tree those stages wrote

Any deviation is a nonzero exit. A target the interpreter cannot import is
SKIPPED, not failed, so a partial install still gates the half it has; name it
in ``require`` and the skip becomes a failure. Exit is 0 when everything that
could run passed.

Timing is meaningless here (Sabir's SimulatedAir advances simulated time
instantly), so every sabir session is tagged sim_time and the gate asserts
that the rate figures are *suppressed* rather than believing them.
"""

from __future__ import annotations

import functools
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Iterable
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import config as config_mod
from . import report
from .metrics import SessionMetrics
from .responder import Responder

#: The modems live in the same distribution creance does, so the interpreter
#: running the bench is the one that runs them. ``CREANCE_MODEM_PYTHON`` aims
#: the gate at another one — an external checkout, or a build under a different
#: numpy — without changing anything else here.
MODEM_PY = os.environ.get("CREANCE_MODEM_PYTHON", sys.executable)
KESTREL_SERVER = ("-m", "hfmodem.kestrel.host.run_server")
KESTREL_PAIR = ("-m", "hfmodem.kestrel.arq.run_pair")
SABIR_SERVER = ("-m", "hfmodem.sabir.host.run_server")

INIT_CALL = "N0CRE"
RESP_CALL = "K7CRE"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Deviation(Exception):
    """A stage did not do what M1 promises."""


@dataclass
class Stage:
    name: str
    status: str
    detail: str = ""
    dur_s: float = 0.0


@functools.cache
def importable(module: str) -> bool:
    """Can ``MODEM_PY`` actually run this server?

    Asked of the interpreter rather than of the filesystem: a path that exists
    proves nothing about whether the stack behind it imports, and a probe that
    tests for a file is the one that goes quietly False when the file moves."""
    return subprocess.run([MODEM_PY, "-c", f"import {module}"],
                          capture_output=True).returncode == 0


def have_kestrel() -> bool:
    return importable(KESTREL_SERVER[1])


def have_sabir() -> bool:
    return importable(SABIR_SERVER[1])


def free_ports(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def _require(cond, msg: str) -> None:
    if not cond:
        raise Deviation(msg)


def _kestrel_modem(name: str, cmd: int, data: int) -> list[str]:
    return ["[[modem]]", f'name = "{name}"', f"cmd_port = {cmd}",
            f"data_port = {data}", 'bandwidth = "2300"', 'rev = "loopback"',
            "[modem.quirks]", "iamalive_s = 2", "preconnect_queue = true", ""]


def _native_modem(name: str, port: int) -> list[str]:
    """One framed-CBOR socket per simulated station."""
    return ["[[modem]]", f'name = "{name}"', 'dialect = "hostapi"',
            f"cmd_port = {port}", "data_port = 0", 'rev = "virtual-air"',
            "sim_time = true", ""]


class Selftest:
    def __init__(self, *, size: str | int = "10k", keep: bool = False,
                 verbose: bool = False, require: Iterable[str] = (),
                 out=sys.stderr) -> None:
        self.size = size
        self.keep = keep
        self.require = {r.strip() for r in require if r.strip()}
        self.verbose = verbose
        self.out = out
        self.tmp = Path(tempfile.mkdtemp(prefix="creance-selftest-"))
        self.results = self.tmp / "results"
        self.stages: list[Stage] = []
        self.procs: list[tuple[str, subprocess.Popen]] = []
        self.responder: Responder | None = None
        self._responder_thread: threading.Thread | None = None

    # -- plumbing ----------------------------------------------------------

    def say(self, msg: str) -> None:
        print(msg, file=self.out, flush=True)

    def check(self, name: str, fn) -> bool:
        t0 = time.monotonic()
        try:
            detail = fn() or ""
            status = PASS
        except Deviation as exc:
            status, detail = FAIL, str(exc)
        except Exception as exc:                     # a crash is a deviation
            status, detail = FAIL, f"{type(exc).__name__}: {exc}"
        stage = Stage(name, status, detail, time.monotonic() - t0)
        self.stages.append(stage)
        self.say(f"{stage.status}  {name}  ({stage.dur_s:.1f}s)"
                 + (f"\n      {detail}" if detail else ""))
        return status != FAIL

    def skip(self, name: str, why: str) -> None:
        """Record a stage that did not run.

        A skip is honest when a target is genuinely optional and a lie when the
        target is the point of the run: a path that stops resolving — a rename, a
        moved venv — costs an entire graded target and the gate still exits 0.
        So a skipped target named in ``require`` is a failure, not a skip."""
        if name.split()[0] in self.require:
            self.stages.append(Stage(name, FAIL, why))
            self.say(f"{FAIL}  {name}\n      required target absent: {why}")
            return
        self.stages.append(Stage(name, SKIP, why))
        self.say(f"{SKIP}  {name}\n      {why}")

    def cli(self, *argv: str) -> int:
        from . import cli
        if self.verbose:
            self.say("$ creance " + " ".join(argv))
        return cli.main(list(argv))

    def spawn(self, name: str, argv: list[str]) -> None:
        log = open(self.tmp / f"{name}.log", "wb")
        proc = subprocess.Popen(argv, cwd=str(self.tmp), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
        log.close()
        self.procs.append((name, proc))

    def sessions(self, **match) -> list[SessionMetrics]:
        """Sessions in the throwaway results tree matching every field, newest
        last (sids are timestamp-prefixed)."""
        found = [m for m in report.scan(self.results)
                 if all(getattr(m, k) == v for k, v in match.items())]
        return sorted(found, key=lambda m: m.sid)

    def latest(self, **match) -> SessionMetrics:
        found = self.sessions(**match)
        _require(found, f"no session recorded for {match}")
        return found[-1]

    # -- config ------------------------------------------------------------

    def write_configs(self, kestrel_ports, sabir_ports) -> None:
        def write(path: Path, site: str, mycall: str, modems: list[str]) -> Path:
            path.write_text("\n".join(
                ["[site]", f'name = "{site}"', f'mycall = "{mycall}"',
                 f'results_dir = "{self.results}"', "",
                 "[responder]", "hello_timeout_s = 30", "max_session_s = 300", "",
                 "[initiator]", "connect_timeout_s = 120", ""] + modems),
                encoding="utf-8")
            return path

        modems = []
        if kestrel_ports:
            modems += _kestrel_modem("kestrel-loopback", *kestrel_ports)
        if sabir_ports:
            a_port, b_port = sabir_ports
            modems += _native_modem("sabir-A", a_port)
            modems += _native_modem("sabir-B", b_port)
            self.resp_cfg = write(self.tmp / "responder.toml", "resp", RESP_CALL,
                                  _native_modem("sabir-B", b_port))
            self.camp_cfg = write(self.tmp / "campaign.toml", "camp", INIT_CALL,
                                  _native_modem("sabir-A", a_port))
        self.init_cfg = write(self.tmp / "initiator.toml", "init", INIT_CALL,
                              modems)

    # -- stages ------------------------------------------------------------

    def stage_conform_kestrel(self) -> str:
        rc = self.cli("conform", "-c", str(self.init_cfg),
                      "--modem", "kestrel-loopback", "--dst", RESP_CALL,
                      "--golden", "kestrel-loopback")
        _require(rc == 0, "deviations from the kestrel-loopback golden table "
                          "(listed above)")
        return "no deviation from the kestrel-loopback golden table"

    def stage_conform_sabir(self) -> str:
        rc = self.cli("conform", "-c", str(self.init_cfg),
                      "--pair", "sabir-A", "sabir-B", "--dst", RESP_CALL,
                      "--golden", "sabir-hostapi-pair")
        _require(rc == 0, "deviations from the sabir-pair golden table "
                          "(listed above)")
        return "no deviation from the sabir-pair golden table"

    def start_responder(self) -> str:
        cfg = config_mod.load(self.resp_cfg)
        self.responder = Responder(cfg, results_root=self.results)
        self.responder.start()
        _require(self.responder.modem_states.get("sabir-B") == "up",
                 f"responder could not bring up sabir-B: "
                 f"{self.responder.modem_states}")
        self._responder_thread = threading.Thread(
            target=self.responder.serve, name="selftest-responder", daemon=True)
        self._responder_thread.start()
        return f"listening on sabir-B as {RESP_CALL}"

    def _await_responder(self, count: int, timeout: float = 60.0) -> SessionMetrics:
        deadline = time.monotonic() + timeout
        while self.responder.sessions_completed < count:
            _require(time.monotonic() < deadline,
                     f"responder finished {self.responder.sessions_completed} "
                     f"session(s), expected {count}")
            time.sleep(0.2)
        return self.responder.last_metrics

    def _cwp_stage(self, scenario: str, count: int) -> str:
        rc = self.cli("run", "-c", str(self.init_cfg), "--modem", "sabir-A",
                      "--dst", RESP_CALL, "--scenario", scenario,
                      "--size", str(self.size), "--label", "selftest")
        _require(rc == 0, f"creance run {scenario} exited {rc}")
        m = self.latest(site="init", modem="sabir-A", scenario=scenario)
        _require(m.outcome == "ok", f"initiator outcome {m.outcome!r}, expected ok")
        _require(m.end_sha_match is True and m.report_sha_match is True,
                 f"integrity unproven: end_sha_match={m.end_sha_match} "
                 f"report_sha_match={m.report_sha_match}")
        _require(m.far_bytes, "far-end REPORT carried no byte count")
        _require(m.sim_time and "sim_time" in m.suppressed,
                 "session against simulated air is not tagged sim_time")
        _require(m.goodput_bps_far is None and m.drain_bps_local is None
                 and m.ptt_duty is None,
                 "sim_time session still reports rate/duty figures")
        _require((Path(m_dir := self._session_dir(m)) / "session.json").exists(),
                 f"no session.json under {m_dir}")

        far = self._await_responder(count)
        _require(far.outcome == "ok",
                 f"responder outcome {far.outcome!r}, expected ok")
        _require(far.label == "" and far.scenario == scenario,
                 f"responder recorded scenario {far.scenario!r}")
        return (f"{m.bytes_tx} B tx, {far.bytes_rx} B rx at the far end, "
                f"sha256 agreed both ways, rates suppressed (sim_time)")

    def _session_dir(self, m: SessionMetrics) -> Path:
        return self.results / m.site / m.wall_start[:10] / m.sid

    def stage_unidir(self) -> str:
        return self._cwp_stage("unidir", 1)

    def stage_echo(self) -> str:
        return self._cwp_stage("echo", 2)

    def stage_echo_peer(self) -> str:
        rc = self.cli("run", "-c", str(self.init_cfg),
                      "--modem", "kestrel-loopback", "--dst", RESP_CALL,
                      "--scenario", "unidir", "--size", "4k")
        _require(rc == 0, f"creance run against the loopback exited {rc}")
        m = self.latest(site="init", modem="kestrel-loopback")
        _require(m.outcome == "echo_peer",
                 f"outcome {m.outcome!r}, expected echo_peer")
        integrity = [f for f in m.findings if f.get("check") == "echo_integrity"]
        _require(integrity, "no echo_integrity finding recorded")
        _require(integrity[0].get("verdict") == "PASS",
                 f"echo_integrity {integrity[0].get('verdict')}: {integrity[0]}")
        return (f"{integrity[0].get('bytes_tx')} B echoed back byte-exact "
                "through the CWP framer")

    def stage_campaign(self) -> str:
        # --interval 0 overrides the sweep's own cool-down: over the virtual
        # air there is nothing to cool down, and 30 s a session would dominate
        rc = self.cli("campaign", "-c", str(self.camp_cfg), "--all",
                      "--dst", RESP_CALL, "--interval", "0",
                      "--label", "selftest-campaign")
        _require(rc == 0, f"creance campaign exited {rc}")
        runs = self.sessions(site="camp")
        _require(len(runs) >= 4, f"campaign ran {len(runs)} session(s), expected 4")
        texts = sorted((self.results / "camp").rglob("campaign-*.txt"))
        _require(texts, "no aggregate report written")
        _require("scenario" in texts[-1].read_text(encoding="utf-8"),
                 f"{texts[-1]} is not an aggregate table")
        bad = [m for m in runs if m.outcome != "ok"]
        outcomes = ", ".join(f"{m.scenario}={m.outcome}" for m in runs)
        _require(not bad, f"campaign outcomes: {outcomes}")
        return (f"{len(runs)} sessions, all ok; aggregate at "
                f"{texts[-1].relative_to(self.results)}")

    def stage_report(self) -> str:
        rc = self.cli("report", str(self.results))
        _require(rc == 0, f"creance report exited {rc}")
        return f"aggregated {len(report.scan(self.results))} sessions"

    # -- driver ------------------------------------------------------------

    def run(self) -> int:
        self.say(f"creance selftest: work in {self.tmp}")
        kestrel, sabir = have_kestrel(), have_sabir()
        if not kestrel:
            self.skip("kestrel loopback",
                      f"{MODEM_PY} cannot import {KESTREL_SERVER[1]}")
        if not sabir:
            self.skip("sabir virtual pair",
                      f"{MODEM_PY} cannot import {SABIR_SERVER[1]}")
        if not (kestrel or sabir):
            self.say("\ncreance selftest: nothing to test")
            return 0

        kestrel_ports = free_ports(2) if kestrel else None
        sabir_ports = free_ports(2) if sabir else None
        try:
            if kestrel:
                self.spawn("kestrel", [
                    MODEM_PY, *KESTREL_SERVER,
                    "--cmd-port", str(kestrel_ports[0]),
                    "--data-port", str(kestrel_ports[1]),
                    "--iamalive-interval", "2", "--quiet"])
            if sabir:
                self.spawn("sabir", [
                    MODEM_PY, *SABIR_SERVER, "--ports",
                    *map(str, sabir_ports), "--profile", "clean",
                    "--snr", "90", "--quiet"])
            self.write_configs(kestrel_ports, sabir_ports)
            self._await_ports(kestrel_ports, sabir_ports)

            if kestrel:
                self.check("conform kestrel-loopback", self.stage_conform_kestrel)
            if sabir:
                self.check("conform sabir-pair", self.stage_conform_sabir)
                if self.check("responder on sabir-B", self.start_responder):
                    self.check(f"cwp unidir {self.size} over the pair",
                               self.stage_unidir)
                    self.check(f"cwp echo {self.size} over the pair",
                               self.stage_echo)
            if kestrel:
                self.check("echo_peer at the loopback", self.stage_echo_peer)
            if sabir:
                self.check("campaign --all on sabir-A", self.stage_campaign)
            self.check("report over the results tree", self.stage_report)
        finally:
            self.teardown()
        return self.summarize()

    def _await_single(self, ports, timeout: float = 60.0) -> None:
        """Single-socket servers: one connect is the whole probe, so there is
        no half-attach hazard to avoid here."""
        import socket as _socket
        deadline = time.monotonic() + timeout
        for port in ports:
            while time.monotonic() < deadline:
                try:
                    _socket.create_connection(("127.0.0.1", port), 1.0).close()
                    break
                except OSError:
                    time.sleep(0.2)
            else:
                raise Deviation(f"structured server never came up on {port}")

    def _await_ports(self, kestrel_ports, sabir_ports, timeout: float = 60.0) -> None:
        """Probe each target using its actual one- or two-socket transport."""
        from hfhost.supervisor import probe
        pairs = []
        if kestrel_ports:
            pairs.append(("kestrel", *kestrel_ports))
        if sabir_ports:
            self._await_single(sabir_ports, timeout)
        deadline = time.monotonic() + timeout
        for name, cmd_port, data_port in pairs:
            while not probe("127.0.0.1", cmd_port, data_port, 1.0):
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{name}: never started listening")
                time.sleep(0.25)

    def teardown(self) -> None:
        if self.responder is not None:
            self.responder.stop()
            if self._responder_thread is not None:
                self._responder_thread.join(timeout=10.0)
        for name, proc in self.procs:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if self.keep:
            self.say(f"\ncreance selftest: kept {self.tmp}")
        else:
            shutil.rmtree(self.tmp, ignore_errors=True)

    def summarize(self) -> int:
        # A required target this run never even considered is the loudest kind of
        # absence: not a stage that skipped, but a name nothing answered to. That
        # is what a rename looks like from here, so it fails rather than passing
        # quietly on the strength of the targets that happen to still resolve.
        # Stage names read "conform kestrel-loopback", "responder on sabir-B" —
        # the target is somewhere inside, not the first word.
        missing = [n for n in sorted(self.require)
                   if not any(n in s.name for s in self.stages)]
        for name in missing:
            self.stages.append(Stage(f"{name} (required)", FAIL,
                                     "required target is not a target this "
                                     "selftest knows how to run"))
            self.say(f"{FAIL}  {name} (required)\n      not a known target")
        by = {status: [s for s in self.stages if s.status == status]
              for status in (PASS, FAIL, SKIP)}
        self.say(f"\ncreance selftest: {len(by[PASS])} passed, "
                 f"{len(by[FAIL])} failed, {len(by[SKIP])} skipped")
        for s in by[FAIL]:
            self.say(f"  {s.status} {s.name}: {s.detail}")
        return 1 if by[FAIL] else 0


def run(*, keep: bool = False, size: str | int = "10k",
        verbose: bool = False, require: Iterable[str] = (),
        out=sys.stderr) -> int:
    return Selftest(size=size, keep=keep, verbose=verbose,
                    require=require, out=out).run()
