# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""creance command line: argparse subcommands over the harness modules.

Every subcommand loads a site config, drives one of the orchestration layers
(responder, initiator, campaign, conformance, report) and turns the result
into an exit code: 0 success, 1 a substantive failure (session failed, golden
deviation, modem unreachable), 2 a usage or environment error.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import signal
import sys
import threading
import time
from pathlib import Path

from . import campaign as campaign_mod
from . import conformance as conf
from . import conformance_hostapi as conf_h
from . import config as config_mod
from . import chat as chat_mod
from . import initiator, payloads, report, scenarios
from hfhost.client import ModemClient, ModemError
from hfhost.hostapi import HostApiClient
from .config import Config, ConfigError
from .report import GOOD_OUTCOMES
from .responder import Responder
from hfhost.supervisor import Supervisor, probe
from hfhost.transcript import Transcript

#: how long to wait for a modem this command started to begin listening
READY_TIMEOUT_S = 30.0


class Size(int):
    """A transfer size in bytes that remembers how it was written, so `--size
    10k` still reads as "10k" in the log line. Converting at the parser is the
    point: validating here and converting downstream would let a bad size pass
    argparse, connect over the air, and only then die deep in the session."""

    def __new__(cls, text) -> "Size":
        self = super().__new__(cls, payloads.size_bytes(text))
        self.text = str(text)
        return self

    def __str__(self) -> str:
        return self.text


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load(path: str) -> Config:
    try:
        return config_mod.load(path)
    except OSError as exc:
        raise ConfigError(f"{path}: {exc.strerror or exc}") from None


def _known(cfg: Config, *names: str) -> None:
    have = {m.name for m in cfg.modems}
    for name in names:
        if name not in have:
            raise ConfigError(f"unknown modem {name!r}; configured: "
                              + ", ".join(sorted(have)))


def _results_root(args, cfg: Config) -> Path:
    return Path(getattr(args, "results", None) or cfg.site.results_dir)


def _one_line(m) -> str:
    bits = [m.sid or "-", m.modem or "-", m.scenario or "-", m.outcome or "-"]
    if m.bytes_tx or m.bytes_rx:
        bits.append(f"{m.bytes_tx}B tx / {m.bytes_rx}B rx")
    if m.goodput_bps_far is not None:
        bits.append(f"{m.goodput_bps_far / 1000:.2f} kbps far")
    return "  ".join(bits)


def _verbose(args) -> bool:
    return bool(getattr(args, "verbose", False))


def _await_ready(cfg: Config, supervisor: Supervisor, names, host: str,
                 timeout: float = READY_TIMEOUT_S) -> list[str]:
    """Start each modem and wait until its port pair accepts. A just-spawned
    server is alive well before it is listening, and a session attaches once —
    so the wait belongs here, ahead of the first CONNECT. Returns the modems
    that came up; one dead modem must not cost an overnight campaign the runs
    the others could have carried."""
    ready: list[str] = []
    for name in names:
        mcfg = cfg.modem(name)
        deadline = time.monotonic() + timeout
        while not supervisor.ensure(name) or not probe(host, mcfg.cmd_port,
                                                       mcfg.data_port, 1.0):
            if time.monotonic() >= deadline:
                _err(f"creance: {name}: not listening on "
                     f"{host}:{mcfg.cmd_port}/{mcfg.data_port} "
                     f"within {timeout:.0f}s")
                break
            time.sleep(0.25)
        else:
            supervisor.acquire(name)
            ready.append(name)
    return ready


def _with_rev(cfg: Config, rev: str | None) -> Config:
    """Override every modem's recorded rev for this invocation — the
    regression loop re-runs the same matrix against a rebuilt modem, and
    hand-editing the site TOML between runs is how labels go wrong."""
    if not rev:
        return cfg
    return dataclasses.replace(cfg, modems=tuple(
        dataclasses.replace(m, rev=rev) for m in cfg.modems))


def _attach(client: ModemClient, budget: float) -> None:
    """Attach, retrying until the budget is spent: a modem this command just
    started needs a moment before it is listening."""
    deadline = time.monotonic() + budget
    while True:
        try:
            client.attach()
            return
        except (ModemError, OSError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)


# -- respond -----------------------------------------------------------------

def _install_signals(stop: threading.Event,
                     signals=(signal.SIGINT, signal.SIGTERM)) -> None:
    def handler(signum, frame) -> None:
        stop.set()
    for sig in signals:
        try:
            signal.signal(sig, handler)
        except ValueError:
            pass          # not the main thread: the embedder owns shutdown


def _progress(r: Responder, stop: threading.Event, verbose: bool):
    """serve()'s until-callback, doubling as the console reporter: it runs
    once per loop iteration, which is exactly when the state can have moved."""
    seen = {"state": r.state, "done": r.sessions_completed,
            "modems": dict(r.modem_states)}

    def tick() -> bool:
        if r.state != seen["state"]:
            seen["state"] = r.state
            _err(f"  [{r.state}]" + (f" {r.winner}" if r.winner else ""))
        if verbose and r.modem_states != seen["modems"]:
            for name, state in r.modem_states.items():
                if seen["modems"].get(name) != state:
                    _err(f"  modem {name}: {state}")
            seen["modems"] = dict(r.modem_states)
        if r.sessions_completed != seen["done"]:
            seen["done"] = r.sessions_completed
            if r.last_metrics is not None:
                _err("  " + _one_line(r.last_metrics))
        return stop.is_set()

    return tick


def cmd_respond(args) -> int:
    cfg = _load(args.config)
    r = Responder(cfg, host=args.host, results_root=args.results)
    stop = threading.Event()
    _install_signals(stop)
    _err(f"creance: responding as {cfg.site.mycall} at site {cfg.site.name} on "
         + ", ".join(m.name for m in cfg.modems))
    _err(f"creance: results under {r.results_root}; SIGINT/SIGTERM to stop")
    try:
        r.serve(until=_progress(r, stop, _verbose(args)))
    finally:
        _err("creance: stopping")
        r.stop()
    _err(f"creance: {r.sessions_completed} session(s) served")
    return 0


# -- run ---------------------------------------------------------------------

def cmd_chat(args) -> int:
    cfg = _load(args.config)
    _known(cfg, args.modem)
    if args.dst is not None and args.dst == cfg.site.mycall:
        raise ConfigError(f"--dst {args.dst} is this station's own callsign")
    outcome = chat_mod.chat(cfg, args.modem, dst=args.dst, host=args.host,
                            connect_timeout=args.connect_timeout)
    if outcome == "ok":
        return 0
    _err(f"creance: chat ended: {outcome}")
    return 1


def cmd_run(args) -> int:
    cfg = _with_rev(_load(args.config), args.rev)
    _known(cfg, args.modem)
    params: dict = {"payload": args.payload}
    if args.size is not None:
        params["size"] = int(args.size)
    if args.duration is not None:
        params["duration"] = args.duration

    root = _results_root(args, cfg)
    if _verbose(args):
        _err(f"creance: {args.scenario} {params} -> {args.dst} via {args.modem}; "
             f"results under {root}")
    supervisor = Supervisor(cfg, str(root / cfg.site.name), host=args.host)
    failed = 0
    try:
        if not _await_ready(cfg, supervisor, [args.modem], args.host):
            raise ModemError(f"{args.modem}: never started listening")
        for i in range(args.repeat):
            if i and args.interval > 0:
                time.sleep(args.interval)
            m = initiator.run_session(cfg, args.modem, args.dst, args.scenario,
                                      params, label=args.label,
                                      results_root=root, supervisor=supervisor,
                                      host=args.host)
            print(report.session_report(m))
            if m.outcome not in GOOD_OUTCOMES:
                failed += 1
    finally:
        supervisor.stop_all()
    if failed:
        _err(f"creance: {failed}/{args.repeat} session(s) did not end "
             + "/".join(GOOD_OUTCOMES))
        return 1
    return 0


# -- campaign ----------------------------------------------------------------

def cmd_campaign(args) -> int:
    cfg = _with_rev(_load(args.config), args.rev)
    if args.all and not args.dst:
        raise ConfigError("--all requires --dst")
    runs = (campaign_mod.load_plan(args.plan) if args.plan
            else campaign_mod.derive_all(cfg, args.dst))
    if args.interval is not None:
        runs = [dataclasses.replace(r, interval_s=args.interval) for r in runs]
    names = list(dict.fromkeys(r.modem for r in runs))
    _known(cfg, *names)
    root = _results_root(args, cfg)
    if _verbose(args):
        _err(f"creance: {len(runs)} run(s) over {', '.join(names)}; "
             f"results under {root}")
    supervisor = Supervisor(cfg, str(root / cfg.site.name), host=args.host)
    try:
        ready = _await_ready(cfg, supervisor, names, args.host)
        if not ready:
            raise ModemError("no configured modem is reachable")
        dropped = [n for n in names if n not in ready]
        for name in dropped:
            _err(f"creance: dropping every run on {name}: not reachable")
        runs = [r for r in runs if r.modem in ready]
        metrics, text = campaign_mod.run_campaign(
            cfg, runs, label=args.label, deadline_s=args.deadline,
            results_root=root, host=args.host)
    finally:
        supervisor.stop_all()
    for m in metrics:
        _err("  " + _one_line(m))
    print(text)
    ok = sum(1 for m in metrics if m.outcome in GOOD_OUTCOMES)
    rate = ok / len(metrics) if metrics else 0.0
    _err(f"creance: {ok}/{len(metrics)} session(s) "
         + "/".join(GOOD_OUTCOMES)
         + (f"; skipped {', '.join(dropped)}" if dropped else ""))
    floor = 1.0 if args.fail_under is None else args.fail_under / 100
    if rate < floor:
        _err(f"creance: {rate:.0%} succeeded, below the {floor:.0%} gate")
        return 1
    return 0


# -- conform -----------------------------------------------------------------

def _iamalive_window(quirks) -> float | None:
    """Passive observation window for the IAMALIVE cadence probe. A short
    cadence is worth waiting out; a 60 s one would stall the suite, so it goes
    unjudged (ABSENT) instead."""
    interval = quirks.iamalive_s
    return 2 * interval if 0 < interval <= 10 else None


def cmd_conform(args) -> int:
    cfg = _load(args.config)
    names = [args.modem] if args.modem else list(args.pair)
    _known(cfg, *names)
    if args.dst == cfg.site.mycall:
        raise ConfigError(f"--dst {args.dst} must differ from site mycall")

    root = _results_root(args, cfg)
    out = (root / cfg.site.name / time.strftime("%Y-%m-%d")
           / f"conform-{time.strftime('%Y%m%dT%H%M%S')}")
    transcript = Transcript(str(out / "transcript.jsonl"), "conform",
                            echo=_verbose(args))
    supervisor = Supervisor(cfg, str(out), host=args.host)
    # one budget for attach and for every probe wait; a pair runs over the air
    # with two half-duplex turnarounds per exchange, so it gets longer
    paired = len(names) == 2
    budget = args.timeout if args.timeout is not None else (60.0 if paired else 30.0)
    timing = dict(connect_timeout_s=budget, drain_timeout_s=budget,
                  data_timeout_s=budget, abort_timeout_s=min(budget, 30.0),
                  reply_timeout_s=min(budget, 5.0),
                  payload_n=256 if paired else 512)
    dialects = {cfg.modem(n).dialect for n in names}
    if len(dialects) > 1:
        raise ConfigError("a conformance run grades one dialect at a time; "
                          f"{' and '.join(sorted(names))} span {sorted(dialects)}")
    structured = dialects == {"hostapi"}
    catalog = conf_h if structured else conf

    clients: list = []
    try:
        for name in names:
            if not supervisor.ensure(name):
                raise ModemError(f"{name}: not reachable")
            supervisor.acquire(name)
            if structured:
                client = HostApiClient(cfg.modem(name), transcript=transcript)
                client.attach(timeout=budget)
            else:
                client = ModemClient(cfg.modem(name), args.host, transcript,
                                     attach_timeout_s=budget)
                _attach(client, budget)
            clients.append(client)
        if structured:
            findings = conf_h.run(conf_h.Ctx(
                clients[0], mycall=cfg.site.mycall, dst=args.dst,
                reply_timeout_s=min(budget, 5.0)))
            if len(clients) == 2:
                # Fresh attachments for the pair run. The single catalog is
                # deliberately destructive -- oversize_frame_refused provokes a
                # close, and several probes leave the modem having refused
                # commands -- so a catalog that breaks the connection on purpose
                # does not get to hand its client on. Measured: reusing them
                # leaves the pair unable to link at all.
                for cl in clients:
                    cl.close()
                clients = [HostApiClient(cfg.modem(n), transcript=transcript)
                           for n in names]
                for cl in clients:
                    cl.attach(timeout=budget)
                a, b = clients
                findings += conf_h.run_paired(
                    conf_h.Ctx(a, mycall=cfg.site.mycall,
                               reply_timeout_s=min(budget, 5.0)),
                    conf_h.Ctx(b, mycall=args.dst,
                               reply_timeout_s=min(budget, 5.0)))
        elif len(clients) == 1:
            window = (args.observe if args.observe is not None
                      else _iamalive_window(clients[0].cfg.quirks))
            findings = conf.run_single(conf.ProbeContext(
                clients[0], mycall=cfg.site.mycall, dst=args.dst,
                iamalive_window_s=window, **timing))
        else:
            window = (args.observe if args.observe is not None
                      else _iamalive_window(clients[0].cfg.quirks))
            a, b = clients
            findings = conf.run_single(conf.ProbeContext(
                a, mycall=cfg.site.mycall, dst=args.dst, peer=b,
                iamalive_window_s=window, **timing))
            findings += conf.run_paired(
                conf.ProbeContext(a, mycall=cfg.site.mycall, peer=b, **timing),
                conf.ProbeContext(b, mycall=args.dst, **timing))
    finally:
        for client in clients:
            client.close()
        for name in names:
            supervisor.release(name)
        supervisor.stop_all()
        transcript.close()

    print(conf.render(findings))
    _err(f"creance: transcript at {out / 'transcript.jsonl'}")
    if args.golden:
        tables = catalog.GOLDEN
        # A structured pair run executes BOTH catalogs, so the expected set is
        # the union of both tables. Comparing sixteen findings against the pair
        # table alone reports every single-ended probe as unexpected.
        if args.golden.endswith("-pair"):
            base = args.golden[:-len("-pair")]
            if base in tables:
                tables = dict(tables)
                tables[args.golden] = {**tables[base], **tables[args.golden]}
        deviations = conf.diff_golden(args.golden, findings, tables)
        if deviations:
            print(f"\ndeviations from golden table {args.golden!r}:")
            for d in deviations:
                print(f"  {d}")
            return 1
        print(f"\nno deviation from golden table {args.golden!r}")
        return 0
    _err("creance: no --golden table, so this run is graded on raw FAIL only"
         + (f"; --golden {args.modem} has one" if args.modem in catalog.GOLDEN
            else ""))
    fails = [f.probe for f in findings if f.verdict == conf.FAIL]
    if fails:
        _err("creance: FAIL on " + ", ".join(fails))
        return 1
    return 0


# -- report ------------------------------------------------------------------

def cmd_report(args) -> int:
    metrics = report.scan(args.results_dir, since=args.since)
    for attr in ("modem", "scenario", "label"):
        want = getattr(args, attr)
        if want:
            metrics = [m for m in metrics if getattr(m, attr) == want]
    if args.sid:
        match = next((m for m in metrics if m.sid == args.sid), None)
        if match is None:
            _err(f"creance: no session {args.sid!r} under {args.results_dir}")
            return 1
        print(report.session_report(match))
        return 0
    if not metrics:
        _err(f"creance: no sessions under {args.results_dir}"
             + (f" since {args.since}" if args.since else "")
             + (" matching the filters" if (args.modem or args.scenario
                                            or args.label) else ""))
        return 1
    print(report.aggregate(metrics))
    return 0


# -- modems ------------------------------------------------------------------

def _last_seen(root: Path) -> dict[str, float]:
    """Age in seconds of the newest session per modem, read from the results
    tree — the only idle signal that survives across CLI invocations."""
    now = time.time()
    ages: dict[str, float] = {}
    for m in report.scan(root):
        if not (m.modem and m.wall_start):
            continue
        try:
            age = now - time.mktime(time.strptime(m.wall_start,
                                                  "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            continue
        ages[m.modem] = min(ages.get(m.modem, age), age)
    return ages


def _age(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def cmd_modems(args) -> int:
    cfg = _load(args.config)
    root = _results_root(args, cfg)
    ages = _last_seen(root)
    supervisor = Supervisor(cfg, str(root / cfg.site.name), host=args.host)
    transcript = Transcript(os.devnull, "modems")   # a health check leaves none
    rows: list[tuple[str, ...]] = [("modem", "process", "attach", "version",
                                    "last session")]
    unreachable: list[str] = []
    try:
        for mcfg in cfg.modems:
            up = supervisor.ensure(mcfg.name)
            proc = supervisor.process_for(mcfg.name)
            state = proc.state if proc is not None else "external"
            attach, version = "unreachable", "-"
            if up:
                client = ModemClient(mcfg, args.host, transcript,
                                     attach_timeout_s=args.timeout)
                try:
                    _attach(client, args.timeout)
                    attach = "ok"
                    if mcfg.quirks.version_reply:
                        version = client.request_version(args.timeout) or "-"
                except (ModemError, OSError) as exc:
                    attach = "fail"
                    if _verbose(args):
                        _err(f"  {mcfg.name}: {exc}")
                finally:
                    client.close()
            if attach != "ok":
                unreachable.append(mcfg.name)
            rows.append((mcfg.name, state, attach, version,
                         _age(ages.get(mcfg.name))))
    finally:
        supervisor.stop_all()
        transcript.close()
    print(report.table(rows))
    if unreachable:
        _err("creance: unreachable: " + ", ".join(unreachable))
        return 1
    return 0


# -- monitor -----------------------------------------------------------------

def cmd_monitor(args) -> int:
    from hfhost.audio import AudioError, KiwiSource, WavReplaySource
    from .monitor import default_specs, device_source, run as run_monitor

    if args.wav:
        source = WavReplaySource(args.wav, realtime=not args.fast)
    elif args.kiwi:
        host, _, port = args.kiwi.partition(":")
        if args.freq is None:
            raise ConfigError("--kiwi needs --freq (dial kHz)")
        source = KiwiSource(host, args.freq, port=int(port or 8073))
    else:
        source = device_source(args.device)

    if args.deep and not args.wav:
        raise ConfigError("--deep is offline only: a wideband decode costs seconds "
                          "per burst and would back up a live stream. Use it with --wav.")
    specs = default_specs(args.base, deep=args.deep)
    if args.modems:
        want = {m.strip() for m in args.modems.split(",") if m.strip()}
        specs = [s for s in specs if s.name in want]
    # `timeout` (as onair.sh's `listen` uses to bound a pass) sends SIGTERM,
    # whose default disposition kills the process outright -- straight past the
    # `finally` that prints the capture summary. Routed through the stop event
    # instead, it ends the pass the same way the source running out would.
    stop = threading.Event()
    _install_signals(stop, signals=(signal.SIGTERM,))
    try:
        with source:
            run_monitor(source, specs, window_s=args.window,
                        summary_every_s=args.summary_every, stop=stop)
    except AudioError as exc:
        _err(f"creance: {exc}")
        return 2
    return 0


# -- selftest ----------------------------------------------------------------

def cmd_selftest(args) -> int:
    from . import selftest
    return selftest.run(keep=args.keep, size=args.size, verbose=_verbose(args),
                        require=(args.require or "").split(","))


# -- parser ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS so a subparser's default cannot clobber the global flag
    common.add_argument("-v", "--verbose", action="store_true",
                        default=argparse.SUPPRESS, help="more detail on stderr")

    p = argparse.ArgumentParser(
        prog="creance", parents=[common],
        description="Conformance and performance harness for VARA-dialect HF "
                    "modems (kestrel, shrike, sabir).")
    sub = p.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_, description=help_,
                              parents=[common])

    def site(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-c", "--config", required=True, metavar="TOML",
                        help="site config")
        sp.add_argument("--host", default="127.0.0.1",
                        help="modem host (default 127.0.0.1)")
        sp.add_argument("--results", metavar="DIR",
                        help="results root (default: [site] results_dir)")

    sp = add("respond", "Run the responder daemon: listen on every configured "
                        "modem, detect the inbound protocol, serve the "
                        "requested scenario, return to listening.")
    site(sp)
    sp.set_defaults(func=cmd_respond)

    sp = add("run", "Run one session as initiator.")
    site(sp)
    sp.add_argument("--modem", required=True, help="modem to dial from")
    sp.add_argument("--dst", required=True, metavar="CALL",
                    help="callsign to connect to")
    sp.add_argument("--scenario", required=True, choices=sorted(scenarios.SCENARIOS))
    size = sp.add_mutually_exclusive_group()
    size.add_argument("--size", type=Size, metavar="N",
                      help="transfer size: 4096, 10k, 1M (default 10k)")
    size.add_argument("--duration", type=float, metavar="S",
                      help="send for S seconds instead of a fixed size")
    sp.add_argument("--payload", default="prbs9",
                    choices=sorted(payloads.GENERATORS),
                    help="payload generator (default prbs9)")
    sp.add_argument("--repeat", type=int, default=1, metavar="N")
    sp.add_argument("--interval", type=float, default=0.0, metavar="S",
                    help="seconds between repeats")
    sp.add_argument("--label", default="", metavar="TAG",
                    help="recorded in session.json for regression grouping")
    sp.add_argument("--rev", metavar="REV",
                    help="override the configured modem rev for these runs")
    sp.set_defaults(func=cmd_run)

    sp = add("chat", "Interactive terminal-to-terminal chat over one modem. "
                     "--dst CALL to place a call; omit it to listen for one.")
    site(sp)
    sp.add_argument("--modem", required=True, help="modem to chat over")
    sp.add_argument("--dst", metavar="CALL",
                    help="callsign to call; omit to listen for an inbound call")
    sp.add_argument("--connect-timeout", type=float, default=90.0, metavar="S",
                    help="seconds to wait for the link to come up (default 90)")
    sp.set_defaults(func=cmd_chat)

    sp = add("campaign", "Drive a matrix of sessions unattended, then report.")
    site(sp)
    what = sp.add_mutually_exclusive_group(required=True)
    what.add_argument("--plan", metavar="TOML", help="campaign plan")
    what.add_argument("--all", action="store_true",
                      help="derive the default sweep for every configured "
                           "modem (needs --dst)")
    sp.add_argument("--dst", metavar="CALL", help="callsign for --all")
    sp.add_argument("--label", default="", metavar="TAG")
    sp.add_argument("--deadline", type=float, metavar="S",
                    help="start no further session after S seconds")
    sp.add_argument("--interval", type=float, metavar="S",
                    help="cool-down between sessions, overriding the plan "
                         f"(--all derives {campaign_mod.ALL_INTERVAL_S:g}s)")
    sp.add_argument("--rev", metavar="REV",
                    help="override the configured modem rev for this campaign")
    sp.add_argument("--fail-under", type=float, metavar="PCT",
                    help="exit 0 while at least PCT%% of sessions succeed "
                         "(default: every session must)")
    sp.set_defaults(func=cmd_campaign)

    sp = add("conform", "Run the conformance probe catalog against a modem, or "
                        "a linked pair, and grade it.")
    site(sp)
    which = sp.add_mutually_exclusive_group(required=True)
    which.add_argument("--modem", metavar="NAME", help="single-modem suite")
    which.add_argument("--pair", nargs=2, metavar=("A", "B"),
                       help="single suite on A plus the paired suite over A-B")
    sp.add_argument("--golden", choices=sorted({**conf.GOLDEN, **conf_h.GOLDEN}),
                    metavar="TARGET",
                    help="gate on deviations from the golden verdict table ("
                         + ", ".join(sorted(conf.GOLDEN)) + ")")
    sp.add_argument("--dst", default="K7CRE", metavar="CALL",
                    help="far-end callsign the probes use (default K7CRE)")
    sp.add_argument("--observe", type=float, metavar="S",
                    help="passive observation window for the cadence probes")
    sp.add_argument("--timeout", type=float, metavar="S",
                    help="budget for attach and for each probe wait "
                         "(default 30, or 60 with --pair)")
    sp.set_defaults(func=cmd_conform)

    sp = add("report", "Summarize a results tree.")
    sp.add_argument("results_dir", metavar="RESULTS_DIR")
    sp.add_argument("--since", metavar="YYYY-MM-DD")
    sp.add_argument("--sid", metavar="SID", help="print one session in full")
    sp.add_argument("--modem", metavar="NAME")
    sp.add_argument("--scenario", metavar="NAME", choices=sorted(scenarios.SCENARIOS))
    sp.add_argument("--label", metavar="TAG")
    sp.set_defaults(func=cmd_report)

    sp = add("modems", "Health of every configured modem. Starts spawnable "
                       "modems that are not running and briefly attaches to "
                       "each, so do not point it at modems a responder holds.")
    site(sp)
    sp.add_argument("--timeout", type=float, default=10.0, metavar="S",
                    help="attach timeout (default 10)")
    sp.set_defaults(func=cmd_modems)

    sp = add("monitor", "Listen to one live audio source (rig input, KiwiSDR, "
                        "or WAV replay), fan it to every modem's receive-only "
                        "monitor in real time, and print a merged, "
                        "confidence-graded view of what is on the frequency.")
    where = sp.add_mutually_exclusive_group()
    where.add_argument("--device", default="USB Audio Device", metavar="NAME",
                       help="live rig audio input, by name (default "
                            "'USB Audio Device')")
    where.add_argument("--kiwi", metavar="HOST[:PORT]",
                       help="stream a remote KiwiSDR instead of a local rig")
    where.add_argument("--wav", metavar="FILE",
                       help="replay a recording (test/dev source only)")
    sp.add_argument("--freq", type=float, metavar="KHZ",
                    help="KiwiSDR dial frequency in kHz (with --kiwi)")
    sp.add_argument("--fast", action="store_true",
                    help="replay a --wav as fast as possible, not at real time")
    sp.add_argument("--deep", action="store_true",
                    help="run each modem's expensive decode too (kestrel's wideband "
                         "over). Offline only — requires --wav")
    sp.add_argument("--modems", metavar="A,B",
                    help="restrict to these monitors (default: all available)")
    sp.add_argument("--base", metavar="DIR",
                    help="source tree the runners resolve against "
                         "(default: the one creance is installed from)")
    sp.add_argument("--window", type=float, default=20.0, metavar="S",
                    help="activity window for the rolling summary (default 20)")
    sp.add_argument("--summary-every", type=float, default=10.0, metavar="S",
                    help="seconds of audio between summaries (default 10)")
    sp.set_defaults(func=cmd_monitor)

    sp = add("selftest", "M1 acceptance gate: spawn the kestrel loopback and "
                         "the sabir virtual pair, then run conformance, CWP "
                         "sessions and a campaign against them.")
    sp.add_argument("--keep", action="store_true",
                    help="keep the temporary configs and results tree")
    sp.add_argument("--size", type=Size, default="10k", metavar="N",
                    help="payload size for the session stages (default 10k)")
    sp.add_argument("--require", metavar="A,B",
                    help="targets that must be present: if one of these is "
                         "missing its stages FAIL instead of skipping, so a "
                         "moved path costs a red run rather than silent coverage")
    sp.set_defaults(func=cmd_selftest)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        _err(f"creance: {exc}")
        return 2
    except (ModemError, OSError) as exc:
        _err(f"creance: {exc}")
        return 2
    except KeyboardInterrupt:
        _err("creance: interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
