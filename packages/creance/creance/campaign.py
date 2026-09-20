# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Autonomous campaign runner: drive a matrix of sessions through initiator.py
without supervision, continuing past failures, and emit an aggregate report.

A plan is a TOML list of [[run]] tables (modem, dst, scenario, params, repeat,
interval_s, retries). --all derives the default sweep — connect, unidir@1k,
unidir@10k, echo@10k — for every configured modem against a caller-supplied
dst. Nothing prints; the aggregate text comes back and lands in
results/<site>/<date>/campaign-<ts>.txt, and the run ends by enforcing the
retention caps on the tree it just grew.
"""

from __future__ import annotations

import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import report
from .config import Config, ConfigError
from .initiator import run_session
from .metrics import SessionMetrics
from .responder import sweep_results
from hfhost.supervisor import Supervisor

#: cool-down between --all sessions. Non-zero on purpose: back-to-back
#: CONNECTs dial a far end that may still be draining the last session.
ALL_INTERVAL_S = 30.0

_ALL_SWEEP = (
    ("connect", {}),
    ("unidir", {"size": "1k"}),
    ("unidir", {"size": "10k"}),
    ("echo", {"size": "10k"}),
)


@dataclass(frozen=True, slots=True)
class Run:
    modem: str
    dst: str
    scenario: str
    params: dict = field(default_factory=dict)
    repeat: int = 1
    interval_s: float = 0.0
    retries: int = 0
    label: str = ""


def load_plan(path: str | Path) -> list[Run]:
    path = Path(path)
    with open(path, "rb") as fh:
        try:
            raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from None
    entries = raw.get("run")
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"{path}: no [[run]] entries")
    runs: list[Run] = []
    allowed = {"modem", "dst", "scenario", "params", "repeat", "interval_s",
               "retries", "label"}
    for i, table in enumerate(entries):
        ctx = f"[[run]] #{i + 1}"
        if not isinstance(table, dict):
            raise ConfigError(f"{ctx}: must be a table")
        unknown = sorted(set(table) - allowed)
        if unknown:
            raise ConfigError(f"{ctx}: unknown key(s) {', '.join(unknown)}")
        for req in ("modem", "dst", "scenario"):
            if req not in table:
                raise ConfigError(f"{ctx}: missing {req}")
        params = table.get("params", {})
        if not isinstance(params, dict):
            raise ConfigError(f"{ctx}: params must be a table")
        ctx = f"[[run]] {table['modem']}/{table['scenario']}"
        try:
            runs.append(Run(
                modem=str(table["modem"]), dst=str(table["dst"]),
                scenario=str(table["scenario"]), params=dict(params),
                repeat=int(table.get("repeat", 1)),
                interval_s=float(table.get("interval_s", 0.0)),
                retries=int(table.get("retries", 0)),
                label=str(table.get("label", ""))))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{ctx}: {exc}") from None
    return runs


def derive_all(config: Config, dst: str) -> list[Run]:
    if not dst:
        raise ConfigError("--all requires a dst")
    return [Run(modem=m.name, dst=dst, scenario=scn, params=dict(p),
                interval_s=ALL_INTERVAL_S)
            for m in config.modems for scn, p in _ALL_SWEEP]


def run_campaign(config: Config, plan: str | Path | list[Run] | None = None, *,
                 all_dst: str | None = None, label: str = "",
                 deadline_s: float | None = None,
                 results_root: str | Path | None = None,
                 host: str = "127.0.0.1") -> tuple[list[SessionMetrics], str]:
    if isinstance(plan, list):
        runs = plan
    elif plan is not None:
        runs = load_plan(plan)
    elif all_dst is not None:
        runs = derive_all(config, all_dst)
    else:
        raise ConfigError("run_campaign needs a plan or all_dst")

    root = Path(results_root) if results_root is not None else Path(config.site.results_dir)
    supervisor = Supervisor(config, str(root / config.site.name))
    start = time.monotonic()
    metrics: list[SessionMetrics] = []

    def remaining() -> float | None:
        return None if deadline_s is None else deadline_s - (time.monotonic() - start)

    try:
        for run in runs:
            if (left := remaining()) is not None and left <= 0:
                break
            for _ in range(max(1, run.repeat)):
                # The session gets the campaign's remaining time as its own
                # cap, so one stalling peer cannot overrun the deadline.
                if (left := remaining()) is not None and left <= 0:
                    break
                m = _run_with_retries(config, run, label, root, supervisor, host,
                                      left)
                metrics.append(m)
                if run.interval_s > 0:
                    if (left := remaining()) is None:
                        time.sleep(run.interval_s)
                    elif left <= 0:
                        break
                    else:
                        time.sleep(min(run.interval_s, left))
    finally:
        supervisor.stop_all()

    text = report.aggregate(metrics) if metrics else "no sessions\n"
    out = root / config.site.name / time.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"campaign-{time.strftime('%Y%m%dT%H%M%S')}.txt").write_text(
        text, encoding="utf-8")
    # An initiator-only box runs no responder daemon, so nothing else would
    # ever enforce the [responder] retention caps on its results tree.
    sweep_results(root, max_age_s=config.responder.retention_max_age_s,
                  max_bytes=config.responder.retention_max_bytes)
    return metrics, text


def _run_with_retries(config: Config, run: Run, label: str, root: Path,
                      supervisor: Supervisor, host: str,
                      budget_s: float | None = None) -> SessionMetrics:
    lbl = run.label or label
    deadline = None if budget_s is None else time.monotonic() + budget_s
    m: SessionMetrics | None = None
    for _ in range(run.retries + 1):
        cap = config.initiator.max_session_s
        if deadline is not None:
            cap = min(cap, deadline - time.monotonic())
            if cap <= 0 and m is not None:      # retries share the budget
                break
        m = run_session(config, run.modem, run.dst, run.scenario, run.params,
                        label=lbl, results_root=root, supervisor=supervisor,
                        max_session_s=max(cap, 0.1), host=host)
        if not (m.outcome or "").startswith("failed"):
            return m
    return m
