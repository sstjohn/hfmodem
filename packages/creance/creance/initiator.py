# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Single-session orchestration at the initiator.

run_session builds the results directory and transcript, ensures the modem is
up, attaches with the desired state (MYCALL/BW/COMPRESSION), CONNECTs the dst,
waits for CONNECTED, then runs the scenario's initiator half (scenarios.py)
and DISCONNECTs. The whole session runs under initiator.max_session_s (or a
tighter caller budget): the watchdog trips the cancel event, every scenario
primitive raises SessionCancelled, and the session lands as aborted:watchdog
instead of hanging a campaign. Every exit path — success, refusal, timeout, or
crash —
writes session.json via metrics.compute + report.write_session and releases
the supervisor. The outcome vocabulary is the plan's:
ok / refused / plain_peer / echo_peer / report_missing / failed:<reason> /
aborted:watchdog.
"""

from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path

from . import scenarios
from hfhost.client import ModemClient, ModemError
from .link import CONNECTED, DISCONNECTED, Link, open_link
from .config import Config
from .metrics import SessionMetrics, compute
from .report import write_session
from .scenarios import CwpSession
from hfhost.supervisor import Supervisor
from hfhost.transcript import Transcript, read


def _make_sid(modem: str) -> str:
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{modem}-{secrets.token_hex(2)}"


def _session_dir(results_root: Path, site: str, sid: str) -> Path:
    day = time.strftime("%Y-%m-%d")
    d = results_root / site / day / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_session(config: Config, modem_name: str, dst: str, scenario: str,
                params: dict | None = None, *, label: str = "",
                results_root: str | Path | None = None,
                supervisor: Supervisor | None = None,
                max_session_s: float | None = None,
                host: str = "127.0.0.1") -> SessionMetrics:
    params = dict(params or {})
    budget = (config.initiator.max_session_s if max_session_s is None
              else max_session_s)
    mcfg = config.modem(modem_name)
    sid = _make_sid(modem_name)
    root = Path(results_root) if results_root is not None else Path(config.site.results_dir)
    sid_dir = _session_dir(root, config.site.name, sid)
    transcript = Transcript(str(sid_dir / "transcript.jsonl"), sid)
    wall_start = time.strftime("%Y-%m-%dT%H:%M:%S")

    meta: dict = {
        "sid": sid, "site": config.site.name, "modem": modem_name,
        "version": "", "rev": mcfg.rev, "label": label, "scenario": scenario,
        "params": params, "peer_sid": "", "peer_call": "",
        "wall_start": wall_start, "wall_end": "",
        "sim_time": mcfg.sim_time, "outcome": None,
    }

    def finish(outcome: str) -> SessionMetrics:
        if session is not None:
            # the peer named itself in its HELLO_ACK: the only key that joins
            # our record to the far site's record of the same exchange, and
            # worth having on the failure paths too
            meta["peer_sid"], meta["peer_call"] = session.peer_sid, session.peer_call
        meta["outcome"] = outcome
        meta["wall_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        transcript.note(modem_name, "session_end", outcome=outcome)
        records, _ = read(str(sid_dir / "transcript.jsonl"))
        m = compute(records, meta)
        write_session(sid_dir, m)
        return m

    cancel = threading.Event()
    # Every recv has a per-frame timeout, so a peer trickling one frame per
    # timeout would keep the session alive forever; this is the hard stop.
    watchdog = threading.Timer(budget, cancel.set)
    watchdog.daemon = True
    watchdog.start()
    owns_supervisor = supervisor is None
    if owns_supervisor:
        supervisor = Supervisor(config, str(sid_dir))
    client: ModemClient | None = None
    session: CwpSession | None = None
    try:
        if not supervisor.ensure(modem_name):
            raise ModemError(f"{modem_name}: not reachable")
        supervisor.acquire(modem_name)

        client = open_link(mcfg, host, transcript,
                           attach_timeout_s=config.initiator.connect_timeout_s)
        client.configure(config.site.mycall)
        client.attach()

        meta["version"] = client.request_version() or ""

        client.connect(dst)
        connected = client.wait_for(CONNECTED,
                                    config.initiator.connect_timeout_s,
                                    cancel=cancel)
        if connected is None:
            try:
                client.abort()
            except ModemError:
                pass
            return finish("aborted:watchdog" if cancel.is_set()
                          else "failed:connect_timeout")

        session = CwpSession(client, transcript, sid=sid,
                             mycall=config.site.mycall, cancel=cancel,
                             epoch=client.epoch, recv_timeout_s=budget)
        result = scenarios.initiate(session, scenario, params,
                                    on_plain_peer=config.initiator.on_plain_peer)
        _disconnect(client)
        return finish(result.outcome)
    except (ModemError, OSError) as exc:
        transcript.error(modem_name, f"session failed: {exc}")
        if client is not None:
            _disconnect(client)
        return finish(f"failed:{_reason(exc)}")
    except Exception as exc:                       # never lose the transcript
        transcript.error(modem_name, f"session error: {exc!r}")
        if client is not None:
            _disconnect(client)
        return finish(f"failed:{_reason(exc)}")
    finally:
        watchdog.cancel()
        cancel.set()
        if client is not None:
            client.close()
        if supervisor is not None:
            supervisor.release(modem_name)
            if owns_supervisor:
                supervisor.stop_all()
        transcript.close()


def _disconnect(client: Link) -> None:
    try:
        client.disconnect()
        client.wait_for(DISCONNECTED, 10.0)
    except (ModemError, OSError):
        pass


def _reason(exc: Exception) -> str:
    return type(exc).__name__.lower()
