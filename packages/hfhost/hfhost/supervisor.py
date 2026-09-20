# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Process supervision for spawned modems: start-on-demand, restart with
backoff, circuit breaker, spawn groups (one process fronting several modem
entries), deaf-modem idle recycle.

Health signals, in order of trust:
  1. poll() on a spawned process — primary, side-effect-free.
  2. probe() — a paired TCP connect+close on BOTH ports, used only when there
     is no process handle. Never probe the cmd port alone: both servers accept
     the cmd socket then block accepting the data socket, so a lone cmd
     connection (even closed immediately — it waits in the listen backlog)
     eventually wedges their accept loop. A paired probe costs at worst one
     zero-length app session.

No monitor thread: callers (responder INIT loop, client reconnect escalation)
drive policy through ensure_running()/kill_and_respawn().
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import threading
import time
from collections import Counter

from .config import ModemConfig, ModemSource

_LOG_CAP = 5 * 1024 * 1024


class SpawnError(RuntimeError):
    pass


def _close(sock: socket.socket) -> None:
    try:
        sock.close()
    except OSError:
        pass


_REAP_S = 5.0

#: `hfmodem.core.ptt.UNKEY_BUDGET_S`, mirrored rather than imported — hfhost
#: imports no modem. A modem asking a slow or silent rigctld spends the whole
#: budget before the line is down.
_MODEM_UNKEY_BUDGET_S = 6.0

#: What a modem gets to put its transmitter down in before SIGKILL, which is the
#: budget plus room to exit after it. Measured SIGTERM-to-exit here is 1.3 ms
#: with no handler and 3.8 ms with one, rising to 75 ms worst case when the GIL
#: is held by a numpy FFT — how fast a healthy child exits, not how long a child
#: legitimately needs to put the key down, and it is the second quantity this
#: has to clear. A ceiling rather than a cost: spent only by a process that
#: ignores SIGTERM or is still working through its ladder, against a stop or a
#: recycle that follows hours of silence or a run of multi-second attach
#: failures.
_UNKEY_GRACE_S = _MODEM_UNKEY_BUDGET_S + 1.0


def _reap(proc: subprocess.Popen, timeout: float = _REAP_S) -> None:
    """Bounded wait after a kill(). A managed service may fork; an orphaned
    grandchild that inherited stdout/stderr would hold a pipe's write end
    open forever, and communicate() (or a bare wait() reading one) would
    block on it even though the killed process is long gone."""
    try:
        proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"pid {proc.pid} killed but its output is still held open") from exc


def _terminate(proc: subprocess.Popen, grace: float) -> None:
    """SIGTERM, then SIGKILL once the grace runs out.

    A modem that keys a transmitter puts the line down in its SIGTERM handler,
    and SIGKILL is the one signal it cannot unkey through: handler, atexit and
    keying watchdog all die with it, leaving the carrier up with nothing left
    running to drop it. Deassert-on-process-death is a hope rather than a
    mechanism — unmeasured on this station's adapter — and a daemon that holds
    the key is not known to drop it when its client goes away."""
    proc.terminate()
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        _reap(proc)


def probe(host: str, cmd_port: int, data_port: int = 0,
          timeout: float = 1.0) -> bool:
    """Liveness probe.

    For the two-socket dialect this is deliberately *paired*: connect cmd then
    data and close both. See the module docstring for why — a half-attach wedges
    those servers, so a cmd-only probe is unsafe. If the cmd connect succeeds
    and data fails, the server is already broken and the stale cmd backlog entry
    may wedge it later; unavoidable, and moot for spawned processes where poll()
    is the health signal.

    A ``data_port`` of 0 means a single-socket modem, where there is no pair to
    keep and one connect is the whole probe."""
    try:
        cmd = socket.create_connection((host, cmd_port), timeout=timeout)
    except OSError:
        return False
    if not data_port:
        _close(cmd)
        return True
    try:
        data = socket.create_connection((host, data_port), timeout=timeout)
    except OSError:
        _close(cmd)
        return False
    _close(data)
    _close(cmd)
    return True


class ModemProcess:
    """One spawned modem process and its restart policy.

    States: running | backoff (exited, restart pending) | down (circuit
    breaker tripped: breaker_threshold consecutive exits with uptime under
    fast_fail_s; retried every down_retry_s) | stopped (deliberate)."""

    def __init__(self, name: str, spawn: tuple[str, ...], *,
                 cwd: str | None = None, log_path: str | None = None,
                 env: dict | None = None,
                 fast_fail_s: float = 10.0, breaker_threshold: int = 3,
                 down_retry_s: float = 300.0,
                 backoff_initial_s: float = 1.0, backoff_max_s: float = 30.0,
                 max_idle_recycle_s: float | None = None) -> None:
        self.name = name
        # spawn is argv from config; a lone string still gets shlex treatment
        self.argv = shlex.split(spawn[0]) if len(spawn) == 1 else list(spawn)
        self.cwd = os.path.expanduser(cwd) if cwd else None
        self.log_path = log_path
        self.env = env
        self.fast_fail_s = fast_fail_s
        self.breaker_threshold = breaker_threshold
        self.down_retry_s = down_retry_s
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s
        self.max_idle_recycle_s = max_idle_recycle_s

        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._started_at = 0.0
        self._fast_fails = 0
        self._backoff = backoff_initial_s
        self._next_start_at = 0.0
        self._down = False
        self._stopped = True
        self._last_activity = time.monotonic()

    # -- introspection -----------------------------------------------------

    @property
    def pid(self) -> int | None:
        proc = self._proc
        return proc.pid if proc is not None else None

    @property
    def alive(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def state(self) -> str:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return "running"
            if self._down:
                return "down"
            if self._stopped:
                return "stopped"
            return "backoff"

    # -- lifecycle ---------------------------------------------------------

    def start(self, now: float | None = None) -> None:
        """Unconditional spawn (no policy). SpawnError on exec failure."""
        with self._lock:
            if self.alive:
                return
            log = self._open_log()
            try:
                self._proc = subprocess.Popen(
                    self.argv, cwd=self.cwd, env=self.env,
                    stdin=subprocess.DEVNULL,
                    stdout=log if log is not None else subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    start_new_session=True)
            except OSError as exc:
                raise SpawnError(f"{self.name}: {exc}") from exc
            finally:
                if log is not None:
                    log.close()
            self._started_at = time.monotonic() if now is None else now
            self._last_activity = self._started_at
            self._stopped = False

    def _open_log(self):
        if not self.log_path:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        try:
            if os.path.getsize(self.log_path) > _LOG_CAP:
                os.replace(self.log_path, self.log_path + ".1")
        except OSError:
            pass
        return open(self.log_path, "ab")

    def ensure_running(self, now: float | None = None) -> bool:
        """Apply restart policy; True when the process is (now) running."""
        now = time.monotonic() if now is None else now
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                if self._fast_fails and now - self._started_at >= self.fast_fail_s:
                    self._fast_fails = 0
                    self._backoff = self.backoff_initial_s
                return True
            if proc is not None:
                self._note_exit(now)
            if now < self._next_start_at:
                return False
            if self._down:
                # half-open retry: one more fast failure re-trips immediately
                self._fast_fails = self.breaker_threshold - 1
                self._down = False
            self.start(now)
            return self.alive

    def _note_exit(self, now: float) -> None:
        uptime = now - self._started_at
        self._proc = None
        if uptime < self.fast_fail_s:
            self._fast_fails += 1
            self._next_start_at = now + self._backoff
            self._backoff = min(self._backoff * 2, self.backoff_max_s)
            if self._fast_fails >= self.breaker_threshold:
                self._down = True
                self._next_start_at = now + self.down_retry_s
        else:
            self._fast_fails = 0
            self._backoff = self.backoff_initial_s
            self._next_start_at = now + self._backoff

    def kill_and_respawn(self, now: float | None = None, *,
                         grace: float = _UNKEY_GRACE_S) -> None:
        """Deliberate recycle (attach-failure escalation, deaf-modem recycle):
        stop the process, reset the breaker, spawn fresh.

        Both triggers are cases where the modem is presumed unresponsive, which
        is exactly when it is most likely to be stuck mid-transmit rather than
        idle — so the stop is `_terminate`'s, not a bare kill, and the grace is
        the unkey budget both here and in `stop()`. The escalation is the
        respawn, which happens even when the stop could not be completed: a
        `_reap` that reports a held-open pipe has still killed the process, and
        leaving the entry in `"backoff"` with no handle is a modem that never
        comes back."""
        with self._lock:
            proc, self._proc = self._proc, None
            self._fast_fails = 0
            self._backoff = self.backoff_initial_s
            self._down = False
            self._next_start_at = 0.0
            try:
                if proc is not None and proc.poll() is None:
                    _terminate(proc, grace)
            finally:
                self.start(now)

    def stop(self, term_timeout_s: float = _UNKEY_GRACE_S) -> None:
        """Clean shutdown: SIGTERM, wait, SIGKILL. Resets all policy state."""
        with self._lock:
            proc, self._proc = self._proc, None
            self._stopped = True
            self._down = False
            self._fast_fails = 0
            self._backoff = self.backoff_initial_s
            self._next_start_at = 0.0
        if proc is None or proc.poll() is not None:
            return
        _terminate(proc, term_timeout_s)

    # -- deaf-modem recycle ------------------------------------------------

    def note_activity(self, now: float | None = None) -> None:
        self._last_activity = time.monotonic() if now is None else now

    def idle_recycle_due(self, now: float | None = None) -> bool:
        if self.max_idle_recycle_s is None or not self.alive:
            return False
        now = time.monotonic() if now is None else now
        return now - self._last_activity >= self.max_idle_recycle_s


class Supervisor:
    """Owns every spawned process for a site config. Modems sharing a
    spawn_group share one refcounted ModemProcess."""

    def __init__(self, config: ModemSource, log_dir: str, *,
                 host: str = "127.0.0.1", probe_timeout_s: float = 0.5,
                 fast_fail_s: float = 10.0, breaker_threshold: int = 3,
                 down_retry_s: float = 300.0,
                 backoff_initial_s: float = 1.0,
                 backoff_max_s: float = 30.0) -> None:
        self.config = config
        self.log_dir = log_dir
        self.host = host
        self.probe_timeout_s = probe_timeout_s
        self._tuning = dict(fast_fail_s=fast_fail_s,
                            breaker_threshold=breaker_threshold,
                            down_retry_s=down_retry_s,
                            backoff_initial_s=backoff_initial_s,
                            backoff_max_s=backoff_max_s)
        self._lock = threading.Lock()
        self._procs: dict[str, ModemProcess] = {}
        self._holders: dict[str, Counter] = {}

    def _key(self, m: ModemConfig) -> str | None:
        if m.spawn_group is not None:
            return f"group:{m.spawn_group}"
        if m.spawn is not None:
            return f"modem:{m.name}"
        return None

    def process_for(self, modem_name: str) -> ModemProcess | None:
        """The (shared) ModemProcess behind a modem, or None if the modem is
        externally launched. Creates the handle lazily; does not start it."""
        m = self.config.modem(modem_name)
        key = self._key(m)
        if key is None:
            return None
        with self._lock:
            proc = self._procs.get(key)
            if proc is None:
                proc = self._build(m)
                self._procs[key] = proc
            return proc

    def _build(self, m: ModemConfig) -> ModemProcess:
        if m.spawn_group is not None:
            group = next(g for g in self.config.spawn_groups
                         if g.name == m.spawn_group)
            members = [mm for mm in self.config.modems
                       if mm.spawn_group == group.name]
            idle = [mm.quirks.max_idle_recycle_s for mm in members
                    if mm.quirks.max_idle_recycle_s is not None]
            return ModemProcess(
                group.name, group.spawn, cwd=group.cwd,
                log_path=os.path.join(self.log_dir, f"modem-{group.name}.log"),
                max_idle_recycle_s=min(idle) if idle else None, **self._tuning)
        return ModemProcess(
            m.name, m.spawn, cwd=m.cwd,
            log_path=os.path.join(self.log_dir, f"modem-{m.name}.log"),
            max_idle_recycle_s=m.quirks.max_idle_recycle_s, **self._tuning)

    # -- refcounted acquire/release ----------------------------------------

    def acquire(self, modem_name: str) -> ModemProcess | None:
        """Register interest in a modem's process and ensure it is up.
        Nests: one release() per acquire(). Returns None for
        externally-launched modems."""
        proc = self.process_for(modem_name)
        if proc is None:
            return None
        key = self._key(self.config.modem(modem_name))
        with self._lock:
            self._holders.setdefault(key, Counter())[modem_name] += 1
        try:
            self.ensure(modem_name)
        except Exception:
            self.release(modem_name)
            raise
        return proc

    def release(self, modem_name: str) -> None:
        """Drop one acquire()'s worth of interest; the process stops when the
        last one is released."""
        m = self.config.modem(modem_name)
        key = self._key(m)
        if key is None:
            return
        with self._lock:
            holders = self._holders.get(key)
            if holders is not None and holders[modem_name] > 0:
                holders[modem_name] -= 1
                if not holders[modem_name]:
                    del holders[modem_name]
            still_held = bool(holders)
            proc = self._procs.get(key)
        if proc is not None and not still_held:
            proc.stop()

    # -- health ------------------------------------------------------------

    def ensure(self, modem_name: str, now: float | None = None) -> bool:
        """True when the modem's ports should be attachable: process alive,
        an external instance already owns the ports (never double-spawn), or
        a fresh start-on-demand succeeded per the restart policy."""
        m = self.config.modem(modem_name)
        proc = self.process_for(modem_name)
        if proc is None:
            return probe(self.host, m.cmd_port, m.data_port, self.probe_timeout_s)
        if proc.alive:
            return True
        if probe(self.host, m.cmd_port, m.data_port, self.probe_timeout_s):
            return True
        return proc.ensure_running(now)

    def kill_and_respawn(self, modem_name: str) -> None:
        proc = self.process_for(modem_name)
        if proc is not None:
            proc.kill_and_respawn()

    def note_activity(self, modem_name: str) -> None:
        proc = self.process_for(modem_name)
        if proc is not None:
            proc.note_activity()

    def idle_recycle_due(self, modem_name: str, now: float | None = None) -> bool:
        proc = self.process_for(modem_name)
        return proc.idle_recycle_due(now) if proc is not None else False

    def stop_all(self) -> None:
        """Stop every process, then report the ones that could not be stopped.

        One modem with a leaked grandchild must not strand the rest — including
        one holding the key — running and un-SIGTERMed, and a second failure
        must not be lost behind the first."""
        with self._lock:
            procs = list(self._procs.values())
            self._holders.clear()
        failures = []
        for proc in procs:
            try:
                proc.stop()
            except Exception as exc:
                exc.add_note(f"stopping {proc.name}")
                failures.append(exc)
        if failures:
            raise ExceptionGroup("some modems could not be stopped", failures)
