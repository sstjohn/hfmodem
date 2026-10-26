# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import errno
import fcntl
import os
import signal
import subprocess
import threading
import time

from hfmodem.core import claim as hardware
from hfmodem.core import coexist
from hfmodem.core.native import NativeChild, REAP_GRACE_S
from hfmodem.core.rigs import CAT_BACKENDS
from hfmodem.station import client as station_client

from . import logbook, operator, roster, station

ADMISSION_TIMEOUT_S = 30.0
_local = threading.local()
_descriptor_lock = threading.RLock()


def exit_code(error):
    """Keep the station client's public statuses at every console boundary."""
    if isinstance(error, station_client.ClientError):
        return error.exit_code
    if isinstance(error, hardware.ClaimBusy):
        return station_client.wire.EXIT_CLAIMED
    return 2


def attach(station_file=None):
    return station_client.attach(station_file or operator.station_file(),
                                 data_root=operator.data_root())


def own(station_file=None, *, receive=False):
    return attach(station_file).own(receive=receive)


@contextmanager
def release_confirmation():
    """An accepted handoff applies to this campaign, including later arms."""
    previous = getattr(_local, 'confirmed_releases', None)
    _local.confirmed_releases = set()
    try:
        yield
    finally:
        _local.confirmed_releases = previous


def release(station_file=None, *, timeout=45.0, confirm=None):
    station_file = station_file or operator.station_file()
    # A legacy session needs no daemon. Probe without attach(), which would
    # start one solely to ask it to give the hardware back.
    client = station_client.Client(station_file)
    deadline = time.monotonic() + timeout
    try:
        snapshot = client.state()
    except OSError as error:
        if error.errno == errno.ECONNREFUSED:
            return None
        raise
    if snapshot['service']['state'] != 'observing' and snapshot.get('owner'):
        confirmed = getattr(_local, 'confirmed_releases', None)
        key = client.station_file
        if confirmed is None or key not in confirmed:
            confirm = confirm or (lambda state: input(
                'The station has an active session. Release it for this legacy session? [y/N] '
            ).strip().lower() == 'y')
            try:
                accepted = confirm(snapshot)
            except EOFError:
                raise RuntimeError('station release declined: no input; nothing claimed') from None
            if not accepted:
                raise RuntimeError('station release declined; nothing claimed')
            if confirmed is not None:
                confirmed.add(key)
    while snapshot['service']['state'] != 'observing':
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('station did not reach observing before the release deadline')
        try:
            client.release(timeout=remaining)
        except station_client.CommandFailed as error:
            if error.result.get('reason') != 'transitioning':
                raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('station did not reach observing before the release deadline')
        snapshot = client.state(timeout=min(2, remaining))
        if snapshot['service']['state'] != 'observing':
            time.sleep(min(.1, max(0, deadline - time.monotonic())))
    return client


@contextmanager
def uninterrupted():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def defer_interrupts():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    caught = []
    previous = {sig: signal.signal(sig, lambda signum, frame: caught.append(signum))
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if caught:
        raise KeyboardInterrupt


@contextmanager
def stopping():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    def stop(signum, frame):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def lease(mode, *, env=None):
    with _descriptor_lock:
        if getattr(_local, 'held', False):
            hardware.inherited(9)
            yield hardware.ClaimDescriptor(9)
            return
        environment = dict(os.environ, **(env or {}))
        selection = station.resolve_selection(environ=environment)
        release(environment.get('HFMODEM_STATION'))
        cfg = selection._STATION_CONFIG
        devices = hardware.Devices(cat_port=selection.CAT_PORT or None,
            ptt_port=selection.PTT_PORT or None, capture=selection.AUDIO_IN,
            playback=selection.AUDIO_OUT, cat=CAT_BACKENDS[cfg.rig.cat] if cfg else 'serial')
        saved = None
        try:
            saved = fcntl.fcntl(9, fcntl.F_DUPFD_CLOEXEC, 10)
        except OSError as error:
            if error.errno != 9:
                raise
        acquired = None
        old = os.environ.get('RIG_CLAIMED')
        try:
            acquired = hardware.acquire('console', getattr(mode, 'name', str(mode)), devices=devices)
            if acquired.fd != 9:
                os.dup2(acquired.fd, 9, inheritable=False)
                hardware.close(acquired)
            acquired = hardware.ClaimDescriptor(9)
            _local.held = True
            _local.children = []
            os.environ['RIG_CLAIMED'] = str(hardware.claim_path())
            with stopping():
                yield acquired
        finally:
            with uninterrupted():
                try:
                    if acquired is not None:
                        try:
                            end_children()
                        except BaseException:
                            # Surviving inheritors must keep their possession evidence.
                            hardware.close(acquired)
                            raise
                        else:
                            hardware.release(acquired)
                finally:
                    _local.held = False
                    _local.children = []
                    if old is None:
                        os.environ.pop('RIG_CLAIMED', None)
                    else:
                        os.environ['RIG_CLAIMED'] = old
                    if saved is not None:
                        os.dup2(saved, 9, inheritable=False)
                        os.close(saved)


def remember(process):
    child = NativeChild(process.pid, hardware.process_started(process.pid),
                        getattr(process, 'stdin', None), getattr(process, 'stdout', None),
                        getattr(process, 'stderr', None), _process=process)
    process._claim_child = child
    if getattr(_local, 'held', False):
        _local.children.append(child)
    coexist.event('child_spawned', pid=child.pid, started=child.started)
    return process


def parents():
    done = subprocess.run(['ps', '-axo', 'pid=,ppid='], text=True,
                          capture_output=True, timeout=5, check=True)
    return {int(pid): int(parent) for pid, parent in
            (line.split() for line in done.stdout.splitlines() if line.strip())}


def cleanup_tree(pid, *, group=False):
    path = hardware.mutex_path()
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        try:
            rows = hardware.holders(fd, cleanup=True)
        except hardware.ClaimError:
            if not os.pread(fd, 4096, 0):
                return False
            raise
    finally:
        os.close(fd)
    if not rows:
        return False
    ancestry = parents()
    for row in rows:
        if hardware.process_started(row.pid) != row.started:
            continue
        if group:
            try:
                if os.getpgid(row.pid) == pid:
                    return True
            except ProcessLookupError:
                continue
        seen, current = set(), row.pid
        while current and current not in seen:
            if current == pid:
                return True
            seen.add(current)
            current = ancestry.get(current)
    return False


def end_group(process):
    """Stop a launcher's private session, including children without fd 9."""
    child = process._claim_child
    # Establish the group's identity while its remembered leader is alive;
    # an old pid alone must never authorize signalling a reused process group.
    if not child._alive(refresh_identity=True):
        return
    if os.getpgid(child.pid) != child.pid:
        raise hardware.ClaimError(f'pid {child.pid} is not a private process group')
    with uninterrupted():
        for sig in (signal.SIGTERM, signal.SIGKILL):
            # fd-8 offset matching proves cleanup ownership. Check the group as
            # well as ancestry: a dead launcher can leave reparented cleanup.
            if cleanup_tree(child.pid, group=True):
                return False
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + REAP_GRACE_S
            while time.monotonic() < deadline:
                process.poll()  # Reap the leader before checking its group.
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    return
                time.sleep(.05)
        raise hardware.ClaimError(f'process group {child.pid} did not terminate')


def end_child(process, *, interrupt=False):
    child = process if isinstance(process, NativeChild) else process._claim_child
    if not child.alive():
        return
    with uninterrupted():
        if interrupt:
            child._signal(signal.SIGINT)
            if child.wait_gone(REAP_GRACE_S):
                return
        child.terminate()
        if child.wait_gone(REAP_GRACE_S):
            return
        if cleanup_tree(child.pid):
            return False
        child.kill()
        if not child.wait_gone(REAP_GRACE_S):
            raise hardware.ClaimError(f'child pid {child.pid} did not terminate')


def end_children(*, allow_cleanup=False):
    for child in getattr(_local, 'children', ()):
        if end_child(child) is False and not allow_cleanup:
            raise hardware.ClaimError(f'cleanup still running in pid {child.pid}; not killed')
    rows = hardware.holders(9)
    if not rows:
        return
    ancestry = parents()
    ancestors, current = set(), os.getpid()
    while current and current not in ancestors:
        ancestors.add(current)
        current = ancestry.get(current)
    for row in rows:
        if row.pid not in ancestors:
            child = NativeChild(row.pid, row.started, None, None, None)
            if end_child(child) is False and not allow_cleanup:
                raise hardware.ClaimError(f'cleanup still running in pid {row.pid}; not killed')


@contextmanager
def launch():
    admission = hardware.inherited(9, launch=True)
    try:
        yield admission
    finally:
        admission.failed()


def assess(mode, script, *, timeout, env):
    with lease(mode, env=env), launch():
        process = None
        try:
            with defer_interrupts():
                process = remember(subprocess.Popen(
                    ['bash', '-c', script], cwd=station.ROOT, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(9,),
                    env=dict(os.environ, **env)))
            stdout, stderr = process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
        finally:
            if process is not None:
                end_child(process)


def spacing_note(mode, call, now):
    """How long since the last arm at this gateway, and whether it answered.

    Reported, never refused on. The spacing figure is the operator's to weigh,
    not the console's to enforce.
    """
    rule = roster.spacing(mode)
    for record in logbook.records(mode):
        if record['call'] != call:
            continue
        ended = datetime.fromisoformat(record.get('ended_utc') or record['started_utc'])
        answered = bool((record.get('verdict') or {}).get('reached'))
        since = (now - ended).total_seconds()
        said = (f'  {record["arm"]} ended {since:.0f}s ago and '
                f'{"answered" if answered else "did not answer"}')
        if rule is None:
            return said + f'; this roster carries no spacing figure for {call}.'
        wait = rule['after_an_answered_arm_s'] if answered else rule['after_an_unanswered_arm_s']
        return said + '; ' + (
            f'{wait - since:.0f}s short of the {wait}s that has worked at {call}.'
            if since < wait else f'past the {wait}s that has worked there.')
    return None
