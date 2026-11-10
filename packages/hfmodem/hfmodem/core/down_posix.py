# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import tomllib

from . import claim, posix
from .down import Inventory, Refusal, Settings
from .native import NativeChild, REAP_GRACE_S
from .rigs import CAT_BACKENDS, resolve_profile, refuse_device_in_test, profile_cat

fcntl = posix.module('fcntl')


def load_settings(path=None):
    from . import config
    path = path or os.environ.get('HFMODEM_STATION')
    body = tomllib.loads(Path(path).read_text()) if path else {}
    cfg = config.load(path) if body.get('schema') else None
    rig = body.get('rig', {})
    name, profile = (cfg.selected_rig, cfg.resolved_profile) if cfg else resolve_profile(
        legacy_model=rig.get('model'), profiles=rig.get('profiles'))[:2]
    ports = [os.environ.get(env) or (profile.get(field) if name != 'ft891' or
             cfg and cfg.resolution_sources.get(field) != 'rig-profile' else None)
             for field, env in (('cat_port', 'CAT_PORT'), ('ptt_port', 'PTT_PORT'))]
    refuse_device_in_test(*ports)
    backend = CAT_BACKENDS[rig.get('cat', profile_cat(name))]
    settings = Settings(Path(path) if path else None, body.get('station', {}).get('port', 8737),
                        backend, rig.get('host', '127.0.0.1'), rig.get('port', 4532), *ports, profile)
    audio = (cfg.audio.input, cfg.audio.output) if cfg else (profile.get('audio_device', ''),) * 2
    return settings, audio


def within(pid, ancestor, rows):
    seen = set()
    while pid in rows and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        pid = rows[pid][0]
    return False


def station(command):
    return bool(re.search(r'(?:^|[/\s])hfmodem(?:\.cli)?\s+station(?:\s|$)', command))


def listeners(port, *, timeout=5):
    result = subprocess.run(['lsof', '-t', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN'],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise claim.ClaimError('listener inventory unavailable: ' + result.stderr.strip())
    return {int(pid) for pid in result.stdout.split()}



@dataclass(frozen=True)
class Entry:
    child: NativeChild

    @property
    def pid(self):
        return self.child.pid

    @property
    def identity(self):
        return self.child.pid, self.child.started


class Stopper:
    def __init__(self, entry, backend):
        self.entry, self.backend = entry, backend
        self.requested_at = time.monotonic()
        self.deadline = self.requested_at + REAP_GRACE_S
        self.kill_deadline = None
        self.result = None
        self.backend.stopping[entry.pid] = entry.child.started
        if entry.child.alive():
            print(f'pid {entry.pid} -- SIGTERM', flush=True)
            entry.child.terminate()

    def poll(self):
        if self.result is not None:
            return self.result
        if not self.entry.child.alive():
            self.backend.ended[self.entry.pid] = self.entry.child.started
            self.result = True
        elif self.kill_deadline is None and time.monotonic() >= self.deadline:
            rows = self.backend.table()
            cleanup = {r.pid for r in claim.holders(self.backend.cleanup, cleanup=True)}
            if any(within(pid, self.entry.pid, rows) for pid in cleanup):
                raise claim.ClaimError(f'cleanup in progress (pid {self.entry.pid}); no escalation')
            print(f'pid {self.entry.pid} -- SIGKILL', flush=True)
            self.entry.child.kill()
            self.kill_deadline = time.monotonic() + 5
        elif self.kill_deadline is not None and time.monotonic() >= self.kill_deadline:
            raise Refusal(f'termination unresolved for pid {self.entry.pid}')
        return self.result


class Posix:
    reports_termination = True

    def __init__(self, settings, audio, client):
        self.settings, self.audio, self.client = settings, audio, client
        self.root = os.getpid()
        self.lineage, self.ended, self.issued, self.stopping = {}, {}, {}, {}
        self.inherited = self.released = False
        self.cleanup = self.descriptor = None
        self.signals = {}

    def received(self):
        if os.environ.get('RIG_CLAIMED'):
            self.descriptor = claim.received()
        return self.descriptor

    def recover(self):
        def interrupted(signum, frame):
            raise Refusal(f'cleanup interrupted by signal {signum}', 128 + signum)
        self.signals = {sig: signal.signal(sig, interrupted)
                        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}

    def gate(self):
        try:
            self.cleanup = claim.mutex(timeout=30)
            return self.cleanup
        except claim.ClaimBusy as exc:
            raise Refusal(str(exc), 2) from exc

    def issue(self):
        self.issued = {pid: row[3] for pid, row in self.table().items()}

    def stop(self, entry):
        return Stopper(entry, self)

    def wait(self, stoppers, timeout):
        time.sleep(timeout)

    def proof(self):
        claim.verify_cleanup(self.cleanup)
        return nullcontext()

    def close(self):
        for sig, handler in self.signals.items():
            signal.signal(sig, handler)

    def table(self, deadline=None):
        deadline = time.monotonic() + 5 if deadline is None else deadline
        error = 'deadline expired'
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(['ps', '-axo', 'pid=,ppid=,uid=,lstart=,command='],
                                        capture_output=True, text=True,
                                        timeout=min(5, max(.001, deadline - time.monotonic())))
            except subprocess.TimeoutExpired as exc:
                # A discovery failure like any other, so a deadline's final pass can retry it.
                raise claim.ClaimError(f'process inventory unavailable: {exc}') from exc
            if result.returncode == 0 and not result.stderr.strip() and time.monotonic() < deadline:
                rows = self.rows(result.stdout, deadline)
                if rows is not None:
                    break
                error = 'a listed parent exited while its child was identified'
                continue
            # Exit 1 does not certify stdout as complete. Discard it and retake
            # the table within the caller's bound, before recording any lineage.
            error = f'ps exit {result.returncode}: {result.stderr.strip()}'
            if result.returncode not in (0, 1):
                raise claim.ClaimError('process inventory unavailable: ' + error)
            time.sleep(min(.05, max(0, deadline - time.monotonic())))
        else:
            raise claim.ClaimError('process inventory unavailable: ' + error)
        if self.root not in rows:
            raise claim.ClaimError('process inventory unavailable')
        for pid, (parent, _, _, started) in rows.items():
            self.lineage.setdefault(pid, (started, parent, rows[parent][3] if parent in rows else None))
        return rows

    def rows(self, listing, deadline):
        """The listing's rows, or None when one that exited before its identity was
        read was the parent of a row still listed: that gap would cut its ancestry.

        Linux ps derives lstart from the boot time, which a wall-clock step moves, so
        there each identity is the start-tick token claims carry. The listing's own
        ps has exited by now and is dropped like any other childless row.
        """
        linux = sys.platform.startswith('linux')
        rows, exited = {}, set()
        for line in listing.splitlines():
            pid, parent, uid, *fields = line.strip().split(None, 8)
            if len(fields) != 6:
                raise claim.ClaimError("partial process metadata")
            pid, started = int(pid), "_".join(fields[:5])
            if linux:
                if time.monotonic() >= deadline:
                    raise claim.ClaimError('process inventory unavailable: deadline expired')
                try:
                    started = claim.process_started(pid)
                except claim.ClaimError:
                    if claim._exited(pid):
                        exited.add(pid)
                        continue
                    raise
            rows[pid] = (int(parent), int(uid), fields[-1], started)
        if linux and time.monotonic() >= deadline:
            raise claim.ClaimError('process inventory unavailable: deadline expired')
        if any(parent in exited for parent, *_ in rows.values()):
            return None
        return rows

    def inheritor(self, pid, matched, identities):
        started, parent, parent_started = self.lineage.get(pid, (None, None, None))
        stopping = self.stopping | self.ended
        return (pid in matched and started == identities[pid] and parent in stopping
                and parent_started in (None, stopping[parent]))


    def legacy_target(self, pid, rows, matched, delegated):
        parent, uid, command, started = rows[pid]
        if uid != os.getuid() or pid in delegated:
            return False
        words = command.split()
        daemon = any(Path(w).name == 'rigctld' for w in words) and any(
            w == '-m' and i + 1 < len(words) and words[i + 1] == str(self.settings.profile["model"])
            for i, w in enumerate(words))
        # A repository path and PPID 1 prove neither radio ownership nor a legacy
        # modem identity. Detached corpus/analysis workers can have both. Unknown
        # actual device holders remain fail-closed in inventory(), never signalled.
        return (pid in matched or daemon or bool(re.search(
            r'(?:hfmodem\.)?(?:besra\.host\.run_server|shrike\.onair|lanner\.qso|sabir\.onair)|'
            r'(?:^|[/\s])(?:run_server|kestrel_connect|kiwi_witness)(?:\.py)?(?:\s|$)|creance\.cli', command)))


    def inventory(self, deadline):
        claim.verify_cleanup(self.cleanup)
        settings = self.settings
        devices = claim.Devices(settings.cat_port, settings.ptt_port, *self.audio, settings.backend)
        nodes = []
        for endpoint in (devices.cat_port if settings.backend == 'serial' else None, devices.ptt_port):
            nodes.extend(claim.serial_nodes(endpoint))
        audio = []
        for endpoint, direction in zip(self.audio, ('capture', 'playback')):
            try:
                audio.extend(claim.audio_nodes(endpoint, direction))
            except Exception:
                pass
        rows = self.table(deadline)
        identities = {pid: row[3] for pid, row in rows.items()}
        ancestors = {pid for pid in rows if within(self.root, pid, rows)}
        delegates = listeners(settings.port) if settings.backend == 'rigctld' else set()
        for pid in delegates:
            if pid not in rows or 'rigctld' not in rows[pid][2]:
                raise claim.ClaimError(f'delegated CAT listener unidentified: pid {pid}')
            print(f'delegated rigctld pid {pid} on {settings.host}:{settings.port}', flush=True)
        stations = {pid for pid, row in rows.items() if station(row[2])} | listeners(settings.station_port)
        protected = {pid for pid in rows if any(within(pid, s, rows) for s in stations)}
        device_rows = claim.openers(nodes, deadline=deadline) if nodes else []
        audio_rows = claim.openers(audio, deadline=deadline) if audio else []
        device_rows += [r for r in audio_rows if Path(r.command).name not in
                        {'pipewire', 'pulseaudio', 'coreaudiod', 'wireplumber', 'jackd'}]
        device_pids = {r.pid for r in device_rows} - delegates
        probe = None
        try:
            descriptor = self.descriptor
            if descriptor is None:
                probe = os.open(claim.claim_path(), os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = False
                except BlockingIOError:
                    held = True
                descriptor = claim.ClaimDescriptor(probe)
            else:
                held = True
            published = claim.holder(descriptor) if held else None
            matched = {r.pid for r in claim.holders(descriptor, deadline=deadline)} if held else set()
            if time.monotonic() >= deadline:
                raise claim.ClaimError('cleanup inventory deadline expired')
        finally:
            if probe is not None:
                os.close(probe)
        candidates = device_pids | matched | stations | {
            pid for pid in rows if self.legacy_target(pid, rows, set(), delegates)}
        if not candidates <= rows.keys():
            # A holder younger than this table: unsettled, so the next pass takes it again.
            return Inventory([], ('process metadata missing',))
        if stations:
            if not settings.path:
                raise claim.ClaimError('station found without a station file; release it explicitly')
            client = self.client
            service = client.Client(settings.path, port=settings.station_port)
            try:
                snapshot = service.state(timeout=min(2, max(.001, deadline - time.monotonic())))
                if snapshot['service']['state'] != 'observing' or matched & protected and not self.released:
                    try:
                        service.release(timeout=max(.001, deadline - time.monotonic()))
                        self.released = True
                    except client.CommandFailed as exc:
                        if exc.result.get('reason') != 'transitioning':
                            raise
                    return Inventory([], releasing=True)
            except Exception as exc:
                raise claim.ClaimError('station not released, held by pid '
                                       + ', '.join(map(str, sorted(stations))) + f': {exc}') from exc
            if (device_pids | matched) & protected:
                raise claim.ClaimError('observing station still holds a device or claim')
        if not self.inherited and published and published.pid in rows:
            if not claim.process_matches(published.pid, published.started):
                raise claim.ClaimError('claim publisher pid reused; refusing cleanup')
            raise Refusal(f'pid {published.pid} ({published.command}) holds the claim; '
                          'interrupt it and run down again', 7)
        targets, unknown = [], []
        for pid in sorted(candidates - delegates - protected):
            if pid in ancestors:
                if pid in device_pids:
                    unknown.append(pid)
                continue
            if self.issued.get(pid) != identities[pid] and not self.inheritor(pid, matched, identities):
                raise claim.ClaimError(f'pid {pid} started after this down was issued')
            if self.legacy_target(pid, rows, matched, delegates):
                targets.append(Entry(NativeChild(pid, identities[pid], None, None, None)))
            else:
                unknown.append(pid)
        if unknown:
            raise claim.ClaimError('unidentified device/claim holder(s): ' + ', '.join(map(str, unknown)))
        if not targets:
            claim.check_devices(devices, deadline=deadline)
        if time.monotonic() >= deadline:
            raise claim.ClaimError('cleanup inventory deadline expired')
        return Inventory(targets)
