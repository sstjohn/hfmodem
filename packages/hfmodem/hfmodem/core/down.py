# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from functools import partial
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import socket
import tempfile
import threading
import time

from . import claim

SETTLE_S = 45.0
# Pauses between CAT proof tries, one fewer than the tries. With a fresh open
# per try, the whole proof spans about 2 s.
CAT_RETRY_S = (.5, 1.0)
POLL_S = .05
TARGET_ROLES = {'python', 'supervisor', 'native', 'one-shot', 'recorder'}


class Refusal(claim.ClaimError):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Settings:
    path: Path
    station_port: int
    backend: str
    host: str
    port: int
    cat_port: str | None
    ptt_port: str | None
    profile: dict

    @classmethod
    def load(cls, path=None):
        from . import config
        from .rigs import CAT_BACKENDS
        if path is None:
            root = Path(__file__).resolve().parents[4]
            default = root / 'station.toml' if (root / 'packages/hfmodem').is_dir() else Path('station.toml')
            path = os.environ.get(config.STATION_ENV) or default
        path = Path(path).expanduser().resolve()
        cfg = config.load(path)
        return cls(path, cfg.station.port, CAT_BACKENDS[cfg.rig.cat], cfg.rig.host,
                   cfg.rig.port, claim.serial_port(cfg.resolved_profile.get('cat_port')),
                   claim.serial_port(cfg.rig.ptt.port), cfg.resolved_profile)

    @property
    def devices(self):
        return {port for port in (self.cat_port if self.backend == 'serial' else None,
                                  self.ptt_port) if port}

    def cat_proof(self):
        if self.backend == 'none':
            print('CAT none', flush=True)
            return 'none'
        try:
            if self.backend == 'serial':
                from .cat import CatDesync, QueryRefused, YaesuCat
                # A radio still busy after unkeying answers '?;', and a reply
                # nothing asked for poisons the driver, so each try opens afresh:
                # the open drains whatever the last one left behind.
                for pause in (*CAT_RETRY_S, None):
                    driver = YaesuCat(self.cat_port, self.profile)
                    # CAT opens in its worker; carry this proof's refusal scope into it.
                    driver._run = partial(copy_context().run, driver._run)
                    try:
                        driver.open()
                        hz = driver.freq()
                        break
                    except (QueryRefused, CatDesync):
                        if pause is None:
                            raise
                    finally:
                        driver.close()
                    time.sleep(pause)
            else:
                deadline = time.monotonic() + 2
                with socket.create_connection((self.host, self.port), timeout=2) as connection:
                    connection.sendall(b'f\n')
                    reply = b''
                    while b'\n' not in reply and len(reply) < 128:
                        connection.settimeout(max(.001, deadline - time.monotonic()))
                        data = connection.recv(128)
                        if not data or time.monotonic() >= deadline:
                            break
                        reply += data
                    text = reply.split(b'\n', 1)[0].strip()
                    hz = int(text) if text.isdigit() else 0
            if hz <= 0:
                raise claim.ClaimError('CAT frequency readback invalid')
        except Exception:
            print('CAT failed', flush=True)
            raise
        print('CAT ok', flush=True)
        return 'ok'


def _record(line, cat):
    target = claim.state_dir() / 'cleanup.json'
    fd, temporary = tempfile.mkstemp(prefix='cleanup-', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(dict(pid=os.getpid(), at=time.time(), line=line, cat=cat), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass
class Inventory:
    targets: list
    readers: tuple[str, ...] = ()
    releasing: bool = False


class Flow:
    def __init__(self, settings, backend):
        self.settings, self.backend = settings, backend
        self.stoppers = {}
        self.max_unwind = 0
        self.offers = SimpleQueue()
        backend.offer = self.offers.put

    def accept(self, entry, deadline):
        if entry.identity in self.stoppers:
            return
        if time.monotonic() >= deadline:
            raise Refusal(f'pid {entry.pid} found after cleanup inventory did not settle within 45 s')
        stopper = self.backend.stop(entry)
        self.stoppers[entry.identity] = stopper
        self.max_unwind = max(self.max_unwind, stopper.deadline - stopper.requested_at)

    def offered(self, deadline):
        while True:
            try:
                entry = self.offers.get_nowait()
            except Empty:
                return
            self.accept(entry, deadline)

    def poll(self):
        for stopper in self.stoppers.values():
            terminated = stopper.kill_deadline is not None
            result = stopper.poll()
            if (not terminated and stopper.kill_deadline is not None
                    and not getattr(self.backend, 'reports_termination', False)):
                print(f'pid {stopper.entry.pid} -- terminated', flush=True)
            if result is not None and not result:
                raise Refusal(f'termination unresolved for pid {stopper.entry.pid}: {result.detail}')

    def pending(self):
        return [stopper for stopper in self.stoppers.values() if stopper.result is None]

    def settle(self, acquire=None):
        deadline = time.monotonic() + SETTLE_S
        self.backend.issue()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='down-inventory') as worker:
            while True:
                future = worker.submit(self.backend.inventory, max(deadline, time.monotonic() + POLL_S))
                while not future.done():
                    self.offered(deadline)
                    self.poll()
                    self.backend.wait(self.pending(), POLL_S)
                try:
                    inventory = future.result()
                except claim.ClaimError:
                    if time.monotonic() < deadline:
                        raise
                    # Discovery honours the deadline the window just reached; the
                    # window's last inventory takes the final pass's bound instead.
                    inventory = self.backend.inventory(
                        min(time.monotonic() + 5, deadline + self.max_unwind + 5))
                self.offered(deadline)
                self.poll()
                for entry in inventory.targets:
                    self.accept(entry, deadline)
                self.poll()
                pending = self.pending()
                unsettled = pending or inventory.targets or inventory.readers or inventory.releasing
                if unsettled and time.monotonic() >= deadline:
                    # Accepted unwind bounds survive the inventory window.
                    while self.pending():
                        self.poll()
                        self.backend.wait(self.pending(), POLL_S)
                    if inventory.readers or inventory.releasing:
                        raise Refusal('cleanup inventory did not settle within 45 s: ' +
                                      ', '.join(inventory.readers or ('station not released',)))
                    limit = min(time.monotonic() + 5, deadline + self.max_unwind + 5)
                    if time.monotonic() >= limit:
                        raise Refusal('cleanup inventory did not settle within 45 s')
                    final = self.backend.inventory(limit)
                    if final.targets or final.readers or final.releasing:
                        raise Refusal('cleanup inventory did not settle within 45 s')
                elif unsettled:
                    self.backend.wait(pending, min(.1, max(0, deadline - time.monotonic())))
                    continue
                if acquire is None:
                    return
                try:
                    return acquire()
                except claim.ClaimBusy as exc:
                    if time.monotonic() >= deadline:
                        raise Refusal(f'cleanup inventory did not settle within 45 s: {exc}') from exc
                self.backend.wait([], min(.1, max(0, deadline - time.monotonic())))

    def run(self):
        gate = descriptor = None
        inherited = False
        try:
            claim.bootstrap()
            descriptor = self.backend.received()
            inherited = descriptor is not None
            self.backend.inherited = inherited
            self.backend.recover()
            gate = self.backend.gate()
            if descriptor is None:
                descriptor = self.settle(lambda: claim.acquire('cleanup', 'down', cleanup=gate))
            else:
                self.settle()
            with self.backend.proof():
                line = claim.line_proof(claim.line_factory(self.settings.ptt_port))
                cat = self.settings.cat_proof()
            _record(line, cat)
            print('CLEANUP COMPLETE', flush=True)
            return 0
        finally:
            try:
                if descriptor is not None:
                    (claim.close if inherited else claim.release)(descriptor)
            finally:
                try:
                    if gate is not None:
                        claim.mutex_release(gate)
                finally:
                    self.backend.close()


class _Tcp4(C.Structure):
    _fields_ = [(name, W.DWORD) for name in ('state', 'address', 'port', 'remote', 'remote_port', 'pid')]


class _Tcp6(C.Structure):
    _fields_ = [('address', C.c_ubyte * 16), ('scope', W.DWORD), ('port', W.DWORD),
                ('remote', C.c_ubyte * 16), ('remote_scope', W.DWORD),
                ('remote_port', W.DWORD), ('state', W.DWORD), ('pid', W.DWORD)]


def listeners(port, host=None, *, deadline=None, cancelled=lambda: False):
    query = C.WinDLL('iphlpapi', use_last_error=True).GetExtendedTcpTable
    query.argtypes = [C.c_void_p, C.POINTER(W.DWORD), W.BOOL, W.ULONG, C.c_int, W.ULONG]
    query.restype = W.DWORD
    def check():
        if cancelled() or deadline is not None and time.monotonic() >= deadline:
            raise claim.ClaimError('listener inspection interrupted or timed out')
    check()
    addresses = {address[4][0].split('%')[0] for address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM,
        flags=socket.AI_NUMERICHOST if deadline is not None else 0)} if host else None
    result = set()
    for family, row_type in ((socket.AF_INET, _Tcp4), (socket.AF_INET6, _Tcp6)):
        check()
        size = W.DWORD(0)
        code = query(None, C.byref(size), False, family, 3, 0)
        if code not in (0, 122):
            raise claim.ClaimError(f'GetExtendedTcpTable listener inventory unavailable: {code}')
        while True:
            check()
            if size.value > 64 * 1024 * 1024:
                raise claim.ClaimError('GetExtendedTcpTable listener inventory too large')
            data = C.create_string_buffer(max(4, size.value))
            code = query(data, C.byref(size), False, family, 3, 0)
            if code != 122:
                break
        if code:
            raise claim.ClaimError(f'GetExtendedTcpTable listener inventory unavailable: {code}')
        count = W.DWORD.from_buffer(data).value
        if 4 + count * C.sizeof(row_type) > len(data):
            raise claim.ClaimError('GetExtendedTcpTable partial listener inventory')
        for row in (row_type * count).from_buffer(data, 4):
            raw = bytes(row.address) if family == socket.AF_INET6 else int(row.address).to_bytes(4, 'little')
            address = socket.inet_ntop(family, raw)
            if socket.ntohs(row.port & 0xffff) == port and (addresses is None or address in addresses or not any(raw)):
                result.add(row.pid)
    check()
    return result


class Windows:
    def __init__(self, settings, client):
        from . import winproc
        self.wp = winproc
        self.settings, self.client = settings, client
        self.inherited = False
        self.entries, self.end_entries = {}, {}
        self.issued, self.roots = set(), set()
        self.stations = set()
        self.lineage = winproc.Lineage()
        self.evidence, self.reported = {}, set()
        self.released = False
        self.release_done = threading.Event()
        self.release_started = False
        self.release_error = None
        self.resources = {}
        self.offer = lambda entry: None

    def say(self, message):
        if message not in self.reported:
            print(message, flush=True)
            self.reported.add(message)

    def received(self):
        record = self.wp.read_launch()
        if record is None or 'claim' not in record['handles']:
            if os.environ.get('RIG_CLAIMED') or record is not None and record['role'] == 'cleanup':
                raise Refusal('cleanup has no received claim')
            return None
        if record['role'] != 'cleanup':
            raise Refusal('down requires a cleanup launch record')
        return claim.received()

    def identify_recorded(self, identity, deadline):
        while True:
            try:
                return self.wp.open(identity['pid'])
            except self.wp.ProcessAccessError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                # A concurrent down can make even identifying opens fail during exit.
                time.sleep(min(POLL_S, remaining))

    def live_launches(self, deadline=None):
        retry_deadline = time.monotonic() + self.wp.KILL_WAIT_S
        if deadline is not None:
            retry_deadline = min(retry_deadline, deadline)
        return self.wp.live_launches(identify=lambda identity: self.identify_recorded(
            identity, retry_deadline))

    def recover_inspectors(self):
        abandoned = {}
        try:
            for _, record in self.wp.launch_records():
                if (record['role'] != 'inspector' or record['phase'] != 'unresolved'
                        or not self.wp._confirmed_exit(record['launcher'])):
                    continue
                child = record['child']
                try:
                    entry = self.wp.open(child['pid'])
                except claim.ClaimError:
                    continue
                if entry is not None:
                    if entry.started == child['started']:
                        abandoned[entry.pid] = entry
                    else:
                        entry.close()
            _, unresolved = self.wp._recover_inspectors(claim.state_dir())
            if unresolved in abandoned and abandoned[unresolved].wait(5):
                _, unresolved = self.wp._recover_inspectors(claim.state_dir())
            if unresolved is not None:
                raise Refusal(f'inspection unavailable: pid {unresolved} (inspector) has not exited')
        finally:
            for entry in abandoned.values():
                entry.close()

    def recover(self):
        self.recover_inspectors()
        for _, record in self.wp.launch_records():
            if record['role'] == 'inspector' or record['phase'] != 'created':
                continue
            self.wp._inspect_checkpoint('down_recovery_read')
            if not self.wp._confirmed_exit(record['launcher']):
                continue
            identity = record['child']
            deadline = time.monotonic() + self.wp.KILL_WAIT_S
            error = None
            try:
                child = self.wp.open(identity['pid'], self.wp.END)
            except self.wp.ProcessAccessError as exc:
                error = exc
                try:
                    child = self.identify_recorded(identity, deadline)
                except claim.ClaimError as exc:
                    raise Refusal(f"cannot identify recorded child pid {identity['pid']}: {exc}") from exc
            if child is None:
                continue
            with child:
                if child.started != identity['started'] or child.wait():
                    continue
                fresh = self.wp.read_launch(child)
                if fresh is None or fresh['phase'] != 'created':
                    continue
                if error is None:
                    try:
                        child.terminate()
                    except self.wp.ProcessAccessError as exc:
                        error = exc
                if not child.wait(max(0, deadline - time.monotonic())):
                    if error is not None:
                        raise Refusal(str(error)) from error
                    raise Refusal(f'termination unresolved for pid {child.pid}')
                self.wp._update_launch(child, 'exited')
                self.say(f'pid {child.pid} -- recovered never-run child')
        _, unverified = self.live_launches()
        if unverified:
            raise Refusal('cannot identify recorded child pid ' + ', '.join(map(str, sorted(unverified))))
        self.orphan_admissions()

    def orphan_admissions(self):
        records, _ = self.live_launches()
        for record in records:
            if (record['role'] not in ('cleanup', 'inspector', 'recorder') and
                    record['phase'] == 'resuming' and self.wp._confirmed_exit(record['launcher'])):
                child, launcher = record['child'], record['launcher']
                raise Refusal(f"pid {child['pid']} ({Path(child['image']).name}) was launched and never acknowledged; "
                              f"its launcher pid {launcher['pid']} has exited. End it "
                              f"(`Stop-Process -Id {child['pid']}`) and run down again.", 2)

    def publication(self, *, cleanup=False):
        from . import winclaim
        data = winclaim.read_record(claim.mutex_path() if cleanup else claim.claim_path())
        if not data:
            return None
        return claim._parse_line(data, cleanup=cleanup)

    def busy_gate(self):
        self.orphan_admissions()
        rows = self.wp.using(claim.mutex_path())
        published = self.publication(cleanup=True)
        if published is not None and any(pid == published.pid for pid, _ in rows):
            if not self.wp._confirmed_exit(dict(pid=published.pid, started=published.started)):
                with_entry = self.wp.open(published.pid)
                if with_entry is None:
                    raise Refusal(f'cannot identify cleanup publisher pid {published.pid}')
                with with_entry:
                    if with_entry.started == published.started and not with_entry.wait():
                        return True, f'cleanup in progress (pid {published.pid} {Path(with_entry.image).name})'
        records, _ = self.live_launches()
        names = {pid: Path(image).name or 'cannot identify' for pid, image in rows if pid != os.getpid()}
        for record in records:
            if record['role'] not in ('cleanup', 'inspector', 'recorder') and record['phase'] in ('created', 'resuming'):
                child = record['child']
                names[child['pid']] = Path(child['image']).name
        return False, 'admission in progress (pids ' + ', '.join(f'{pid} {name}' for pid, name in sorted(names.items())) + ')'

    def gate(self):
        try:
            self.gate_fd = claim.mutex(timeout=30, busy=self.busy_gate)
            return self.gate_fd
        except claim.ClaimBusy as exc:
            self.orphan_admissions()
            _, message = self.busy_gate()
            raise Refusal(message, 2) from exc

    def refresh(self):
        entries, unverified = self.wp.pinned()
        for entry in entries:
            previous = self.entries.get(entry.pid)
            if previous is None:
                self.entries[entry.pid] = entry
            else:
                if previous.identity != entry.identity:
                    entry.close()
                    raise Refusal(f'process identity changed while pinned: pid {entry.pid}')
                entry.close()
        self.lineage.update(self.entries.values())
        return unverified

    def issue(self):
        self.refresh()
        self.issued = {entry.identity for entry in self.entries.values() if not entry.wait()}
        self.wp._inspect_checkpoint('down_issued')

    def relevant(self, pid, source):
        entry = self.entries.get(pid)
        if entry is None:
            self.refresh()
            entry = self.entries.get(pid)
        if entry is None:
            raise Refusal(f'cannot identify pid {pid} ({source})')
        return entry

    def same_account(self, entry):
        from . import winclaim
        api, tokens = self.wp._api(), []
        try:
            for handle in (api.GetCurrentProcess(), entry.handle):
                token = W.HANDLE()
                if not api.OpenProcessToken(handle, 8, C.byref(token)):
                    return False
                tokens.append(token)
            if not api.EqualSid(winclaim._token_sid(tokens[0], 1), winclaim._token_sid(tokens[1], 1)):
                return False
            sessions = []
            for token in tokens:
                value, size = W.DWORD(), W.DWORD()
                if not api.GetTokenInformation(token, 12, C.byref(value), C.sizeof(value), C.byref(size)):
                    return False
                sessions.append(value.value)
            return sessions[0] == sessions[1]
        finally:
            for token in tokens:
                winclaim.close_handle(token)

    def station(self, deadline):
        client = self.client
        pids = listeners(self.settings.station_port)
        service = client.Client(self.settings.path, port=self.settings.station_port)
        try:
            snapshot = service.state(timeout=min(2, max(.001, deadline - time.monotonic())))
        except OSError:
            if not pids:
                return set(), False
            raise Refusal('station not released, held by pid ' + ', '.join(map(str, sorted(pids))))
        except client.ClientError as exc:
            raise Refusal(f'station not released: {exc}') from exc
        pid = snapshot['service'].get('pid')
        if type(pid) is not int or pid not in pids:
            raise Refusal('station listener identity missing or inconsistent')
        if self.release_started and self.release_done.is_set():
            self.release_started = False
            if self.release_error is not None:
                raise Refusal(f'station not released: {self.release_error}')
        if self.release_started:
            return pids, True
        if snapshot['service']['state'] != 'observing' or not self.released:
            self.release_started = True
            self.release_done.clear()
            def release():
                try:
                    service.release(timeout=max(.001, deadline - time.monotonic()))
                    self.released = True
                except client.CommandFailed as exc:
                    if exc.result.get('reason') != 'transitioning':
                        self.release_error = exc
                except Exception as exc:
                    self.release_error = exc
                finally:
                    self.release_done.set()
            threading.Thread(target=release, name='down-station-release', daemon=True).start()
            return pids, True
        return pids, False

    def inventory(self, deadline):
        claim.verify_cleanup(self.gate_fd)
        self.refresh()
        records, unresolved = self.live_launches(deadline)
        claim_pids = {pid for pid, _ in self.wp.using(claim.claim_path())}
        seen_claim_pids = set(claim_pids)
        gate_pids = {pid for pid, _ in self.wp.using(claim.mutex_path())}
        stations, releasing = self.station(deadline)
        if stations and not releasing:
            claim_pids = {pid for pid, _ in self.wp.using(claim.claim_path())}
            seen_claim_pids |= claim_pids
        delegates = listeners(self.settings.port, self.settings.host) if self.settings.backend == 'rigctld' else set()
        own = self.relevant(os.getpid(), 'down')
        protected = self.lineage.ancestors(own) | {own.identity}
        for pid in stations:
            entry = self.relevant(pid, 'station listener')
            self.stations.add(entry.identity)
        station_tree = self.stations.copy()
        for identity in self.stations:
            station_tree |= self.lineage.descendants(identity)
        protected |= station_tree
        for pid in delegates:
            entry = self.relevant(pid, 'delegated CAT listener')
            protected.add(entry.identity)
            self.say(f'delegated rigctld pid {pid} on {self.settings.host}:{self.settings.port}')
        publisher = self.publication(cleanup=True)
        if publisher is not None:
            entry = self.relevant(publisher.pid, 'cleanup publisher')
            if entry.started == publisher.started and not entry.wait() and entry.pid in gate_pids:
                protected.add(entry.identity)
        recorded, inspectors = {}, set()
        for record in records:
            child = record['child']
            entry = self.relevant(child['pid'], f"recorded {record['role']}")
            if child['pid'] in unresolved:
                raise Refusal(f"cannot identify recorded child pid {child['pid']} ({child['image']})")
            if entry.started != child['started'] or entry.wait():
                continue
            recorded[entry.identity] = record
            if record['role'] == 'inspector':
                inspectors.add(entry.identity)
                self.say(f'pid {entry.pid} (inspector) excluded from cleanup targets')
                if record['phase'] == 'unresolved':
                    raise Refusal(f'inspection unavailable: pid {entry.pid} (inspector) has not exited')
            elif record['role'] in TARGET_ROLES:
                self.roots.add(entry.identity)
        publication = self.publication() if claim_pids else None
        if not self.inherited and publication is not None:
            entry = self.entries.get(publication.pid)
            if (entry is not None and entry.started == publication.started and not entry.wait()
                    and entry.pid in claim_pids and entry.identity not in recorded
                    and entry.identity not in station_tree):
                raise Refusal(f'pid {entry.pid} ({publication.command}) holds the claim; interrupt it and run down again', 7)
        for pid in sorted(seen_claim_pids):
            entry = self.relevant(pid, 'claim holder')
            if (entry.identity not in protected and entry.identity not in recorded
                    and not self.lineage.ancestors(entry) & self.roots):
                self.say(f'unidentified claim holder pid {entry.pid} {Path(entry.image).name}')
        for identity, record in recorded.items():
            entry = self.entries[identity[0]]
            if record['role'] in TARGET_ROLES and identity not in protected and not entry.wait():
                self.authorize(entry, (), claim_pids)
                self.offer(entry)
        candidates = claim_pids | gate_pids | {identity[0] for identity in station_tree | recorded.keys()}
        for pid in tuple(candidates):
            entry = self.relevant(pid, 'claim or gate holder')
            candidates |= {identity[0] for identity in self.lineage.descendants(entry)}
        for root in self.roots:
            candidates |= {identity[0] for identity in self.lineage.descendants(root)}
        entries = [self.relevant(pid, 'candidate') for pid in sorted(candidates)]
        entries = [entry for entry in entries if not entry.wait() and entry.identity != own.identity]
        walk = self.wp.device_holders(entries, self.settings.devices, deadline=min(deadline, time.monotonic() + 5)) if entries and self.settings.devices else None
        if walk is not None and walk.status == 'unavailable':
            raise Refusal(walk.message or 'inspection unavailable')
        positive = {entry.pid: walk.entries.get(entry.pid, self.wp.DeviceWalk()).devices for entry in entries} if walk else {}
        self.evidence = {}
        from .serialline import same_port
        recorded_ports = list(self.wp.recorded_devices(records, publication))
        roles = {identity: role for identity, role, _ in recorded_ports}
        for identity, role, devices in recorded_ports:
            entry = self.entries.get(identity[0])
            if entry is not None and entry.identity == identity and not entry.wait():
                for port in self.settings.devices:
                    if any(same_port(port, device) for device in devices):
                        self.evidence[port] = (f'{port} is held by your hfmodem '
                            f'{role} (pid {identity[0]})')
        for entry in entries:
            if entry.wait():
                continue
            for port in positive.get(entry.pid, ()):
                role = roles.get(entry.identity)
                if role is not None:
                    message = f'{port} is held by your hfmodem {role} (pid {entry.pid})'
                else:
                    message = f'{port} is open in pid {entry.pid} ({Path(entry.image).name})'
                self.evidence[port] = message
                if entry.identity in inspectors:
                    self.say(message + ' (inspector duplicate)')
        if not releasing and any(entry.identity in station_tree and (entry.pid in claim_pids or positive.get(entry.pid)) for entry in entries):
            raise Refusal('observing station still holds a device or claim')
        targets, readers = [], []
        for entry in entries:
            if entry.wait():
                continue
            ports = positive.get(entry.pid, ())
            if entry.identity in inspectors:
                continue
            if entry.identity in protected:
                if ports and entry.pid not in delegates and entry.identity not in station_tree:
                    raise Refusal('; '.join(self.evidence[port] for port in ports))
                continue
            record = recorded.get(entry.identity)
            target = record is not None and record['role'] in TARGET_ROLES
            target |= bool((entry.pid in claim_pids or ports) and self.lineage.ancestors(entry) & self.roots)
            if not target and entry.pid not in claim_pids and not ports:
                continue
            if entry.identity not in self.issued:
                raise Refusal(f'pid {entry.pid} started after this down was issued')
            if target:
                self.authorize(entry, ports, claim_pids)
                targets.append(entry)
            elif ports:
                raise Refusal('; '.join(self.evidence[port] for port in ports))
            else:
                name = f'unidentified claim holder pid {entry.pid} {Path(entry.image).name}'
                if walk is not None:
                    result = walk.entries.get(entry.pid, self.wp.DeviceWalk())
                    if not result.complete:
                        detail = result.message or walk.message or (
                            f'negative device evidence incomplete ({walk.status}; '
                            f'{len(result.handles)} handles checked, {result.counts["blocked"]} blocked)')
                        if walk.status == 'deadline' and time.monotonic() >= deadline:
                            raise Refusal(f'cleanup inventory did not settle within 45 s: {name}; {detail}')
                        raise Refusal(f'{name}; {detail}')
                self.say(name + ' -- waiting for plain reader')
                readers.append(name)
        return Inventory(targets, tuple(readers), releasing)

    def authorize(self, entry, ports, claim_pids):
        if entry.identity not in self.issued:
            raise Refusal(f'pid {entry.pid} started after this down was issued')
        resource = ', '.join(ports) or ('rig.claim' if entry.pid in claim_pids else 'recorded process')
        self.resources[entry.identity] = resource
        if not self.same_account(entry):
            raise Refusal(f'pid {entry.pid} ({Path(entry.image).name}) holds {resource} and this account cannot end it; '
                          'end it from an elevated shell or Task Manager')

    def stop(self, entry):
        try:
            ending = self.wp.open(entry.pid, self.wp.END)
        except self.wp.ProcessAccessError as exc:
            raise Refusal(f'pid {entry.pid} ({Path(entry.image).name}) holds '
                          f'{self.resources.get(entry.identity, "rig.claim")} and this account cannot end it; '
                          'end it from an elevated shell or Task Manager') from exc
        if ending is None or ending.identity != entry.identity:
            if ending is not None:
                ending.close()
            raise Refusal(f'cannot end pinned target pid {entry.pid}')
        self.end_entries[entry.identity] = ending
        return self.wp.Stopper(ending, report=lambda channel: print(
            f'pid {entry.pid} -- stop requested' if channel == 'stop channel' else
            f'pid {entry.pid} ({Path(entry.image).name}) -- {channel}', flush=True))

    def wait(self, stoppers, timeout):
        if not stoppers:
            time.sleep(timeout)
            return
        deadline = min(time.monotonic() + timeout, *(stopper.next_deadline for stopper in stoppers))
        handles = [stopper.entry.handle for stopper in stoppers]
        chunks = [handles[index:index + 64] for index in range(0, len(handles), 64)]
        for index, chunk in enumerate(chunks):
            result = self.wp._api().WaitForMultipleObjects(len(chunk), (W.HANDLE * len(chunk))(*chunk),
                False, self.wp._milliseconds(max(0, deadline - time.monotonic()) if index == len(chunks) - 1 else 0))
            if result == 0xffffffff:
                raise claim.ClaimError('WaitForMultipleObjects failed during cleanup')
            if result != 258:
                return

    def proof(self):
        return self.wp.port_refusals(self.evidence)

    def close(self):
        for entry in (*self.end_entries.values(), *self.entries.values()):
            entry.close()


def run(path, client):
    try:
        if os.name == 'nt':
            settings = Settings.load(path)
            backend = Windows(settings, client)
        else:
            from .down_posix import Posix, load_settings
            settings, audio = load_settings(path)
            backend = Posix(settings, audio, client)
        return Flow(settings, backend).run()
    except Exception as exc:
        print('NOT CONFIRMED: ' + str(exc).removeprefix('NOT CONFIRMED: '), flush=True)
        return exc.code if isinstance(exc, Refusal) else 1
