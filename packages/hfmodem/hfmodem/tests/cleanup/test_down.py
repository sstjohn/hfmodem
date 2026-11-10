# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from concurrent.futures import Future
from contextlib import nullcontext
import os
from types import SimpleNamespace as NS

import pytest

from hfmodem import cli
from hfmodem.core import claim, down, winproc


@pytest.mark.skipif(os.name == 'nt', reason='POSIX backend selection')
@pytest.mark.parametrize('argument', [[], ['candidate.toml']])
def test_posix_down_uses_shared_flow(argument, monkeypatch):
    from hfmodem.core import down_posix
    settings, audio, backend = object(), object(), NS()
    def load(path):
        assert str(path) == argument[0] if argument else path is None
        return settings, audio
    def posix(selected, devices, client):
        assert (selected, devices, client.__name__) == (settings, audio, 'hfmodem.station.client')
        return backend
    monkeypatch.setattr(down_posix, 'load_settings', load)
    monkeypatch.setattr(down_posix, 'Posix', posix)
    monkeypatch.setattr(down.Flow, 'run', lambda self: 0 if
                        (self.settings, self.backend) == (settings, backend) else 99)
    assert cli.main(['down', *argument]) == 0


class Immediate:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


@pytest.mark.parametrize('unresolved', [False, True])
def test_shutdown_deadlines_survive_rounds(monkeypatch, capsys, unresolved):
    clock, requests, kills = [100.0], [], []
    timer = NS(monotonic=lambda: clock[0])
    monkeypatch.setattr(down, 'time', timer)
    monkeypatch.setattr(winproc, 'time', timer)
    monkeypatch.setattr(down, 'ThreadPoolExecutor', Immediate)

    class Entry:
        def __init__(self, pid, grace):
            self.pid, self.grace, self.identity = pid, grace, (pid, 'win:1')
            self.ends = float('inf')

        def wait(self):
            return clock[0] >= self.ends

        def terminate(self):
            kills.append((self.pid, clock[0]))
            if not unresolved:
                self.ends = clock[0]

    entries = [Entry(1, 50), Entry(2, 50)]
    def request(entry):
        requests.append((entry.pid, clock[0]))
        return entry.grace
    monkeypatch.setattr(winproc, 'request_stop', request)

    class Backend:
        def issue(self):
            pass

        def inventory(self, deadline):
            available = entries if clock[0] >= 110 else entries[:1]
            return down.Inventory([entry for entry in available if not entry.wait()])

        def stop(self, entry):
            return winproc.Stopper(entry)

        def wait(self, stoppers, timeout):
            clock[0] += timeout

    flow = down.Flow(None, Backend())
    if unresolved:
        with pytest.raises(down.Refusal, match='termination unresolved for pid 1'):
            flow.settle()
        assert len(kills) == 1
    else:
        flow.settle()
        assert clock[0] < 162
        assert {pid for pid, _ in kills} == {1, 2}
        assert all(at >= dict(requests)[pid] + 51 for pid, at in kills)
    assert [pid for pid, _ in requests] == [1, 2]
    assert len(kills) == len({pid for pid, _ in kills})


@pytest.mark.parametrize('inherited', [False, True])
@pytest.mark.parametrize('failure', [None, 'line', 'cat'])
def test_proof_and_descriptor_lifetime(monkeypatch, tmp_path, inherited, failure):
    actions = []
    descriptor, gate = object(), object()
    monkeypatch.setattr(claim, 'bootstrap', lambda: actions.append('bootstrap'))
    monkeypatch.setattr(claim, 'state_dir', lambda: tmp_path)
    def acquire(*args, **kwargs):
        assert kwargs == {'cleanup': gate}
        actions.append('acquire')
        return descriptor
    monkeypatch.setattr(claim, 'acquire', acquire)
    monkeypatch.setattr(claim, 'line_factory', lambda port: 'factory')
    def line(factory):
        assert factory == 'factory'
        actions.append('line')
        if failure == 'line':
            raise claim.ClaimError('line failed')
        return 'low'
    monkeypatch.setattr(claim, 'line_proof', line)
    monkeypatch.setattr(claim, 'close', lambda d: actions.append(('close', d)))
    monkeypatch.setattr(claim, 'release', lambda d: actions.append(('release', d)))
    monkeypatch.setattr(claim, 'mutex_release', lambda d: actions.append(('gate-release', d)))
    def cat():
        actions.append('cat')
        if failure == 'cat':
            raise claim.ClaimError('CAT failed')
        return 'none'
    backend = NS(received=lambda: descriptor if inherited else None,
                 recover=lambda: actions.append('recover'), gate=lambda: gate,
                 proof=nullcontext, close=lambda: actions.append('backend-close'))
    flow = down.Flow(NS(ptt_port='COM41', cat_proof=cat), backend)
    def settle(acquire=None):
        actions.append('settle')
        return acquire() if acquire else None
    monkeypatch.setattr(flow, 'settle', settle)
    if failure:
        with pytest.raises(claim.ClaimError):
            flow.run()
        assert not (tmp_path / 'cleanup.json').exists()
    else:
        assert flow.run() == 0
        import json
        record = json.loads((tmp_path / 'cleanup.json').read_text())
        assert set(record) == {'pid', 'at', 'line', 'cat'}
        assert record['line'] == 'low' and record['cat'] == 'none'
    assert ('acquire' in actions) != inherited
    assert actions[-3:] == [('close' if inherited else 'release', descriptor),
                            ('gate-release', gate), 'backend-close']


def test_proof_uses_existing_port_evidence(monkeypatch):
    monkeypatch.setattr(winproc, '_port_holder', lambda *args: pytest.fail('proof started another inventory'))
    with winproc.port_refusals({'COM41': 'COM41 is open in pid 12 (foreign.exe)'}):
        port = r'\\.\COM41' if os.name == 'nt' else 'COM41'
        assert winproc.port_refusal(port, deadline=0) == 'COM41 is open in pid 12 (foreign.exe)'
        assert winproc.port_refusal('COM42', deadline=0) == 'COM42 is open in another process'
    assert winproc._port_refusals.get() is None


def test_cat_worker_uses_only_its_proof_evidence(monkeypatch, tmp_path):
    import serial
    from hfmodem.core import serialline
    from hfmodem.core.cat import RigError
    from hfmodem.core.rigs import RIGS
    monkeypatch.setattr(serialline, 'os', NS(**{**vars(serialline.os), 'name': 'nt'}))
    walked = []
    monkeypatch.setattr(winproc, '_port_holder', lambda *a: walked.append(a))

    class Port:
        def __init__(self, **kwargs):
            pass

        def open(self):
            failure = OSError('denied')
            failure.winerror = 5
            raise failure

        def close(self):
            pass

    monkeypatch.setattr(serial, 'Serial', Port)
    settings = down.Settings(tmp_path / 'station.toml', 0, 'serial', '', 0,
                             'COM41', None, RIGS['ft891'])
    for evidence in ('COM41 is held by your hfmodem station (pid 42)',
                     'COM41 is open in pid 43 (foreign.exe)'):
        with winproc.port_refusals({'COM41': evidence}):
            with pytest.raises(RigError) as caught:
                settings.cat_proof()
            assert str(caught.value) == f'CAT open failed: {evidence}'
        assert winproc._port_refusals.get() is None
    assert walked == []


def test_settings_resolve_station_profile_and_environment(tmp_path, monkeypatch):
    path = tmp_path / 'station.toml'
    path.write_text('''schema = 1
[station]
port = 18737
mycall = "K1ABC"
control = "local"
regulatory = "unregulated"
because = "fake endpoints"
transmit = false
[rig]
model = "ft891"
cat = "rigctld"
host = "127.0.0.1"
port = 14532
[rig.profiles.ft891]
cat_port = "COM41"
ptt_port = "COM42"
''')
    for key in ('HFMODEM_RIG', 'CAT_PORT', 'PTT_PORT'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('HFMODEM_STATION', str(path))
    settings = down.Settings.load()
    assert settings.path == path
    assert (settings.station_port, settings.backend, settings.host, settings.port) == (18737, 'rigctld', '127.0.0.1', 14532)
    assert settings.devices == {'COM42'} and settings.cat_port == 'COM41'
    monkeypatch.setenv('PTT_PORT', 'COM43')
    assert down.Settings.load(path).ptt_port == 'COM43'
    monkeypatch.delenv('PTT_PORT')
    path.write_text(path.read_text().replace('ptt_port = "COM42"', 'ptt_port = ""'))
    assert down.Settings.load(path).ptt_port is None


@pytest.mark.wall_clock
def test_recorded_target_is_requested_before_inspection_finishes(monkeypatch):
    import threading
    stopped = threading.Event()
    entry = NS(pid=1, identity=(1, 'win:1'), wait=lambda: True)
    monkeypatch.setattr(winproc, 'request_stop', lambda entry: 2)

    class Backend:
        def issue(self):
            pass

        def inventory(self, deadline):
            self.offer(entry)
            assert stopped.wait(2), 'inventory delayed the stop request'
            return down.Inventory([])

        def stop(self, entry):
            stopped.set()
            return winproc.Stopper(entry)

        def wait(self, stoppers, timeout):
            stopped.wait(timeout)

    down.Flow(None, Backend()).settle()
    assert stopped.is_set()


@pytest.mark.parametrize('ancestry', [False, True])
def test_record_only_inventory_does_not_pin_children(tmp_path, monkeypatch, ancestry):
    import json
    directory = tmp_path / 'rig.claim.launch.d'
    directory.mkdir()
    path = directory / '12-100.json'
    record = dict(child=dict(pid=12, started='win:100', image='python.exe'),
                  launcher=dict(pid=10, started='win:90', image='python.exe'),
                  role='python', phase='created')
    if ancestry:
        record['ancestry'] = [record['launcher'],
                              dict(pid=8, started='win:80', image='console.exe')]
    path.write_text(json.dumps(record))
    monkeypatch.setattr(claim, 'state_dir', lambda: tmp_path)
    monkeypatch.setattr(winproc, '_read_json', lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(winproc, 'open', lambda *args: pytest.fail('record read pinned a child'))
    assert list(winproc.launch_records()) == [(path, record)]


@pytest.mark.parametrize('ancestry', [False, True])
def test_record_inventory_refuses_missing_launcher(tmp_path, monkeypatch, ancestry):
    import json
    directory = tmp_path / 'rig.claim.launch.d'
    directory.mkdir()
    record = dict(child=dict(pid=12, started='win:100', image='python.exe'),
                  role='python', phase='created')
    if ancestry:
        record['ancestry'] = [dict(pid=10, started='win:90', image='python.exe')]
    (directory / '12-100.json').write_text(json.dumps(record))
    monkeypatch.setattr(claim, 'state_dir', lambda: tmp_path)
    monkeypatch.setattr(winproc, '_read_json', lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(winproc, 'open', lambda *args: pytest.fail('record read pinned a child'))
    with pytest.raises(claim.ClaimError, match='invalid launch ancestry'):
        list(winproc.launch_records())


@pytest.mark.parametrize('kind', ['launch', 'ports'])
def test_record_inventory_accepts_vanished_snapshot(tmp_path, monkeypatch, kind):
    from hfmodem.core import winclaim
    directory = tmp_path / f'rig.claim.{kind}.d'
    directory.mkdir()
    path = directory / '12-100.json'
    path.touch()
    monkeypatch.setattr(claim, 'state_dir', lambda: tmp_path)
    opened = winclaim.open_handle
    def vanished(path, *args):
        path.unlink()
        if os.name == 'nt':
            return opened(path, *args)
        return os.open(path, os.O_RDONLY)
    monkeypatch.setattr(winclaim, 'open_handle', vanished)
    inventory = winproc.launch_records() if kind == 'launch' else winproc.recorded_devices([], None)
    assert list(inventory) == []


@pytest.mark.parametrize('kind', ['launch', 'ports'])
def test_record_readers_do_not_create_directories(tmp_path, monkeypatch, kind):
    from hfmodem.core import winclaim
    path = tmp_path / f'rig.claim.{kind}.d' / '12-100.json'
    monkeypatch.setattr(claim, 'state_dir', lambda: tmp_path)
    if os.name != 'nt':
        def missing(path, *args):
            raise FileNotFoundError(path)
        monkeypatch.setattr(winclaim, 'open_handle', missing)
    assert winproc._read_json(path) is None
    assert list(winproc.launch_records()) == []
    assert winproc.live_launches() == ([], set())
    assert list(winproc.recorded_devices([], None)) == []
    winproc._prune_record(path)
    if kind == 'launch':
        entry = winproc.Pinned(None, 12, 'win:100', 'python.exe')
        entry.wait = lambda timeout=0: True
        assert winproc.read_launch(entry) is None
        assert winproc._update_launch(entry, grace=15) is None
        assert winproc.advance_launch(entry, 'exited') is None
        assert winproc._reap_inspector(entry, path) == 'exited'
    assert not path.parent.exists()


@pytest.mark.parametrize('kind', ['launch', 'ports'])
def test_record_pruning_unlinks_while_reader_is_open(tmp_path, monkeypatch, kind):
    import ctypes as C
    from hfmodem.core import winclaim
    path = tmp_path / f'rig.claim.{kind}.d' / '12-100.json'
    path.parent.mkdir()
    path.write_text('record')
    locks = []
    monkeypatch.setattr(winproc, '_record_lock', lambda path: locks.append(path) or nullcontext(True))
    monkeypatch.setattr(winproc, '_replace', lambda *args: pytest.fail('pruning renamed a record'))
    if os.name == 'nt':
        import msvcrt
        reader = os.fdopen(msvcrt.open_osfhandle(winclaim.open_handle(path, 1, 7), os.O_RDONLY))
    else:
        def deleting(handle, kind, data, size):
            assert kind == 21
            assert C.cast(data, C.POINTER(winproc.W.DWORD)).contents.value == 3
            assert size == C.sizeof(winproc.W.DWORD)
            path.unlink()
            return True
        monkeypatch.setattr(winclaim, 'open_handle', lambda path, access, share: os.open(path, os.O_RDONLY))
        monkeypatch.setattr(winclaim, 'close_handle', os.close)
        monkeypatch.setattr(winproc, '_api', lambda: NS(SetFileInformationByHandle=deleting))
        reader = path.open()
    with reader:
        winproc._prune_record(path)
        assert not path.exists()
        winproc._prune_record(path)
        assert reader.read() == 'record'
    assert locks == [path, path]
    assert not list(path.parent.iterdir())


@pytest.mark.parametrize('phase', [None, 'resuming', 'exited', 'unresolved'])
@pytest.mark.parametrize('present', [True, False])
def test_launch_update_accepts_completed_recovery(tmp_path, monkeypatch, phase, present):
    entry = winproc.Pinned(None, 12, 'win:100', 'python.exe')
    record = dict(child=winproc._identity(entry), role='inspector', phase='exited') if present else None
    monkeypatch.setattr(claim, 'state_dir', lambda: tmp_path)
    monkeypatch.setattr(winproc, '_record_lock', lambda path: nullcontext(True))
    monkeypatch.setattr(winproc, 'read_launch', lambda entry: record)
    monkeypatch.setattr(winproc, '_write_json', lambda *args: pytest.fail('completed recovery rewritten'))
    assert winproc._update_launch(entry, phase, grace=15) == record
    if record is not None:
        assert record['phase'] == 'exited' and 'grace' not in record
    assert winproc.advance_launch(entry, 'exited') == record
    if not present:
        with pytest.raises(claim.ClaimError, match='missing launch record'):
            winproc.advance_launch(entry, 'resuming')


@pytest.mark.parametrize('present', [True, False])
def test_inspector_reap_accepts_completed_recovery(tmp_path, monkeypatch, present):
    entry = winproc.Pinned(None, 12, 'win:100', 'python.exe')
    entry.wait = lambda timeout=0: False
    record = dict(child=winproc._identity(entry), phase='exited') if present else None
    monkeypatch.setattr(winproc, '_record_lock', lambda path: nullcontext(True))
    monkeypatch.setattr(winproc, '_read_json', lambda path: record)
    monkeypatch.setattr(winproc, '_write_json', lambda *args: pytest.fail('completed recovery rewritten'))
    assert winproc._reap_inspector(entry, tmp_path / '12-100.json') == 'exited'


@pytest.fixture
def reader_inventory(monkeypatch):
    reader = NS(pid=12, started='win:10', identity=(12, 'win:10'), image='python.exe',
                wait=lambda timeout=0: False)
    own = NS(pid=os.getpid(), identity=(os.getpid(), 'win:20'), wait=lambda timeout=0: False)
    backend = down.Windows(NS(devices={'COM41'}, backend='none', station_port=12345), None)
    backend.entries = {entry.pid: entry for entry in (own, reader)}
    backend.issued = {entry.identity for entry in (own, reader)}
    backend.gate_fd = 99
    backend.lineage = NS(ancestors=lambda entry: set(), descendants=lambda entry: set())
    monkeypatch.setattr(backend, 'refresh', lambda: None)
    monkeypatch.setattr(backend, 'station', lambda deadline: (set(), False))
    monkeypatch.setattr(backend, 'publication', lambda **kwargs: None)
    monkeypatch.setattr(claim, 'verify_cleanup', lambda fd: None)
    monkeypatch.setattr(winproc, 'live_launches', lambda **kwargs: ([], set()))
    monkeypatch.setattr(winproc, 'using', lambda path: [(reader.pid, reader.image)] if path == claim.claim_path() else [])
    return backend, reader


@pytest.mark.parametrize('disappears', ['walk', 'station-refresh', None])
def test_observed_reader_is_named_before_it_disappears(reader_inventory, monkeypatch, capsys, disappears):
    backend, reader = reader_inventory
    if disappears == 'station-refresh':
        monkeypatch.setattr(backend, 'station', lambda deadline: ({123}, False))
        station = NS(pid=123, identity=(123, 'win:1'), wait=lambda: False)
        backend.entries[123] = station
        calls = []
        def using(path):
            if path != claim.claim_path():
                return []
            calls.append(path)
            return [(reader.pid, reader.image)] if len(calls) == 1 else []
        monkeypatch.setattr(winproc, 'using', using)
    def walk(entries, devices, **kwargs):
        if disappears == 'walk':
            reader.wait = lambda timeout=0: True
        return winproc.DeviceHolders({entry.pid: winproc.DeviceWalk(complete=True) for entry in entries}, 'complete')
    monkeypatch.setattr(winproc, 'device_holders', walk)
    inventory = backend.inventory(down.time.monotonic() + 45)
    name = f'unidentified claim holder pid {reader.pid} python.exe'
    assert name in capsys.readouterr().out
    assert inventory.readers == ((name,) if disappears is None else ())
    assert not inventory.targets


@pytest.mark.parametrize('status', ['busy', 'incomplete', 'deadline'])
@pytest.mark.parametrize('expired', [False, True])
def test_incomplete_reader_evidence_names_its_cause(reader_inventory, monkeypatch, status, expired):
    backend, reader = reader_inventory
    monkeypatch.setattr(down, 'time', NS(monotonic=lambda: 100))
    monkeypatch.setattr(winproc, 'device_holders', lambda *args, **kwargs:
        winproc.DeviceHolders({reader.pid: winproc.DeviceWalk(handles={4: 'blocked'})}, status))
    with pytest.raises(down.Refusal) as caught:
        backend.inventory(100 if expired else 145)
    assert f'unidentified claim holder pid {reader.pid}' in str(caught.value)
    assert f'({status}; 1 handles checked, 1 blocked)' in str(caught.value)
    assert ('did not settle within 45 s' in str(caught.value)) == (expired and status == 'deadline')


def test_plain_reader_settle_expiry(reader_inventory, monkeypatch):
    backend, reader = reader_inventory
    clock = [100.0]
    monkeypatch.setattr(down, 'time', NS(monotonic=lambda: clock[0]))
    monkeypatch.setattr(down, 'ThreadPoolExecutor', Immediate)
    monkeypatch.setattr(backend, 'issue', lambda: None)
    monkeypatch.setattr(backend, 'wait', lambda stoppers, timeout: clock.__setitem__(0, clock[0] + timeout))
    monkeypatch.setattr(winproc, 'device_holders', lambda *args, **kwargs:
        winproc.DeviceHolders({reader.pid: winproc.DeviceWalk(complete=True)}, 'complete'))
    with pytest.raises(down.Refusal, match='did not settle within 45 s: unidentified claim holder pid 12'):
        down.Flow(None, backend).settle()
    assert clock[0] == pytest.approx(145)


@pytest.mark.parametrize('final', ['busy', 'expired'])
def test_inventory_cut_by_the_settle_deadline_gets_the_final_bound(monkeypatch, final):
    clock = [100.0]
    bounds = []
    monkeypatch.setattr(down, 'time', NS(monotonic=lambda: clock[0]))
    monkeypatch.setattr(down, 'ThreadPoolExecutor', Immediate)

    class Backend:
        def issue(self):
            pass

        def inventory(self, deadline):
            # Discovery takes a second, and stops at the deadline it is given.
            bounds.append(deadline)
            if deadline - clock[0] < 1 or final == 'expired' and len(bounds) > 1 and clock[0] >= 145:
                clock[0] = deadline
                raise claim.ClaimError('holder discovery deadline expired')
            clock[0] += 1
            return down.Inventory([])

        def wait(self, stoppers, timeout):
            clock[0] += timeout

    def acquire():
        raise claim.ClaimBusy('rig claimed: pid 12')

    with pytest.raises(claim.ClaimError) as caught:
        down.Flow(None, Backend()).settle(acquire)
    if final == 'busy':
        assert str(caught.value) == 'cleanup inventory did not settle within 45 s: rig claimed: pid 12'
        assert bounds[-1] - bounds[-2] == pytest.approx(5 - down.POLL_S)
    else:
        assert str(caught.value) == 'holder discovery deadline expired'


@pytest.mark.parametrize('station_exits', [False, True])
def test_station_protection_survives_listener_loss(reader_inventory, monkeypatch, station_exits):
    backend, _ = reader_inventory
    station = winproc.Pinned(123, 123, 'win:1', 'station.exe')
    child = winproc.Pinned(124, 124, 'win:2', 'recorder.exe')
    own = winproc.Pinned(125, os.getpid(), 'win:20', 'down.exe')
    backend.entries.update((entry.pid, entry) for entry in (station, child, own))
    backend.issued.update((station.identity, child.identity))
    backend.lineage = winproc.Lineage()
    backend.lineage.links[child.identity] = station.identity
    monkeypatch.setattr(station, 'wait', lambda: False)
    monkeypatch.setattr(child, 'wait', lambda: False)
    monkeypatch.setattr(own, 'wait', lambda: False)
    monkeypatch.setattr(winproc, 'using', lambda path: [])
    monkeypatch.setattr(winproc, 'live_launches', lambda **kwargs: ([dict(
        child=dict(pid=child.pid, started=child.started, image=child.image),
        role='recorder', phase='acknowledged')], set()))
    monkeypatch.setattr(winproc, 'device_holders', lambda *args, **kwargs:
        winproc.DeviceHolders({}, 'complete'))
    monkeypatch.setattr(backend, 'station', lambda deadline: ({station.pid}, False))
    monkeypatch.setattr(backend, 'same_account', lambda entry: True)
    monkeypatch.setattr(backend, 'stop', lambda entry: pytest.fail('protected child stopped'))
    monkeypatch.setattr(child, 'terminate', lambda: pytest.fail('protected child terminated'))
    flow = down.Flow(None, backend)
    first = backend.inventory(down.time.monotonic() + 45)
    assert not first.targets
    monkeypatch.setattr(backend, 'station', lambda deadline: (set(), False))
    monkeypatch.setattr(station, 'wait', lambda: station_exits)
    monkeypatch.setattr(backend, 'issue', lambda: None)
    flow.settle()
    assert backend.stations == {station.identity}
    assert not flow.stoppers


@pytest.mark.parametrize('blocker', ['reader', 'incomplete', 'device', 'late'])
def test_final_open_reclassifies_reopened_reader(reader_inventory, monkeypatch, blocker):
    backend, reader = reader_inventory
    clock, issues, attempts, inspections = [100.0], [], [], []
    descriptor = object()
    monkeypatch.setattr(down, 'time', NS(monotonic=lambda: clock[0]))
    monkeypatch.setattr(down, 'ThreadPoolExecutor', Immediate)
    monkeypatch.setattr(backend, 'issue', lambda: issues.append(clock[0]))
    monkeypatch.setattr(backend, 'wait', lambda stoppers, timeout: clock.__setitem__(0, clock[0] + timeout))
    def using(path):
        if path == claim.claim_path() and len(attempts) == 1:
            return [(reader.pid, reader.image)]
        return []
    monkeypatch.setattr(winproc, 'using', using)
    def walk(entries, devices, **kwargs):
        inspections.append(clock[0])
        if len(inspections) == 2:
            monkeypatch.setattr(winproc, 'using', lambda path: [])
        return winproc.DeviceHolders({reader.pid: winproc.DeviceWalk(
            complete=blocker != 'incomplete', devices={'COM41'} if blocker == 'device' else set())}, 'complete')
    monkeypatch.setattr(winproc, 'device_holders', walk)
    def acquire():
        attempts.append(clock[0])
        if len(attempts) == 1:
            if blocker == 'late':
                backend.issued.remove(reader.identity)
            raise claim.ClaimBusy('reader reopened rig.claim')
        assert len(inspections) == 2
        return descriptor
    flow = down.Flow(None, backend)
    if blocker == 'reader':
        assert flow.settle(acquire) is descriptor
        assert len(attempts) == 2
    else:
        reason = {'incomplete': 'negative device evidence incomplete',
                  'device': 'COM41 is open in pid 12', 'late': 'started after this down was issued'}[blocker]
        with pytest.raises(down.Refusal, match=reason):
            flow.settle(acquire)
        assert len(attempts) == 1
    assert issues == [100]


@pytest.mark.parametrize('observed', [False, True])
def test_final_open_retries_share_original_settle_deadline(reader_inventory, monkeypatch, observed):
    backend, reader = reader_inventory
    clock, issues, attempts = [100.0], [], []
    monkeypatch.setattr(down, 'time', NS(monotonic=lambda: clock[0]))
    monkeypatch.setattr(down, 'ThreadPoolExecutor', Immediate)
    monkeypatch.setattr(backend, 'issue', lambda: issues.append(clock[0]))
    monkeypatch.setattr(backend, 'wait', lambda stoppers, timeout: clock.__setitem__(0, clock[0] + timeout))
    monkeypatch.setattr(winproc, 'using', lambda path: [])
    def acquire():
        attempts.append(clock[0])
        if len(attempts) == 1:
            clock[0] += 44
            if observed:
                monkeypatch.setattr(winproc, 'using', lambda path:
                    [(reader.pid, reader.image)] if path == claim.claim_path() else [])
        raise claim.ClaimBusy('reader reopened rig.claim')
    monkeypatch.setattr(winproc, 'device_holders', lambda *args, **kwargs:
        winproc.DeviceHolders({reader.pid: winproc.DeviceWalk(complete=True)}, 'complete'))
    with pytest.raises(down.Refusal, match='did not settle within 45 s'):
        down.Flow(None, backend).settle(acquire)
    assert clock[0] == pytest.approx(145)
    assert issues == [100]
    if observed:
        assert attempts == [100]
    else:
        assert len(attempts) > 1


@pytest.mark.parametrize('denied', ['open', 'terminate'])
@pytest.mark.parametrize('signalled', [True, False])
@pytest.mark.parametrize('initial_denials', [0, 2])
def test_created_recovery_waits_out_access_denied(monkeypatch, capsys, signalled, denied, initial_denials):
    clock, terms, waits, updates = [100.0], [], [], []
    monkeypatch.setattr(down, 'time', NS(monotonic=lambda: clock[0],
        sleep=lambda timeout: clock.__setitem__(0, clock[0] + timeout)))
    class Child(winproc.Pinned):
        def wait(self, timeout=0):
            waits.append(timeout)
            clock[0] += min(timeout, 2) if signalled else timeout
            return signalled and clock[0] >= 103

        def terminate(self):
            terms.append(self.identity)
            clock[0] += 1
            raise winproc.ProcessAccessError(self.pid, winproc.END)

        def close(self):
            pass

    child = Child(99, 12, 'win:10', 'python.exe')
    record = dict(role='python', phase='created', launcher=dict(pid=11, started='win:1'),
                  child=dict(pid=child.pid, started=child.started))
    backend = down.Windows(None, None)
    monkeypatch.setattr(backend, 'recover_inspectors', lambda: None)
    monkeypatch.setattr(backend, 'orphan_admissions', lambda: None)
    monkeypatch.setattr(winproc, '_inspect_checkpoint', lambda point: None)
    monkeypatch.setattr(winproc, '_confirmed_exit', lambda identity: True)
    monkeypatch.setattr(winproc, 'launch_records', lambda: [(None, record)])
    monkeypatch.setattr(winproc, 'live_launches', lambda **kwargs: ([], set()))
    opens = []
    def opening(pid, rights=winproc.IDENTIFY):
        opens.append(rights)
        if denied == 'open' and rights == winproc.END:
            clock[0] += 1
            raise winproc.ProcessAccessError(pid, rights)
        if denied == 'open' and len(opens) <= initial_denials + 1:
            raise winproc.ProcessAccessError(pid, rights)
        return child
    monkeypatch.setattr(winproc, 'open', opening)
    monkeypatch.setattr(winproc, 'read_launch', lambda entry: record)
    monkeypatch.setattr(winproc, '_update_launch', lambda entry, phase: updates.append(phase))
    if signalled:
        backend.recover()
        assert updates == ['exited']
        assert 'recovered never-run child' in capsys.readouterr().out
    else:
        with pytest.raises(down.Refusal, match='cannot end pid 12: access denied'):
            backend.recover()
        assert not updates
    assert terms == ([child.identity] if denied == 'terminate' else [])
    delay = initial_denials * down.POLL_S if denied == 'open' else 0
    assert opens == ([winproc.END] if denied == 'terminate' else
                     [winproc.END] + [winproc.IDENTIFY] * (initial_denials + 1))
    assert waits == pytest.approx([0, 4 - delay])
    assert clock[0] == pytest.approx(103 + delay if signalled else 105)


@pytest.mark.parametrize('reason', ['gone', 'reused', 'signalled', 'denied'])
@pytest.mark.parametrize('initial_denials', [0, 2])
def test_created_recovery_denied_open_checks_incarnation(monkeypatch, reason, initial_denials):
    opens, waits, closed = [], [], []
    clock, sleeps = [100.0], []
    def sleep(timeout):
        sleeps.append(timeout)
        clock[0] += timeout
    monkeypatch.setattr(down, 'time', NS(monotonic=lambda: clock[0], sleep=sleep))
    child = winproc.Pinned(99, 12, 'win:11' if reason == 'reused' else 'win:10', 'python.exe')
    def wait(timeout=0):
        waits.append(timeout)
        return reason == 'signalled'
    child.wait = wait
    child.terminate = lambda: pytest.fail('identify handle used for termination')
    monkeypatch.setattr(winproc.winclaim, 'close_handle', closed.append)
    record = dict(role='python', phase='created', launcher=dict(pid=11, started='win:1'),
                  child=dict(pid=12, started='win:10'))
    backend = down.Windows(None, None)
    monkeypatch.setattr(backend, 'recover_inspectors', lambda: None)
    monkeypatch.setattr(backend, 'orphan_admissions', lambda: None)
    monkeypatch.setattr(winproc, '_inspect_checkpoint', lambda point: None)
    monkeypatch.setattr(winproc, '_confirmed_exit', lambda identity: True)
    monkeypatch.setattr(winproc, 'launch_records', lambda: [(None, record)])
    monkeypatch.setattr(winproc, 'live_launches', lambda **kwargs: ([], set()))
    monkeypatch.setattr(winproc, 'read_launch', lambda entry: pytest.fail('completed recovery read again'))
    def opening(pid, rights=winproc.IDENTIFY):
        assert pid == 12
        opens.append(rights)
        if rights == winproc.END or reason == 'denied' or len(opens) <= initial_denials + 1:
            raise winproc.ProcessAccessError(pid, rights)
        return None if reason == 'gone' else child
    monkeypatch.setattr(winproc, 'open', opening)
    if reason == 'denied':
        with pytest.raises(down.Refusal, match='cannot identify recorded child pid 12: cannot identify pid 12: access denied'):
            backend.recover()
    else:
        backend.recover()
    assert opens[0] == winproc.END
    assert opens[1:] == [winproc.IDENTIFY] * (len(sleeps) + 1)
    if reason == 'denied':
        assert clock[0] == 100 + winproc.KILL_WAIT_S
        assert all(0 < timeout <= down.POLL_S for timeout in sleeps)
    else:
        assert len(sleeps) == initial_denials
    assert waits == ([0] if reason == 'signalled' else [])
    assert closed == ([99] if reason in ('reused', 'signalled') else [])


@pytest.mark.parametrize('outcome', ['gone', 'reused', 'signalled', 'live', 'denied'])
@pytest.mark.parametrize('cleanup', [False, True])
def test_record_bookkeeping_identification_retry(monkeypatch, outcome, cleanup):
    clock, attempts, pruned, closed = [100.0], [], [], []
    monkeypatch.setattr(down, 'time', NS(monotonic=lambda: clock[0],
        sleep=lambda timeout: clock.__setitem__(0, clock[0] + timeout)))
    record = dict(child=dict(pid=12, started='win:10'))
    monkeypatch.setattr(winproc, 'launch_records', lambda: [('record.json', record)])
    monkeypatch.setattr(winproc, '_prune_record', pruned.append)
    child = winproc.Pinned(99, 12, 'win:11' if outcome == 'reused' else 'win:10', 'python.exe')
    child.wait = lambda: outcome == 'signalled'
    monkeypatch.setattr(winproc.winclaim, 'close_handle', closed.append)
    def opening(pid, rights=winproc.IDENTIFY):
        assert pid == 12 and rights == winproc.IDENTIFY
        attempts.append(clock[0])
        if len(attempts) <= 2 or outcome == 'denied':
            raise winproc.ProcessAccessError(pid, rights)
        return None if outcome == 'gone' else child
    monkeypatch.setattr(winproc, 'open', opening)
    records, unresolved = down.Windows(None, None).live_launches() if cleanup else winproc.live_launches()
    if not cleanup or outcome == 'denied':
        assert records == [record] and unresolved == {12}
        assert not pruned and not closed
        assert clock[0] == (105 if cleanup else 100)
        if cleanup:
            assert len(attempts) > 2
        else:
            assert len(attempts) == 1
    else:
        assert len(attempts) == 3 and not unresolved
        assert clock[0] == pytest.approx(100 + 2 * down.POLL_S)
        assert records == ([record] if outcome == 'live' else [])
        assert pruned == ([] if outcome == 'live' else ['record.json'])
        assert closed == ([] if outcome == 'gone' else [99])


@pytest.mark.parametrize('deadline', [None, 99, 102, 145])
@pytest.mark.parametrize('inventory', [False, True])
def test_record_bookkeeping_denials_share_deadline(monkeypatch, deadline, inventory):
    clock, attempts, passes = [100.0], [], []
    monkeypatch.setattr(down, 'time', NS(monotonic=lambda: clock[0],
        sleep=lambda timeout: clock.__setitem__(0, clock[0] + timeout)))
    records = [dict(role='python', child=dict(pid=pid, started='win:10', image='python.exe'))
               for pid in range(12, 24)]
    monkeypatch.setattr(winproc, 'launch_records', lambda: list(enumerate(records)))
    monkeypatch.setattr(winproc, '_prune_record', lambda path: pytest.fail('unresolved record pruned'))
    def opening(pid):
        attempts.append((pid, clock[0]))
        raise winproc.ProcessAccessError(pid, winproc.IDENTIFY)
    monkeypatch.setattr(winproc, 'open', opening)
    backend = down.Windows(NS(backend='none'), None)
    if inventory:
        bookkeeping = backend.live_launches
        def live_launches(limit):
            result = bookkeeping(limit)
            passes.append(result)
            return result
        monkeypatch.setattr(backend, 'live_launches', live_launches)
        monkeypatch.setattr(claim, 'verify_cleanup', lambda fd: None)
        monkeypatch.setattr(backend, 'refresh', lambda: None)
        monkeypatch.setattr(backend, 'station', lambda limit: (set(), False))
        monkeypatch.setattr(backend, 'publication', lambda **kwargs: None)
        monkeypatch.setattr(winproc, 'using', lambda path: [])
        own = winproc.Pinned(None, os.getpid(), 'win:20', 'down.exe')
        backend.entries = {own.pid: own, 12: NS(pid=12)}
        backend.gate_fd = 99
        with pytest.raises(down.Refusal, match='cannot identify recorded child pid 12'):
            backend.inventory(145 if deadline is None else deadline)
    else:
        passes.append(backend.live_launches(deadline))
    assert passes == [(records, set(range(12, 24)))]
    expected = max(100, min(105, deadline if deadline is not None else 145))
    assert clock[0] == pytest.approx(expected)
    assert {pid for pid, _ in attempts} == set(range(12, 24))
    assert all(at == pytest.approx(expected) for pid, at in attempts if pid != 12)


@pytest.mark.parametrize('signalled', [True, False])
def test_down_termination_waits_out_access_denied(monkeypatch, capsys, signalled):
    clock, terms, requests = [100.0], [], []
    monkeypatch.setattr(down, 'time', NS(monotonic=lambda: clock[0]))
    monkeypatch.setattr(winproc, 'time', down.time)
    monkeypatch.setattr(down, 'ThreadPoolExecutor', Immediate)
    entry = NS(pid=12, identity=(12, 'win:10'), wait=lambda: signalled and clock[0] >= 114)
    def terminate():
        terms.append(clock[0])
        raise winproc.ProcessAccessError(entry.pid, winproc.END)
    def request(target):
        requests.append(target.identity)
        return 2
    entry.terminate = terminate
    monkeypatch.setattr(winproc, 'request_stop', request)
    backend = NS(issue=lambda: None,
                 inventory=lambda deadline: down.Inventory([] if entry.wait() else [entry]),
                 stop=lambda entry: winproc.Stopper(entry),
                 wait=lambda stoppers, timeout: clock.__setitem__(0, clock[0] + timeout))
    flow = down.Flow(None, backend)
    if signalled:
        flow.settle()
        assert 114 <= clock[0] < 115
    else:
        with pytest.raises(down.Refusal, match='termination unresolved for pid 12: .*cannot end pid 12: access denied'):
            flow.settle()
        assert 117 <= clock[0] < 118
    assert requests == [entry.identity] and len(terms) == 1
    assert capsys.readouterr().out.count('pid 12 -- terminated') == 1


@pytest.mark.parametrize('denied', [False, True])
def test_process_teardown_pins_and_reaps_all_test_children(tmp_path, monkeypatch, denied):
    from .test_down_windows import Processes
    ended, closed, waited, signals = [], [], [], set()
    class Entry(winproc.Pinned):
        def wait(self, timeout=0):
            return self.identity in signals

        def terminate(self):
            ended.append(self.identity)
            if denied and self.pid == 10:
                raise winproc.ProcessAccessError(self.pid, winproc.END)
            signals.add(self.identity)

        def close(self):
            closed.append(self.handle)

    root = Entry(110, 10, 'win:10', 'launcher.exe')
    monkeypatch.setattr(claim, 'state_dir', lambda: tmp_path)
    monkeypatch.setattr(winproc, '_parents', lambda: {20: 10, 30: 999})
    def census():
        return [Entry(pid + 1000, pid, f'win:{pid}', 'python.exe') for pid in (20, 30, 40)], set()
    monkeypatch.setattr(winproc, 'pinned', census)
    monkeypatch.setattr(winproc, 'launch_records', lambda: [(None, dict(child=dict(pid=pid, started=stamp)))
        for pid, stamp in ((30, 'win:30'), (40, 'win:1'))])
    monkeypatch.setattr(winproc, 'open', lambda pid, rights: Entry(pid + 2000, pid, f'win:{pid}', 'python.exe'))
    monkeypatch.setattr(winproc, '_pin', lambda handle, pid, rights: root)
    monkeypatch.setattr(winproc.winclaim, 'duplicate', lambda handle: handle + 1)
    monkeypatch.setattr('subprocess.Popen', lambda *args, **kwargs:
        NS(pid=10, _handle=109, wait=lambda timeout: waited.append(timeout)))
    owned = Processes(tmp_path)
    process = owned.popen(['launcher'])
    assert process.entry is root
    if denied:
        with pytest.raises(ExceptionGroup, match='test process teardown failed'):
            owned.close()
    else:
        owned.close()
    assert set(ended) == {(10, 'win:10'), (20, 'win:20'), (30, 'win:30')}
    assert len(ended) == 3 and waited == [5]
    assert {110, 2020, 2030, 2040, 1020, 1030, 1040} <= set(closed)
    assert not owned.entries and not owned.tracked


def test_inventory_consumes_identity_checked_port_evidence(reader_inventory, tmp_path, monkeypatch):
    import json
    backend, reader = reader_inventory
    directory = tmp_path / 'rig.claim.ports.d'
    directory.mkdir()
    (directory / '12-10.json').write_text(json.dumps(dict(
        process=dict(pid=reader.pid, started=reader.started, image=reader.image), ports=['COM41'])))
    monkeypatch.setattr(claim, 'state_dir', lambda: tmp_path)
    monkeypatch.setattr(winproc, '_read_json', lambda path: json.loads(path.read_text()))
    pinned = winproc.Pinned(None, reader.pid, reader.started, reader.image)
    pinned.wait = reader.wait
    monkeypatch.setattr(winproc, 'open', lambda pid: pinned)
    monkeypatch.setattr(winproc, 'device_holders', lambda *a, **k:
        winproc.DeviceHolders({reader.pid: winproc.DeviceWalk(complete=True)}, 'complete'))
    backend.inventory(down.time.monotonic() + 45)
    assert backend.evidence == {'COM41': 'COM41 is held by your hfmodem process (pid 12)'}


@pytest.mark.parametrize('source', ['none', 'claim', 'launch', 'both'])
def test_port_role_uses_launch_then_claim_then_process(tmp_path, monkeypatch, source):
    import json
    entry = winproc.Pinned(None, 12, 'win:100', 'python.exe')
    entry.wait = lambda timeout=0: False
    directory = tmp_path / 'rig.claim.ports.d'
    directory.mkdir()
    (directory / '12-100.json').write_text(json.dumps(dict(
        process=winproc._identity(entry), ports=['COM41'])))
    monkeypatch.setattr(claim, 'state_dir', lambda: tmp_path)
    monkeypatch.setattr(winproc, '_read_json', lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(winproc, 'open', lambda pid: entry)
    publisher = NS(pid=12, started='win:100', kind='station') if source in ('claim', 'both') else None
    records = [dict(child=winproc._identity(entry), role='python', phase='acknowledged')] if source in ('launch', 'both') else []
    evidence = [(identity, role, ports) for identity, role, ports in
                winproc.recorded_devices(records, publisher) if ports]
    role = {'none': 'process', 'claim': 'station', 'launch': 'modem', 'both': 'modem'}[source]
    assert evidence == [(entry.identity, role, ['COM41'])]


@pytest.mark.skipif(os.name == 'nt', reason='POSIX signals')
@pytest.mark.parametrize('name', ['SIGINT', 'SIGTERM', 'SIGHUP'])
def test_posix_cleanup_restores_handlers_and_preserves_signal_status(name):
    import signal
    from hfmodem.core.down_posix import Posix
    backend = Posix(None, (), None)
    sig = getattr(signal, name)
    previous = signal.getsignal(sig)
    backend.recover()
    try:
        with pytest.raises(down.Refusal) as stopped:
            signal.getsignal(sig)(sig, None)
        assert stopped.value.code == 128 + sig
    finally:
        backend.close()
    assert signal.getsignal(sig) is previous



@pytest.mark.posix_only('the fake Yaesu is a pty, and Windows has none')
@pytest.mark.parametrize('answers,error,queries', [
    (['?;', '?;', 'FA007103800;'], None, 3),
    (['?;'] * 9, '^radio refused CAT query$', 3),
    (['?;FA007103800;', 'FA007103800;'], None, 2),
    (['?;FA007103800;'] * 9, '^unsolicited or ambiguous CAT reply', 3),
    ([''], '^CAT reply timed out$', 1),
])
def test_cat_proof_retries_a_busy_radio_within_its_bound(answers, error, queries,
                                                          monkeypatch, tmp_path):
    import serial
    import time
    from hfmodem.core.cat import RigError
    from hfmodem.core.rigs import RIGS
    from hfmodem.tests.core.fakeyaesu import FakeYaesu, PtySerial
    answers = iter(answers)
    fake = FakeYaesu(replies={'FA;': lambda command: next(answers)})
    monkeypatch.setattr(serial, 'Serial', PtySerial)
    settings = down.Settings(tmp_path / 'station.toml', 0, 'serial', '', 0,
                             fake.port, None, RIGS['ft891'])
    started = time.monotonic()
    try:
        if error is None:
            assert settings.cat_proof() == 'ok'
        else:
            with pytest.raises(RigError, match=error):
                settings.cat_proof()
    finally:
        fake.close()
    assert time.monotonic() - started < 2.5
    assert fake.log.count('FA;') == queries
