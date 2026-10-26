# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The qualification stand-in exception never grants general device access.

All kernel and device metadata is scripted; these tests open no device nodes.
The real pair's line behavior remains the coexist validation test's obligation.
"""
import importlib.util
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest


def _info(mode, uid=0, **kwargs):
    return SimpleNamespace(st_mode=mode, st_uid=uid, st_nlink=1, st_dev=5,
                           st_ino=405, st_rdev=0, st_ctime_ns=1, **kwargs)


@pytest.fixture
def system(monkeypatch):
    source = Path(__file__).parents[1] / 'hermetic_py/hermetic_guard.py'
    spec = importlib.util.spec_from_file_location('tty0tty_guard_test', source)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    env = {'HFMODEM_QUALIFICATION_PHASE': 'coexist', 'HFMODEM_COEXIST_LINE_PROOF': '1'}
    uname = SimpleNamespace(nodename='lima-hfmodem-ci', release='test-kernel')
    metadata = {path: _info(stat.S_IFDIR | 0o755) for path in
                ('/', '/dev', '/var', '/var/lib', '/var/lib/hfmodem-ci')}
    metadata[guard.TTY0TTY_RECEIPT] = _info(stat.S_IFREG | 0o644)
    for minor, path in enumerate(guard.TTY0TTY_PAIR):
        info = _info(stat.S_IFCHR | 0o600, uid=501)
        info.st_ino += minor
        info.st_rdev = os.makedev(235, minor)
        metadata[path] = info
    receipt = dict(schema_version=1, vm='hfmodem-ci', kernel=uname.release,
                   tty0tty=dict(upstream='https://github.com/freemed/tty0tty.git',
                                commit=guard.TTY0TTY_COMMIT, pair=list(guard.TTY0TTY_PAIR),
                                owner_uid=501, srcversion='84EBB0DA875588B10C1ABA2'))
    texts = {'/sys/module/tty0tty/srcversion': receipt['tty0tty']['srcversion'] + '\n',
             '/proc/tty/drivers': 'tty0tty /dev/tnt 235 0-7 serial\n',
             '/sys/class/tty/tnt0/dev': '235:0\n', '/sys/class/tty/tnt1/dev': '235:1\n'}
    resolved = {f'/sys/dev/char/235:{minor}': f'/sys/devices/virtual/tty/tnt{minor}'
                for minor in range(2)}

    def read(path):
        value = json.dumps(receipt) if path == guard.TTY0TTY_RECEIPT else texts[path]
        if isinstance(value, Exception):
            raise value
        return value

    def lstat(path):
        try:
            return metadata[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    fake_os = SimpleNamespace(**vars(os))
    fake_os.environ = env
    fake_os.uname = lambda: uname
    fake_os.getuid = lambda: 501
    fake_os.lstat = lstat
    fake_os.path = SimpleNamespace(realpath=lambda path: resolved.get(path, path))
    hooks = []
    monkeypatch.setattr(guard, 'os', fake_os)
    monkeypatch.setattr(guard, 'sys', SimpleNamespace(platform='linux', addaudithook=hooks.append))
    monkeypatch.setattr(guard, '_text', read)

    def trip(message):
        raise PermissionError(message)

    monkeypatch.setattr(guard, 'trip', trip)
    return SimpleNamespace(guard=guard, env=env, uname=uname, metadata=metadata,
                           receipt=receipt, texts=texts, resolved=resolved, hooks=hooks)


def _open(guard, path, event='open'):
    guard.guard(event, (path, None, 0))


@pytest.mark.parametrize('path_type', [str, os.fsencode, Path])
def test_only_the_validated_pair_is_permitted(system, path_type):
    guard = system.guard
    guard.arm()
    assert len(system.hooks) == 1
    for path in guard.TTY0TTY_PAIR:
        _open(guard, path_type(path))
    guard.disarm()
    assert guard._tty0tty == {}


@pytest.mark.parametrize('opt_in', [None, '', '0', 'false', 'true', 'yes', '01', '1 ', ' 1', '2'])
def test_unrequested_pair_never_probes_platform_or_provisioning(system, monkeypatch, opt_in):
    if opt_in is None:
        del system.env['HFMODEM_COEXIST_LINE_PROOF']
    else:
        system.env['HFMODEM_COEXIST_LINE_PROOF'] = opt_in

    def forbidden(*args):
        pytest.fail('portable validation probed line-proof metadata')

    class PortablePlatform:
        addaudithook = staticmethod(system.hooks.append)

        @property
        def platform(self):
            return forbidden()

    monkeypatch.setattr(system.guard, 'sys', PortablePlatform())
    monkeypatch.setattr(system.guard.os, 'uname', forbidden)
    for name in ('_trusted', '_text', '_tty0tty_node'):
        monkeypatch.setattr(system.guard, name, forbidden)
    system.guard.arm()
    assert system.guard._tty0tty == {}
    with pytest.raises(PermissionError, match='/dev/tnt0'):
        _open(system.guard, '/dev/tnt0')


@pytest.mark.parametrize('opt_in', [None, '0', 'malformed'])
def test_withdrawing_request_revokes_cached_pair_before_metadata_probe(system, monkeypatch, opt_in):
    system.guard.arm()
    assert system.guard._tty0tty
    if opt_in is None:
        del system.env['HFMODEM_COEXIST_LINE_PROOF']
    else:
        system.env['HFMODEM_COEXIST_LINE_PROOF'] = opt_in

    def forbidden(*args):
        pytest.fail('withdrawn line request probed device metadata')

    for name in ('_trusted', '_text', '_tty0tty_node'):
        monkeypatch.setattr(system.guard, name, forbidden)
    with pytest.raises(PermissionError, match='/dev/tnt0'):
        _open(system.guard, '/dev/tnt0')


@pytest.mark.parametrize('phase', [None, '', 'inventory', 'parallel', 'serial', 'wall_clock'])
def test_pair_stays_tripwired_outside_explicit_coexist_execution(system, phase):
    system.env['HFMODEM_QUALIFICATION_PHASE'] = phase
    system.guard.arm()
    assert system.guard._tty0tty == {}
    with pytest.raises(PermissionError, match='/dev/tnt0'):
        _open(system.guard, '/dev/tnt0')


@pytest.mark.parametrize('defect', [
    'platform', 'host', 'schema', 'vm', 'kernel', 'upstream', 'commit', 'pair',
    'owner', 'srcversion', 'loaded_module', 'missing_module', 'driver', 'major',
    'duplicate_driver', 'sysfs_device', 'sysfs_identity', 'receipt_owner',
    'receipt_writable', 'receipt_symlink', 'parent_writable', 'parent_owner',
    'dev_writable', 'dev_owner', 'node_owner', 'node_mode', 'node_regular',
    'node_symlink', 'node_major', 'node_minor', 'node_links', 'missing_node',
])
def test_untrusted_or_foreign_standins_receive_no_exception(system, defect):
    guard = system.guard
    module = system.receipt['tty0tty']
    node = system.metadata['/dev/tnt0']
    receipt = system.metadata[guard.TTY0TTY_RECEIPT]
    if defect == 'platform':
        guard.sys.platform = 'darwin'
    elif defect == 'host':
        system.uname.nodename = 'another-guest'
    elif defect in ('schema', 'vm', 'kernel'):
        system.receipt[{'schema': 'schema_version'}.get(defect, defect)] = 'foreign'
    elif defect in ('upstream', 'commit', 'pair', 'owner', 'srcversion'):
        module[{'owner': 'owner_uid'}.get(defect, defect)] = 'foreign'
    elif defect == 'loaded_module':
        system.texts['/sys/module/tty0tty/srcversion'] = '0' * 24
    elif defect == 'missing_module':
        system.texts['/sys/module/tty0tty/srcversion'] = FileNotFoundError()
    elif defect in ('driver', 'major', 'duplicate_driver'):
        system.texts['/proc/tty/drivers'] = {
            'driver': 'foreign /dev/tnt 235 0-7 serial\n',
            'major': 'tty0tty /dev/tnt 236 0-7 serial\n',
            'duplicate_driver': system.texts['/proc/tty/drivers'] * 2}[defect]
    elif defect == 'sysfs_device':
        system.texts['/sys/class/tty/tnt0/dev'] = '236:0'
    elif defect == 'sysfs_identity':
        system.resolved['/sys/dev/char/235:0'] = '/sys/devices/pci/serial/tty/tnt0'
    elif defect.startswith('receipt_'):
        if defect == 'receipt_owner':
            receipt.st_uid = 501
        else:
            receipt.st_mode = (stat.S_IFLNK | 0o777 if defect == 'receipt_symlink'
                               else stat.S_IFREG | 0o666)
    elif defect in ('parent_writable', 'parent_owner', 'dev_writable', 'dev_owner'):
        parent = system.metadata['/dev' if defect.startswith('dev_') else '/var/lib/hfmodem-ci']
        if defect.endswith('writable'):
            parent.st_mode |= 0o022
        else:
            parent.st_uid = 501
    elif defect == 'node_owner':
        node.st_uid = 502
    elif defect == 'node_mode':
        node.st_mode |= 0o020
    elif defect in ('node_regular', 'node_symlink'):
        node.st_mode = (stat.S_IFREG if defect == 'node_regular' else stat.S_IFLNK) | 0o600
    elif defect in ('node_major', 'node_minor'):
        node.st_rdev = os.makedev(236, 0) if defect == 'node_major' else os.makedev(235, 2)
    elif defect == 'node_links':
        node.st_nlink = 2
    elif defect == 'missing_node':
        del system.metadata['/dev/tnt1']
    guard.arm()
    assert guard._tty0tty == {}
    with pytest.raises(PermissionError):
        _open(guard, '/dev/tnt0')


@pytest.mark.parametrize('path', ['/dev/tnt2', '/dev/tnt00', '/dev/tnt0extra',
                                  '/dev/ttyUSB0', '/dev/ttyS0', '/dev/cu.radio',
                                  '/dev/snd/pcmC0D0c'])
def test_valid_pair_does_not_allow_other_devices_or_same_prefixes(system, path):
    system.guard.arm()
    with pytest.raises(PermissionError):
        _open(system.guard, path)


@pytest.mark.parametrize('alias', ['/tmp/line', '/dev/line', 'tnt0', '/dev/./tnt0'])
def test_aliases_to_valid_nodes_stay_tripwired(system, alias):
    system.resolved[alias] = '/dev/tnt0'
    system.guard.arm()
    with pytest.raises(PermissionError):
        _open(system.guard, alias)


@pytest.mark.parametrize('change', ['inode', 'ctime', 'device', 'symlink', 'module',
                                   'dev_parent', 'phase', 'missing'])
def test_approved_paths_are_rechecked_before_each_open(system, change):
    guard = system.guard
    guard.arm()
    _open(guard, '/dev/tnt0')
    node = system.metadata['/dev/tnt0']
    if change == 'inode':
        node.st_ino += 1
    elif change == 'ctime':
        node.st_ctime_ns += 1
    elif change == 'device':
        node.st_rdev = os.makedev(4, 64)
    elif change == 'symlink':
        node.st_mode = stat.S_IFLNK | 0o777
    elif change == 'module':
        system.texts['/sys/module/tty0tty/srcversion'] = '0' * 24
    elif change == 'dev_parent':
        system.metadata['/dev'].st_mode |= 0o022
    elif change == 'phase':
        system.env['HFMODEM_QUALIFICATION_PHASE'] = 'parallel'
    elif change == 'missing':
        del system.metadata['/dev/tnt0']
    with pytest.raises(PermissionError):
        _open(guard, '/dev/tnt0')


def test_pair_exception_never_allows_device_directory_creation(system):
    system.guard.arm()
    with pytest.raises(PermissionError):
        _open(system.guard, '/dev/tnt0', event='os.mkdir')


def test_rearming_cannot_retain_a_prior_phase_exception(system):
    system.guard.arm()
    system.env['HFMODEM_QUALIFICATION_PHASE'] = 'serial'
    system.guard.arm()
    assert len(system.hooks) == 1
    assert system.guard._tty0tty == {}
    with pytest.raises(PermissionError):
        _open(system.guard, '/dev/tnt0')
