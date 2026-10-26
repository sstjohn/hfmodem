# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fixtures for the console's offline checks. Nothing here opens a device."""
import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import signal
from types import SimpleNamespace

import pytest

from . import (ardop, arm, claim, fingerprints, logbook, modes, multimode, operator, pactor, residency,
               roster, rtty, station, vara)
from hfmodem.core import capture_identity
from hfmodem.core.native import NativeChild
from hfmodem.tests.evidence import private_source_tree, require_input

FLOWN = Path(__file__).resolve().parents[2] / 'working'
FLOWN_ROSTERS = {'pactor': 'pactor-level-air-0916', 'vara': 'vara-level-air-0915',
                 'ardop': 'ardop-air-0922'}
NOW = datetime(2026, 9, 16, 5, 0, tzinfo=timezone.utc)


def flown_records():
    require_input(pactor.PROFILE.arms / 'pactor-current-ve3kpg-80-sense-20260916T060745Z/provenance.json')
    return logbook.records(pactor.PROFILE)


class Arm:
    """A launcher that prints one transcript and exits clean."""

    pid = os.getpid()

    def __init__(self, transcript=''):
        self.stdout = iter([transcript])

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


def visible(directory):
    """What a refused arm left besides its capture bookkeeping."""
    return sorted(p.name for p in Path(directory).iterdir() if not p.name.startswith('.'))


def args(**changes):
    flown = pactor.PROFILE.levels['current']
    values = dict(peer='KB5LZK', band=40, level='p3-100', sense=True, pounce=0, force=False,
                  follow_offset='all', entry_sl=3, traffic_sl=1, repeat_gear=3,
                  keep_slots='all',
                  control_placement='pulse-center', control_tail='off',
                  control_stagger='off', crest_limit='off', speed_policy='current', changeover_p1_cs=False, no_wait_greeting=False,
                  long_cycle=False, p3_fallback=False,
                  pactor1_only=False, setup_phase='reply', p1_mail=False, unannounced=False,
                  tx_drive=flown['tx_drive'], p1_drive=flown['p1_drive'],
                  send=None, witness=True,
                  witness_seconds=arm.WITNESS_SECONDS, kiwi=None, retune=False,
                  observe_after_link_down=True, tap_tail=arm.TAP_TAIL_S,
                  qualify=None, modem_arg=[])
    return argparse.Namespace(**dict(values, **changes))


def transcript(armed=True, decoded=0, tail='session ended: test only\n'):
    """What an arm's log looks like to the outcome classifier."""
    return (('!! TRANSMIT ARMED -- a licensed operator must be on frequency\n' if armed else
             '!! channel in use, and the 0s wait budget is spent\n')
            + '** CONNECTED to KB5LZK **\n' * bool(decoded)
            + (f'control signals decoded {decoded} '
               f'({decoded} PACTOR-1 at zero errors, 0 PACTOR-3)\n' if armed else '')
            + tail)


def environment(*, selection=None):
    """What a healthy station reads back, for tests that do not reach one."""
    return dict(codec_gains={station.AUDIO_DEVICE: roster.gain_hint(pactor.PROFILE, 40, NOW),
                             station.TAP_DEVICE: 0.593},
                rig=dict(levels=dict(AGC='2', PREAMP='0', RF='1.000000', IF='0',
                                     NB='0.000000', STRENGTH='12'),
                         funcs=dict(NB='0', NR='0', ANF='1'),
                         dial_hz='7100100', mode='PKTUSB 3000', ptt='0'))


def readings(plan):
    """What check_rig returns on a healthy rig, for tests that do not reach one."""
    env = environment()
    return dict(dial_hz=plan['dial_hz'], dial_off_by_hz=0, rfpower='1.000000',
                codec_gain=env['codec_gains'][station.AUDIO_DEVICE], **env)


def record(mode, call, minutes_ago, reached):
    started = NOW - timedelta(minutes=minutes_ago)
    name = f'pactor-current-{call.lower()}-40-sense-{started:%Y%m%dT%H%M%SZ}'
    directory = mode.arms / name
    directory.mkdir()
    (directory / 'provenance.json').write_text(json.dumps(dict(
        arm=name, directory=str(directory), call=call, band_m=40, centre_hz=7101600,
        dial_hz=7100100, mode='sense', level=dict(name='current'), returncode=0,
        started_utc=started.isoformat(), ended_utc=started.isoformat(),
        verdict=dict(reached=reached, line='test only'))))
    return name


def probed(entries, env, python=None):
    """What the residency probe reads back where a test does not mean to run one.

    The working tree -- which is where this station's arms have in fact resolved
    every module they have ever loaded.
    """
    return {'hfmodem.shrike.onair':
            dict(file=str(arm.PACKAGE / 'hfmodem/shrike/onair.py'), sha256='9' * 64)}


def seed_operator_data(root, *, private):
    """Seed historical private expectations only where the source promises them.

    Public offline checks start with empty operator storage. No station record is
    needed to exercise ownership, configuration or import/export behavior.
    """
    for name, flown in FLOWN_ROSTERS.items():
        directory = root / 'working' / name
        directory.mkdir(parents=True)
        if private:
            shutil.copy(FLOWN / flown / 'peers.json', directory)
    messages = root / 'working' / 'vara' / 'messages'
    if private:
        shutil.copytree(FLOWN / 'vara-level-air-0915' / 'messages', messages)
    else:
        messages.mkdir()
    (root / 'operator.toml').write_text('delivery_to = "recipient@example.test"\n')
    return root


@pytest.fixture(scope='session')
def operator_data(tmp_path_factory):
    """Isolated operator storage; private source trees also seed flown records."""
    return seed_operator_data(tmp_path_factory.mktemp('operator-data'),
                              private=private_source_tree(FLOWN.parent))


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch, tmp_path, operator_data):
    def forbidden(*called, **kwargs):
        pytest.fail(f'A test reached a subprocess: {called}')
    monkeypatch.setattr(station.subprocess, 'run', forbidden)
    monkeypatch.setattr(station.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(arm, 'git_head', lambda: dict(head='0' * 40, dirty=False))
    monkeypatch.setattr(residency, 'probe', probed)
    monkeypatch.delenv('MYCALL', raising=False)
    monkeypatch.delenv('IDCALL', raising=False)
    monkeypatch.setattr(pactor, 'MAIL_PASSWORD_FILE', tmp_path / 'absent-password')
    monkeypatch.setenv('WINLINK_PASSWORD', 'offline-fake-launcher-only')
    monkeypatch.delenv('WINLINK_PASSWORD_FILE', raising=False)
    monkeypatch.setenv('HFMODEM_STATION', str(Path(__file__).with_name('fixtures') / 'station.toml'))
    monkeypatch.setenv('HFMODEM_DATA_DIR', str(operator_data))
    monkeypatch.delenv('HFMODEM_RIG', raising=False)
    # These modules resolved their station and data roots at import, from
    # whatever environment the run started in.
    selection = station.resolve_selection(allow_missing=True)
    for name in station._DEFAULT_FIELDS:
        monkeypatch.setattr(station, name, getattr(selection, name, ''))
    # The plans spell the fixture rig's own ports; a modem parsed in process
    # resolves its ports from here, and a test run may resolve no real device.
    monkeypatch.setenv('CAT_PORT', '/dev/null')
    monkeypatch.setenv('PTT_PORT', '/dev/null')
    for module in (pactor, vara, ardop):
        monkeypatch.setattr(module, 'HERE', operator.mode_root(module.PROFILE.name))
        monkeypatch.setattr(module, 'PROFILE', modes.resolve(module.PROFILE.name))
    monkeypatch.setattr(multimode, 'RECORDS', operator.data_root() / 'campaigns')
    monkeypatch.setattr(fingerprints, 'FOUND', operator.data_root() / 'fingerprints.json')
    monkeypatch.setattr(rtty, 'ADIF', operator.data_root() / 'working' / 'qso-log.adi')
    monkeypatch.setattr(rtty, 'HOLDS', SimpleNamespace(
        name='rtty', lock=operator.mode_root('rtty-qso') / 'arm.lock',
        arms=operator.mode_root('rtty-qso') / 'sessions'))


@pytest.fixture
def flyable(monkeypatch):
    """A rig that reads back and a launcher that says nothing, so `fly` runs through."""
    monkeypatch.setattr(station, 'check_rig', readings)
    monkeypatch.setattr(station, 'read_environment', environment)
    monkeypatch.setattr(station, 'start_tap', lambda plan, directory, *, writer=None: None)
    monkeypatch.setattr(station.subprocess, 'Popen', lambda *a, **k: Arm())


def fixture_profile(profile, directory):
    """Give offline tests their own copy of the historical expectation roster."""
    source = FLOWN / FLOWN_ROSTERS[profile.name] / 'peers.json'
    if not private_source_tree(FLOWN.parent):
        require_input(source)
    target = directory / 'peers.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text())
    return replace(profile, roster=target)


@pytest.fixture
def vara_mode(tmp_path):
    """The VARA profile with its arms and its lock somewhere disposable."""
    profile = fixture_profile(vara.PROFILE, tmp_path)
    profile = replace(profile, arms=tmp_path / 'arms', lock=tmp_path / 'arm.lock')
    profile.arms.mkdir()
    return profile


@pytest.fixture
def mode(monkeypatch, tmp_path):
    """The flown PACTOR profile with its arms and its lock somewhere disposable.

    EVERY mode is redirected, not only the one under test. `console log`, the
    spacing note and the multimode choice all read across modes, and a test that
    reached a real arms directory would pass or fail on what the radio did in the
    minutes before it ran -- which it did, the moment VARA gave the logbook a
    second mode to walk.
    """
    monkeypatch.setenv('MYCALL', 'W9SSJ')  # identity of these offline expectation fixtures
    monkeypatch.setenv('HFMODEM_DELIVERY_TO', 'recipient@example.test')
    resolve = modes.resolve
    spare = {}

    def redirected(name):
        """Resolve for real -- an unadapted mode must still refuse by name -- then redirect.

        Lazily, because a test may set a profile to None mid-run to exercise that
        refusal, and a table built at setup would answer from the cache instead.
        """
        profile = resolve(name)
        if name not in spare:
            arms = tmp_path / name / 'arms'
            arms.mkdir(parents=True, exist_ok=True)
            profile = fixture_profile(profile, tmp_path / name)
            spare[name] = replace(profile, arms=arms, lock=tmp_path / name / 'arm.lock')
        return spare[name]

    monkeypatch.setattr(modes, 'resolve', redirected)
    monkeypatch.setattr(rtty, 'HOLDS', SimpleNamespace(
        name='rtty', lock=tmp_path / 'rtty' / 'arm.lock', arms=tmp_path / 'rtty' / 'sessions'))
    allocate = capture_identity.allocate_arm
    monkeypatch.setattr(capture_identity, 'allocate_arm',
                        lambda repo, **kwargs: allocate(
                            tmp_path, **kwargs | {'transcripts': tmp_path / 'working'}))
    return redirected(pactor.PROFILE.name)


@pytest.fixture
def fake_claim(monkeypatch, tmp_path, no_hardware):
    monkeypatch.setenv('HFMODEM_COEXIST_HARNESS', '1')
    monkeypatch.setenv('HFMODEM_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.delenv('HFMODEM_COEXIST_LOG', raising=False)
    monkeypatch.delenv('HFMODEM_COEXIST_BARRIERS', raising=False)
    monkeypatch.setattr(claim.hardware, 'process_started', lambda pid: f'fake-{pid}')
    monkeypatch.setattr(claim.hardware, '_command', lambda pid: 'fake console')
    monkeypatch.setattr(claim.hardware, 'check_devices', lambda *a, **k: ())
    monkeypatch.setattr(claim.hardware, 'openers', lambda *a, **k: [])
    client = SimpleNamespace(state=lambda **kw: {'service': {'state': 'observing'}},
                             release=lambda **kw: None)
    monkeypatch.setattr(claim.station_client, 'attach', lambda *a, **kw: client)
    monkeypatch.setattr(claim.station_client.Client, 'state', lambda self, **kw: client.state(**kw))
    monkeypatch.setattr(claim.station_client.Client, 'release', lambda self, **kw: client.release(**kw))
    monkeypatch.setattr(arm, '_down_once', lambda env, timeout: (0, 'PTT LINE low\nCLEANUP COMPLETE\n'))
    original = claim.remember
    def remember(process):
        if hasattr(process, 'pid') and not isinstance(process, Arm):
            return original(process)
        process._claim_child = SimpleNamespace(alive=lambda: False, _signal=lambda sig: None)
        return process
    monkeypatch.setattr(claim, 'remember', remember)
    end = claim.end_child
    monkeypatch.setattr(claim, 'end_child', lambda process, **kw:
                        end(process, **kw) if hasattr(process, 'pid') and not isinstance(process, Arm) else None)
    init = Arm.__init__
    def initialize(self, transcript=''):
        init(self, transcript)
        self.stdout = iter(['CLAIM ADMITTED\n', transcript,
                            'PTT LINE low\n', 'CLEANUP COMPLETE\n'])
    monkeypatch.setattr(Arm, '__init__', initialize)
    return client


@pytest.fixture
def fake_processes(monkeypatch, fake_claim):
    sequence = iter(range(100001, 101001))
    def remember(process):
        if not hasattr(process, 'pid'):
            process.pid = next(sequence)
        child = NativeChild(process.pid, f'fake-{process.pid}', None, None, None, _process=process)
        process._claim_child = child
        claim._local.children.append(child)
        return process
    def send(child, sig):
        if child._process.poll() is not None:
            return False
        if sig == signal.SIGKILL:
            child._process.kill()
        else:
            child._process.terminate()
        return True
    monkeypatch.setattr(claim, 'remember', remember)
    monkeypatch.setattr(NativeChild, '_alive', lambda child, **kw: child._process.poll() is None)
    monkeypatch.setattr(NativeChild, '_signal', send)
