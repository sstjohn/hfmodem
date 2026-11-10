# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Windows capabilities of the operator console."""
from __future__ import annotations

import errno
import sys


UNPORTED = {
    'pactor': 'the launcher',
    'pactor-answer': 'the launcher',
    'preflight': 'the launcher',
    'vara': 'the launcher',
    'ardop': 'the launcher',
    'ardop-mail': 'the launcher',
    'rpr-mail': 'the launcher',
    'vara-mail': 'the launcher',
    'sabir': 'the launcher',
    'vara-txcal': 'the launcher',
    'pounce': 'the launcher',
    'sense': 'channel sense and campaign scans',
    'kiwi': 'the Kiwi witness',
    'voice': 'in-process voice keying',
    'hflink': 'in-process HFLINK keying',
    'selcall': 'in-process selcall keying',
    'cold_start': 'cold-starting a station from the console',
    'cat': 'console CAT access',
}
WORKING = (
    'check/power/listen/down with the Python launcher',
    'replay monitor',
    'log',
    'logbook',
    'fingerprints',
    'data',
    'attach/own/release to a station that is already running',
    'dry runs without CAT',
)
GAIN = 'codec gain control is macOS-only; set the level in Windows Sound settings'


def windows():
    return sys.platform == 'win32'


def check(path, needs):
    if windows() and needs in UNPORTED:
        print(f'{path} needs {UNPORTED[needs]}, which is not yet ported to Windows. '
              f'On Windows today: {", ".join(WORKING)}. Use macOS or Linux for flights.',
              file=sys.stderr)
        raise SystemExit(2)


def flight(path, args):
    if getattr(args, 'kiwi', None) is not None:
        check(f'{path} --kiwi', 'kiwi')
    if getattr(args, 'prepare_radio', False):
        check(f'{path} --prepare-radio', 'cat')
    if not getattr(args, 'dry_run', False):
        if getattr(args, 'sense', False) or getattr(args, 'pounce', 0):
            check(path, 'sense')
        verb = {'campaign': 'pounce', 'rpr': 'rpr-mail'}.get(path.split()[0], path.split()[0])
        if path in {'vara chat', 'vara txcal'}:
            verb = 'vara-' + path.split()[1]
        check(path, verb)


def monitor(args):
    if args.list_modes:
        return
    path = f'{args.monitor_mode} monitor' if getattr(args, 'monitor_mode', None) else 'monitor'
    if args.kiwi:
        check(f'{path} --kiwi', 'kiwi')
    if not (args.wav or args.arm):
        check(path, 'cat')


def sabir(args):
    if args.action == 'proof':
        flight('sabir proof', args)
        if args.channel is None:
            check('sabir proof without --channel', 'cat')


def attached(station_file, path='attach/own'):
    if not windows() or 'cold_start' not in UNPORTED:
        return None
    from hfmodem.station import client as station_client
    client = station_client.Client(station_file)
    try:
        client.state()
    except OSError as exc:
        if exc.errno == errno.ECONNREFUSED:
            check(path, 'cold_start')
        raise
    return client


def codec(device=None, scope='input'):
    if windows():
        return dict(status='unsupported', device=device, scope=scope, value=None,
                    writable=False, error=GAIN)
    return None


def gain():
    result = codec()
    if result is not None:
        print(result['error'])
        return result
    return None
