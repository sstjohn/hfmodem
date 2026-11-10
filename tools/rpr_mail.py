# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One station-owned Winlink exchange over brubru connected AX.25."""
import argparse
from datetime import date, datetime, time
import json
import math
import os
from pathlib import Path
import socket
import signal
import sys
import tomllib

from hfmodem.core.pipeio import Reader


def toml_value(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, list):
        return '[' + ', '.join(map(toml_value, value)) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(f'{toml_value(k)} = {toml_value(v)}' for k, v in value.items()) + '}'
    raise TypeError(f'Unsupported TOML value {type(value).__name__}')


def write_toml(body):
    lines = []
    def table(data, prefix=()):
        if prefix:
            lines.append('[' + '.'.join(toml_value(k) for k in prefix) + ']')
        for key, value in data.items():
            if not isinstance(value, dict):
                lines.append(f'{toml_value(key)} = {toml_value(value)}')
        for key, value in data.items():
            if isinstance(value, dict):
                table(value, (*prefix, key))
    table(body)
    return '\n'.join(lines) + '\n'


def private_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        if isinstance(text, str):
            text = text.encode('utf-8')
        stream.write(text)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def derive_station(source, centre, directory):
    body = tomllib.loads(Path(source).read_text(encoding='utf-8'))
    station = body.setdefault('station', {})
    station['port'] = free_port()
    # The launcher has already resolved explicit identity overrides from the
    # operator's station defaults; freeze that same identity for this arm.
    for field, variable in (('mycall', 'MYCALL'), ('idcall', 'IDCALL')):
        if variable in os.environ:
            station[field] = os.environ[variable]
    body.setdefault('rig', {}).update(centre_hz=centre, cat='rigctld', host='127.0.0.1', port=4532)
    audio = body.setdefault('audio', {})
    audio['capture_dir'] = str(directory)
    for chain in audio.get('chains', {}).values():
        chain['capture_dir'] = str(directory)
    protocols = body.setdefault('protocols', {})
    # Only the arm's modem may own a host or emit on this held channel.
    for protocol in protocols.values():
        protocol['enabled'] = False
    rpr = protocols.setdefault('rpr', {})
    rpr.update(enabled=True, host='mail')
    rpr.pop('port', None)
    rpr.setdefault('options', {})['tx_events'] = str(directory/'tx_events.jsonl')
    return body


def prepare(source, centre, directory, *, send=None, to='', subject=''):
    from hfmodem.core import config
    from hfmodem.winlink import load_outbound
    directory = Path(directory).resolve()
    body = derive_station(source, centre, directory)
    for path in send or []:
        load_outbound(path, body['station']['mycall'], to=to, subject=subject)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory/'station.toml'
    private_write(path, write_toml(body))
    config.load(path)
    return path


STOP_MARGIN_S = 5.


def stop_bound(path, *, cwid_optional=False):
    """Budget close, one committed burst, closing CW and lifecycle overhead."""
    from hfmodem.core import config, cwid, identification
    from hfmodem.brubru.host.mail import STOP_CLOSE_S, RETIRE_MARGIN_S
    cfg = config.load(path)
    settle = cfg.rig.ptt.settle_s
    closing = 0.
    if cfg.station.transmit:
        identity = identification.from_station(cfg.station)
        if identity.closing_required('rpr', optional=cwid_optional):
            closing = cwid.duration(identity.cw_call) + settle
    return math.ceil(cfg.rig.max_key_s + settle + STOP_CLOSE_S + RETIRE_MARGIN_S
                     + closing + STOP_MARGIN_S)


def run_station(path, gateway, *, mail_send=None, mail_fetch=False, mail_to='',
                mail_subject='', mail_sid='', mail_out=None, mail_timeout=1200,
                cwid_optional=False, launcher_control=False):
    from hfmodem.core import config, identification
    from hfmodem.station.process import Station
    from hfmodem.station.service import Service
    from hfmodem.brubru.host.mail import run_exchange
    from hfmodem.winlink import B2FSession, load_outbound
    from hfmodem.winlink.client import progress_to_stdout
    cfg = config.load(path)
    if launcher_control and os.name == 'nt':
        from hfmodem.core import winproc
        winproc.declare_grace(stop_bound(path, cwid_optional=cwid_optional))
    outbox = [load_outbound(p, cfg.station.mycall, to=mail_to, subject=mail_subject)
              for p in mail_send or []]
    session = B2FSession(cfg.station.mycall, target=gateway, password=config.mail_password(),
                         outbox=outbox, client_sid=config.client_sid(mail_sid, path=path))
    mail_out = Path(mail_out) if mail_out else Path(path).resolve().parent/'mail'
    result = 1
    def exchange(station):
        nonlocal result
        try:
            runtime = station.hosts['rpr'].server
            result = run_exchange(runtime, session, gateway, timeout=mail_timeout,
                                  stopped=lambda: bool(station._signal_number) or station._stop.is_set(),
                                  outdir=mail_out,
                                  burst_limit=cfg.rig.max_key_s + cfg.rig.ptt.settle_s)
        finally:
            station._stop.set()
    identity = identification.from_station(cfg.station) if cfg.station.transmit else None
    if identity is not None:
        identity.announce(optional=cwid_optional)
    service = Service(path)
    try:
        service.bind()
        # The arm already owns admission; match ardop-mail without station measurements.
        station = Station(cfg, station_file=path, ownership_preflight=False, graceful_signals=True,
                          closing_cwid=bool(identity and identity.closing_required(
                              'rpr', optional=cwid_optional)))
        station.service = service
        service.station = station
        control = StationControl(sys.stdin.buffer, station) if launcher_control else None
        try:
            with progress_to_stdout():
                code = station.run(on_ready=exchange)
        finally:
            if control is not None and not control.close():
                raise RuntimeError('RPR launcher control reader did not stop')
        return code or result
    finally:
        service.close()


class StationControl(Reader):
    def __init__(self, stream, station):
        self.station, self.requests = station, 0
        super().__init__(stream)

    def request(self):
        self.requests += 1
        # Ctrl-C reaches both processes in either order. The pipe's first
        # request must not count as a second signal in the station.
        if self.requests == 1:
            self.station._stop.set()
            self.station._own_cancel.set()
            return
        if not self.station._signals:
            self.station._signal(signal.SIGTERM, None)
        self.station._signal(signal.SIGINT, None)

    def _run(self):
        pending = b''
        try:
            for block in self._blocks():
                pending += block
                while b'\n' in pending:
                    word, pending = pending.split(b'\n', 1)
                    if word == b'stop':
                        self.request()
            if not self.stop.is_set() and not self.requests:
                self.request()
        except BaseException as exc:
            self.error = exc
        finally:
            self.done.set()


def mail_flags(parser):
    parser.add_argument('--mail-send', type=Path, action='append', default=[])
    parser.add_argument('--mail-fetch', action='store_true')
    parser.add_argument('--mail-to', default='')
    parser.add_argument('--mail-subject', default='')
    parser.add_argument('--mail-sid', default='')
    parser.add_argument('--mail-out', type=Path)
    parser.add_argument('--mail-timeout', type=float, default=1200)
    parser.add_argument('--cwid-optional', action='store_true')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('source', type=Path)
    prep.add_argument('centre', type=int)
    prep.add_argument('directory', type=Path)
    mail_flags(prep)
    run = sub.add_parser('run')
    run.add_argument('path', type=Path)
    run.add_argument('gateway')
    run.add_argument('--launcher-control', action='store_true', help=argparse.SUPPRESS)
    mail_flags(run)
    bound = sub.add_parser('stop-bound')
    bound.add_argument('path', type=Path)
    bound.add_argument('--cwid-optional', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.source, args.centre, args.directory, send=args.mail_send,
                to=args.mail_to, subject=args.mail_subject)
    elif args.action == 'stop-bound':
        print(stop_bound(args.path, cwid_optional=args.cwid_optional))
    else:
        options = vars(args).copy()
        options.pop('action')
        raise SystemExit(run_station(**options))


if __name__ == '__main__':
    main()
