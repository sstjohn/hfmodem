"""Foreground ALE station and held-channel selcall operator console.

All keying goes through Station's existing rig, identity and transmit arbiter.
The UI uses the native host dialects in-process; no mail client is needed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import tomllib
import uuid

from hfmodem.core import band, config
from hfmodem.core.rig import RigError
from hfmodem.station.process import Station
from . import claim, modes, station

HELP = """Commands (text after the address needs no quotation marks):
  status | heard | channels       station state / recent ALE stations / dial list
  call CALL [text]                ALE call, optionally carrying AMD text
  msg CALL text                  native ALE AMD text (up to 90 characters)
  dtm CALL text | dbm CALL text   native ALE data text with ARQ
  net HFN | net HFL               call the amateur net
  scan on | scan off              resume scanning / hold this channel
  channel INDEX|KHZ               hold a configured channel
  sound | hangup                 manual sound / end ALE links
  selcall ADDRESS                4- or 6-digit call on the held channel
  page ADDRESS text              native selcall page (up to 32 characters)
  ack ADDRESS                    selcall acknowledgment
  beacon ADDRESS                 selcall beacon request
  help | quit                    show this help / stop and close the rig
Incoming text is printed automatically. OK means accepted, not delivered.
DTM/DBM report SENT after ARQ; AMD and pages need a reply to prove receipt.
"""


def parser(family='hflink'):
    p = argparse.ArgumentParser(prog=f'console {family}', description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    run = sub.add_parser('station', help='scan, answer and exchange native text in the foreground')
    run.add_argument('--config', default=os.environ.get(config.STATION_ENV, str(station.ROOT / 'station.toml')))
    run.add_argument('--rig', choices=('ft891', 'ftdx10'))
    freq = run.add_mutually_exclusive_group(required=family != 'selcall')
    freq.add_argument('--channels', nargs='+', type=float, metavar='KHZ', help='USB dial frequencies to scan')
    freq.add_argument('--freq', type=float, metavar='KHZ', help='one USB dial frequency to hold')
    sounding = run.add_mutually_exclusive_group(required=family != 'selcall')
    sounding.add_argument('--pilot', nargs='+', type=float, metavar='KHZ', help='sound hourly on these scan channels only')
    sounding.add_argument('--no-sound', action='store_true', help='answer calls without automatic sounding (mobile/portable or HFL)')
    run.add_argument('--address', nargs='+', metavar='ADDRESS', help='own ALE address(es), or own numeric selcall address(es) for selcall station')
    run.add_argument('--selcall', nargs='+', metavar='ADDRESS', help='also receive selcall on a single held ALE channel')
    run.add_argument('--acknowledge', action='store_true', help='answer individual selcall acknowledgment requests; opt-in, interoperability still needs a rig test')
    run.add_argument('--scan-rate', type=float, choices=(2.0, 5.0), default=2.0)
    run.add_argument('--dots', type=float, default=6.0, metavar='SECONDS', help='selcall dot preamble, default 6 seconds')
    run.add_argument('--verbose', action='store_true', help='show every retune and repeated decoder event')
    run.add_argument('--summary-seconds', type=float, default=30.0, help='quiet-station status interval (default 30; 0 disables)')
    run.add_argument('--log-dir', type=Path, help='new session directory; default below the configured capture directory')
    execute = run.add_mutually_exclusive_group(required=True)
    execute.add_argument('--dry-run', action='store_true', help='show the plan; open no audio, CAT, PTT or ports')
    execute.add_argument('--execute', action='store_true', help='run; configured transmit=true is still required')
    return p


def dial(khz):
    if not math.isfinite(khz) or not 1000 <= khz <= 60000:
        raise ValueError('a channel is an HF USB dial in kHz (for example 7102)')
    # Host names have one decimal place; reject values that would alias them.
    hz = round(khz * 1000)
    if hz % 100:
        raise ValueError('ALE/selcall channel dials use 0.1 kHz steps')
    return hz


def configuration(args, family):
    with Path(args.config).expanduser().open('rb') as f:
        raw = tomllib.load(f)
    dials = [dial(f) for f in (args.channels or ([args.freq] if args.freq is not None else []))]
    if len(set(dials)) != len(dials):
        raise ValueError('duplicate scan channels')
    pilots = [dial(f) for f in args.pilot or []]
    if len(set(pilots)) != len(pilots) or set(pilots) - set(dials):
        raise ValueError('pilot channels must be distinct members of the scan list')
    if not math.isfinite(args.summary_seconds) or args.summary_seconds < 0:
        raise ValueError('--summary-seconds must be finite and nonnegative')
    if not math.isfinite(args.dots) or not 0 <= args.dots <= 60:
        raise ValueError('--dots must be between 0 and 60 seconds')
    if family == 'selcall' and (len(dials) > 1 or pilots or args.selcall):
        raise ValueError('selcall station holds one channel; use --freq and --address')
    addresses = args.address or [raw.get('station', {}).get('mycall', '')]
    sel = addresses if family == 'selcall' else args.selcall
    if sel and len(dials) > 1:
        raise ValueError('selcall needs a held channel: use --freq, or run console selcall station separately')
    if args.acknowledge and not sel:
        raise ValueError('--acknowledge needs own selcall addresses')
    # This station answers and optionally sounds on its own. Apply automatic
    # emission limits without changing the source file or its transmit interlock.
    raw.setdefault('station', {})['control'] = 'automatic'
    if dials:
        raw.setdefault('rig', {})['centre_hz'] = dials[0] + band.DIAL_OFFSET_HZ
    protocols = {}
    if family != 'selcall':
        protocols['ale'] = dict(enabled=True, host='ale', port=0,
                                addresses=addresses, channels=dials,
                                scan_rate=args.scan_rate, conventions='hflink',
                                base=bool(pilots), pilot_channels=pilots,
                                listen=True)
    if sel:
        protocols['selcall'] = dict(enabled=True, host='selcall', port=0,
                                    addresses=sel, dots_s=args.dots,
                                    acknowledge=args.acknowledge, listen=True)
    raw['protocols'] = protocols
    cfg = config.parse(raw, rig_name=args.rig)
    if family == 'selcall' and not args.address:
        raise ValueError('selcall needs --address with your own 4- and/or 6-digit ID')
    if args.execute and not cfg.station.transmit:
        raise ValueError('station transmit=false; --execute does not override it')
    return cfg, dials


def plan(cfg, dials, args):
    ale = cfg.protocols.get('ale')
    sel = cfg.protocols.get('selcall')
    return dict(callsign=cfg.station.mycall, rig=cfg.selected_rig,
                control=cfg.control.value, regulatory=cfg.profile.name,
                transmit=cfg.station.transmit,
                dial_khz=[f / 1000 for f in dials] or 'current radio dial',
                ale_addresses=list(ale.addresses) if ale else [],
                scan_rate=ale.scan_rate if ale and len(dials) > 1 else 0,
                sounding='hourly, clear-channel only' if ale and ale.base else 'off',
                pilot_khz=[f / 1000 for f in ale.pilot_channels] if ale else [],
                selcall_addresses=list(sel.addresses) if sel else [],
                selcall_acknowledge=bool(sel and sel.acknowledge),
                input=cfg.audio.input, output=cfg.audio.output,
                tx_drive=cfg.audio.tx_drive, filter_hz=list(cfg.rig.filter_hz),
                max_key_s=cfg.rig.max_key_s,
                summary_seconds=args.summary_seconds)


class Output:
    """Full journal; only repeated reception/state chatter is thinned on screen."""

    def __init__(self, journal, *, verbose=False, stream=None, clock=time.monotonic):
        self.journal, self.verbose = journal, verbose
        self.stream = stream or sys.stdout
        self.clock = clock
        self.lock = threading.Lock()
        self.last = {}
        self.tunes = 0

    def line(self, protocol, text, *, repeated=False, detail=False):
        now = datetime.now(timezone.utc)
        text = ''.join(c if c.isprintable() else ' ' for c in text)
        with self.lock:
            self.journal.write(json.dumps(dict(utc=now.isoformat(), protocol=protocol, text=text)) + '\n')
            self.journal.flush()
            if detail and not self.verbose:
                return
            key = (protocol, text.split(' BER ')[0] if text.startswith('HEARD ') else text)
            at = self.clock()
            if repeated and not self.verbose and at - self.last.get(key, -math.inf) < 30:
                return
            self.last[key] = at
            # Bound reception chatter bookkeeping for long-running stations.
            if len(self.last) > 2048:
                self.last = {k: t for k, t in self.last.items() if at - t < 30}
            print(f'{now:%H:%M:%SZ} {protocol.upper():7s} {text}', file=self.stream, flush=True)

    def tuned(self, khz):
        self.tunes += 1
        self.line('ale', f'TUNE {khz:g} kHz USB', detail=True)


class Events:
    def __init__(self, host, protocol, output):
        self.host, self.protocol, self.output = host, protocol, output

    def event(self, ev):
        # Keep host formatting and delivery; the observer adds terminal/journal.
        self.host.event(ev)
        lines = list(self.host.log)
        line = lines[-1]
        if self.protocol == 'ale' and ev.kind == 'STATE':
            name = self.host.names[ev.channel] if 0 <= ev.channel < len(self.host.names) else '-'
            line += f' CHANNEL {name}'
        if len(lines) > 1 and line.startswith('WHO '):
            self.output.line(self.protocol, lines[-2], repeated=ev.kind == 'HEARD')
        self.output.line(self.protocol, line, repeated=ev.kind in ('HEARD', 'STATE', 'TUNE_FAILED'))


def command(line, available):
    """Translate operator words to the native dialect without a shell or quoting."""
    if any(ord(c) < 32 or ord(c) > 126 for c in line):
        raise ValueError('commands and messages use printable ASCII on one line')
    word, _, rest = line.strip().partition(' ')
    word = word.lower()
    if word in ('help', 'quit', 'exit', ''):
        return None, word
    aliases = {'msg': 'AMD', 'selcall': 'CALL', 'beacon': 'BEACON', 'page': 'PAGE', 'ack': 'ACK'}
    protocol = 'selcall' if word in ('selcall', 'page', 'ack', 'beacon') else 'ale'
    native = aliases.get(word, word.upper())
    if protocol not in available:
        # In a selcall-only session "call" is also natural.
        if word == 'call' and 'selcall' in available:
            protocol = 'selcall'
        elif word == 'status':
            protocol = next(iter(available))
        else:
            raise ValueError(f'{protocol} is not enabled in this station')
    allowed = {'STATUS', 'HEARD', 'CHANNELS', 'CALL', 'AMD', 'DTM', 'DBM',
               'NET', 'SCAN', 'CHANNEL', 'SOUND', 'HANGUP', 'PAGE', 'ACK', 'BEACON'}
    if native not in allowed:
        raise ValueError('unknown command; type help')
    return protocol, f'{native} {rest}'.rstrip()


class OperatorStation(Station):
    def __init__(self, cfg, *, output, args, dials):
        super().__init__(cfg)
        self.output, self.args, self.dials = output, args, dials
        self.next_summary = math.inf

    def open_rig(self, **kw):
        super().open_rig(**kw)
        try:
            reply = self.rig.cat.command('m')
            mode = self.rig.cat._field(reply, 'Mode')
            if mode not in ('USB', 'PKTUSB'):
                raise RigError(f'HFLINK needs USB/PKTUSB; radio reports {mode!r}. Set USB on the rig.')
            self.output.line('rig', f'{mode}, passband {self.rig.cat._field(reply, "Passband")} Hz; filter unchanged')
            if self.dials:
                self.rig.tune(self.dials[0] + band.DIAL_OFFSET_HZ)
        except BaseException:
            self.rig.close()
            raise

    def ready(self, _station):
        if set(self.hosts) != set(self.cfg.protocols):
            raise RuntimeError('an operator host failed to start; station stopped')
        for name, served in self.hosts.items():
            lane = self.lanes[name]
            lane.modem.observer = Events(served.server, name, self.output)
            send = lane.modem._send_audio

            def observed_send(audio, rate, *, name=name, send=send):
                self.output.line(name, f'TX start ({len(audio) / rate:.1f}s)')
                try:
                    end = send(audio, rate)
                except Exception as exc:
                    self.output.line(name, f'TX failed: {exc}')
                    raise
                self.output.line(name, f'TX refused: {end or "refused"}' if end is None or isinstance(end, str) else 'TX complete')
                return end

            lane.modem._send_audio = observed_send
            if name == 'ale':
                tune = lane.modem._tune_radio

                def observed_tune(ch, tune=tune):
                    at = tune(ch)
                    if at is not None:
                        self.output.tuned(self.dials[ch] / 1000)
                    return at

                lane.modem._tune_radio = observed_tune
            self.output.line(name, served.server.command('STATUS'))
        self.output.line('console', 'Ready. Type help for calls and native text; quit stops the station.')
        self.next_summary = time.monotonic() + self.args.summary_seconds if self.args.summary_seconds else math.inf
        self.operator_thread = threading.Thread(target=self.inputs, name='hflink-operator', daemon=True)
        self.operator_thread.start()

    def inputs(self):
        while not self._stop.is_set():
            if sys.stdin.isatty():
                with self.output.lock:
                    print('hflink> ', end='', flush=True)
            line = sys.stdin.readline()
            if self._stop.is_set():
                return
            if not line:
                self.output.line('console', 'Input closed; stopping.')
                self._stop.set()
                return
            try:
                protocol, native = command(line.rstrip('\n'), self.hosts)
                if protocol is None:
                    if native in ('quit', 'exit'):
                        self.output.line('console', 'Stopping station.')
                        self._stop.set()
                        return
                    if native == 'help':
                        with self.output.lock:
                            print(HELP, flush=True)
                    continue
                self.output.line('command', f'{protocol.upper()} {native}')
                host = self.hosts[protocol].server
                if protocol == 'selcall' and native.startswith('ACK '):
                    parts = native.split()
                    if len(parts) != 2:
                        raise ValueError('ack ADDRESS')
                    answer = host._send_call('ack', parts[1])
                else:
                    answer = host.command(native)
                self.output.line(protocol, answer)
            except Exception as exc:
                self.output.line('console', f'ERR {exc}')

    def _tick(self):
        super()._tick()
        if time.monotonic() >= self.next_summary:
            for name, served in self.hosts.items():
                self.output.line(name, served.server.command('STATUS'))
            if 'ale' in self.lanes:
                ctl = self.lanes['ale'].modem.ctl
                self.output.line('console', f'{self.output.tunes} retunes; {len(ctl.lqa)} station/channel observations')
            self.next_summary = time.monotonic() + self.args.summary_seconds


@contextmanager
def claimed():
    with ExitStack() as stack:
        for name in modes.KNOWN:
            mode = modes.resolve(name)
            mode.lock.parent.mkdir(parents=True, exist_ok=True)
            stack.enter_context(claim.lease(mode))
        yield


def run(argv, *, family='hflink'):
    args = parser(family).parse_args(argv)
    cfg, dials = configuration(args, family)
    description = plan(cfg, dials, args)
    if args.dry_run:
        print(json.dumps(description, indent=2))
        return 0
    print(f'{description["callsign"]} / {description["rig"]}: automatic control, '
          f'{description["regulatory"]}; key-up limit {cfg.rig.max_key_s:g}s')
    print(f'USB dial kHz: {description["dial_khz"]}; sounding {description["sounding"]}')
    if description['pilot_khz']:
        print(f'Sounding pilots kHz: {description["pilot_khz"]}')
    cfg.require_audio('HFLINK station', input=True, output=True, drive=True)
    if not cfg.rig.ptt.port:
        raise ValueError('HFLINK station needs the configured [rig.ptt] port')
    directory = args.log_dir
    if directory is None:
        if not cfg.audio.capture_dir:
            raise ValueError('set [audio] capture_dir or supply --log-dir')
        directory = Path(cfg.audio.capture_dir).expanduser() / f'hflink-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}'
    directory = directory.expanduser()
    with claimed():
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'plan.json').write_text(json.dumps(description, indent=2) + '\n')
        with (directory / 'events.jsonl').open('w') as journal:
            output = Output(journal, verbose=args.verbose)
            output.line('console', f'Full event log: {directory / "events.jsonl"}')
            operator = OperatorStation(cfg, output=output, args=args, dials=dials)
            try:
                return operator.run(on_ready=operator.ready)
            finally:
                output.line('console', 'Station closed.')
