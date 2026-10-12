"""`console rtty qso`: the RTTY terminal on the station rig, and its record. The
launcher's output goes to arm.log and nowhere else, because the terminal owns /dev/tty."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
from types import SimpleNamespace

from . import arm, claim, operator, station

HOLDS = SimpleNamespace(name='rtty', lock=operator.mode_root('rtty-qso') / 'arm.lock',
                        arms=operator.mode_root('rtty-qso') / 'sessions')
ADIF = operator.data_root() / 'working' / 'qso-log.adi'
#: The tap's and the witness's ceiling. A session's length is not known when it
#: starts, and teardown ends both recorders sooner.
SESSION_S = 4 * 3600
STOP_S = 60
OWNER_FD_MIN = 64
ARM_LOG_LINES = 10000
#: What the RTTY path prints about the keying line at its end: the keyer handing
#: the line back (read LOW, or not), and `prove_unkeyed` reading the receiver.
VERDICTS = ('keying line handed back', 'receiver rms', 'CANNOT PROVE PTT IS DOWN', 'PTT VERDICT')


def parser():
    p = argparse.ArgumentParser(
        prog='console rtty qso',
        description='Keyboard-to-keyboard RTTY on the station rig. Nothing transmits until '
                    'you key: Tab keys and sends as you type, Esc turns it over, ^C unkeys and exits '
                    'now, ^Q leaves.')
    p.add_argument('--freq', type=float, metavar='KHZ',
                   help='tune the dial here first; without it the rig stays where it is')
    p.add_argument('--rate', default='45.45/170', metavar='BAUD/SHIFT[/STOP]')
    p.add_argument('--centre', type=float, default=2210.0, metavar='HZ',
                   help='audio centre of our tones; mark is centre + shift/2, high on RF')
    p.add_argument('--contest', metavar='ID', help='contest id: serials and the ADIF record')
    p.add_argument('--log', default=str(ADIF), metavar='FILE', help='the ADIF QSO log')
    p.add_argument('--kiwi', nargs='?', const='', metavar='HOST',
                   help='place a KiwiSDR witness, preferring HOST when one is named')
    p.add_argument('--dry-run', action='store_true', help='print the plan and stop')
    return p


def plan(args, now):
    """The session, decided before anything opens. Nothing here touches hardware."""
    from hfmodem.core import config
    from hfmodem.lanner.rtty import Rate

    Rate.parse(args.rate)
    selection = station.resolve_selection()
    station.warn_receive_only(selection)
    stamp = now.strftime('%Y%m%dT%H%M%SZ')
    directory = HOLDS.arms / stamp
    dial = round(args.freq * 1e3) if args.freq is not None else None
    command = [str(station.ONAIR), 'rtty', str(dial) if dial else '-', '--rate', args.rate,
               '--centre', f'{args.centre:g}', '--adif', str(Path(args.log).resolve()),
               '--session', str(directory)] + (['--contest', args.contest] if args.contest else [])
    env = dict(CAT_PORT=selection.CAT_PORT, PTT_PORT=selection.PTT_PORT,
               HFMODEM_STATION=os.environ.get(config.STATION_ENV)
                              or str(station.ROOT / 'station.toml'))
    if selection.RIG_NAME != 'ft891':
        env['HFMODEM_RIG'] = selection.RIG_NAME
    return dict(arm=f'rtty-qso-{stamp}', directory=str(directory), mode='qso', call='',
                dial_hz=dial, rate=args.rate, tx_centre_hz=args.centre, contest=args.contest,
                adif=str(Path(args.log).resolve()),
                adjustments=dict(kiwi=args.kiwi, witness_seconds=SESSION_S,
                                 retune=dial is not None),
                console_argv=getattr(args, 'console_argv', None), git=arm.git_head(),
                env=env, command=command)


def is_verdict(line):
    return any(mark in line for mark in VERDICTS)


def tell(directory, line):
    """The console's stdout is under the curses screen, so recorder news goes where the
    terminal reads it."""
    with (directory / 'recorders.log').open('a') as out:
        out.write(line.strip() + '\n')


def owner_pipe():
    """The ownership pipe: (read end for the terminal, write end this process keeps).
    The read end is moved to OWNER_FD_MIN or above, clear of the descriptors the
    launcher names for itself (its rig claim takes 9)."""
    heard, owner = os.pipe()
    high = fcntl.fcntl(heard, fcntl.F_DUPFD, OWNER_FD_MIN)
    os.close(heard)
    return high, owner


class Launcher:
    """The launcher's output, read as it comes: its verdicts kept in memory, and
    arm.log written by a thread that owns the file to its close, behind a bounded
    queue, so a full, stalled or failed file costs arm.log lines (counted) and never
    the reading, a verdict or the console's own cleanup."""

    def __init__(self, process, sink):
        self.process, self.sink, self.verdicts = process, sink, []
        self.lost, self.trouble = 0, None
        self._lines = queue.Queue(ARM_LOG_LINES)
        self._writer = threading.Thread(target=self._write, daemon=True)
        self._writer.start()

    def _write(self):
        while (line := self._lines.get()) is not None:
            try:
                self.sink.write(line)
                self.sink.flush()
            except (OSError, ValueError) as exc:
                self._lose(exc)
        try:
            self.sink.close()
        except (OSError, ValueError) as exc:
            self._lose(exc, lines=0)

    def _lose(self, exc, lines=1):
        self.lost += lines
        self.trouble = self.trouble or f'{type(exc).__name__}: {exc}'

    def read(self):
        for line in self.process.stdout:
            if is_verdict(line):
                self.verdicts.append(line.strip())
            try:
                self._lines.put_nowait(line)
            except queue.Full:
                self._lose(OSError('arm.log fell behind'))

    def finish(self, timeout=5):
        """End the writer, waiting at most `timeout`; True when arm.log is closed."""
        try:
            self._lines.put(None, timeout=timeout)
        except queue.Full:
            return False
        self._writer.join(timeout)
        return not self._writer.is_alive()

    def stop(self):
        """Stop a launcher still up (its trap halts the terminal, whose TERM
        unkeys), read it to the end and wait for it; None when it never finished."""
        if self.process.poll() is None:
            self.process.terminate()
        reader = threading.Thread(target=self.read, daemon=True)
        reader.start()
        try:
            code = self.process.wait(timeout=STOP_S)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return None
        reader.join(timeout=5)
        return code

    @property
    def verdict(self):
        return ' | '.join(self.verdicts[-2:])

    def tell(self):
        """The keying-line verdict, said the moment the launcher has settled and
        before anything that touches the disk."""
        print(f'  keying line: {self.verdict}' if self.verdict else
              '  keying line: confirmation unavailable -- check the transmitter.', flush=True)

    def losses(self):
        if self.lost or self.trouble:
            return dict(lines=self.lost, first_error=self.trouble)
        return None


def record(plan, directory, start, held, stop):
    start(plan, directory, say=lambda line: tell(directory, line), spawned=held.append, stop=stop)


def fly(plan):
    directory = Path(plan['directory'])
    with claim.lease(HOLDS):
        read = plan['dial_hz'] or station.cat('f', selection=station.selection_for_plan(plan))
        dial = int(read) if str(read or '').isdigit() else None
        plan = dict(plan, dial_hz=dial, started_utc=datetime.now(timezone.utc).isoformat(),
                    readings=station.check_rig(dict(plan, dial_hz=dial or 0)))
        directory.mkdir(parents=True)
        record_path = directory / 'provenance.json'
        record_path.write_text(json.dumps(plan, indent=2) + '\n')
        for line in station.spell_environment(plan['readings'],
                                              selection=station.selection_for_plan(plan)):
            print(line, flush=True)
        print(arm.spelled(plan), flush=True)
        starts = [station.start_tap]
        if plan['adjustments']['kiwi'] is not None:
            if dial:
                starts.append(station.place_kiwi)
            else:
                tell(directory, 'NO WITNESS: the rig gave no dial to place it on')
        held, recorders, stop, code = [], [], threading.Event(), None
        launcher = sink = interrupt = owner = None
        interrupted = False

        def operator_stop(_signum, _frame):
            nonlocal interrupted
            interrupted = True
            # Popen must finish handing us the child before an interrupt can
            # unwind: otherwise a newly spawned launcher would have no owner.
            if launcher is not None:
                raise KeyboardInterrupt

        try:
            for start in starts:
                recorders.append(threading.Thread(target=record,
                                                  args=(plan, directory, start, held, stop),
                                                  daemon=True))
                recorders[-1].start()
            sink = (directory / 'arm.log').open('w')
            # A stop directed only to the console must end the launcher too.
            interrupt = signal.signal(signal.SIGINT, operator_stop)
            # The terminal watches the read end: it reaches EOF the moment this
            # process is gone, however it went, reaped or not.
            heard, owner = owner_pipe()
            try:
                process = subprocess.Popen(
                    plan['command'], cwd=station.ROOT, pass_fds=(heard,),
                    env=dict(os.environ, **plan['env'], HFMODEM_OWNER_FD=str(heard)),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            finally:
                os.close(heard)
            try:
                launcher, sink = Launcher(process, sink), None
                if interrupted:
                    raise KeyboardInterrupt
                launcher.read()
                code = process.wait()
            except BaseException:
                # Further interrupts must not cut short unkeying and its verdict.
                signal.signal(signal.SIGINT, lambda *_: None)
                code = launcher.stop()
                raise
            finally:
                launcher.tell()
        finally:
            if owner is not None:
                os.close(owner)
            if interrupt is not None:
                signal.signal(signal.SIGINT, interrupt)
            if sink is not None:
                sink.close()
            if launcher is not None:
                if not launcher.finish():
                    plan['arm_log_incomplete'] = True
                    print('  arm.log writer did not finish; its end may be missing.', flush=True)
                if lost := launcher.losses():
                    plan['arm_log_lost'] = lost
                    print(f'  arm.log is missing {lost["lines"]} launcher line(s)'
                          + (f' ({lost["first_error"]})' if lost['first_error'] else '')
                          + '.', flush=True)
            stop.set()
            for thread in recorders:
                thread.join(timeout=5)
            for recorder in held:
                if recorder.poll() is None:
                    recorder.terminate()
                    recorder.wait(timeout=60)
            try:
                plan['readings_at_end'] = station.read_environment(
                    selection=station.selection_for_plan(plan))
            except Exception as trouble:
                plan['readings_at_end'] = dict(error=f'{type(trouble).__name__}: {trouble}')
            plan.update(ended_utc=datetime.now(timezone.utc).isoformat(), returncode=code,
                        verdict=dict(line=(launcher and launcher.verdict)
                                     or 'no keying-line verdict from the launcher'))
            try:
                record_path.write_text(json.dumps(plan, indent=2) + '\n')
            except OSError as trouble:
                print(f'  provenance.json was not written at the end: {trouble}', flush=True)
        status = directory / 'recorders.log'
        if status.is_file():
            print(status.read_text().rstrip(), flush=True)
        print(f'  {plan["arm"]}: exit {code}\n  {directory}', flush=True)
        return code


def run(argv):
    args = parser().parse_args(argv)
    args.console_argv = ['rtty', 'qso', *argv]
    built = plan(args, datetime.now(timezone.utc))
    if args.dry_run:
        print(json.dumps(built, indent=2))
        print(arm.spelled(built))
        return 0
    return fly(built)
