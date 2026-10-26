# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
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
import re
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
#: A session's band change, as its terminal says it in arm.log.
BAND_LINE = re.compile(r'BAND (\d+) m (\d+) Hz')


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
        self.moved = lambda band, dial: None
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
        return arm.pump(self.process, self, self)

    def write(self, line):
        if is_verdict(line):
            self.verdicts.append(line.strip())
        elif band := BAND_LINE.fullmatch(line.strip()):
            self.moved(int(band[1]), int(band[2]))
        try:
            self._lines.put_nowait(line)
        except queue.Full:
            self._lose(OSError('arm.log fell behind'))

    def feed(self, line):
        return ()

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
            self.process._claim_child.terminate()
        def drain():
            try:
                self.read()
            except (OSError, RuntimeError, ValueError) as error:
                self._lose(error, lines=0)
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            code = self.process.wait(timeout=STOP_S)
        except subprocess.TimeoutExpired:
            claim.end_child(self.process)
            return None
        reader.join(timeout=5)
        return code

    @property
    def verdict(self):
        return ' | '.join(self.verdicts[-2:])

    def tell(self):
        """The keying-line verdict, said the moment the launcher has settled and
        before anything that touches the disk."""
        self.reported = True
        print(f'  keying line: {self.verdict}' if self.verdict else
              '  keying line: confirmation unavailable -- check the transmitter.', flush=True)

    def losses(self):
        if self.lost or self.trouble:
            return dict(lines=self.lost, first_error=self.trouble)
        return None


def record(plan, directory, start, spawned, stop, **where):
    start(plan, directory, say=lambda line: tell(directory, line), spawned=spawned, stop=stop, **where)


def end_recorder(process):
    """Retire an owned recorder, including one that exited during placement."""
    if process.poll() is not None:
        return
    if not hasattr(process, '_claim_child'):
        try:
            claim.remember(process)
        except claim.hardware.ClaimError:
            # Placement can reap a failed recorder between poll and identity.
            # A live but unidentifiable process must still fail closed.
            if process.poll() is None:
                raise
            return
    claim.end_child(process)


class Witness:
    """One worker owns initial placement and every later replacement in order.

    The log reader only queues a move, so a slow witness search never holds up
    terminal output or shutdown. The shared stop event cancels placement at exit.
    """
    def __init__(self, directory, held, stop):
        self.directory, self.held, self.stop = directory, held, stop
        self.moves, self.current = queue.Queue(), []

    def move(self, plan, *, stem='witness'):
        self.moves.put((plan, stem))

    def spawned(self, process):
        self.held.append(process)
        self.current.append(process)

    def run(self):
        while not self.stop.is_set():
            try:
                plan, stem = self.moves.get(timeout=.1)
            except queue.Empty:
                continue
            if self.stop.is_set():
                break
            try:
                for old in self.current:
                    end_recorder(old)
                self.current.clear()
                station.place_kiwi(plan, self.directory, spawned=self.spawned, stop=self.stop,
                                   say=lambda line: tell(self.directory, line), stem=stem)
            except Exception as error:
                tell(self.directory, f'NO WITNESS: replacement failed: {error}')


def fly(plan, *, holds=None, header=None):
    directory = Path(plan['directory'])
    with claim.lease(holds or HOLDS, env=plan.get('env')):
        held, recorders, stop = [], [], threading.Event()
        launcher = sink = owner = admission = process = None
        code = None
        plan = dict(plan, started_utc=datetime.now(timezone.utc).isoformat())
        directory.mkdir(parents=True)
        record_path = directory / 'provenance.json'
        try:
            read = plan['dial_hz'] or station.cat('f', selection=station.selection_for_plan(plan))
            dial = int(read) if str(read or '').isdigit() else None
            plan.update(dial_hz=dial, readings=station.check_rig(dict(plan, dial_hz=dial or 0)))
            record_path.write_text(json.dumps(plan, indent=2) + '\n')
            selection = station.selection_for_plan(plan)
            preflight = [*station.spell_environment(plan['readings'], selection=selection),
                         arm.spelled(plan)]
            loud = station.codec_warning(plan['readings'], selection=selection)
            print('\n'.join(filter(None, [header(plan), loud])) if header else '\n'.join(preflight),
                  flush=True)
            def start(begin, spawned, plan=plan, **where):
                recorders.append(threading.Thread(target=record,
                    args=(plan, directory, begin, spawned, stop), kwargs=where, daemon=True))
                recorders[-1].start()

            witness = Witness(directory, held, stop)
            if plan['adjustments']['kiwi'] is not None:
                recorders.append(threading.Thread(target=witness.run, daemon=True))
                recorders[-1].start()

            def moved(band, dial):
                plan.setdefault('bands', []).append(dict(band_m=band, dial_hz=dial,
                                                         utc=datetime.now(timezone.utc).isoformat()))
                if plan['adjustments']['kiwi'] is not None:
                    witness.move(dict(plan, band_m=band, dial_hz=dial),
                                 stem=f'witness-{len(plan["bands"])}-{band}m')

            start(station.start_tap, held.append)
            if plan['adjustments']['kiwi'] is not None:
                if dial:
                    witness.move(plan)
                else:
                    tell(directory, 'NO WITNESS: the rig gave no dial to place it on')
            sink = (directory / 'arm.log').open('w')
            if header:
                sink.write('\n'.join(preflight) + '\n')
            admission = claim.hardware.inherited(9, launch=True)
            heard, owner = owner_pipe()
            try:
                with claim.defer_interrupts():
                    process = claim.remember(subprocess.Popen(
                        plan['command'], cwd=station.ROOT, pass_fds=(9, heard),
                        env=dict(os.environ, **plan['env'], HFMODEM_OWNER_FD=str(heard)),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
                    process._claim_admission = admission
            finally:
                os.close(heard)
            try:
                launcher, sink = Launcher(process, sink), None
                launcher.moved = moved
                code = launcher.read()
            except BaseException:
                with claim.uninterrupted():
                    if launcher is not None:
                        code = launcher.stop()
                raise
            finally:
                if launcher is not None:
                    launcher.tell()
        finally:
            with claim.uninterrupted(), arm._preserve_primary_failure():
                if owner is not None:
                    os.close(owner)
                try:
                    if process is not None:
                        if launcher is None and sink is not None:
                            launcher, sink = Launcher(process, sink), None
                        if process.poll() is None and launcher is not None:
                            code = launcher.stop()
                        claim.end_child(process)
                        if launcher is not None and not getattr(launcher, 'reported', False):
                            launcher.tell()
                finally:
                    if admission is not None:
                        admission.failed()
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
                    end_recorder(recorder)
                if any(thread.is_alive() for thread in recorders):
                    raise RuntimeError('recorder startup did not stop; cleanup incomplete')
                claim.end_children(allow_cleanup=True)
                inherited = arm.cleanup_result(''.join(getattr(process, '_cleanup_lines', [])), code)
                plan['cleanup'] = inherited if inherited['complete'] else arm.bring_down(directory, env=plan['env'])
                if code == 0 and not plan['cleanup']['complete']:
                    code = 1
                try:
                    plan['readings_at_end'] = station.read_environment(selection=station.selection_for_plan(plan))
                except Exception as trouble:
                    plan['readings_at_end'] = dict(error=f'{type(trouble).__name__}: {trouble}')
                    plan['console finalization failed'] = str(trouble)
                plan.update(ended_utc=datetime.now(timezone.utc).isoformat(), returncode=code,
                    verdict=dict(line=(launcher and launcher.verdict) or 'no keying-line verdict from the launcher'))
                try:
                    record_path.write_text(json.dumps(plan, indent=2) + '\n')
                except OSError as trouble:
                    print(f'  console finalization failed: provenance.json was not written at the end: {trouble}', flush=True)
        status = directory / 'recorders.log'
        if status.is_file():
            print(status.read_text().rstrip(), flush=True)
        print(f'  {plan["arm"]}: exit {code}\n  {directory}', flush=True)
        for line in arm.faults(directory, code):
            print('  ' + line, flush=True)
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
