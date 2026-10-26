#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Kestrel chat terminal using the VarAC protocol, launched under the console claim."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import queue
import select
import sys
import termios
import threading
import time

from hfmodem.contact.varac import (BANDWIDTHS, CALLING_HZ, DEFAULT_BW, Client, Flow, HostError,
                                   Mailbox, Session, emit_tag, utc)
from hfmodem.station.contactlog import ContactLog


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mycall', type=str.upper, required=True,
                   help='the digital call, as the fly path takes it: a tactical call such as X-W9SSJ is fine')
    p.add_argument('--idcall', help='CW call for the closing ID; default IDCALL, [station] idcall, then --mycall')
    p.add_argument('--band', type=int, choices=tuple(CALLING_HZ), required=True)
    p.add_argument('--bw', type=int, choices=BANDWIDTHS, default=DEFAULT_BW,
                   help=f'VARA bandwidth for monitoring, /cq and /call (default {DEFAULT_BW})')
    for field in ('name', 'qth', 'loc', 'services'):
        p.add_argument('--' + field)
    p.add_argument('--bbs-directory', help='serve this directory and advertise BBS; overrides station.toml')
    p.add_argument('--no-bbs', action='store_true', help='disable the configured BBS export')
    p.add_argument('--pskreporter', action='store_true',
                   help='opt in to public PSKReporter CQ/beacon uploads; requires --loc GRID')
    p.add_argument('--monitor-wide', action='store_true',
                   help='set the dial off each channel worked by the offset that fits the most VarAC '
                        'channels in the passband, so the slot monitor hears more of them; our audio '
                        'is moved to match both ways')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--rigctld', default='127.0.0.1:4532')
    p.add_argument('--expect-model', required=True)
    p.add_argument('--ptt-device', required=True)
    p.add_argument('--tx-device', required=True)
    p.add_argument('--rx-device', required=True)
    return p


def dial_offset(args):
    from hfmodem.contact.slots import wide_offset
    return wide_offset() if args.monitor_wide else 0


def validate(args):
    from hfmodem.core import identification
    if getattr(args, 'pskreporter', False):
        from hfmodem.station.pskreporter import Config
        Config(args.mycall, args.loc or '', enabled=True)
    args.identity = identification.resolve(args.mycall, args.idcall)
    args.identity.announce()
    for field in ('name', 'qth', 'loc', 'services'):
        value = getattr(args, field, None)
        if value is not None:
            emit_tag('S' if field == 'services' else field.upper(), value)


def chat_server():
    """A host that reports CQ frames whatever its listen state, so the session can
    monitor with auto-answer off; its port bytes are otherwise stock's."""
    from hfmodem.kestrel.host.server import HostSession, VaraServer

    class ReportingSession(HostSession):
        def modem_cqframe(self, call, bw, sn):
            self.modem_bitrate(1, 18, False)
            self.modem_snr(sn)
            self._event('command', f'CQFRAME {call} {bw}')

    class ChatServer(VaraServer):
        session_class = ReportingSession

    return ChatServer


class Host:
    def __init__(self, transport, *, runtime_factory=None, server_factory=None):
        from kestrel_host import AudioHostRuntime
        self.transport = transport
        self.runtime = (runtime_factory or AudioHostRuntime)(transport)
        self.server_factory = server_factory or chat_server()
        self.server = None
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            while not self.stop.is_set():
                self.runtime.step()
        except BaseException as error:
            self.error = error
            self.stop.set()

    def _host_log(self, kind, detail):
        if kind == 'rejected':
            self.transport.log(f'host command rejected: {detail}')

    def __enter__(self):
        try:
            self.server = self.server_factory(host='127.0.0.1', cmd_port=0, data_port=0, kiss_port=0,
                                              modem_factory=self.runtime.modem_factory,
                                              log=self._host_log)
            self.transport.start()
            self.server.start_background()
            self.thread.start()
        except BaseException:
            self.close()
            raise
        return self

    def check(self):
        if self.error is not None:
            raise HostError(f'Kestrel host stopped: {self.error}') from self.error

    def close(self):
        self.stop.set()
        try:
            if self.server is not None:
                self.server.stop()
        finally:
            if self.thread.ident is not None:
                self.thread.join(timeout=10)
            try:
                if self.thread.is_alive():
                    raise HostError('Kestrel runtime did not stop')
                if self.runtime.modem:
                    self.runtime.modem.stop()
                    self.runtime.modem.poll()
            finally:
                self.transport.stop()

    def __exit__(self, *exc):
        self.close()


def tune(rig, dial, bandwidth, say=print):
    from hfmodem.core.band import QSY_TOLERANCE_HZ
    from hfmodem.core.rigs import RIGS, mode_passband, resolve_rig
    if rig.keyed:
        raise HostError('Cannot QSY a keyed transmitter')
    try:
        rig.set_freq(dial)
        rig.set_mode('PKTUSB', mode_passband(RIGS[resolve_rig()], 3000))
        state = rig.state()
    except OSError as error:
        raise HostError(f'CAT failed during QSY to {dial} Hz: {error}') from error
    if (state['frequency_hz'] is None or abs(state['frequency_hz'] - dial) > QSY_TOLERANCE_HZ
            or state['mode'] != 'PKTUSB' or (state['passband_hz'] or 0) < 500):
        raise HostError(f'QSY did not take: requested {dial} Hz PKTUSB, read {state}')
    if state['passband_hz'] < bandwidth:
        say(f'warning: the rig passband reads {state["passband_hz"]} Hz, narrower than the '
            f'{bandwidth} Hz session')


def change_band(rig, dial, bandwidth, say=print, settle=1.0):
    """The rig to another band's calling frequency. The FTDX10's band stack brings back
    that band's own mode, width and roofing, so it gets the arm's whole `tune` again,
    read back as the arm reads it; another rig gets a QSY. The ATU's match memory is not
    on the CAT and a tune keys a carrier, so the operator is asked to press TUNE."""
    from hfmodem.core import ftdx10
    from hfmodem.core.band import QSY_TOLERANCE_HZ
    from hfmodem.core.rigs import resolve_rig
    if resolve_rig() != 'ftdx10':
        return tune(rig, dial, bandwidth, say)
    if rig.keyed:
        raise HostError('Cannot change band on a keyed transmitter')
    try:
        reply = ftdx10.normal(rig.batch(ftdx10.session_batch(dial)))
        got = ftdx10.session_frequency(reply, daemon=True)
        if not got or abs(int(got) - dial) > QSY_TOLERANCE_HZ:
            raise HostError(f'band change did not take: asked {dial} Hz, the rig reads {got or "no dial"}')
        say(f'  dial {got} verified')
        if not ftdx10.session_report(reply, daemon=True, say=say):
            raise HostError('band change setup unverified; transmitter remains disarmed')
        # A reply inside the batch that set the filters answers before the rig has settled.
        time.sleep(settle)
        if not ftdx10.readback_report(ftdx10.normal(rig.batch(ftdx10.READBACK)), daemon=True, say=say):
            raise HostError('band change filters unverified; transmitter remains disarmed')
        on = ftdx10.tuner(ftdx10.normal(rig.batch(ftdx10.TUNER)))
    except OSError as error:
        raise HostError(f'CAT failed during the band change to {dial} Hz: {error}') from error
    if on is not False:
        say(f'tuner {"on" if on else "unread"}: if the ATU holds no match for {dial / 1e6:.3f} MHz, '
            'press TUNE on the rig; the session keys no tune carrier')
    return dict(verified=True, tuner_on=on, atu_match_known=False)


class Log(io.TextIOWrapper):
    def fileno(self):
        # input() edits a line only when sys.stdout claims fd 1, which now holds the terminal.
        return 1


class Terminal:
    """The operator's line, edited by readline in a reader thread; screen lines print above it.

    The console pipes stdout to arm.log, so that pipe moves behind sys.stdout and the
    terminal takes fds 0 and 1, where readline reads and draws."""
    PROMPT = '> '

    def __init__(self, held, history):
        tty = os.open('/dev/tty', os.O_RDWR)
        held.callback(os.close, tty)
        held.callback(termios.tcsetattr, tty, termios.TCSANOW, termios.tcgetattr(tty))
        sys.stdout.flush()
        log = os.dup(1)
        os.dup2(tty, 0)
        os.dup2(tty, 1)
        sys.stdout = Log(io.FileIO(log, 'w'), line_buffering=True)
        self.screen = held.enter_context(open(os.dup(tty), 'w', buffering=1))
        self.history, self.lines, self.editing = history, queue.Queue(), False
        import readline  # libedit settles at import whether fds 0 and 1 are a terminal
        self.edit = readline
        if readline.backend == 'editline':
            readline.parse_and_bind('bind ^R em-inc-search-prev')
        readline.set_history_length(1000)
        try:
            readline.read_history_file(history)
        except OSError:
            pass
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            self.editing = True
            try:
                line = input(self.PROMPT) + '\n'
            except EOFError:
                line = ''
            self.editing = False
            if line.strip():
                self.edit.write_history_file(self.history)
            self.lines.put(line)
            if not line:
                return

    def readline(self):
        """The next operator line, '' at EOF, or None when nothing waits."""
        try:
            return self.lines.get_nowait()
        except queue.Empty:
            return None

    def write(self, text):
        if self.editing:
            # libedit keeps the line just sent, newline and all, as its buffer until the next keystroke.
            typed = self.edit.get_line_buffer()
            text = '\r\x1b[K' + text + self.PROMPT + ('' if typed.endswith('\n') else typed)
        self.screen.write(text)

    def flush(self):
        self.screen.flush()


class Diagnostics:
    """Bounded runtime-to-terminal mailbox; audio threads never write the screen."""
    def __init__(self):
        self.events = queue.Queue(maxsize=256)
        self.recovery = queue.Queue(maxsize=16)
        self.dropped = 0

    def put(self, kind, **details):
        if kind == 'link_recovery':
            # Keep recovery/closing notices despite a flood of verbose frames.
            # If the operator thread stalls, retain the newest bounded history.
            try:
                self.recovery.put_nowait((kind, details))
            except queue.Full:
                try:
                    self.recovery.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.recovery.put_nowait((kind, details))
                except queue.Full:
                    pass
            return
        try:
            self.events.put_nowait((kind, details))
        except queue.Full:
            self.dropped += 1

    def drain(self, session):
        for _ in range(16):
            try:
                kind, details = self.recovery.get_nowait()
            except queue.Empty:
                break
            session.diagnostic(kind, **details)
        if self.dropped:
            dropped, self.dropped = self.dropped, 0
            session.diagnostic('dropped', count=dropped)
        for _ in range(256):
            try:
                kind, details = self.events.get_nowait()
            except queue.Empty:
                break
            session.diagnostic(kind, **details)


def terminal(session, tty):
    """Operator lines go to the session between its ticks; EOF or /quit ends it."""
    failure = None
    try:
        while True:
            session.tick()
            line = tty.readline()
            if line is None:
                continue
            if not line or not session.command(line):
                break
    except KeyboardInterrupt:
        pass
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            session.quit()
        except BaseException as cleanup_error:
            if failure is not None and cleanup_error is not failure:
                # Preserve the initiating fault while retaining the close failure
                # in the traceback. The outer runtime/keyer teardown still runs.
                raise failure from cleanup_error
            raise


def received_retune_adapter(runtime, client, tuner, unkeyed, *, completed=None):
    """Bridge cooperative application work to the sole modem owner's boundary."""
    def retune_steps(target, generation):
        modem = runtime.modem
        if modem is None:
            raise HostError('Receiver retune has no modem owner')
        valid = lambda: (client.link_generation == generation
                         and client.connected and not client.closed)
        try:
            request = runtime.request_retune(
                lambda: tuner(target), modem=modem, station=modem.station,
                receive_cursor=modem.received_cursor, unkeyed=unkeyed, valid=valid)
        except (ValueError, queue.Full) as error:
            raise HostError(f'Receiver retune cannot be queued: {error}') from error
        try:
            while not request.done:
                if not valid():
                    raise HostError('Link changed during receiver retune')
                yield .05
            if request.error is not None:
                raise HostError(f'Receiver retune failed: {request.error}') from request.error
            if request.cancelled:
                raise HostError('Receiver retune cancelled')
        finally:
            request.cancel()
            # cancel() waits for an executing owner transaction. A completed
            # move remains real even if /stop arrives before the next app tick.
            if request.done and not request.cancelled and request.error is None:
                if completed is not None:
                    completed(target)
    return retune_steps


def close_reporter(reporter, say, emit):
    before = reporter.status()['submitted']
    result = reporter.close()
    emit('pskreporter_shutdown', **result)
    say(f"PSKReporter shutdown: {result['submitted'] - before} final UDP reports submitted; "
        f"{result['unsent_at_shutdown']} unsent, {result['inflight_at_shutdown']} in flight/unknown. "
        "Server receipt is unconfirmed.")


def run(args, rig, admission):
    import numpy as np
    from hfmodem.contact.slots import SlotWatch
    from hfmodem.core import config
    from kestrel_connect import AudioVaraIO

    offset = dial_offset(args)

    with ExitStack() as held:
        from hfmodem.contact.bbs import HostCatalog
        from hfmodem.contact.bbs_server import configured_directory
        root = configured_directory('' if args.no_bbs else args.bbs_directory)
        catalog = None
        if root:
            catalog = HostCatalog()
            held.callback(catalog.close)
            catalog.enable(root)
        tty = Terminal(held, args.session.parent.parent / 'history')
        transcript = held.enter_context((args.session / 'host.jsonl').open('w'))
        events = held.enter_context((args.session / 'events.jsonl').open('a'))
        contacts = held.enter_context((args.session / 'contacts.jsonl').open('a'))
        contact_log = ContactLog(args.session)
        saves = []
        def check_saves():
            for saved in saves:
                try:
                    saved.result(timeout=10)
                except Exception as error:
                    print(f'Contact recording save failed: {error}', flush=True)
        # Registered before Host: its recorder drains and closes first.
        held.callback(check_saves)
        session = None
        reporter = None
        if getattr(args, 'pskreporter', False):
            from hfmodem.station.pskreporter import Config, Reporter
            reporter = Reporter(Config(args.mycall, args.loc, enabled=True))
            held.callback(lambda: close_reporter(reporter, say, emit))

        def say(text):
            tty.write(text + '\n')
            tty.flush()

        def emit(kind, **record):
            if kind == 'heard' and reporter is not None:
                reporter.accept_heard(record, live=True)
            line = json.dumps(dict(kind=kind, utc=utc(), **record), ensure_ascii=False) + '\n'
            for stream in (events, contacts) if kind == 'contact' else (events,):
                stream.write(line)
                stream.flush()
            if kind == 'contact':
                recorder = getattr(audio, 'rec', None)
                native = contact_log.update(record, sample=getattr(recorder, 'position', None), offset_hz=offset)
                if record.get('ended_utc') and getattr(audio, 'rec', None) is not None:
                    saves.append(audio.rec.after_pending(lambda native=native: contact_log.finish(native)))
            if kind == 'band':
                # arm.log's reader, the console, moves the KiwiSDR witness on this line.
                print(f'BAND {record["band_m"]} m {record["dial_hz"]} Hz', flush=True)

        def event(item):
            transcript.write(json.dumps(dict(utc=utc(), kind=item.kind, raw=item.raw),
                                        ensure_ascii=False) + '\n')
            transcript.flush()
            if session is not None:
                session.handle(item)

        audio = AudioVaraIO(args.tx_device, args.rx_device, config.tx_drive(), rig,
                            record=args.session / 'receive.wav', offset_hz=offset)
        diagnostics = Diagnostics()
        audio.diagnostic = diagnostics.put
        calling = CALLING_HZ[args.band]
        slots = SlotWatch(calling, calling, offset=offset, mycall=args.mycall)
        held.callback(slots.close)
        # Our own transmission comes back through the codec; the watch hears silence then.
        audio.taps.append(lambda frame: slots.put(np.zeros_like(frame) if rig.keyed else frame))
        held.callback(rig.shutdown)
        host = held.enter_context(Host(audio))
        if offset:
            say(f'monitor-wide: the dial sits {abs(offset)} Hz {"below" if offset < 0 else "above"} '
                f'each channel; we key and hear the channel at {1500 - offset} Hz audio')

        def closing():
            if (rec := getattr(audio, 'rec', None)) is not None:
                say(f'receive audio is being stored in the background (log: {rec.store_log})')
        held.callback(closing)
        admission.opened()
        owner = int(os.environ['HFMODEM_OWNER_FD']) if 'HFMODEM_OWNER_FD' in os.environ else None

        def ready(sockets, timeout):
            host.check()
            if session is not None:
                diagnostics.drain(session)
            while not rig.meter_reports.empty():
                emit('tx_meters', **rig.meter_reports.get())
            if rig.retired:
                raise HostError('Station session was retired')
            readers = sockets + ([owner] if owner is not None else [])
            readable = select.select(readers, [], [], timeout)[0]
            if owner is not None and owner in readable:
                if not os.read(owner, 64):
                    rig.shutdown()
                    raise HostError('Console owner exited')
                readable.remove(owner)
            return readable

        with Client.open(command_port=host.server.cmd_port, data_port=host.server.data_port,
                         kiss_port=getattr(host.server, "kiss_port", None), event=event, ready=ready) as client:
            client.initialize(args.mycall, args.bw)
            flow = Flow(client, lambda hz: tune(rig, hz + offset, args.bw, say), band=args.band,
                        profile={field.upper(): getattr(args, field) for field in ('name', 'qth', 'loc')},
                        services=args.services)
            flow.received_retune_steps = received_retune_adapter(
                host.runtime, client, flow.tuner,
                lambda: not rig.keyed and rig.line_state() is False,
                completed=flow.received_tuned)
            flow.dial = CALLING_HZ[args.band]
            session = Session(client, flow, say=say, emit=emit, arm=lambda on: setattr(rig, 'armed', on),
                              line_state=rig.line_state, bandwidth=args.bw, reporter=reporter,
                              caller=lambda: getattr(host.runtime.modem, 'attempt_caller', None),
                              mailbox=Mailbox(args.session.parent.parent / 'vmail'), slots=slots,
                              band_tuner=lambda hz: change_band(rig, hz + offset, args.bw, say),
                              bbs_catalog=catalog)
            if catalog is not None:
                say(f'BBS hosting enabled: {root}; advertised in advanced beacons and opening reports')
            say('PSKReporter ' + ('on: validated CQ/beacon decodes, five-minute batches; '
                                    '/pskreporter status or off' if reporter is not None else 'off'))
            say('commands: ' + ', '.join(Session.COMMANDS) + '; anything else is sent when linked')
            try:
                session.monitor()
                terminal(session, tty)
            except Exception as error:
                say(f'error: {error}')
                raise
        return 0


def main(argv=None):
    args = parser().parse_args(argv)
    validate(args)
    from hfmodem.core.claim import inherited
    from hfmodem.core.ptt import arming_refusal
    from vara_rig_bridge import Rig, _install_unkey_handlers

    if refusal := arming_refusal(args.ptt_device):
        raise ValueError(refusal)
    admission = inherited(9, launch=True)
    rig = None
    try:
        rig = Rig(args.rigctld, False, print, ptt_device=args.ptt_device, line_ptt=True)
        _install_unkey_handlers(rig)
        model, dial = rig.identify()
        if args.expect_model.lower() not in model.lower() or not dial.isdigit():
            raise HostError(f'Rig identification failed: {model!r}, dial {dial!r}')
        tune(rig, CALLING_HZ[args.band] + dial_offset(args), args.bw)
        return run(args, rig, admission)
    finally:
        try:
            if rig is not None:
                rig.shutdown()
                print('PTT VERDICT: ' + ('keying line handed back LOW' if rig.released_low else
                      'CANNOT PROVE PTT IS DOWN'), flush=True)
        finally:
            admission.failed()


if __name__ == '__main__':
    raise SystemExit(main())
