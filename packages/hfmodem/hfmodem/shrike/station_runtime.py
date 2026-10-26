# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A serial PTC owner over the station's native PACTOR sample clock.

The worker owns the parser, ARQ and existing onair driver. The station thread
only hands it indexed capture. Neither this module nor the native producer may
open a card, set a dial, or operate PTT independently of the station arbiter.
This runtime supports Normal calls with PACTOR-1/3 through POSIX PTYs or
authenticated serial-supervisor streams serviced on the same native owner.
"""
from __future__ import annotations

from pathlib import Path
import math
import os
import queue
import tempfile
import threading
import time
from types import SimpleNamespace

from hfmodem.core.identification import CallIdentity
from hfmodem.core.operations import LeaseTimeout

from .arq import State, CONTYPE_ACCEPTS, parse_connect_arg
from .station_io import StationInput


class _AttachmentEnded(ConnectionAbortedError):
    pass


class StationPtcRuntime:
    """One owner, sequential fresh attachments, and fresh contact capture."""

    def __init__(self, link, *, drive_reference_mv=3000):
        if (isinstance(drive_reference_mv, bool)
                or not isinstance(drive_reference_mv, (int, float))
                or not math.isfinite(drive_reference_mv) or drive_reference_mv <= 0):
            raise ValueError('PACTOR drive reference must be finite and positive')
        self.link = link
        station = link.station
        if station.rig is None or station.arbiter is None:
            raise ValueError('PTC station runtime requires the station transmit arbiter')
        if station.arbiter.identity is None or station.arbiter.identity.mode != 'cw':
            raise ValueError('PTC station runtime requires configured CW identification')
        from hfmodem.station.contact import ContactOwner
        self.contact = ContactOwner(station.arbiter, 'pactor')
        self.drive_reference_mv = drive_reference_mv
        self._stop = threading.Event()
        self._serve = threading.Event()
        self._ready = threading.Event()
        self._port_lock = threading.Lock()
        self._thread = None
        self._port = None
        self._endpoint = None
        self._busy = False
        self.failure = None
        self.host = self.tx = None
        self.endpoint = ''
        self.recovery_root = None
        self._retained = 0
        self._managed = False
        self._attachments = queue.Queue(maxsize=1)

    @property
    def active(self):
        host = self.host
        return self._busy or (host is not None and (getattr(host._owner, 'connect_pending', False)
                    or host.arq.state in (State.CONNECTING, State.CONNECTED, State.DISCONNECTING)))

    def start(self):
        if self._thread is not None:
            raise RuntimeError('PTC station owner already started')
        if self._stop.is_set():
            raise RuntimeError('PTC station owner stopped')
        self._thread = threading.Thread(target=self._run, name='pactor-ptc-owner', daemon=True)
        self._thread.start()

    def start_host(self, endpoint='', *, managed=False):
        if managed and endpoint:
            raise ValueError('managed PTC cannot expose a serial alias')
        if not managed and os.name != 'posix':
            raise OSError('PTC station requires a qualified virtual COM bridge on Windows')
        if self._serve.is_set() or self._stop.is_set():
            raise RuntimeError('PTC station endpoint already started or stopped')
        self.recovery_root = Path(tempfile.mkdtemp(prefix='hfmodem-ptc-'))
        self._managed = managed
        self.endpoint = '' if managed else str(endpoint or self.recovery_root / 'ptc')
        if self._thread is None:
            self.start()
        self._serve.set()
        if not self._ready.wait(5):
            self.stop()
            raise TimeoutError('PTC station endpoint startup timed out')
        if self.failure is not None:
            self.stop()
            raise RuntimeError(f'PTC station endpoint failed: {self.failure}') from self.failure
        print(f'PTC recovery directory: {self.recovery_root}', flush=True)

    def serve_stream(self, stream):
        """Broker thread: hand off a stream and join its actual owner retirement.

        No parser, ARQ, capture or radio operation runs on this thread. The
        broker keeps its lease until the native owner has retired the contact,
        closed capture and reset the attachment, including on station failure.
        """
        if not self._managed or self._stop.is_set() or self.failure is not None:
            raise RuntimeError('PTC stream owner unavailable')
        request = SimpleNamespace(stream=stream, done=threading.Event(), failure=None)
        try:
            self._attachments.put_nowait(request)
        except queue.Full:
            raise RuntimeError('PTC stream attachment already pending') from None
        while not request.done.wait(.05):
            if self._thread is None or not self._thread.is_alive():
                raise RuntimeError('PTC stream owner exited before retirement')
        if request.failure is not None:
            raise RuntimeError('PTC stream attachment failed retirement') from request.failure

    def pump(self):
        """Station thread: publication and lane draining share one boundary."""
        with self._port_lock:
            chunks = self.link.lane.poll()
            if self._port is not None:
                for chunk in chunks:
                    self._port.feed(chunk)
        if self.failure is not None:
            raise RuntimeError(f'PTC station owner failed: {self.failure}') from self.failure
        return chunks

    def stop(self):
        self._stop.set()
        self._serve.set()
        with self._port_lock:
            if self._port is not None:
                self._port.cancel()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(5)
            if thread.is_alive():
                raise LeaseTimeout('PTC owner did not retire; station containment required')

    def _other_active(self):
        owner = self.link.station.arbiter.contact_protocol
        return (owner is not None and owner != 'pactor') or any(name != 'pactor' and getattr(lane, 'active', False)
                   for name, lane in self.link.station.lanes.items())

    def _scannable(self):
        host = self.host
        return (not self._stop.is_set() and not self._busy and not self._other_active()
                and host is not None and not host._owner.contact_active
                and host.arq.state in (State.DISCONNECTED, State.LISTENING)
                and host.contact_ready() and not self.link.station.arbiter.identity.due)

    def _poll_host(self):
        if self._stop.is_set():
            raise _AttachmentEnded('PTC station stopping')
        for _ in range(2):
            if not self._endpoint.service():
                raise _AttachmentEnded('PTC application detached')

    def _service(self):
        # Idle service has the same prohibition on nested transmission as a
        # serial poll suspended inside native capture/keyed output.
        return self.link.station.arbiter.run_owner_poll(self._poll_host)

    def _validate_connect(self, own, remote):
        if self._other_active():
            raise ValueError('STATION CHANNEL OWNER BUSY')
        if self.link.station.arbiter.identity.due:
            raise ValueError('STATION IDENTIFICATION PENDING')
        self.tx.validate_host_answer()
        if (self.host.settings.get('CWID') == '4 2'
                and own.upper() != self.link.station.arbiter.identity.mycall):
            raise ValueError('PTC CWID requires the configured station CW identity')
        if remote.strip():
            _, variant = parse_connect_arg(remote)
        elif self.host.arq.last_connect is not None:
            _, variant = self.host.arq.last_connect
        else:
            return                      # ARQ supplies its normal no-last-call error
        if variant != 'normal':
            raise ValueError('STATION PTC SUPPORTS NORMAL CALLS ONLY')
        if self._retained >= 16:
            raise ValueError('PTC RECOVERY STORAGE FULL')
        if not self.contact.admit():
            raise ValueError('STATION CHANNEL OWNER BUSY')

    def _validate_setting(self, name, arg):
        if ((name == 'CWID' and arg == '4 2')
                or name == 'MYCALL' and self.host.settings.get('CWID') == '4 2'):
            call = arg if name == 'MYCALL' else self.host.mycall
            if call.upper() != self.link.station.arbiter.identity.mycall:
                raise ValueError('PTC CWID requires the configured station CW identity')
        if name == 'MYLEVEL' and int(arg) == 2:
            raise ValueError('STATION PTC SUPPORTS MYLEVEL 1 OR 3')
        if name == 'TXDELAY':
            from .onair import ANSWER_D_S
            if int(arg) * .005 + .010 > ANSWER_D_S:
                raise ValueError('TXDelay exceeds the native call-answer turnaround')

    def _retire_port(self):
        with self._port_lock:
            port, self._port = self._port, None
        if port is not None:
            port.close()

    def _new_transport(self, outdir):
        from . import onair
        from .ptc import PtcHost

        self._retire_port()
        station = self.link.station
        # An interrupted output can publish a retirement floor ahead of the
        # card. Keep servicing host cancellation, without inventing capture.
        while station.arbiter.transmit_history().capture_floor > station.audio.samples:
            if self._endpoint is not None and not self._endpoint.closed:
                self._service()
            if self._stop.wait(.01):
                raise _AttachmentEnded('PTC station stopping')
        with self._port_lock:
            port = StationInput(self.link, owner_poll=self._poll_host,
                responding=lambda: bool(self.host and self.host.arq.answering),
                scannable=self._scannable)
            self._port = port
        def independent_ptt(_on):
            raise RuntimeError('native PACTOR cannot operate station PTT')
        def key_failure():
            if station.rig.retired:
                raise ConnectionAbortedError('station rig retired')
        proxy = SimpleNamespace(ptt=independent_ptt, key_failure=key_failure)
        tx = onair.RadioTx(proxy, transmit=True, outdir=outdir,
            drive=station.arbiter.drive, settle=station.arbiter.settle_s,
            max_key=station.rig.max_key_s, drive_reference_mv=self.drive_reference_mv,
            record_audio=False)
        tx.live, tx.p4_probe, tx.defer_p3_cs = port, None, True
        # Use the documented wire order and the native receiver's default.
        # Reversal swaps CS1/CS2 at zero errors; the CLI's MSB experiment is
        # not the station's control-word contract.
        tx.p1_ack_msb = False
        tx.validate_host_connect = self._validate_connect
        tx.validate_host_setting = self._validate_setting
        if self.host is None:
            self.host = PtcHost(tx, mycall=station.cfg.station.mycall)
            self.host.enable_owner(defer_connect=True)
            self.host._store('MYLEVEL', '3')
            self.host._store('MAXERROR', '70')
            self.host._store('MODE', '2')
        else:
            self.host.peer = tx
            tx.attach(self.host)
            for name in ('FSKAMPL', 'PSKAMPL', 'TXDELAY', 'CWSPEED', 'CSDELAY'):
                if name in self.host.settings:
                    tx.set_host_control(name, int(self.host.settings[name]))
        self.tx = tx
        if tx.settle + .010 > onair.ANSWER_D_S:
            raise ValueError('station PTT settle exceeds the native call-answer turnaround')
        tx.p1_status_bits45 = 0 if self.host.settings['MYLEVEL'] == '1' else 3
        tx.prepare_p1_setup()
        tx.sessrx = onair._SessionRx(self.host)
        from . import telemetry
        telemetry.bind(tx, tx.sessrx)
        return port

    def _contact(self, called=None):
        from . import onair

        if not self._answer_timing_ready():
            self._retire_port()
            return
        if not self.contact.admit(receive_epoch=(self._port.channel_epoch if called is not None else None)):
            self._retire_port()
            return
        if self._retained >= 16:
            raise BufferError('PTC recovery storage full; recover retained transfers before restart')
        outdir = Path(tempfile.mkdtemp(prefix='contact-', dir=self.recovery_root))
        host = self.host
        self._busy = True
        identity = self.link.station.arbiter.identity
        port = self._port
        try:
            with host._owner.contact():
                try:
                    if called is None:
                        port = self._new_transport(outdir)
                    else:
                        self.tx.outdir = outdir
                    tx = self.tx
                    level = int(host.settings['MYLEVEL'])
                    tx.p1_status_bits45 = 0 if level == 1 else 3
                    args = onair.build_parser().parse_args(['--transmit', '--mycall', host.arq.mycall
                        if called is None else host.mycall])
                    args.answer = 1 if called is not None else 0
                    args.p1_status_bits45 = tx.p1_status_bits45
                    args.cwid_optional = False
                    identity.begin()
                    onair._drive_session(args, tx=tx, host=host, sessrx=tx.sessrx,
                        live=port, rig=None, dx=host.arq.dxcall, reply=None, outdir=outdir,
                        identity=CallIdentity(args.mycall, identity.mycall, 'station configuration'),
                        called=called, start_admitted=(None if called is not None else
                            lambda: host._owner.start_connect(self._endpoint)), host_controlled=True)
                finally:
                    if port is not None and port.ever_keyed:
                        arbiter = self.link.station.arbiter
                        history = arbiter.transmit_history()
                        identified = arbiter.last_sent.get('ident') == history.end
                        identity.end(identified=identified)
                        # On failed output, the station's existing ID producer
                        # owns the owed identification. Keep this contact busy
                        # while the main station discharges it; no new RF here.
                        deadline = time.monotonic() + self.link.station.rig.max_key_s + 2
                        while identity.due and not self._stop.is_set():
                            self._service()
                            if time.monotonic() >= deadline:
                                raise TimeoutError('PTC contact identification not retired')
                            self._stop.wait(.01)
        finally:
            self._retire_port()
            self._busy = False
            self.contact.release_if(lambda: not self.active)
            retained = host.arq.transfer_unconfirmed
            if retained is not None:
                # The driver's outer retirement can finalize originals after
                # its ordinary report. Never overwrite earlier evidence.
                if (outdir / 'p3-transfer-unconfirmed.json').exists():
                    if ('host_unconfirmed' in retained
                            and not (outdir / 'p3-transfer-unconfirmed-host.bin').exists()):
                        final = outdir / 'retired'
                        final.mkdir()
                        onair._write_unconfirmed_transfer(final, host.arq)
                else:
                    onair._write_unconfirmed_transfer(outdir, host.arq)
            try:
                outdir.rmdir()          # only empty success directories are removed
            except OSError:
                self._retained += 1    # preserve every file, including operator additions

    def _answer_timing_ready(self):
        try:
            self.tx.validate_host_answer()
            return True
        except ValueError:
            return False

    def _idle(self):
        from . import onair

        host = self.host
        while not self._stop.is_set():
            self._service()
            self.contact.release_if(lambda: not self.active)
            if host.arq.scan_held:
                # Retired scanner capture owns the token until release/expiry.
                self._stop.wait(.01)
                continue
            if (not host.contact_ready() or self._other_active()
                    or self.link.station.arbiter.identity.due
                    or not self._answer_timing_ready()):
                self._retire_port()
                self._stop.wait(.01)
                continue
            if not host._owner.connect_pending and host.arq.state != State.LISTENING:
                self._retire_port()
                self._stop.wait(.01)
                continue
            if self._port is None or self._port.closed or self._port.capture_retired:
                self._new_transport(self.recovery_root)
            if host._owner.connect_pending:
                if not self._endpoint.pending_bytes:
                    self._contact()
                continue
            port = self._port
            settings = dict(host.settings)
            sequence = self.link.station.arbiter.transmit_history().sequence
            channel_epoch = port.channel_epoch
            def eligible():
                return (not self._stop.is_set() and not self._other_active()
                    and not host.arq.scan_held and host.arq.state == State.LISTENING
                    and host.contact_ready() and settings == host.settings
                    and channel_epoch == self.link.station.arbiter.receive_boundary[0]
                    and sequence == self.link.station.arbiter.transmit_history().sequence)
            try:
                called = onair._await_call(port, host.mycall, None,
                    strict=host.arq.cfg.conintegrity == 1, eligible=eligible)
            except ConnectionAbortedError:
                if (sequence == self.link.station.arbiter.transmit_history().sequence
                        and channel_epoch == self.link.station.arbiter.receive_boundary[0]):
                    raise
                called = None           # another lane's output retired idle acquisition
            if (called is not None and called[1].variant == 'Normal'
                    and 'normal' in CONTYPE_ACCEPTS[host.arq.cfg.contype]):
                self._contact(called)
            elif not host.arq.scan_held:
                self._retire_port()

    def _run(self):
        request = None
        try:
            self._serve.wait()
            if self._stop.is_set():
                return
            self._new_transport(self.recovery_root)
            if self._managed:
                self._retire_port()
            while not self._stop.is_set():
                if self._managed:
                    from .stream_endpoint import PtcStreamEndpoint
                    self._ready.set()
                    try:
                        request = self._attachments.get(timeout=.05)
                    except queue.Empty:
                        continue
                    self._new_transport(self.recovery_root)
                    endpoint = PtcStreamEndpoint(self.host, request.stream)
                else:
                    from .serial_endpoint import PtcSerialEndpoint
                    endpoint = PtcSerialEndpoint(self.host, endpoint=self.endpoint)
                if endpoint.open() is False:
                    self._retire_port()
                    request.done.set()
                    request = None
                    continue
                self._endpoint = endpoint
                self._ready.set()
                try:
                    self._idle()
                except _AttachmentEnded:
                    if not self._stop.is_set() and not endpoint.hangup:
                        raise
                finally:
                    self._retire_port()
                    endpoint.close()
                if not self._stop.is_set():
                    self.host.reopen_owner()
                    self._new_transport(self.recovery_root)
                    if self._managed:
                        self._retire_port()
                if request is not None:
                    request.done.set()
                    request = None
        except BaseException as exc:
            if not self._stop.is_set():
                self.failure = exc
        finally:
            try:
                self._retire_port()
            except BaseException as exc:
                self.failure = self.failure or exc
            finally:
                try:
                    if self._endpoint is not None:
                        self._endpoint.close()
                    self.contact.close()
                except BaseException as exc:
                    self.failure = self.failure or exc
                finally:
                    self._ready.set()
                    if request is not None:
                        request.failure = self.failure
                        request.done.set()
                    while not self._attachments.empty():
                        pending = self._attachments.get_nowait()
                        pending.failure = self.failure or RuntimeError('PTC station stopped')
                        pending.done.set()


class PtcServer:
    """Lifecycle facade; serial service stays on the native protocol owner."""
    def __init__(self, runtime, *, endpoint=''):
        self.runtime, self.endpoint = runtime, endpoint

    def start(self):
        self.runtime.start_host(self.endpoint)
        self.endpoint = self.runtime.endpoint

    def stop(self, timeout=5.0):
        self.runtime.stop()
        return True
