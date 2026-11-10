# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Claim-aware live capture adapter; replay and network sources need no claim."""

import os
import subprocess
import time

from hfhost.audio import AudioError, CommandSource


class ClaimedSource(CommandSource):
    def __init__(self, cmd, *, descriptor=None):
        super().__init__(cmd)
        self.descriptor = descriptor
        self._claim = None
        self._owned = False
        self._child = None

    def _spawn(self):
        from hfmodem.core import claim, coexist
        from hfmodem.core.native import NativeChild

        admission = None
        reader = writer = None
        try:
            if self.descriptor is not None:
                source = claim.inherited(self.descriptor)
                self._claim = claim.ClaimDescriptor(os.dup(source.fd))
            elif os.environ.get('RIG_CLAIMED'):
                admission = claim.received(launch=True)
                source = admission.claim
                self._claim = claim.ClaimDescriptor(os.dup(source.fd))
            else:
                self._claim = claim.acquire('monitor', 'capture',
                                            devices=claim.Devices(capture=self.cmd[-1]))
                self._owned = True
                if os.name != 'nt' and self._claim.fd != 9:
                    os.dup2(self._claim.fd, 9)
                    claim.close(self._claim)
                    self._claim = claim.ClaimDescriptor(9)
                source = self._claim
            if admission is None or not admission.active:
                admission = claim.inherited(self._claim, launch=True)
            reader, writer = os.pipe()
            env = dict(os.environ)
            if os.name != 'nt':
                import fcntl
                status_fd = fcntl.fcntl(writer, fcntl.F_DUPFD_CLOEXEC, 10)
                os.close(writer)
                writer = status_fd
                env['HFMODEM_CAPTURE_ACK_FD'] = str(writer)
            launched = claim.spawn(admission if os.name == 'nt' else source, self.cmd, role='python',
                handles={'capture_ack': writer}, stdout=subprocess.PIPE, env=env, start_new_session=True)
            self._proc = proc = launched.process
            try:
                self._child = NativeChild(proc.pid, claim.process_started(proc.pid),
                    None, proc.stdout, None, _process=proc, _win_entry=launched.child)
            finally:
                if launched.child is not None:
                    launched.child.close()
            coexist.event('capture_spawn', child_pid=proc.pid,
                          child_started=self._child.started, argv=self.cmd)
            os.close(writer)
            writer = None
            deadline = time.monotonic() + claim.ADMISSION_TIMEOUT
            from hfmodem.core.pipeio import Reader
            import queue
            status = Reader(reader)
            reply = b''
            try:
                while time.monotonic() < deadline:
                    try:
                        reply += status.queue.get(timeout=.1)
                    except queue.Empty:
                        if status.done.is_set():
                            break
                    if b'\n' in reply:
                        break
            finally:
                if not status.close():
                    reader = None
                    raise AudioError('capture acknowledgement reader did not stop')
            if reply != b'CLAIM ADMITTED\n':
                raise AudioError('CLAIM NOT ADMITTED capture did not acknowledge its open')
            admission.opened()
            return proc
        except BaseException:
            # End the opener before closing either admission reference.
            self.close()
            raise
        finally:
            if admission is not None:
                admission.failed()
            for fd in (reader, writer):
                if fd is not None:
                    os.close(fd)

    def close(self):
        from hfmodem.core import claim

        if self._child is not None:
            child = self._child
            child.terminate()
            if not child.wait_gone(12):
                child.kill()
                if not child.wait_gone(5):
                    raise AudioError(f'termination unresolved for capture pid {child.pid}')
            self._proc.wait(timeout=5)
            self._proc.stdout.close()
            self._child = self._proc = None
        elif self._proc is not None:
            # Identity discovery failed: retain the claim, report uncertainty.
            raise AudioError(f'termination unresolved for capture pid {self._proc.pid}')
        if self._claim is not None:
            descriptor, self._claim = self._claim, None
            (claim.release if self._owned else claim.close)(descriptor)
