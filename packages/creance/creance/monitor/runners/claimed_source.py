# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Claim-aware live capture adapter; replay and network sources need no claim."""

import fcntl
import os
import select
import subprocess
import time

from hfhost.audio import AudioError, CommandSource


class ClaimedSource(CommandSource):
    def __init__(self, cmd):
        super().__init__(cmd)
        self._claim = None
        self._owned = False
        self._child = None

    def _spawn(self):
        from hfmodem.core import claim, coexist
        from hfmodem.core.native import NativeChild

        admission = None
        reader = writer = None
        try:
            if os.environ.get('RIG_CLAIMED'):
                claim.inherited(9)
                self._claim = claim.ClaimDescriptor(os.dup(9))
            else:
                self._claim = claim.acquire('monitor', 'capture',
                                            devices=claim.Devices(capture=self.cmd[-1]))
                self._owned = True
                if self._claim.fd != 9:
                    os.dup2(self._claim.fd, 9)
                    claim.close(self._claim)
                    self._claim = claim.ClaimDescriptor(9)
            admission = claim.inherited(self._claim, launch=True)
            reader, writer = os.pipe()
            status_fd = fcntl.fcntl(writer, fcntl.F_DUPFD_CLOEXEC, 10)
            os.close(writer)
            writer = status_fd
            env = dict(os.environ, HFMODEM_CAPTURE_ACK_FD=str(writer))
            self._proc = proc = subprocess.Popen(
                self.cmd, stdout=subprocess.PIPE, env=env, start_new_session=True,
                pass_fds=(9, writer))
            self._child = NativeChild(proc.pid, claim.process_started(proc.pid),
                                      None, proc.stdout, None, _process=proc)
            coexist.event('capture_spawn', child_pid=proc.pid,
                          child_started=self._child.started, argv=self.cmd)
            os.close(writer)
            writer = None
            deadline = time.monotonic() + claim.ADMISSION_TIMEOUT
            reply = b''
            while time.monotonic() < deadline:
                if select.select([reader], [], [], max(0, deadline - time.monotonic()))[0]:
                    chunk = os.read(reader, 4096)
                    if not chunk:
                        break
                    reply += chunk
                    if b'\n' in reply:
                        break
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
