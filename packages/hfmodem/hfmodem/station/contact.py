# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An opaque station contact reservation; no mutex spans modem processing."""
import threading


class ContactOwner:
    def __init__(self, arbiter, protocol):
        self.arbiter, self.protocol = arbiter, protocol
        self._token = None
        self._lock = threading.Lock()

    @property
    def receive_boundary(self):
        return self.arbiter.receive_boundary if self.arbiter is not None else (0, 0)

    @property
    def token(self):
        with self._lock:
            return self._token

    def admit(self, *, receive_epoch=None, authority=None, fence=None):
        if self.arbiter is None:
            return False
        with self._lock:
            binding = {} if authority is None and fence is None else dict(authority=authority, fence=fence)
            token = self.arbiter.claim_contact(self.protocol, self._token,
                                               receive_epoch=receive_epoch, **binding)
            if token is None:
                return False
            self._token = token
            return True

    def release_if(self, idle):
        with self._lock:
            if self._token is not None and idle():
                self.arbiter.release_contact(self.protocol, self._token)
                self._token = None

    def close(self):
        self.release_if(lambda: True)
