# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import imaplib
import ipaddress
import socket

import pytest


def _loopback(address) -> bool:
    if not isinstance(address, tuple):
        return True  # a Unix socket path
    host = address[0]
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Fail a test that reaches a mail server or any host off this machine, before it connects.

    The failure is raised where the call is made and again at teardown, because a
    warm-up thread or a broad `except` can swallow the first."""
    reached = []

    def refuse(what):
        reached.append(what)
        pytest.fail(f'a test reached {what}')

    for name in ('IMAP4', 'IMAP4_SSL'):
        monkeypatch.setattr(imaplib, name, lambda host='', *a, _name=name, **k: refuse(f'imaplib.{_name}({host!r})'))

    create_connection = socket.create_connection

    def guarded_create_connection(address, *args, **kwargs):
        if not _loopback(address):
            refuse(f'socket.create_connection({address!r})')
        return create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket, 'create_connection', guarded_create_connection)
    for method in ('connect', 'connect_ex'):
        original = getattr(socket.socket, method)

        def guarded(self, address, _original=original, _method=method):
            if not _loopback(address):
                refuse(f'socket.{_method}({address!r})')
            return _original(self, address)

        monkeypatch.setattr(socket.socket, method, guarded)
    yield reached
    if reached:
        pytest.fail(f'a test reached the network: {reached}')
