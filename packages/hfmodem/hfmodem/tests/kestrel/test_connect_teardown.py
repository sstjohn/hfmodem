# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What `kestrel_connect` puts down first when a session ends badly.

The tool's only teardown was ``io.stop()``: close the input stream, then join a
recorder thread that is allowed ten seconds. An exception escaping ``connect()``
or ``mail_session()`` with PTT still asserted is reachable — a failed unkey
returns False rather than raising — and the key was held through all of that,
with nothing under it but the atexit hook and a 25 s watchdog that belongs to
another file.

Nothing here opens an audio device or a socket: the rig, the audio and the
session are stubs that record the order they were called in.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from hfmodem.tests.kestrel import corpora

kc = corpora.harness("kestrel_connect")


def _handshake(*_a, **_kw):
    """A stand-in for the handshake, not a bare object: `main` reads the answers
    it could not put a name to off it before it prints the verdict."""
    return SimpleNamespace(unattributed=[])


def test_the_transmitter_comes_down_before_the_audio_teardown(monkeypatch):
    order: list[str] = []

    class Rig:
        keyed = True
        retired = False

        def __init__(self, *_a, **_kw):
            pass

        def identify(self):
            return "Yaesu FT-891", "7100000"

        def unkey_hard(self, why):
            order.append("rig")
            return True

    class Io:
        def __init__(self, *_a, **_kw):
            pass

        def stop(self):
            order.append("audio")

    def died(*_a, **_kw):
        raise RuntimeError("the session died with the key up")

    monkeypatch.setattr(kc, "Rig", Rig)
    monkeypatch.setattr(kc, "AudioVaraIO", Io)
    monkeypatch.setattr(kc, "VaraStationHandshake", _handshake)
    monkeypatch.setattr(kc, "attach_mail", lambda *a, **kw: (None, None))
    monkeypatch.setattr(kc, "_install_unkey_handlers", lambda rig: None)
    monkeypatch.setattr(kc, "connect", died)
    monkeypatch.setattr(sys, "argv",
                        ["kestrel_connect.py", "--gateway", "W1AAA",
                         "--mycall", "W9SSJ", "--rigctld", "127.0.0.1:65000"])

    with pytest.raises(RuntimeError):
        kc.main()

    assert order == ["rig", "audio"], (
        "the key was held through the audio teardown" if order == ["audio"]
        else f"teardown ran {order}")


def test_a_rig_already_taken_down_is_not_put_through_the_ladder_again(monkeypatch):
    """A retired rig has had its forced unkey run and reported. Repeating it says
    nothing new and costs the session's last-ten-lines summary the reason the
    attempt ended — which is the line an operator reads first."""
    order: list[str] = []

    class Rig:
        keyed = True
        retired = True

        def __init__(self, *_a, **_kw):
            pass

        def identify(self):
            return "Yaesu FT-891", "7100000"

        def unkey_hard(self, why):
            order.append("rig")
            return True

    class Io:
        def __init__(self, *_a, **_kw):
            pass

        def stop(self):
            order.append("audio")

    monkeypatch.setattr(kc, "Rig", Rig)
    monkeypatch.setattr(kc, "AudioVaraIO", Io)
    monkeypatch.setattr(kc, "VaraStationHandshake", _handshake)
    monkeypatch.setattr(kc, "attach_mail", lambda *a, **kw: (None, None))
    monkeypatch.setattr(kc, "_install_unkey_handlers", lambda rig: None)
    monkeypatch.setattr(kc, "connect", lambda *a, **kw: False)
    monkeypatch.setattr(sys, "argv",
                        ["kestrel_connect.py", "--gateway", "W1AAA",
                         "--mycall", "W9SSJ", "--rigctld", "127.0.0.1:65000"])

    kc.main()

    assert order == ["audio"], f"the ladder ran again: {order}"


def test_the_audio_comes_down_even_when_the_unkey_itself_raises(monkeypatch):
    """The ordering above was true only while `unkey_hard` never raised. It talks
    to rigctld over a socket, and an unkey that raises rather than returning False
    took `io.stop()` with it — leaking the input device and the recorder join the
    transmitter was deliberately sequenced ahead of."""
    order: list[str] = []

    class Rig:
        keyed = True
        retired = False

        def __init__(self, *_a, **_kw):
            pass

        def identify(self):
            return "Yaesu FT-891", "7100000"

        def unkey_hard(self, why):
            order.append("rig")
            raise OSError("rigctld went away with the key up")

    class Io:
        def __init__(self, *_a, **_kw):
            pass

        def stop(self):
            order.append("audio")

    monkeypatch.setattr(kc, "Rig", Rig)
    monkeypatch.setattr(kc, "AudioVaraIO", Io)
    monkeypatch.setattr(kc, "VaraStationHandshake", _handshake)
    monkeypatch.setattr(kc, "attach_mail", lambda *a, **kw: (None, None))
    monkeypatch.setattr(kc, "_install_unkey_handlers", lambda rig: None)
    monkeypatch.setattr(kc, "connect", lambda *a, **kw: False)
    monkeypatch.setattr(sys, "argv",
                        ["kestrel_connect.py", "--gateway", "W1AAA",
                         "--mycall", "W9SSJ", "--rigctld", "127.0.0.1:65000"])

    with pytest.raises(OSError):
        kc.main()

    assert order == ["rig", "audio"], (
        "the raising unkey took the audio teardown with it" if order == ["rig"]
        else f"teardown ran {order}")
