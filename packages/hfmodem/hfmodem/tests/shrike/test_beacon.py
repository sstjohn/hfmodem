# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""beacon drives the same rig as onair, so it has to speak the same rig.

beacon keys a transmitter and plays into a sound card, so nothing real-hardware
runs here. The join is checked structurally: every method beacon calls on its
rig must exist on ``ota.Rig``. That is not a proxy for a real test -- it is the
exact defect that shipped. ``rig.start()`` was called on a class that has never
had a ``start``, so ``beacon.main`` raised AttributeError before it could key
anything, and no test noticed because no test imports beacon at all.

``main`` itself is driven whole below, with every hardware seam faked at
beacon's own imports, because the next two defects were behavioural: a missing
keying line surfaced as a traceback where ota converts it to a sentence, and a
rigctl that died mid-run let the remaining keyings count themselves as sent.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from hfmodem.core import audio
from hfmodem.core.ptt import PttError
from hfmodem.shrike import beacon, ota

SOURCE = Path(inspect.getfile(beacon))


def _rig_calls(tree: ast.AST) -> set[str]:
    """Names invoked as ``rig.<name>(...)`` anywhere in the module."""
    return {n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "rig"}


def test_every_rig_method_beacon_calls_exists():
    called = _rig_calls(ast.parse(SOURCE.read_text()))
    assert called, "no rig.<method>() calls found -- has beacon been rewritten?"
    missing = sorted(m for m in called if not hasattr(ota.Rig, m))
    assert not missing, (
        f"beacon calls {missing} on ota.Rig, which does not have them -- "
        f"main() raises AttributeError before it can key. Rig offers: "
        f"{sorted(m for m in vars(ota.Rig) if not m.startswith('_'))}")


def test_the_key_is_always_dropped():
    """The finally: that unkeys is the one line here worth pinning. beacon keys in
    a loop, so an exception between keys must still reach rig.stop()."""
    tree = ast.parse(SOURCE.read_text())
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
    assert any("stop" in _rig_calls(ast.Module(body=t.finalbody, type_ignores=[]))
               for t in tries), "no finally: drops the transmitter"


def test_the_sample_rate_is_the_constant():
    assert "48000" not in SOURCE.read_text(), (
        "a literal sample rate drifts on its own; spec.SAMPLE_RATE is the figure")


# -- main, whole, over fakes --------------------------------------------------------


#: A ramp rather than silence, so "the whole burst reached the air" is a claim
#: the audio can actually fail. Every sample is distinct and the last one is the
#: one a discarded tail loses.
_BURST = np.linspace(-1.0, 1.0, 4800, dtype=np.float32)


class _Rig:
    def __init__(self):
        self.stopped = False
        self.failure = None
        self.serial = "/dev/null"

    def set_mode(self, m):
        return True

    def set_freq(self, hz):
        return True

    def ptt(self, up):
        return True

    def key_failure(self):
        return self.failure

    def stop(self):
        self.stopped = True


def _fake_main(monkeypatch, rig, *, ptt_device=lambda r, s: "/dev/null", count=3,
               play=None):
    """Wire beacon.main to fakes at beacon's own names; returns the _play log.

    ``play`` substitutes the transmit call, which is how the one test that wants
    the real `ota._play` gets it — over a fake card rather than over a stub.
    """
    monkeypatch.setattr(beacon.session, "build_session", lambda *a, **kw: _BURST)
    monkeypatch.setattr(beacon, "find_device", lambda name, kind: 0)
    monkeypatch.setattr(beacon, "ptt_device", ptt_device)
    monkeypatch.setattr(beacon, "Rig", lambda *a, **kw: rig)
    played: list = []
    monkeypatch.setattr(beacon, "_play",
                        play or (lambda *a, **kw: played.append(a)))
    monkeypatch.setattr(sys, "argv",
                        ["beacon", "--serial", "/dev/null", "--mycall", "W9SSJ",
                         "--dxcall", "W9SSJ", "--dial", "7100000",
                         "--count", str(count), "--gap", "0"])
    return played


def test_a_missing_keying_line_is_a_sentence_not_a_traceback(monkeypatch):
    """`ptt_device` raises PttError; ota.run converts it to NOT KEYING because
    the operator's fix is a flag, not a stack trace. beacon keys the same rig
    and owes the same conversion."""
    def refuse(r, s):
        raise PttError("PTT port /dev/cu.usbserial-XXXXB1 is not a character device")
    played = _fake_main(monkeypatch, _Rig(), ptt_device=refuse)
    with pytest.raises(SystemExit) as e:
        beacon.main()
    assert "NOT KEYING" in str(e.value), str(e.value)
    assert not played


def test_a_dead_channel_stops_the_count(monkeypatch):
    """A rigctl that dies mid-run must end the beacon, not let the remaining
    keyings count themselves as sent -- the record-of-intentions defect the
    2026-08-10 headline fix removed from ota.run."""
    rig = _Rig()
    rig.failure = "the rigctl that took the key-down exited (status 2)"
    played = _fake_main(monkeypatch, rig)
    with pytest.raises(SystemExit) as e:
        beacon.main()
    assert "NOT CONFIRMED" in str(e.value), str(e.value)
    assert len(played) == 1, "keyings after a dead channel counted themselves as sent"
    assert rig.stopped, "the finally: must still drop the key"


# -- the whole burst, through a card that is behind ---------------------------------

#: What the device is still holding when a write returns. This station's USB
#: codec reports 3.42 ms low / 12.75 ms high output latency, and the high figure
#: is 612 frames at 48 kHz. A card that emits everything the instant it is handed
#: over cannot tell a transmit path that drains from one that guesses, which is
#: how a beacon shipped for weeks throwing the end of every keying away.
_HELD = int(0.01275 * ota.FS)


class _Out:
    """An output stream that is genuinely behind, so draining it can be got wrong."""

    def __init__(self, card, channels):
        self.card, self.channels = card, channels
        self._played: list[np.ndarray] = []
        self._queued = np.zeros((0, channels), np.float32)
        card.calls.append("open")

    def start(self):
        self.card.calls.append("start")

    def write(self, block):
        n = max(len(block) - _HELD, 0)
        self._played.append(block[:n])
        self._queued = block[n:]

    def stop(self):
        """`Pa_StopStream`: returns only once the queue has reached the air."""
        self.card.calls.append("stop")
        self._played.append(self._queued)
        self._queued = self._queued[:0]

    def close(self):
        """Closing a stream nobody stopped is `paAbort` -- the queue is lost."""
        self.card.calls.append("close")
        self.card.air.append(np.concatenate(self._played)[:, 0] if self._played
                             else np.zeros(0, np.float32))


class _Card:
    """Enough of sounddevice to transmit through, and no convenience call."""

    def __init__(self):
        self.air: list[np.ndarray] = []
        self.calls: list[str] = []

    def OutputStream(self, *, channels=1, **kw):
        return _Out(self, channels)

    def play(self, *a, **kw):
        raise AssertionError(
            "sd.play raises CallbackAbort when the array runs out, which is "
            "PortAudio's paAbort: what the device still holds is discarded "
            "rather than played. Transmit through core.audio.play_drained.")


def _on_a_card(monkeypatch) -> _Card:
    card = _Card()
    monkeypatch.setitem(sys.modules, "sounddevice", card)
    monkeypatch.setattr(audio, "_WARMED", set())  # the warmup is per process, not per test
    return card


def test_the_beacon_puts_its_whole_burst_on_the_air(monkeypatch):
    """Every sample handed over reaches the card, tail included.

    The beacon exists to be watched on a remote SDR, and it was transmitting
    through `sd.play` -- whose callback aborts the stream when the array runs
    out, so the last of the audio was discarded before the key was even
    considered. Nothing above the sound card could see that: the sample count,
    the keyed duration and the log were all unchanged.
    """
    card = _on_a_card(monkeypatch)
    rig = _Rig()
    _fake_main(monkeypatch, rig, count=1, play=ota._play)
    beacon.main()
    assert np.array_equal(card.air[-1], _BURST), (
        f"{_BURST.size - card.air[-1].size} of {_BURST.size} samples never left "
        f"the card")


def test_the_stream_is_stopped_before_it_is_closed(monkeypatch):
    """`Stream.stop()` is `Pa_StopStream`, which waits for the pending buffers;
    closing without it is `paAbort`, which is the defect under another name."""
    card = _on_a_card(monkeypatch)
    _fake_main(monkeypatch, _Rig(), count=1, play=ota._play)
    beacon.main()
    assert card.calls[-2:] == ["stop", "close"]
