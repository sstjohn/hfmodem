# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The loopback may emit only events the real core emits, in the roles it emits them.

`LoopbackModem` exists so the host API can be smoke-tested with no radio, no
channel and no scientific stack — it is the only kestrel target that installs and
runs stdlib-only. The hazard is that it is also the only one creance grades, and a
fabricated wire line is indistinguishable from a real one at the client. No
conformance probe can ever catch a modem that invents its own telemetry; the check
has to happen here, where both implementations are in reach.

Hence the rule, and this file enforcing it. It found four violations when written:
`BITRATE` and `SN` synthesised from constants where the real core has no such hook
at all, `PENDING` raised on the initiator's own outbound connect where the real
core raises it only once it has taken the responder role, and `BUSY OFF` before
`CONNECTED` where the real core clears busy at teardown.

The real core's hook set is read from the source rather than by running it: the
observer bridge in `kestrel/arq/modem.py` is the only place it touches a
`ModemObserver`, and parsing keeps this test stdlib-only, which is the whole
property being protected.
"""
from __future__ import annotations

import ast
import functools
import threading
import time
from pathlib import Path

from hfmodem.kestrel.host.modem_core import LoopbackModem, ModemObserver

_ARQ_BRIDGE = Path(__file__).resolve().parents[2] / "kestrel" / "arq" / "modem.py"


def _real_core_hooks() -> set[str]:
    """Every `modem_*` observer hook the real ARQ core can call."""
    tree = ast.parse(_ARQ_BRIDGE.read_text())
    return {n.func.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr.startswith("modem_")}


class _Recorder(ModemObserver):
    """Records the ordered call log."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._lock = threading.Lock()

    def _record(self, name: str, *args, **kwargs) -> None:
        with self._lock:
            self.calls.append((name, args, kwargs))

    @property
    def names(self) -> set[str]:
        with self._lock:
            return {c[0] for c in self.calls}

    def index(self, name: str) -> int:
        """Position of the first call to `name`, or -1."""
        with self._lock:
            return next((i for i, c in enumerate(self.calls) if c[0] == name), -1)


# Generated over the ABC's whole surface, not just its abstract methods: the two
# hooks the fabrications used (`modem_bitrate`, `modem_snr`) are declared with
# no-op bodies, so a recorder that overrode only the abstract ones would sit
# there quietly recording nothing while the loopback invented telemetry.
for _hook in (n for n in dir(ModemObserver) if n.startswith("modem_")):
    setattr(_Recorder, _hook, functools.partialmethod(_Recorder._record, _hook))
_Recorder.__abstractmethods__ = frozenset()


def _drive_full_session() -> _Recorder:
    """Connect, transfer, disconnect — every path that reports to the host."""
    rec = _Recorder()
    modem = LoopbackModem(bandwidth="500")
    modem.start(rec)
    try:
        modem.connect("W9SSJ", "KO2F")
        deadline = time.monotonic() + 5.0
        while not modem.connected and time.monotonic() < deadline:
            time.sleep(0.005)
        assert modem.connected, "loopback never came up; the recording would be vacuous"

        modem.transmit(b"the quick brown fox" * 4)
        deadline = time.monotonic() + 5.0
        while rec.index("modem_data_received") < 0 and time.monotonic() < deadline:
            time.sleep(0.005)

        modem.disconnect()
        deadline = time.monotonic() + 5.0
        while rec.index("modem_disconnected") < 0 and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        modem.stop()
    return rec


def test_session_is_substantial():
    """Guard against passing vacuously: a modem that reports nothing trivially
    satisfies a subset rule."""
    rec = _drive_full_session()
    assert {"modem_ptt", "modem_connected", "modem_buffer",
            "modem_data_received", "modem_disconnected"} <= rec.names


def test_loopback_invents_no_events():
    """The subset rule. `modem_bitrate` and `modem_snr` are the hooks the real
    core has no path to at all — it carries no gearshift and no SNR estimate."""
    fabricated = _drive_full_session().names - _real_core_hooks()
    assert not fabricated, f"loopback emits what no real link produced: {sorted(fabricated)}"


def test_pending_is_a_responder_event():
    """`fsm.py` sets `role = 'responder'` before calling `pending()`. A modem
    connecting outbound never sees its own connect request arrive, so an
    initiator-side PENDING describes a far end that does not exist."""
    assert _drive_full_session().index("modem_pending") < 0


def test_busy_clears_at_teardown_not_before_connect():
    """The real core clears busy in `_finish_disconnected`, after DISCONNECTED.
    Clearing it on the way up reports a free channel mid-session."""
    rec = _drive_full_session()
    off = [i for i, (n, a, _) in enumerate(rec.calls) if n == "modem_busy" and a == (False,)]
    assert off, "busy is raised and never cleared"
    assert min(off) > rec.index("modem_connected"), "BUSY OFF before CONNECTED"
    assert min(off) > rec.index("modem_disconnected"), "BUSY OFF before teardown"
