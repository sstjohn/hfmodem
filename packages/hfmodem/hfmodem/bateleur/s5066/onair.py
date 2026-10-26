# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A STANAG 5066 node on the bateleur modem.

The node is the modem's only user while it runs: it is the modem's
observer, its receivers' events are what the node hears, and what the node
has to send the modem sends. The node keeps no clock, so a ticker posts a
step onto the modem's thread every ``step_s``; everything the node does
happens there, the modem's rule for everything else too.

**Timing.** The EOT the node writes and the budget it fills come from the
mode itself (``Mode.seconds``), so a long interleaver's block quantization
and flush are counted as the modem keys them. Where the data starts inside
that is less exact (half the overhead is taken as the preamble); the EOT
is rounded up in any case, and a far end that waits a little long loses
only time.

**Rates.** A data rate change (C.6.4) retunes the modem: to the 188-110,
STANAG 4539 or MIL-STD-188-110C appendix D mode of that rate and
interleaver, which the far end's receivers find by themselves; those are
also the rates this node accepts.
Under STANAG 4285 and the 39-tone modes, whose receivers must be told the
mode beforehand, the node takes only the rate it is at.
"""
from __future__ import annotations

import threading
import time

from hfmodem.core.hostio import join_threads

from ..modem import BateleurModem, Mode, ModemError, Observer
from ..spec import BAUD, MODES110, MODES4285, PREAMBLE_BLOCK_SYMBOLS
from .. import s4539, wbhf
from .node import Node, Timing

_SUFFIX = {
    "110": {"none": "Z", "short": "S", "long": "L"},
    "4285": {"none": "U", "short": "S", "long": "L"},
    "4539": {"short": "S", "long": "L"},
    "WBHF": {"short": "S", "long": "L"},
}
_TABLES = {"110": MODES110, "4285": MODES4285, "4539": s4539.MODES4539,
           "WBHF": wbhf.MODES_WBHF}
#: Receivers that find the mode from the signal.
SELF_DESCRIBING = ("110", "4539", "WBHF")


def rate_of(mode: Mode) -> tuple[int, str]:
    """(bit/s, interleaver) as a DRC names them."""
    digits = ""
    for ch in mode.name:
        if not ch.isdigit():
            break
        digits += ch
    rate = int(digits)
    if mode.standard == "39":
        return rate, "short"
    if mode.standard in ("16", "FSK"):
        return rate, "none"
    if mode.standard == "4529":
        return rate, {"L": "long", "S": "short"}.get(mode.name[len(digits)], "none")
    tail = mode.name[len(digits):]
    for il, s in _SUFFIX.get(mode.standard, {}).items():
        if tail == s:
            return rate, il
    return rate, "none" if tail in ("N", "U", "Z") else "short"


def mode_for(rate: int, interleave: str, standard: str) -> Mode | None:
    """The mode of ``standard`` sending ``rate`` with ``interleave``, or of a
    self-describing standard that has it; None if there is none."""
    order = [standard] + [s for s in SELF_DESCRIBING if s != standard]
    for std in order:
        if std not in _TABLES:
            continue
        if std != standard and standard not in SELF_DESCRIBING:
            break                               # the far receiver could not follow
        name = f"{rate}{_SUFFIX[std].get(interleave, '?')}"
        if std == "110" and rate == 4800:
            name = "4800N"
        if name in _TABLES[std]:
            return Mode(std, name)
    return None


class ModeTiming(Timing):
    """A modem mode's own timing."""

    def __init__(self, mode: Mode) -> None:
        rate, _ = rate_of(mode)
        overhead = mode.seconds(0)
        if mode.standard == "110":
            m = MODES110[mode.name]
            lead = m.preamble_blocks * PREAMBLE_BLOCK_SYMBOLS / BAUD
        else:
            lead = overhead / 2
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "lead", lead)
        object.__setattr__(self, "tail", max(0.0, overhead - lead))
        object.__setattr__(self, "mode", mode)

    def airtime(self, n: int) -> float:
        return self.mode.seconds(n)

    def most_bytes(self, max_s: float) -> int:
        return self.mode.most_bytes(max_s)

    def at_rate(self, rate: int, interleave: str) -> "Timing":
        m = mode_for(rate, interleave, self.mode.standard)
        return ModeTiming(m) if m is not None else self


class OnAir(Observer):
    """``address``: the node's own; ``mode``: what it sends at until a rate
    change; the rest goes to ``Node``."""

    def __init__(self, modem: BateleurModem, address: int | str, mode: Mode, *,
                 step_s: float = 0.1, clock=time.monotonic, log=None, **node_kw) -> None:
        self.modem = modem
        self.clock = clock
        self.step_s = step_s
        self._log = log or (lambda *_: None)
        node_kw.setdefault("max_tx_s", modem.max_key_s)
        self.node = Node(address, ModeTiming(mode), **node_kw)
        self.node.on_rate = self._rate
        self.node.dts.can_receive = self._can_receive
        self.node.dts.rate = rate_of(mode)
        self._mode = mode
        self._running = False
        self._thread: threading.Thread | None = None
        modem.observer = self
        modem.post(lambda: modem.set_mode(mode))

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._ticker, daemon=True, name="s5066")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        self._running = False
        return join_threads((self._thread,), time.monotonic() + timeout)

    def _ticker(self) -> None:
        busy = threading.Event()
        while self._running:
            time.sleep(self.step_s)
            if busy.is_set():
                continue                        # the modem is still keying the last
            busy.set()

            def step():
                try:
                    self.step()
                finally:
                    busy.clear()
            self.modem.post(step)

    # -- on the modem's thread ---------------------------------------------

    def step(self) -> None:
        got = self.node.poll(self.clock(), defer=True)
        if got is not None:
            data, _secs = got
            try:
                self.modem.transmit(data)
            finally:
                self.node.sent(False)

    def _rate(self, remote: int, rate: int, interleave: str) -> None:
        m = mode_for(rate, interleave, self._mode.standard)
        if m is None:
            self._log(f"s5066: no mode for {rate} bit/s {interleave}")
            return
        self._mode = m
        self.modem.set_mode(m)

    def _can_receive(self, rate: int, interleave: str) -> bool:
        if not self.node._can_receive(rate, interleave):
            return False
        if self._mode.standard in SELF_DESCRIBING:
            return mode_for(rate, interleave, self._mode.standard) is not None
        return (rate, interleave) == rate_of(self._mode)

    # -- the modem's reports -------------------------------------------------

    def rx_started(self, standard, mode, freq_hz) -> None:
        self.node.rx_start(self.clock())

    def rx_data(self, data: bytes) -> None:
        self.node.rx_bytes(self.clock(), data)

    def rx_ended(self, complete, snr_db, why) -> None:
        self.node.rx_end(self.clock(), complete)

    def tx_ended(self, sent, why) -> None:
        self.node.sent(bool(sent), self.clock())
        if not sent:
            self._log(f"s5066: transmission refused: {why}")


def parse_mode(text: str) -> Mode:
    """"110 1200L" and the like, as the host dialect's MODE takes."""
    parts = text.split()
    if len(parts) != 2:
        raise ModemError(f"a mode is a standard and a name, not {text!r}")
    return Mode.parse(*parts)

