# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Regression: the transmitter must be released even when transmit fails."""
import pytest

from hfmodem.kestrel.arq import frames as F
from hfmodem.kestrel.arq.fsm import ArqConfig, ArqFsm, State
from hfmodem.kestrel.vara.vara_arq import VaraStationHandshake


class _BoomIO:
    def __init__(self, fail=True):
        self.ptt = []; self.fail = fail
    def key(self, on): self.ptt.append(bool(on))
    def tx(self, samples):
        if self.fail: raise RuntimeError("audio device died mid-burst")
    def log(self, msg): ...
    def pending(self): ...
    def connected(self, *a): ...

def test_ptt_is_released_when_transmit_raises():
    """A failed key-down leaves a transmitter running on a shared band. This is
    the one failure in the handshake with consequences outside the process."""
    io = _BoomIO()
    hs = VaraStationHandshake(["W9SSJ"], io, bw="2300")
    with pytest.raises(RuntimeError):
        hs.originate("K9WRA", "W9SSJ")
    assert io.ptt, "PTT was never touched"
    assert io.ptt[-1] is False, f"transmitter left keyed: PTT sequence {io.ptt}"

def test_ptt_is_released_on_the_normal_path_too():
    io = _BoomIO(fail=False)
    hs = VaraStationHandshake(["W9SSJ"], io, bw="2300")
    hs.originate("K9WRA", "W9SSJ")
    assert io.ptt[-1] is False
    assert io.ptt.count(True) == io.ptt.count(False), f"unbalanced keying: {io.ptt}"


# -- the same guarantee in the ARQ state machine ------------------------------------
#
# `kestrel/arq/fsm.py` had neither half of it: no `try/finally` around either keying
# site, and `_transmit_over` encoded each block *inside* the keyed region, so a slow
# or failing encode held the transmitter. No rig-backed transport is wired to this
# module today, which is why it never bit — and is a reason to close it cheaply, not
# a reason to leave a keying path without the protection every equivalent site in
# `vara_arq.py` already has.

class _FsmIO:
    """Records keying, and can be told to fail in the transmit call."""

    def __init__(self, fail=False):
        self.ptt = []
        self.fail = fail

    def key(self, on): self.ptt.append(bool(on))
    def tx(self, payload, marker, bw="500", level=None):
        if self.fail:
            raise RuntimeError("audio device died mid-over")
    def tx_token(self, name, bw="500"): ...
    def on_busy(self, on): ...
    def on_pending(self, cancel=False): ...
    def on_connected(self, src, dst, bw): ...
    def on_disconnected(self): ...
    def on_buffer(self, n): ...
    def on_deliver(self, blob): ...
    def log(self, msg): ...


def _connected(io) -> ArqFsm:
    fsm = ArqFsm(io, ArqConfig())
    fsm.src, fsm.dst = "W9SSJ", "K9WRA"
    fsm._set_bw("500")
    fsm.role = "initiator"
    fsm.state = State.CONNECTED
    return fsm


def test_a_data_over_that_dies_mid_transmit_still_brings_the_key_down():
    io = _FsmIO(fail=True)
    fsm = _connected(io)
    with pytest.raises(RuntimeError):
        fsm.on_host_data(b"x" * 200)
    assert io.ptt, "PTT was never touched"
    assert io.ptt[-1] is False, f"transmitter left keyed: PTT sequence {io.ptt}"


def test_a_control_burst_that_dies_mid_transmit_still_brings_the_key_down():
    io = _FsmIO(fail=True)
    fsm = _connected(io)
    with pytest.raises(RuntimeError):
        fsm._send_control(F.ACK, seq=0)
    assert io.ptt[-1] is False, f"transmitter left keyed: PTT sequence {io.ptt}"


def test_an_over_that_cannot_be_encoded_never_keys_at_all(monkeypatch):
    """Encoding happens before the key, so a failure there is not an emission.

    The point is not only that the key comes down. Framing, CRC and FEC are work,
    and work between the key and the first sample is unmodulated carrier on a shared
    band — so the only honest place for it is outside the keyed region entirely,
    where an exception costs a burst rather than a transmitter.
    """
    io = _FsmIO()
    fsm = _connected(io)
    fsm.on_host_data(b"x" * 200)
    assert io.ptt.count(True) == 1, f"one over, one key-up: {io.ptt}"

    def explode(*_a, **_kw):
        raise RuntimeError("encode")

    boom = _FsmIO()
    fsm = _connected(boom)
    fsm._blockq.append((0, b"\x00" * fsm._psize))
    monkeypatch.setattr(F, "encode_data", explode)
    with pytest.raises(RuntimeError):
        fsm._maybe_start_over()
    assert boom.ptt == [], f"the transmitter was keyed to encode a block: {boom.ptt}"
