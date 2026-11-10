# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""End-to-end: two BesraModems over the virtual air.

The full stack — ARQ over real modulate → channel → demodulate — with no radio.
This is the integration gate: a connection, a payload delivered exactly, and a
graceful teardown, driven entirely by audio the two modems render and decode.
"""

from __future__ import annotations


from hfmodem.besra.arq.modem import BesraModem
from hfmodem.besra.host import protocol as P
from hfmodem.besra.host.modem_core import ModemObserver
from hfmodem.besra.sim.air import StepAir
from hfmodem.besra.sim import channel as C


class Recorder(ModemObserver):
    """Captures the modem→host events a real host server would relay."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.states: list[str] = []
        self.connected_to: str | None = None
        self.disconnected = False
        self.rx = bytearray()
        self.status: list[str] = []

    def modem_newstate(self, state): self.states.append(state)
    def modem_connected(self, remote, bw): self.connected_to = remote
    def modem_disconnected(self): self.disconnected = True
    def modem_ptt(self, on): pass
    def modem_buffer(self, n): pass
    def modem_data_received(self, kind, blob): self.rx += blob
    def modem_status(self, text): self.status.append(text)


def _pair(air: StepAir):
    caller = BesraModem(bandwidth=500)
    caller.set_mycall("W9SSJ")
    responder = BesraModem(bandwidth=500)
    responder.set_mycall("K7ABC")
    responder.set_listen(True)
    ca, ra = Recorder("caller"), Recorder("responder")
    caller.start(ca)
    responder.start(ra)
    air.join(caller)
    air.join(responder)
    return caller, responder, ca, ra


def test_connect_transfer_disconnect_clean_channel():
    air = StepAir()
    caller, responder, ca, ra = _pair(air)

    caller.connect("K7ABC")
    air.run(max_time=60)
    assert caller.connected, f"caller state {caller.state}, saw {ca.states}"
    assert responder.connected, f"responder state {responder.state}"
    assert ca.connected_to == "K7ABC" and ra.connected_to == "W9SSJ"
    assert caller.state == P.ArdopState.ISS and responder.state == P.ArdopState.IRS

    caller.transmit(b"hello winlink over ardop")
    air.run(max_time=60)
    assert bytes(ra.rx) == b"hello winlink over ardop", f"got {bytes(ra.rx)!r}"

    caller.disconnect()
    # The IRS answers the DISC repeat, not the first copy, so the air has to be
    # allowed to idle through one repeat interval (`ArqSession._corroborated`).
    air.run(max_time=60, quiet_ticks=80)
    assert ca.disconnected and ra.disconnected
    assert caller.state == P.ArdopState.DISC and responder.state == P.ArdopState.DISC


def test_connect_and_transfer_over_awgn():
    # A moderate-SNR clean-ish channel: the robust FSK data mode should still
    # carry the payload (RS/CRC gate any frame the channel corrupts → retransmit).
    air = StepAir(channel=lambda s: C.add_awgn(s, snr_db=25, seed=7))
    caller, responder, ca, ra = _pair(air)

    caller.connect("K7ABC")
    air.run(max_time=90)
    assert caller.connected and responder.connected

    caller.transmit(b"weak signal work")
    air.run(max_time=90)
    assert bytes(ra.rx) == b"weak signal work"


class Answering(Recorder):
    """A host that answers the data it hears from inside the delivery callback
    — the shape of a B2F mail layer, where every over is written in response to
    the one just read."""

    def __init__(self, name: str, script: dict[bytes, bytes]) -> None:
        super().__init__(name)
        self.modem = None
        self.script = dict(script)

    def modem_data_received(self, kind, blob):
        super().modem_data_received(kind, blob)
        reply = self.script.pop(bytes(blob), None)
        if reply is not None and self.modem is not None:
            self.modem.transmit(reply)


def test_mail_shaped_exchange_three_turnarounds():
    """Request-response over real audio, three link reversals, then a clean
    disconnect: proposal out, answer back, body out, acknowledgement back —
    every payload delivered exactly once, both queues drained."""
    air = StepAir()
    caller = BesraModem(bandwidth=500)
    caller.set_mycall("W9SSJ")
    responder = BesraModem(bandwidth=500)
    responder.set_mycall("K7ABC")
    responder.set_listen(True)
    ca = Answering("caller", {b"FS Y": b"body"})
    ra = Answering("responder", {b"FC proposal": b"FS Y", b"body": b"FF"})
    ca.modem, ra.modem = caller, responder
    caller.start(ca)
    responder.start(ra)
    air.join(caller)
    air.join(responder)

    caller.connect("K7ABC")
    air.run(max_time=60)
    assert caller.connected and responder.connected

    caller.transmit(b"FC proposal")
    air.run(max_time=180)

    assert bytes(ra.rx) == b"FC proposalbody", f"responder got {bytes(ra.rx)!r}"
    assert bytes(ca.rx) == b"FS YFF", f"caller got {bytes(ca.rx)!r}"
    assert caller.queued == 0 and responder.queued == 0

    caller.disconnect()
    air.run(max_time=60)
    assert ca.disconnected and ra.disconnected
    assert caller.state == P.ArdopState.DISC and responder.state == P.ArdopState.DISC


def test_arqbw_reaches_the_next_connect():
    """`ARQBW` pushed after start() governs the next ConReq. The host command
    must land in the live session, not just in a stored number — a call placed
    after `ARQBW 2000MAX` that still goes out as ConReq500 negotiates a quarter
    of the bandwidth the operator asked for, silently."""
    air = StepAir()
    caller = BesraModem(bandwidth=500)
    caller.set_mycall("W9SSJ")
    responder = BesraModem(bandwidth=2000)
    responder.set_mycall("K7ABC")
    responder.set_listen(True)
    ca, ra = Recorder("caller"), Recorder("responder")
    caller.start(ca)
    responder.start(ra)
    air.join(caller)
    air.join(responder)

    caller.set_bandwidth(2000, False)
    caller.connect("K7ABC")
    air.run(max_time=60)

    assert caller.connected and responder.connected
    assert any("SESSION BW = 2000" in s for s in ca.status), ca.status
