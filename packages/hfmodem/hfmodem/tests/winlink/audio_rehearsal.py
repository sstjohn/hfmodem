# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Transport adapters for shared mail tests; no sound devices, serial or RF.

PACTOR scripts the remote decision to grant an upgrade. Its entry and all
subsequent mail are audio. ARDOP uses two modems; VARA uses its scripted peer.
These are integration rehearsals, not commercial-modem interoperability proofs.
"""
import numpy as np

from hfmodem.shrike import pactor1, placement, rxfront, spec
from hfmodem.shrike.arq import IRS, State
from hfmodem.tests.shrike.test_p3_offer import cs_event
from hfmodem.tests.shrike.test_qso import AudioLink, AudioSide
from hfmodem.winlink import B2FSession, MailClient


class _EntrySide(AudioSide):
    def send_entry_packet(self, sl, payload, status, *, acquire=False):
        self._tx("P3 ENTRY", placement.link_packet(
            sl, payload, status, swapped=False, acquire=acquire,
            flush=placement.ENTRY_FLUSH))
        return placement.SPEED_PATHS[sl].crc_bytes - 3


class _GrantedLink(AudioLink):
    SIDE = _EntrySide

    def _receive(self, heard, dst):
        # The blind reader can report the same SL1 field at two alignments.
        # The live _SessionRx admits one packet per receive cycle; preserve
        # that rule here instead of manufacturing two answers to one burst.
        events, packets = [], set()
        for ev in super()._receive(heard, dst):
            if ev.kind == "packet":
                key = (ev.protocol, ev.packet, ev.breakin)
                if key in packets:
                    continue
                packets.add(key)
            events.append(ev)
        return events


class Pactor3AudioRehearsal:
    """MailExchange transport starting at a scripted, accepted P1 announcement.

    The peer's grant is decoded from audio, then its receiver must decode our SL1
    entry before it may start its mailbox. From there every byte, acknowledgment
    and changeover passes through the PACTOR waveform/decoder path. Wall-clock
    deadlines, propagation and a commercial gateway's acquisition are not modeled.
    """
    def __init__(self, mycall, gateway, rms_outbox=None):
        self.link = _GrantedLink(mycall, gateway, verbose=False)
        self.host, self.far = self.link.a, self.link.b
        self.host.p1_grant_only = True
        self.rms = B2FSession(gateway, role="answering", target=mycall,
                              outbox=rms_outbox or [])
        self.client = None
        self.entry_decoded = False

    @property
    def send(self):
        return self.host.arq.on_host_data

    def attach(self, client):
        self.client = client

    def connect(self):
        a, b = self.host, self.far
        # Only the connect answer and remote acceptance of the announcement are
        # scripted. No application bytes are allowed to bypass the audio link.
        a.arq.on_host_connect(a.mycall, b.mycall)
        a.on_rx_event(cs_event(pactor1.CS_SPEED))
        a.tick()
        assert a.arq.tx_seq == 1
        self.link.a_io.pending.clear()
        b.arq.role, b.arq.dxcall = IRS, a.mycall
        b.arq._enter_connected()
        b.arq._expected_seq = 2  # It has accepted announcement counter 1.
        b.protocol = spec.Protocol.PACTOR3  # The granting receiver is ready.
        grant = np.asarray(pactor1.control_signal(pactor1.CS_59A), np.float32)
        grants = [e for e in rxfront.decode_events(self.link._channel(grant))
                  if e.kind == "unassigned" and e.spare == pactor1.CS_59A]
        assert len(grants) == 1
        a.on_rx_event(grants[0])
        a.tick()
        assert a.protocol == spec.Protocol.PACTOR3 and a.arq.entry_pending
        self.link._carry(self.link.a_io, b, "entry")
        self.entry_decoded = self.link.p3_rx == [1] and b.arq.rx_seq == 2
        assert self.entry_decoded, "mail may start only after the SL1 entry decodes"
        a.app = self.client
        b.app = MailClient(self.rms, b.arq.on_host_data)
        a.app.link_up()
        b.app.link_up()
        return self.connected

    def step(self):
        self.host.app_turns()
        self.far.app_turns()
        self.link.exchange(1)

    @property
    def connected(self):
        return self.host.arq.state == State.CONNECTED

    @property
    def stall_note(self):
        return (f"pactor: {self.host.protocol}/{self.host.arq.role} "
                f"peer={self.far.protocol}/{self.far.arq.role}; "
                f"{len(self.link.p3_rx)} decoded P3 packets")

    def disconnect(self):
        if self.connected:
            self.host.arq.on_host_disconnect()
            for _ in range(30):
                self.link.exchange(1)
                if not self.connected:
                    break
        # End a failed rehearsal without leaving virtual workers holding state.
        self.host.arq.on_host_abort()
        self.far.arq.on_host_abort()


def transport_for(protocol, mycall, gateway, messages):
    from hfmodem.station.mail import ArdopLoopback, VaraLoopback
    factory = {"pactor3": Pactor3AudioRehearsal,
               "ardop": ArdopLoopback, "vara": VaraLoopback}[protocol]
    return factory(mycall, gateway, rms_outbox=messages)
