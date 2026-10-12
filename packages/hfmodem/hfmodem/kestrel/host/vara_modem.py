# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Host API binding for the stock-compatible VARA driver.

The audio owner supplies a VaraIO transport and calls ``poll()`` before/after
feeding ``station.on_rx_stream`` (and its normal idle/retry cadence). Host
threads only enqueue commands: radio and driver state stay on the owner thread.
Use this as VaraServer's modem_factory when embedding that driver. The default
host CLI remains the radio-less loopback server.
"""
from __future__ import annotations

import queue

from .modem_core import ModemCore
from ..vara.vara_arq import VaraStationHandshake, VaraState


class _HostIO:
    def __init__(self, modem, transport):
        self.modem = modem
        self.transport = transport

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def key(self, on):
        self.transport.key(on)
        if self.modem.observer:
            self.modem.observer.modem_ptt(on)

    def connected(self, caller, called, bw):
        self.transport.connected(caller, called, bw)
        self.modem._connected = True
        if self.modem.observer:
            self.modem.observer.modem_connected(caller, called, bw)

    def pending(self):
        self.transport.pending()
        if self.modem.observer:
            self.modem.observer.modem_pending()

    def data(self, payload):
        if self.modem.observer:
            self.modem.observer.modem_data_received(payload)
        else:
            self.transport.data(payload)

    def compression_error(self, message):
        if self.modem.observer:
            self.modem.observer.modem_warning(message)
        else:
            self.transport.compression_error(message)


class VaraHostModem(ModemCore):
    """Queue-based binding; ``poll`` belongs to the audio driver's owner thread."""
    def __init__(self, transport, *, bandwidth='2300', high_speed=True):
        self.observer = None
        self._transport = transport
        self._events = queue.Queue()
        self._bandwidth = bandwidth
        self._high_speed = high_speed
        self._compression = 'TEXT'
        self._mycall = []
        self._listen = False
        self._connected = False
        self._closing = False
        self._buffer = None
        self.station = self._new_station()

    def _new_station(self):
        return VaraStationHandshake(self._mycall, _HostIO(self, self._transport),
                                    bw=self._bandwidth, high_speed=self._high_speed,
                                    compression=self._compression)

    def _replace_station(self, *, preserve_data=False):
        pending = self.station.preconnect_data if preserve_data else ()
        self.station = self._new_station()
        self._closing = False
        for payload in pending:
            self.station.send(payload)

    def start(self, observer):
        self.observer = observer

    def stop(self):
        self._events.put(('stop',))

    @property
    def connected(self):
        return self._connected

    def set_mycall(self, calls):
        self._events.put(('calls', list(calls)))

    def set_bandwidth(self, bw):
        self._events.put(('bandwidth', bw))

    def set_compression(self, mode):
        self._events.put(('compression', mode))

    def set_listen(self, on):
        self._events.put(('listen', on))

    def connect(self, src, dst, vias=None):
        self._events.put(('connect', src, dst))

    def transmit(self, blob):
        self._events.put(('data', bytes(blob)))

    def disconnect(self):
        self._events.put(('disconnect',))

    def abort(self):
        self._events.put(('abort',))

    def poll(self):
        """Apply queued host commands and publish driver state on its owner thread."""
        while True:
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            kind = ev[0]
            if kind in ('abort', 'stop'):
                self._replace_station()
                if self._connected and self.observer:
                    self.observer.modem_disconnected()
                self._connected = False
                if kind == 'stop':
                    self.observer = None
                elif self._listen:
                    self.station.listen()
            elif kind == 'calls':
                self._mycall = ev[1]
                self.station.mycalls = list(self._mycall)
            elif kind == 'compression':
                self.station.set_compression(ev[1])
                self._compression = ev[1]
            elif kind == 'bandwidth':
                self._bandwidth = ev[1]  # next connection, never mutate live PHY
            elif kind == 'listen':
                self._listen = ev[1]
                if self.station.state == VaraState.DISCONNECTED:
                    self._replace_station(preserve_data=True)
                self.station.listen(ev[1])
            elif kind == 'connect':
                # Host writes before CONNECT belong to this connection. Preserve
                # their original bytes, not provisional framing from another BW.
                self._replace_station(preserve_data=True)
                self.station.originate(ev[2], ev[1])
            elif kind == 'data':
                self.station.send(ev[1])
            elif kind == 'disconnect':
                self._closing = False
                if self.station.state in (VaraState.CONNECTED, VaraState.DISCONNECTING):
                    if self.station.role == 'initiator':
                        self._closing = True
                    elif self.observer:
                        # Do not promise a close the protocol driver cannot key.
                        self.observer.modem_disconnect_refused(
                            'WRONG DISCONNECT: called-station close is not implemented')
                else:
                    # Defensive DISCONNECT cancels an attempt, never a future link.
                    self._replace_station()
                    if self._listen:
                        self.station.listen()
                    if self.observer:
                        self.observer.modem_disconnected()
        if self._closing and not self.station.data_pending:
            self.station.disconnect()
        live = self.station.state in (VaraState.CONNECTED, VaraState.DISCONNECTING)
        if self._connected and not live:
            self._connected = False
            self._closing = False
            if self.observer:
                self.observer.modem_disconnected()
            if self._listen:
                self._replace_station()
                self.station.listen()
        depth = self.station.pending_bytes
        if depth != self._buffer:
            self._buffer = depth
            if self.observer:
                self.observer.modem_buffer(depth)
