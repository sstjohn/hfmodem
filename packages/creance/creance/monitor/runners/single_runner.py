# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One passive receiver: s16le mono 48 kHz in, monitor records out.

Only receive primitives are constructed. No station, host server, ARQ
controller, rig interface or transmit callback is involved.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

FS = 48000


def record(t, protocol, kind, *, text=None, detail='', grade='tentative', **kw):
    return dict(t=float(t), modem=protocol.lower(), protocol=protocol,
                kind=kind, grade=grade, detail=detail,
                **({'text': text} if text is not None else {}), **kw)


def readable(data):
    """Keep arbitrary bytes recoverable without terminal control sequences."""
    return ''.join(chr(b) if 32 <= b < 127 else f'\\x{b:02x}' for b in data)


class Serial:
    def __init__(self, args):
        from hfmodem.bateleur import stream, charframe
        self.stream, self.charframe = stream, charframe
        self.standard, self.order = args.standard, args.bit_order
        self.protocol = {'110': 'MIL110', '4285': 'STANAG4285', '4529': 'STANAG4529',
                         '4539': 'STANAG4539', '39': '39TONE', '16': '16TONE'}.get(
                             self.standard, self.standard)
        self.fe = stream.FrontEnd(FS)
        choices = {
            '110': lambda: stream.Receiver110(bit_order=self.order),
            '4285': lambda: stream.Receiver4285(args.mode, bit_order=self.order),
            '4529': lambda: stream.Receiver4529(args.mode, bit_order=self.order),
            '4539': lambda: stream.Receiver4539(bit_order=self.order),
            'WBHF': lambda: stream.ReceiverWbhf(bit_order=self.order),
            '39': lambda: stream.Receiver39(args.mode, bit_order=self.order),
            '16': lambda: stream.Receiver16(args.mode, bit_order=self.order),
            'FSK': lambda: stream.ReceiverFsk(args.mode, bit_order=self.order),
        }
        self.receiver = choices[self.standard]()
        self.fmt = charframe.CharFormat.parse(args.framing) if args.framing != 'SYNC' else None
        self.deframer = None
        self.samples = 0
        self.name = args.mode or 'auto'

    def events(self, events):
        s = self.stream
        for e in events:
            if isinstance(e, s.Started):
                self.name = e.mode
                self.deframer = self.charframe.Deframer(self.fmt) if self.fmt else None
                yield record(e.at / 9600, self.protocol, 'START', detail=e.mode,
                             offset_hz=e.freq_hz)
            elif isinstance(e, s.Data):
                data = e.data
                if self.deframer:
                    data = bytes(self.deframer.push(self.charframe.unpack(data, self.order)))
                if data:
                    yield record(self.samples / FS, self.protocol, 'DATA', text=readable(data),
                                 detail=f'{self.name}: {len(data)} B; hex {data.hex()}')
            elif isinstance(e, s.Ended):
                yield record(self.samples / FS, self.protocol, 'END',
                             detail=f'{self.name}: {e.why}, S/N {e.snr_db:.1f} dB')

    def push(self, x, *, draining=False):
        if not draining:
            self.samples += len(x)
        bb = self.fe.push(x)
        real = self.standard in ('39', '16', 'FSK', '4529')
        return list(self.events(self.receiver.push(self.fe.audio if real else bb)))

    def flush(self):
        # Let the live receiver finish a burst crossing the last input block.
        out = self.push(np.zeros(4 * FS), draining=True)
        if hasattr(self.receiver, 'flush'):
            out += list(self.events(self.receiver.flush()))
        return out


class Packet:
    def __init__(self, protocol):
        from hfmodem.core.resample import FirDecimator
        self.protocol = protocol
        if protocol == 'rpr':
            from hfmodem.brubru.rx import Receiver
            self.rate = 9600
        elif protocol == 'ale':
            from hfmodem.bateleur.ale.fsk import DualReceiver as Receiver
            self.rate = 8000
        else:
            from hfmodem.bateleur.selcall.fsk import Receiver
            self.rate = 8000
        self.rx, self.decimator = Receiver(), FirDecimator(FS, self.rate)

    def records(self, rows):
        for r in rows:
            if self.protocol == 'rpr':
                from hfmodem.ax25.frame import parse
                for frame in r.frames:
                    decoded = parse(frame)
                    def callsign(address):
                        call = ''.join(chr(b >> 1) for b in address[:6]).strip()
                        ssid = (address[6] >> 1) & 15
                        return call + (f'-{ssid}' if ssid else '')
                    yield record(r.ref / self.rate, 'RPR', 'FRAME', text=readable(decoded.info),
                                 detail=f'for {callsign(decoded.addresses[0])}; '
                                        f'{len(frame)} B; hex {frame.hex()}', grade='confirmed',
                                 station=callsign(decoded.addresses[1]), role='sender', centre_hz=r.centre)
            elif self.protocol == 'selcall':
                c = r.call
                if c:
                    yield record(r.start / self.rate, 'SELCALL', c.kind.upper(),
                                 station=c.frm, role='sender',
                                 detail=f'for {c.to}; {c.text or c.number}',
                                 grade='tentative' if c.erasures or not c.parity_ok else 'confirmed',
                                 offset_hz=r.freq_hz)
            else:
                from hfmodem.bateleur.ale.fsk import Received
                if isinstance(r, Received):
                    f = r.frame
                    yield record(r.start / self.rate, 'ALE', 'FRAME', station=f.caller,
                                 role='sender', detail=f'for {",".join(f.to)}; ' +
                                 '; '.join(f.amd) + f'; {f.words} words, SINAD {r.sinad_db:.1f} dB',
                                 grade='confirmed' if f.complete else 'tentative', offset_hz=r.freq_hz)
                else:
                    yield record(r.end / self.rate, 'ALE', 'WORD', detail=str(r))

    def push(self, x):
        x = self.decimator.push(x)
        rows = self.rx.feed(x) if self.protocol == 'rpr' else self.rx.push(x)
        return list(self.records(rows))

    def flush(self):
        out = self.push(np.zeros(3 * FS))
        finish = getattr(self.rx, 'finish', None) or getattr(self.rx, 'flush', None)
        return out + (list(self.records(finish())) if finish else [])


class Sabir:
    def __init__(self):
        from hfmodem.sabir import monitor
        from hfmodem.sabir.arq.fsm import ArqConfig
        from hfmodem.sabir.arq.modem import LinkModem
        self.monitor = monitor
        self.segmenter, self.decoder = monitor.Segmenter(), LinkModem(ArqConfig())

    def records(self, bursts):
        for start, burst in bursts:
            for e in self.monitor.decode_burst(self.decoder, start, burst):
                yield record(e.t, 'SABIR', e.kind, detail=e.text,
                             grade='tentative' if e.kind == 'DETECT' else 'confirmed')

    def push(self, x):
        return list(self.records(self.segmenter.push(x)))

    def flush(self):
        return list(self.records(self.segmenter.flush()))


class Mod110:
    """Bounded rolling recordings for the channel-bit-only MOD receiver."""
    def __init__(self):
        from hfmodem.bateleur import mod110
        from hfmodem.core.resample import FirDecimator
        self.mod = mod110
        self.decimator = FirDecimator(FS, 9600)
        self.buf = np.empty(0)
        self.base = self.samples = self.due = 0
        self.limit = int((20 + 2 * mod110.MAX_BLOCKS * mod110.BLOCK_BITS *
                          mod110.FRAME / mod110.UNKNOWN / 2400) * 9600)

    def decode(self, final=False):
        r = self.mod.receive(self.buf)
        if r is None or not final and (not (r.end_marker or r.blocks) or r.end_sample > len(self.buf) - 9600):
            return []
        out = [record((self.base + r.start_sample) / 9600, 'MOD110', 'CHANNEL-BITS',
                      detail=f'{len(r.bits)} bits, repeat {r.repeat}; '
                             'payload coding unknown; packed bits hex ' + np.packbits(r.bits).tobytes().hex())]
        cut = min(len(self.buf), r.end_sample)
        self.buf, self.base = self.buf[cut:], self.base + cut
        return out

    def push(self, x):
        y = self.decimator.push(x)
        self.buf = np.concatenate((self.buf, y))
        self.samples += len(y)
        if len(self.buf) > self.limit:
            cut = len(self.buf) - self.limit
            self.buf, self.base = self.buf[cut:], self.base + cut
        if self.samples < self.due:
            return []
        self.due = self.samples + 5 * 9600
        return self.decode()

    def flush(self):
        return self.decode(final=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True)
    p.add_argument('--standard')
    p.add_argument('--mode')
    p.add_argument('--framing', default='SYNC')
    p.add_argument('--bit-order', default='lsb')
    p.add_argument('--audio-centre', type=float, default=700)
    return p


def receiver(args):
    if args.standard:
        return Serial(args)
    if args.protocol in ('rpr', 'ale', 'selcall'):
        return Packet(args.protocol)
    if args.protocol == 'sabir':
        return Sabir()
    if args.protocol == 'mod110':
        return Mod110()
    if args.protocol == 'cw':
        from hfmodem.monitor.cw import CwReader
        return CwReader(FS, args.audio_centre)
    raise ValueError(f'No passive receiver for {args.protocol}')


def main(argv=None):
    rx = receiver(parser().parse_args(argv))
    def emit(rows):
        for row in rows:
            print(json.dumps(row), flush=True)
    carry = b''
    while raw := sys.stdin.buffer.read(9600):
        raw = carry + raw
        size = len(raw) // 2 * 2
        carry = raw[size:]
        emit(rx.push(np.frombuffer(raw[:size], '<i2').astype(float) / 32768))
    if carry:
        raise ValueError('Incomplete s16le sample')
    emit(rx.flush())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
