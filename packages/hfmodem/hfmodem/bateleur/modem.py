# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One bateleur modem: a transmit queue, the receivers, and one thread.

The modem knows nothing of the station. It is handed card-rate audio to listen
to, a ``send(audio, rate)`` that asks something else to put a burst on the air
and says whether it went, and the longest burst that something will key. What
it reports goes to an observer: a transmission began or ended on the air,
bytes arrived, a burst went out or was refused.

**Everything happens on the modem's own thread.** The host server's command
handler and the station's receive loop are two more threads that would
otherwise reach the receivers; commands and audio cross in on one queue and
nothing else crosses at all — the arrangement ``station.air`` settled on for
sabir, for the same reason.

**A message longer than one key-up is split.** The arbiter refuses a burst
past ``max_key_s`` rather than truncating it, so the modem cuts the message
into pieces that each fit, and sends each as a whole transmission — preamble,
data, end-of-message. A receiver hands the pieces' bytes on in order, so the
far end's application sees one stream. Nothing re-sends a piece that is lost:
these waveforms are one-way, and retransmission is the layer above.
"""
from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass

from .worker import ModemWorker

import numpy as np

from hfmodem.core.occupied import keyed_hz

from .phy import ROLLOFF

from . import charframe, fsk, phy, mil110, s4285, s4529, s4539, stream, tone16, tone39, wbhf
from .spec import FS, MODES110, MODES4285


WAVEFORMS = {"110": mil110, "4285": s4285, "4539": s4539, "39": tone39, "16": tone16, "FSK": fsk, "4529": s4529, "WBHF": wbhf}
MODE_TABLES = {"110": MODES110, "4285": MODES4285, "4539": s4539.MODES4539, "WBHF": wbhf.MODES_WBHF}


#: The audio the station declares bateleur to occupy with the rig's 3 kHz
#: filter, which an FSK mode's tones and a 4529 sub-carrier's pulse must
#: keep inside.
FSK_SPAN_HZ = keyed_hz("bateleur", "3000")


#: The standards that carry a plain bit stream, over which start-stop
#: characters may be sent (``charframe``). FSK and the 39-tone mode frame
#: characters their own way.
ASYNC_STANDARDS = ("110", "4285", "4529", "4539", "WBHF", "16")


class ModemError(ValueError):
    """A request the modem will not act on, with the reason."""


@dataclass(frozen=True)
class Mode:
    standard: str               # "110" | "4285" | "4529" | "4539" | "WBHF" | "39" | "16" | "FSK"
    name: str                   # a key of MODES110, MODES4285, s4539.MODES4539
    #                             or wbhf.MODES_WBHF,
    #                             a 39-tone mode's name ("600/99/A10.8"),
    #                             a 16-tone rate ("2400"), or an FSK mode
    #                             ("75W", "75V3D"; see fsk.mode)

    @classmethod
    def parse(cls, standard: str, name: str) -> "Mode":
        standard = standard.strip()
        if standard == "39":
            try:
                return cls("39", tone39.mode(name.strip()).name)
            except ValueError as exc:
                raise ModemError(f"no 39-tone mode {name.strip()!r}: {exc}") from None
        if standard == "4529":
            try:
                m = s4529.mode(name)
            except ValueError as exc:
                raise ModemError(str(exc)) from None
            # About its sub-carrier, the pulse's whole width, 1200 Bd and
            # the roll-off: it must lie in the span the station declares.
            half = s4529.BAUD * (1 + ROLLOFF) / 2
            if m.carrier - half < FSK_SPAN_HZ[0] or m.carrier + half > FSK_SPAN_HZ[1]:
                raise ModemError(f"4529 about {m.carrier:g} Hz keys {m.carrier - half:.0f}-"
                                 f"{m.carrier + half:.0f} Hz, outside the "
                                 f"{FSK_SPAN_HZ[0]:.0f}-{FSK_SPAN_HZ[1]:.0f} Hz the "
                                 f"station declares")
            return cls("4529", m.full_name)
        if standard.upper() == "FSK":
            try:
                m = fsk.mode(name)
            except ValueError as exc:
                raise ModemError(str(exc)) from None
            # The station declares one span for bateleur to its emission
            # gate; sub-carriers 1, 15 and 16 of the multi-channel plan lie
            # outside it, keying sidebands and all.
            lo, hi = min(t for p in m.pairs() for t in p), max(t for p in m.pairs() for t in p)
            if lo - m.baud < FSK_SPAN_HZ[0] or hi + m.baud > FSK_SPAN_HZ[1]:
                raise ModemError(f"FSK {m.name} keys {lo - m.baud:.0f}-{hi + m.baud:.0f} Hz, "
                                 f"outside the {FSK_SPAN_HZ[0]:.0f}-{FSK_SPAN_HZ[1]:.0f} Hz "
                                 f"the station declares")
            return cls("FSK", m.name)
        if standard == "16":
            try:
                return cls("16", str(tone16.mode(name)))
            except ValueError as exc:
                raise ModemError(f"no 16-tone mode {name.strip()!r}: {exc}") from None
        standard = standard.upper()
        table = MODE_TABLES.get(standard)
        if table is None:
            raise ModemError(f"no standard {standard!r}: 110, 4285, 4529, 4539, WBHF, 39, 16 "
                             "or FSK")
        name = name.strip().upper()
        if name not in table:
            raise ModemError(f"no {standard} mode {name!r}; one of {', '.join(table)}")
        return cls(standard, name)

    def render(self, data: bytes, order: str = "lsb", trim: bool = True) -> np.ndarray:
        kwargs = {"trim": trim} if self.standard in MODE_TABLES else {}
        return WAVEFORMS[self.standard].transmit(data, self.name, order=order, **kwargs)

    @property
    def rate(self):
        if self.standard == "16":
            return int(self.name)
        if self.standard == "39":
            return tone39.mode(self.name).rate
        if self.standard == "FSK":
            return fsk.mode(self.name).baud
        if self.standard == "4529":
            return MODES4285[s4529.mode(self.name).as_4285].rate // 2
        if self.standard == "WBHF":
            return wbhf.MODES_WBHF[self.name].w.rate
        return MODE_TABLES[self.standard][self.name].rate

    def seconds(self, n_bytes: int) -> float:
        """Rendered duration, including every filter tail sample."""
        if self.standard == "FSK":
            mode = fsk.mode(self.name)
            return fsk.bit_count(mode, n_bytes) / mode.baud
        if self.standard == "4529":
            symbols = s4285.symbol_count(MODES4285[s4529.mode(self.name).as_4285], n_bytes)
            return 2 * phy.sample_count(symbols) / FS
        wave = WAVEFORMS[self.standard]
        if self.standard in MODE_TABLES:
            symbols = wave.symbol_count(MODE_TABLES[self.standard][self.name], n_bytes)
            return phy.sample_count(symbols) / FS
        count = wave.element_count(self.name, n_bytes) * wave.T
        if self.standard == "39":
            count += tone39._BANDPASS.size - 1
        return count / FS

    @staticmethod
    def receive(standard, audio, name=None, **kwargs):
        if standard in ("4285", "4529", "39", "16", "FSK"):
            return WAVEFORMS[standard].receive(audio, name, **kwargs)
        return WAVEFORMS[standard].receive(audio, **kwargs)

    def most_bytes(self, max_s: float) -> int:
        """The largest message that keys for at most ``max_s``."""
        lo, hi = 0, 1
        while self.seconds(hi) <= max_s and hi < 1 << 24:
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if self.seconds(mid) <= max_s else (lo, mid)
        return lo


class Observer:
    """What the modem reports. Every method is called on the modem's thread."""

    def rx_started(self, standard: str, mode: str, freq_hz: float) -> None: ...
    def rx_data(self, data: bytes) -> None: ...
    def rx_ended(self, complete: bool, snr_db: float, why: str) -> None: ...
    def tx_started(self, mode: Mode, n_bytes: int, seconds: float) -> None: ...
    def tx_ended(self, sent: bool, why: str) -> None: ...


class BateleurModem(ModemWorker):
    worker_name = "bateleur"

    def __init__(self, send, *, card_rate: int, max_key_s: float,
                 observer: Observer | None = None, log=None, trim: bool = True) -> None:
        self._send = send
        self.max_key_s = max_key_s
        self.card_rate = card_rate
        self._abort_generation = 0
        self._pending_tx = 0
        self._tx_lock = threading.Lock()
        self.trim = trim
        self.observer = observer or Observer()
        self._log = log or (lambda *_: None)
        self.mode = Mode("110", "600S")
        self.bit_order = "lsb"
        #: None: the bytes themselves go on the air (110B C.5.4.1.2); or a
        #: ``charframe.CharFormat``: start-stop words (C.5.4.1.1).
        self.framing: charframe.CharFormat | None = None
        self._deframers: dict = {}
        self.zero_interleave = False
        self.listen = True
        self._fe = stream.FrontEnd(card_rate)
        self._rx110 = stream.Receiver110(bit_order=self.bit_order)
        self._rx4539 = stream.Receiver4539(bit_order=self.bit_order)
        self._rxwbhf = stream.ReceiverWbhf(bit_order=self.bit_order)
        self._rx4285: stream.Receiver4285 | None = None
        #: Always listening: its preamble is its own, though it does not
        #: say the mode. With a 39-tone mode set it decodes in that mode at
        #: once; otherwise it finds the mode from the data.
        self._rx39 = stream.Receiver39(None, bit_order=self.bit_order)
        self._want39: str | None = None
        #: Up while a 16-tone rate is set: the signal does not say the rate.
        self._rx16: stream.Receiver16 | None = None
        #: Up only when an operator selects an FSK mode.
        self._rx4529: stream.Receiver4529 | None = None
        self._rxfsk: stream.ReceiverFsk | None = None
        self._init_worker()
        self.transmitting = False
        self._rx_owner = None
        #: Counted, so a test can assert on them.
        self.sent = self.refused = self.received = 0

    # -- from any thread ---------------------------------------------------

    def feed(self, audio: np.ndarray) -> None:
        """Card-rate samples, every one once, in order."""
        self._feed((np.asarray(audio, dtype=np.float32),))

    def stop(self, timeout: float = 5.0) -> bool:
        self.abort()
        return super().stop(timeout)

    # -- on the modem's thread ---------------------------------------------

    @property
    def receiving(self):
        return any(rx is not None and rx.receiving for rx in
                   (self._rx110, self._rx4539, self._rxwbhf, self._rx4285, self._rx39, self._rx16, self._rxfsk, self._rx4529))

    def reset_receive(self):
        self._fe = stream.FrontEnd(self.card_rate)
        self._rebuild()

    def abort(self):
        with self._tx_lock:
            self._abort_generation += 1

    def queue_transmit(self, data):
        with self._tx_lock:
            if self._pending_tx + len(data) > 65536:
                raise ModemError("transmit queue exceeds 65536 bytes")
            self._pending_tx += len(data)
            generation = self._abort_generation

        def send():
            try:
                self.transmit(data, generation=generation)
            finally:
                with self._tx_lock:
                    self._pending_tx -= len(data)
        try:
            self.post(send)
        except Exception:
            with self._tx_lock:
                self._pending_tx -= len(data)
            raise

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        if mode.standard == "4285":
            if self._rx4285 is None or self._rx4285.mode != MODES4285[mode.name]:
                self._rx4285 = stream.Receiver4285(mode.name, bit_order=self.bit_order)
        # A 39-tone mode set is the one its receiver decodes in, from the
        # next transmission: one it is in the middle of goes on as it began.
        self._want39 = mode.name if mode.standard == "39" else None
        if mode.standard == "16":
            if self._rx16 is None or self._rx16.rate != int(mode.name):
                self._rx16 = stream.Receiver16(mode.name, bit_order=self.bit_order)
        elif self._rx16 is not None and not self._rx16.receiving:
            self._rx16 = None
        if mode.standard == "4529":
            if self._rx4529 is None or self._rx4529.mode != mode.name:
                self._rx4529 = stream.Receiver4529(mode.name, bit_order=self.bit_order)
        elif self._rx4529 is not None and not self._rx4529.receiving:
            self._rx4529 = None
        if mode.standard == "FSK":
            if self._rxfsk is None or self._rxfsk.mode != mode.name:
                self._rxfsk = stream.ReceiverFsk(mode.name, bit_order=self.bit_order)
        elif self._rxfsk is not None and not self._rxfsk.receiving:
            self._rxfsk = None
        # 188-110, 4539 and appendix D announce themselves, so their
        # receivers stay up whatever is set.

    def set_bit_order(self, order: str) -> None:
        order = order.strip().lower()
        if order not in ("lsb", "msb"):
            raise ModemError(f"bit order is LSB or MSB, not {order!r}")
        self.bit_order = order
        self._rebuild()

    def set_framing(self, text: str) -> None:
        """"SYNC" for the bytes themselves, or a character format ("8N1")
        for start-stop words, both ways."""
        t = text.strip().upper()
        if t == "SYNC":
            self.framing = None
        else:
            try:
                self.framing = charframe.CharFormat.parse(t)
            except ValueError as exc:
                raise ModemError(str(exc)) from None
        self._deframers = {}

    def _async(self, standard: str) -> bool:
        return self.framing is not None and standard in ASYNC_STANDARDS

    def set_zero_interleave(self, on: bool) -> None:
        """Take a short 188-110 preamble to mean the interleaver is bypassed.
        The standard leaves this to be agreed with the far end beforehand."""
        self.zero_interleave = bool(on)
        self._rebuild()

    def _rebuild(self) -> None:
        if self._rx_owner is not None:
            self.observer.rx_ended(False, float("nan"), "receiver reset")
            self._rx_owner = None
        self._rx110 = stream.Receiver110(bit_order=self.bit_order,
                                         zero_interleave=self.zero_interleave)
        self._rx4539 = stream.Receiver4539(bit_order=self.bit_order)
        self._rxwbhf = stream.ReceiverWbhf(bit_order=self.bit_order)
        if self._rx4285 is not None:
            self._rx4285 = stream.Receiver4285(self._rx4285.mode, bit_order=self.bit_order)
        self._rx39 = stream.Receiver39(self._rx39.mode, bit_order=self.bit_order)
        if self._rx16 is not None:
            self._rx16 = stream.Receiver16(self._rx16.rate, bit_order=self.bit_order)
        if self._rxfsk is not None:
            self._rxfsk = stream.ReceiverFsk(self._rxfsk.mode, bit_order=self.bit_order)
        if self._rx4529 is not None:
            self._rx4529 = stream.Receiver4529(self._rx4529.mode, bit_order=self.bit_order)

    def transmit(self, data: bytes, *, generation=None) -> None:
        """Send ``data`` in the current mode, in as many key-ups as it takes."""
        generation = self._abort_generation if generation is None else generation
        mode = self.mode
        room = mode.most_bytes(self.max_key_s)
        if self.framing is not None and self._async(mode.standard):
            room = room * 8 // self.framing.word_bits
        if room <= 0:
            self.refused += 1
            self.observer.tx_ended(False, f"{mode.standard} {mode.name} cannot key "
                                          f"even an empty message inside max_key_s "
                                          f"{self.max_key_s:.0f} s")
            return
        at = 0
        while at < len(data) or (at == 0 and not data):
            if generation != self._abort_generation:
                self.observer.tx_ended(False, "aborted")
                return
            piece = data[at:at + room]
            sent = piece
            if self.framing is not None and self._async(mode.standard):
                sent = charframe.pack(charframe.frame(piece, self.framing), self.bit_order)
            audio = mode.render(sent, self.bit_order, self.trim)
            secs = audio.size / FS
            if secs > self.max_key_s:
                if room <= 1 or not piece:
                    self.refused += 1
                    self.observer.tx_ended(False, "rendered message exceeds max_key_s")
                    return
                room = max(1, min(len(piece) - 1, room // 2))
                continue
            self.observer.tx_started(mode, len(piece), secs)
            self.transmitting = True
            try:
                why = self._send(audio, FS)
            finally:
                self.transmitting = False
            if why:
                self.refused += 1
                self.observer.tx_ended(False, why)
                return
            self.sent += 1
            self.observer.tx_ended(True, "")
            at += len(piece)
            if not piece:
                break

    def _audio(self, samples: np.ndarray) -> None:
        bb = self._fe.push(samples)
        if not self.listen:
            return
        have = self._rx39.mode.name if self._rx39.mode is not None else None
        if have != self._want39 and not self._rx39.receiving:
            self._rx39 = stream.Receiver39(self._want39, bit_order=self.bit_order)
        chunk_owner = self._rx_owner
        receivers = (
            ("_rx110", bb, lambda r: stream.Receiver110(
                bit_order=self.bit_order, zero_interleave=self.zero_interleave)),
            ("_rx4539", bb, lambda r: stream.Receiver4539(bit_order=self.bit_order)),
            ("_rxwbhf", bb, lambda r: stream.ReceiverWbhf(bit_order=self.bit_order)),
            ("_rx39", self._fe.audio, lambda r: stream.Receiver39(r.mode, bit_order=self.bit_order)),
            ("_rx4285", bb, lambda r: stream.Receiver4285(r.mode, bit_order=self.bit_order)),
            ("_rx16", self._fe.audio, lambda r: stream.Receiver16(r.rate, bit_order=self.bit_order)),
            ("_rxfsk", self._fe.audio, lambda r: stream.ReceiverFsk(r.mode, bit_order=self.bit_order)),
            ("_rx4529", self._fe.audio, lambda r: stream.Receiver4529(r.mode, bit_order=self.bit_order)),
        )
        for name, samples, rebuild in receivers:
            receiver = getattr(self, name)
            if receiver is None:
                continue
            try:
                events = receiver.push(samples)
            except Exception:
                setattr(self, name, rebuild(receiver))
                self._log(f"bateleur: {name[3:]} receiver restarted after chunk failure\n"
                          + traceback.format_exc())
                if self._rx_owner == name:
                    self.observer.rx_ended(False, float("nan"), "receiver reset")
                    self._rx_owner = None
                continue
            for e in events:
                if isinstance(e, stream.Started) and (chunk_owner is None or
                                                    (chunk_owner == name and self._rx_owner is None)):
                    chunk_owner = self._rx_owner = name
                    src = name[3:].upper()
                    if self._async(src):
                        self._deframers[src] = charframe.Deframer(self.framing)
                    self.observer.rx_started(e.standard, e.mode, e.freq_hz)
                elif name == self._rx_owner:
                    if isinstance(e, stream.Data):
                        data = e.data
                        src = name[3:].upper()
                        if self._async(src):
                            deframer = self._deframers.setdefault(src, charframe.Deframer(self.framing))
                            data = deframer.push(charframe.unpack(data, self.bit_order))
                        if data:
                            self.observer.rx_data(data)
                    elif isinstance(e, stream.Ended):
                        self.received += 1
                        self.observer.rx_ended(e.complete, e.snr_db, e.why)
                        self._rx_owner = None
