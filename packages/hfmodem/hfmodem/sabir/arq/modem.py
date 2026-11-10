# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Sample-level link endpoint: the ARQ FSM bound to the two waveform families.

One transmission ("over") is a control burst -- the 8.72 s noncoherent
floor burst or the 0.711 s fast coherent OFDM burst, per the FSM's tier --
optionally followed by the OFDM data body the header describes. A fast
header may carry two blocks: an ACK piggybacked ahead of the DATA header.
All segments are normalised to unit RMS over their active region for consistent
simulation power accounting; a radio must budget peak drive for each waveform's
PAPR. They are pushed to ``outbox``
as one analytic sample stream.

``on_air`` is the receive front-end: try the fast header (one block, then
two -- each length is a protocol constant, and every block self-CRCs, so a
wrong guess just fails), fall back to the floor decode, then demodulate the
body with the gear/loading/codeword-set the header names. The burst need
not begin at sample 0: both tiers report the sample they synced at, so the
same front-end reads a transmitter's own array and a bracket an energy
segmenter cut out of a stream.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace
from hashlib import sha256

from hfmodem.sabir.fec import QCLDPC
from hfmodem.sabir.floor.mfsk import SYM, FLOOR_GEARS, FloorModem
from hfmodem.sabir.phy.modem import (FS, GEARS, GUARD_HEAD, GUARD_TAIL, N_GROUPS,
                                     Phy, carrier_groups)

from . import layout, wire
from .fastctl import FastControl
from .fsm import ArqConfig, ArqFsm
from .profiles import BY_ID

HEADER_SAMPLES = (GUARD_HEAD + GUARD_TAIL
                  + FloorModem().n_symbols(wire.BLOCK_BYTES) * SYM)

# How far into what arrives a burst may begin. A burst handed over in process
# starts at sample 0; one an energy segmenter found in a stream starts behind
# its pre-roll -- four quiet frames, plus the frame the burst opened in
# (``monitor.SEG_PAD``, ``monitor.SEG_FRAME``). The header windows widen by
# this rather than scanning the whole burst, and cannot widen much further: a
# fast header is followed by its OFDM body, whose leading ZC segment completes
# 5632 samples past the header's end and would outbid the header's own
# preamble in the correlator from there on.
PREROLL = 5 * 1024


def _unit_rms(wav: np.ndarray) -> np.ndarray:
    body = wav[GUARD_HEAD : wav.size - GUARD_TAIL]
    return wav / np.sqrt(np.mean(np.abs(body) ** 2))


class LinkModem:
    def __init__(self, cfg: ArqConfig, clock=None, dd: int = 1, session_id_factory=None):
        cfg.control_s = HEADER_SAMPLES / FS
        self.fsm = ArqFsm(self, cfg, clock, session_id_factory=session_id_factory)
        self.dd = dd
        self.floor = FloorModem()
        self._connect_parts = []
        self._connect_remaining = 0
        self.outbox: list[np.ndarray] = []
        self.airtime_s = 0.0
        self.delivered = bytearray()
        self.datagrams: list[bytes] = []
        self.events: list[str] = []
        self._phys: dict[str, Phy] = {}
        self._cw_n = {name: QCLDPC(GEARS[name].code).n for name in GEARS}
        self.fast = FastControl(self._phy("workhorse"))
        self.short_fast = FastControl(self._phy("workhorse"), wire.CONNECTIONLESS_BYTES)
        self.ctrl_sends = {"fast": 0, "floor": 0}

    def _phy(self, name: str) -> Phy:
        if name not in self._phys:
            self._phys[name] = Phy(GEARS[name])
        return self._phys[name]

    def _push(self, wav: np.ndarray) -> float:
        self.outbox.append(wav)
        self.airtime_s += wav.size / FS
        return wav.size / FS

    def _header_wave(self, payload: bytes, fast: bool) -> np.ndarray:
        self.ctrl_sends["fast" if fast else "floor"] += 1
        carrier = self.short_fast if len(payload) == wire.CONNECTIONLESS_BYTES else self.fast
        return _unit_rms(carrier.transmit(payload) if fast
                         else self.floor.transmit(payload))

    # -- ArqIO --------------------------------------------------------------
    def send_control(self, ctrl: wire.Control) -> float:
        if ctrl.type in (wire.CONNECT, wire.CONNECT_ACK):
            self._connect_remaining = ctrl.n_ext
            self._connect_parts = []
            if ctrl.n_ext:
                self._connect_parts = [self._header_wave(ctrl.pack(), False)]
                return (1 + ctrl.n_ext) * HEADER_SAMPLES / FS
        elif ctrl.type == wire.CAPS and self._connect_remaining:
            self._connect_parts.append(self._header_wave(ctrl.pack(), False))
            self._connect_remaining -= 1
            if not self._connect_remaining:
                self._push(np.concatenate(self._connect_parts))
                self._connect_parts = []
            return 0.0
        fast = (self.fsm.bootstrap_fast if ctrl.type in (wire.CONNECT, wire.CONNECT_ACK)
                else self.fsm.ctrl_fast)
        return self._push(self._header_wave(ctrl.pack(), fast))

    def send_beacon(self, beacon: wire.Beacon) -> float:
        """Emit a connectionless presence beacon on the robust floor waveform."""
        return self._push(self._header_wave(beacon.pack(), fast=False))

    def send_datagram(self, data: bytes, profile_name="workhorse", *, fast_header=False):
        """One independently decodable packet. No session and no ACK.

        A caller chooses a receiver-supported profile by an application/net
        contract; there is no fictitious negotiation on a one-way channel.
        """
        from .profiles import PROFILES
        profile = replace(PROFILES[profile_name], frame_cws=64)
        if profile.bandwidth > self.fsm.cfg.bandwidth_hz:
            raise ValueError("datagram exceeds configured bandwidth")
        if fast_header and self.fsm.cfg.bandwidth_hz < 1500:
            raise ValueError("fast header exceeds configured bandwidth")
        codec = self.fsm.profile_codec(profile.name)
        if not data or len(data) > 64 * codec.data_bytes:
            raise ValueError("datagram exceeds profile capacity")
        chunks = codec.chunk(data)
        coded = np.stack([codec.encode_cw(ch) for ch in chunks])
        body, ns = self.encode_body(profile, coded)
        hdr = wire.DatagramHeader(profile.id, ns, len(chunks), len(data), sha256(data).digest()[:8])
        return self._push(np.concatenate([self._header_wave(hdr.pack(), fast_header), body]))

    def _datagram(self, hdr, body):
        profile = BY_ID.get(hdr.gear)
        if profile is None or profile.bandwidth > self.fsm.cfg.bandwidth_hz:
            return
        profile = replace(profile, frame_cws=64)
        codec = self.fsm.profile_codec(profile.name)
        if hdr.n_cw != (hdr.size + codec.data_bytes - 1) // codec.data_bytes:
            return
        try:
            llr, _ = self.decode_body(profile, body, hdr.n_symbols, hdr.n_cw)
            chunks, _ = codec.decode_cws(llr)
            if any(ch is None for ch in chunks):
                return
            data = b"".join(chunks)
            if len(data) == hdr.size and sha256(data).digest()[:8] == hdr.token:
                self.on_datagram(data)
        except ValueError:
            return

    def on_datagram(self, data):
        self.datagrams.append(data)

    def send_data(self, seq, gear, present, n_cw, nibbles, coded, offset=0) -> float:
        gid = gear & wire.GEAR_MASK
        profile = BY_ID[gid]
        body, n_syms = self.encode_body(profile, coded, nibbles)
        ctrl = wire.Control(wire.DATA, self.fsm.session, seq=seq, gear=gear,
                            mask=wire.cw_mask(present),
                            aux=wire.data_aux(n_syms, n_cw, nibbles), offset=offset)
        fast = self.fsm.ctrl_fast
        pb = self.fsm.pb_ack if fast else None
        hdr = ctrl.pack() if pb is None else pb.pack() + ctrl.pack()
        return self._push(np.concatenate([
            self._header_wave(hdr, fast),
            body]))

    def encode_body(self, profile, coded, nibbles=(0,) * 8):
        if profile.floor:
            fm = FloorModem(FLOOR_GEARS[profile.floor])
            return (np.concatenate([_unit_rms(fm.transmit_coded(cw)) for cw in coded]),
                    len(coded) * fm.n_symbols(22))
        phy = self._phy(profile.name)
        loading = wire.expand_loading(nibbles, phy.gear)
        bits, n_syms = layout.assemble(phy, coded, profile.grouped,
                                      phy.gear.repeat, loading)
        return _unit_rms(phy.transmit(bits, loading=loading)), n_syms

    def body_duration(self, profile, n_syms, n_present, nibbles=(0,) * 8):
        """Derive a bounded body duration only after validating its full geometry."""
        if not 0 < n_present <= 64:
            raise ValueError("invalid codeword count")
        if profile.floor:
            fm = FloorModem(FLOOR_GEARS[profile.floor])
            if n_syms != n_present * fm.n_symbols(22) or any(nibbles):
                raise ValueError("invalid narrow DATA geometry")
            return n_present * fm.duration_s(22)
        phy = self._phy(profile.name)
        if any(n not in (0, 1, 2, 3, 4) for n in nibbles) or (profile.id >= 16 and any(nibbles)):
            raise ValueError("unsupported loading map")
        codec = self.fsm.profile_codec(profile.name)
        loading = wire.expand_loading(nibbles, phy.gear)
        _, expected = layout.assemble(phy, np.zeros((n_present, codec.code.n), dtype=int),
                                     profile.grouped, phy.gear.repeat, loading)
        if n_syms != expected:
            raise ValueError("DATA symbol count does not match codeword geometry")
        from hfmodem.sabir.phy.preamble import SEG_LEN
        return (GUARD_HEAD + 2 * SEG_LEN + n_syms * (phy.n_fft + phy.cp)
                + phy.window + GUARD_TAIL) / FS

    def decode_body(self, profile, body, n_syms, n_present, nibbles=(0,) * 8):
        codec = self.fsm.profile_codec(profile.name)
        if not 0 < n_present <= 64:
            raise ValueError("invalid codeword count")
        if profile.floor:
            fm = FloorModem(FLOOR_GEARS[profile.floor])
            if n_syms != n_present * fm.n_symbols(22) or any(nibbles):
                raise ValueError("invalid narrow DATA geometry")
            step = fm.n_symbols(22) * SYM + GUARD_HEAD + GUARD_TAIL
            rows = []
            for i in range(n_present):
                llr, _ = fm.demod(body[i * step:(i + 1) * step + PREROLL], 22)
                rows.append(np.zeros(codec.code.n) if llr is None else llr)
            return np.stack(rows), None
        phy = self._phy(profile.name)
        if any(n not in (0, 1, 2, 3, 4) for n in nibbles):
            raise ValueError("unsupported loading map")
        if profile.id >= 16 and any(nibbles):
            raise ValueError("extended profile requires uniform loading")
        loading = wire.expand_loading(nibbles, phy.gear)
        # Validate geometry before allocating FFT/channel surfaces from an
        # untrusted header. Both endpoints derive the exact same cell count.
        _, expected = layout.assemble(phy, np.zeros((n_present, codec.code.n), dtype=int),
                                       profile.grouped, phy.gear.repeat, loading)
        if n_syms != expected:
            raise ValueError("DATA symbol count does not match codeword geometry")

        def extract(llr):
            return layout.extract(phy, llr, n_present, codec.code.n,
                                  profile.grouped, phy.gear.repeat, loading, n_syms)

        def feedback(llr):
            chunks, _ = codec.decode_cws(extract(llr))
            if not any(ch is not None for ch in chunks):
                return None
            coded = np.stack([codec.encode_cw(ch) if ch is not None
                              else np.zeros(codec.code.n, dtype=int) for ch in chunks])
            known = np.stack([np.full(codec.code.n, ch is not None, dtype=int)
                             for ch in chunks])
            bits, _ = layout.assemble(phy, coded, profile.grouped, phy.gear.repeat, loading)
            mask, _ = layout.assemble(phy, known, profile.grouped, phy.gear.repeat, loading)
            return bits, mask.astype(bool)

        _, llr, res = phy.receive(body, n_symbols=n_syms, loading=loading, dd=self.dd,
                                  blank=self.fsm.cfg.impulse_blank,
                                  feedback=feedback if self.fsm.cfg.feedback_iters else None,
                                  fb_iters=self.fsm.cfg.feedback_iters)
        if res.n_symbols < n_syms:
            raise ValueError("DATA body truncated")
        return extract(llr), self._group_snr(phy, loading, res)

    def connected(self, session, peer):
        self.events.append(f"CONNECTED {peer} session={session}")

    def disconnected(self):
        self.events.append("DISCONNECTED")

    def state_changed(self, state: str):
        pass

    def deliver(self, blob: bytes):
        self.delivered += blob

    def log(self, msg: str):
        self.events.append(msg)

    # -- receive front-end ---------------------------------------------------
    def on_air(self, samples: np.ndarray) -> None:
        blocks, body_at, hdr_fast = self._header(samples)
        self.fsm.begin_burst()
        for ctrl in sorted(blocks, key=lambda b: b.type != wire.CAPS):
            if ctrl.type == wire.DATAGRAM:
                self._datagram(ctrl, samples[body_at:])
            elif ctrl.type == wire.DATA:
                self._data(ctrl, samples[body_at:], hdr_fast)
            elif wire.BEACON_LO <= ctrl.type <= wire.BEACON_HI:
                self.on_beacon(ctrl)
            else:
                self.fsm.on_control(ctrl, hdr_fast=hdr_fast)

    def on_beacon(self, beacon: wire.Beacon) -> None:
        """A connectionless presence beacon arrived. The bare LinkModem logs
        it; a host-bound modem surfaces it as PeerObserved."""
        self.events.append(f"BEACON {beacon.call}")

    def _header(self, samples: np.ndarray):
        """Decode the header region: ([control blocks], body offset, fast?).

        A one-block fast try on a two-block burst (or vice versa) sees a
        half-deinterleaved stream and fails its CRC screen, so the tier and
        block count come out of the CRCs, not out of any prior agreement.

        Where the burst starts is a decode result, not an assumption: both
        tiers report the sample they synced at, and the body is placed from there. In process that offset is zero and every
        window is the fixed slice it always was; off the air it is the
        segmenter's pre-roll, and everything downstream of a wrong one reads
        noise.
        """
        for carrier, n in ((self.fast, 1), (self.fast, 2), (self.short_fast, 1)):
            need = carrier.n_samples(n)
            raw, at = carrier.receive(samples[:need + PREROLL], n, dd=self.dd)
            if raw is None:
                continue
            width = carrier.block_bytes
            blocks = [wire.Control.unpack(raw[i:i + width]) for i in range(0, len(raw), width)]
            if all(b is not None for b in blocks):
                return blocks, at + need, True
        ctrl, at = self._floor_block(samples, 0, PREROLL)
        if ctrl is None:
            raw, st = self.floor.receive(samples[:HEADER_SAMPLES + PREROLL], wire.CONNECTIONLESS_BYTES)
            ctrl = wire.Control.unpack(raw) if raw is not None else None
            if ctrl is None:
                return [], HEADER_SAMPLES, False
            at = max(0, int(st["start"]) - GUARD_HEAD)
            short_samples = GUARD_HEAD + GUARD_TAIL + self.floor.n_symbols(wire.CONNECTIONLESS_BYTES) * SYM
            return [ctrl], at + short_samples, False
        blocks = [ctrl]
        if ctrl.type in (wire.CONNECT, wire.CONNECT_ACK):
            for idx in range(ctrl.n_ext):
                extension, _ = self._floor_block(samples, at + (idx + 1) * HEADER_SAMPLES)
                if extension is not None and extension.type == wire.CAPS:
                    blocks.append(extension)
        return blocks, at + HEADER_SAMPLES, False

    def _floor_block(self, samples: np.ndarray, at: int, slack: int = 1024):
        """One floor block near ``at`` -> (it or None, where it really was).

        ``slack`` is how far past ``at`` the block may begin; the trailing
        The search begins wherever the segmenter's bracket placed the burst.
        """
        raw, st = self.floor.receive(samples[at: at + HEADER_SAMPLES + slack],
                                     wire.BLOCK_BYTES)
        if raw is None:
            return None, at
        # the floor syncs on the tone stream; the burst opens a guard earlier
        return wire.Control.unpack(raw), max(0, at + int(st["start"]) - GUARD_HEAD)

    def _data(self, ctrl: wire.Control, body: np.ndarray,
              hdr_fast: bool) -> None:
        if ctrl.session != self.fsm.session or self.fsm.state != "CONNECTED":
            return
        profile = self.fsm.rx_profile(ctrl.gear)
        if profile is None:
            return
        n_syms, n_cw, nibbles = wire.parse_data_aux(ctrl.aux)
        if not 0 < n_cw <= 64 or int.from_bytes(ctrl.mask, "little") >> n_cw:
            return
        present = wire.mask_indices(ctrl.mask, n_cw)
        try:
            duration = self.body_duration(profile, n_syms, len(present), nibbles)
        except ValueError:
            return
        if not self.fsm.on_data_header(ctrl, duration):
            return
        try:
            cw_llrs, group_snr = self.decode_body(profile, body, n_syms, len(present), nibbles)
        except ValueError:
            cw_llrs = group_snr = None
        self.fsm.on_data(ctrl, cw_llrs, group_snr, hdr_fast)

    def _group_snr(self, phy, loading, res) -> list[float | None]:
        grp = carrier_groups(phy.gear.n_carriers)
        # Estimate masked groups from their still-present pilots so temporary
        # notches can recover without requiring a gear change.
        live = np.ones(phy.gear.n_carriers, dtype=bool)
        out = []
        for g in range(N_GROUPS):
            sel = (grp == g) & live
            if not sel.any():
                out.append(None)
            else:
                snr = res.carrier_snr
                if loading is not None and not (loading[sel] > 0).any():
                    snr = res.pilot_snr
                med = max(float(np.median(snr[sel])), 1e-6)
                out.append(10 * np.log10(med))
        return out
