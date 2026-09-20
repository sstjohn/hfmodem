# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The beacon: narrow-tone MFSK for the WSPR/FT8-class weak-signal floor.

Every weak-signal mode threshold sits near the same ~+5 dB Eb/N0; deep
negative SNR-in-reference-bandwidth is bought by lowering the bit rate and
narrowing the detection bandwidth. The interactive floor (``floor.mfsk``)
spends that budget on Doppler robustness -- 93.75 Hz tones shrug off 30 Hz
polar spread but bottom out near -15 dB. This mode flips the trade: the same
4-FSK/Costas/CA-TBCC machinery on tones 1.5-5.9 Hz apart, symbols 0.17-0.68 s
long, reaching WSPR territory. Narrow tones cannot survive Doppler spread
comparable to their spacing, and second-long symbols cannot carry an ARQ
turnaround -- so this is a non-interactive beacon / FEC-broadcast mode, the
bottom rung of the range<->throughput knob. The two-way link still bottoms
at the wide floor.

Payload is a fixed 9-byte block: callsign (6 chars, base-38, 32 b) + 4-char
Maidenhead grid (15 b) + status byte, CRC-16 terminated -- 55 payload bits
per burst. Sync is the same 7x7 Costas array, self-timed in-band, so no
GPS/NTP slotting is required; the acquisition envelope is the full +/-75 Hz
CFO but only +/-0.25 Hz/s drift (a beacon's oscillator, not a scanning VFO).
A time-slotted decode path (known burst epoch shrinks the sync search to a
few cells) would buy roughly the last 1-2 dB; deliberately not built.

The receiver channelizes to 375 Hz complex rate around the 1500 Hz band
centre (FFT brick-wall, exact at these block sizes), so sync and detection
cost the same as the wide floor despite 30x longer bursts. Depth rungs
lengthen symbols and add in-burst repeats (square-law energy combining,
the classical noncoherent diversity); ``receive_combining`` soft-combines
LLRs across separately received bursts -- the one-to-many broadcast path,
no ACKs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hfmodem.sabir.fec.tbcc import CATBCC
from hfmodem.sabir.floor.mfsk import DATA_UNITS, GRAY, NOISE_UNITS, _plan
from hfmodem.sabir.frame import pn9
from hfmodem.sabir.frame.codec import crc16, crc_ok
from hfmodem.sabir.phy.modem import FS, GUARD_HEAD, GUARD_TAIL
from hfmodem.sabir.phy.rate import CENTER_HZ

D = 128                       # RX decimation: 375 Hz complex rate
FS_DEC = FS / D
F_CENTER = CENTER_HZ             # beacon band centre = tone unit 3
SEG = 24                      # data symbols per Costas segment: drift anchors
FINE = 4                      # fine bins per grid unit
CFO_MAX = 75.0                # coarse CFO search half-width, Hz
DRIFT_MAX = 0.25              # Hz/s: beacon-grade oscillator envelope
EDGE = 1024                   # burst amplitude ramp, samples

BEACON_BYTES = 9


@dataclass(frozen=True)
class BeaconGear:
    name: str
    sym: int                  # samples per symbol at FS
    repeat: int


BEACON_GEARS = {
    "beacon_short": BeaconGear("nb-mfsk4-5.9Hz-x1", 8192, 1),
    "beacon_med":   BeaconGear("nb-mfsk4-2.9Hz-x1", 16384, 1),
    "beacon_deep":  BeaconGear("nb-mfsk4-1.5Hz-x1", 32768, 1),
    "beacon_deep2": BeaconGear("nb-mfsk4-1.5Hz-x2", 32768, 2),
}


# -- payload -------------------------------------------------------------------
CALL_CHARS = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"


@dataclass(frozen=True)
class BeaconPayload:
    callsign: str             # up to 6 of [A-Z 0-9 /]
    grid: str                 # 4-char Maidenhead, e.g. "FN31"
    status: int = 0           # application-defined byte

    def pack(self) -> bytes:
        call = self.callsign.upper().ljust(6)
        if len(call) != 6 or any(ch not in CALL_CHARS for ch in call):
            raise ValueError(f"callsign {self.callsign!r} does not fit "
                             "6 chars of [A-Z 0-9 /]")
        g = self.grid.upper()
        if not (len(g) == 4 and "A" <= g[0] <= "R" and "A" <= g[1] <= "R"
                and g[2:].isdigit()):
            raise ValueError(f"grid {self.grid!r} is not a 4-char locator")
        c = 0
        for ch in call:
            c = c * 38 + CALL_CHARS.index(ch)
        gv = ((ord(g[0]) - 65) * 18 + ord(g[1]) - 65) * 100 + int(g[2:])
        v = (c << 23) | (gv << 8) | (self.status & 0xFF)
        body = v.to_bytes(7, "big")
        return body + crc16(body).to_bytes(2, "big")

    @classmethod
    def unpack(cls, buf: bytes) -> "BeaconPayload | None":
        if len(buf) != BEACON_BYTES or not crc_ok(buf):
            return None
        v = int.from_bytes(buf[:7], "big")
        c, gv, status = v >> 23, (v >> 8) & 0x7FFF, v & 0xFF
        if c >= 38**6 or gv >= 32400:
            return None
        chars = []
        for _ in range(6):
            c, i = divmod(c, 38)
            chars.append(CALL_CHARS[i])
        sq, dig = divmod(gv, 100)
        f1, f2 = divmod(sq, 18)
        return cls("".join(reversed(chars)).strip(),
                   chr(65 + f1) + chr(65 + f2) + f"{dig:02d}", status)


# -- DSP helpers ---------------------------------------------------------------
def _boxcar_same(f: np.ndarray, L: int) -> np.ndarray:
    """Zero-padded centred moving average (convolve-'same' at cumsum cost)."""
    pad = np.concatenate([np.zeros(L // 2), f, np.zeros(L - L // 2)])
    c = np.concatenate([[0.0], np.cumsum(pad)])
    return (c[L:] - c[:-L])[: f.size] / L


def _channelize(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Brick-wall extract FS_DEC of band around F_CENTER at the decimated
    rate. Returns (stream, f_lo): absolute frequency f lands at f - f_lo."""
    n = x.size + (-x.size) % D
    X = np.fft.fft(x, n)
    m = n // D
    k0 = int(round((F_CENTER - FS_DEC / 2) * n / FS))
    return np.fft.ifft(X[(k0 + np.arange(m)) % n]), k0 * FS / n


# -- the modem -----------------------------------------------------------------
class BeaconModem:
    def __init__(self, gear: BeaconGear = BEACON_GEARS["beacon_deep"],
                 threshold: float = 6.0):
        self.gear = gear
        self.tb = CATBCC()
        self.threshold = threshold
        self.S = gear.sym // D                # symbol length, decimated
        self.grid = FS / gear.sym             # tone spacing, Hz
        self.base_hz = F_CENTER - 3 * self.grid

    # -- shape ------------------------------------------------------------
    def n_symbols(self, n_bytes: int) -> int:
        cp, _, dp = _plan(8 * n_bytes * self.gear.repeat, SEG)
        return int(max(cp[-1], dp[-1])) + 1

    def duration_s(self, n_bytes: int) -> float:
        return self.n_symbols(n_bytes) * self.gear.sym / FS

    def _perm(self, n: int) -> np.ndarray:
        return np.arange(n).reshape(16, n // 16).T.ravel()

    # -- transmit ----------------------------------------------------------
    def transmit(self, block: bytes) -> np.ndarray:
        """CRC-terminated block -> complex analytic burst, unit envelope."""
        coded = self.tb.encode(block)
        n = coded.size
        coded = (coded ^ pn9(n))[self._perm(n)]
        stream = np.tile(coded, self.gear.repeat)
        tones = GRAY[2 * stream[0::2] + stream[1::2]]
        cp, cu, dp = _plan(tones.size, SEG)
        units = np.empty(self.n_symbols(len(block)), dtype=np.int64)
        units[cp] = cu
        units[dp] = 2 * tones
        f = np.repeat(self.base_hz + units * self.grid, self.gear.sym)
        f = _boxcar_same(f, self.gear.sym // 16)
        wave = np.exp(2j * np.pi * np.cumsum(f) / FS)
        ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(EDGE) + 0.5) / EDGE))
        wave[:EDGE] *= ramp
        wave[-EDGE:] *= ramp[::-1]
        return np.concatenate([np.zeros(GUARD_HEAD, dtype=np.complex128),
                               wave,
                               np.zeros(GUARD_TAIL, dtype=np.complex128)])

    # -- receive -----------------------------------------------------------
    def demod(self, samples: np.ndarray, n_bytes: int = BEACON_BYTES
              ) -> tuple[np.ndarray | None, dict]:
        """Sync + noncoherent detection -> deinterleaved, de-whitened LLRs.

        LLRs are normalized by the capture's own noise estimate, so summing
        the outputs of several captures is the broadcast soft combiner.
        """
        x = np.asarray(samples, dtype=np.complex128)
        z, f_lo = _channelize(x)
        f_nom = self.base_hz - f_lo
        cp, cu, dp = _plan(8 * n_bytes * self.gear.repeat, SEG)
        n_sym = self.n_symbols(n_bytes)
        det = self._sync(z, f_nom, cp, cu, n_sym)
        if det is None:
            return None, {"sync": None}
        s0, cfo_hz, drift, metric = det
        t = np.arange(z.size - s0) / FS_DEC
        y = z[s0:] * np.exp(-2j * np.pi * (cfo_hz * t + 0.5 * drift * t**2))
        if y.size < n_sym * self.S:
            return None, {"sync": None}
        E = self._energies(y[: n_sym * self.S].reshape(n_sym, self.S), f_nom)
        Ed = E[dp]
        n0 = max(np.median(Ed[:, list(NOISE_UNITS)]) / np.log(2), 1e-12)
        e = Ed[:, list(DATA_UNITS)][:, GRAY]
        # in-burst repeats carry identical tones: square-law combine energies
        # before the LLR nonlinearity (classical noncoherent diversity)
        R = self.gear.repeat
        e = e.reshape(R, -1, 4).sum(axis=0)
        gamma = max(np.mean(e.sum(axis=1)) / (R * n0) - 4.0, 0.05)
        s = gamma / (1.0 + gamma) / n0 * e
        llr = np.empty((e.shape[0], 2))
        llr[:, 0] = np.logaddexp(s[:, 0], s[:, 1]) - np.logaddexp(s[:, 2], s[:, 3])
        llr[:, 1] = np.logaddexp(s[:, 0], s[:, 2]) - np.logaddexp(s[:, 1], s[:, 3])
        n = 16 * n_bytes
        deint = np.empty(n)
        deint[self._perm(n)] = llr.ravel()
        deint *= 1 - 2 * pn9(n)
        return deint, {"cfo_hz": cfo_hz, "drift_hz_s": drift,
                       "start": s0 * D, "metric": metric, "noise": n0}

    def receive(self, samples: np.ndarray, n_bytes: int = BEACON_BYTES,
                check=crc_ok) -> tuple[bytes | None, dict]:
        llr, stats = self.demod(samples, n_bytes)
        if llr is None:
            return None, stats
        block, dec = self.tb.decode(llr, n_bytes, check)
        return block, stats | dec

    def _energies(self, syms: np.ndarray, f_nom: float) -> np.ndarray:
        """Per-symbol tone energies on the unit grid: (n_sym, 7)."""
        t = np.arange(self.S) / FS_DEC
        C = np.exp(-2j * np.pi * np.outer(
            f_nom + np.arange(7) * self.grid, t))
        return np.abs(syms @ C.T / self.S) ** 2

    def _costas_metric(self, z, s0, f_nom, cp, cu, f_off, blocks=None):
        t = np.arange(self.S) / FS_DEC
        m = 0.0
        for i in (range(cp.size) if blocks is None else blocks):
            lo = s0 + cp[i] * self.S
            if lo < 0 or lo + self.S > z.size:
                continue
            ref = np.exp(-2j * np.pi * (f_nom + cu[i] * self.grid + f_off) * t)
            m += abs(np.dot(z[lo : lo + self.S], ref)) ** 2
        return m / self.S**2

    def _sync(self, z, f_nom, cp, cu, n_sym):
        """Coarse spectrogram search + fine timing + per-block CFO/drift fit.

        Returns (start_sample_dec, cfo_hz, drift_hz_s, metric) or None. The
        detection metric is a z-score of the peak against the search field,
        so one threshold serves every rung's sync-symbol count.
        """
        S = self.S
        hop = S // 4
        if z.size < n_sym * S:
            return None
        n_hops = (z.size - S) // hop + 1
        view = np.lib.stride_tricks.as_strided(
            z, (n_hops, S), (z.itemsize * hop, z.itemsize))
        Q = int(np.ceil(CFO_MAX * FINE / self.grid))
        q = np.arange(-Q, 6 * FINE + Q + 1)
        t = np.arange(S) / FS_DEC
        W = np.exp(-2j * np.pi * np.outer(f_nom + q * self.grid / FINE, t))
        E = np.abs(W @ view.T) ** 2 / S**2
        # Costas rejects a wrong lag by ~7:1, but a selective fade swings the
        # received level 10-15 dB inside one burst -- so accumulating raw
        # energy lets a lag whose windows land on the fade peaks outbid the
        # true one. Score each window against its own in-band level.
        E /= np.maximum(np.median(E, axis=0), 1e-12)

        t_max = n_hops - (n_sym - 1) * 4
        if t_max <= 0:
            return None
        M = np.zeros((2 * Q + 1, t_max))
        for i in range(cp.size):
            rows = Q + FINE * cu[i] + np.arange(-Q, Q + 1)
            off = cp[i] * 4
            M += E[rows, off : off + t_max]
        peak = np.unravel_index(np.argmax(M), M.shape)
        med = np.median(M)
        metric = float((M[peak] / med - 1.0) * np.sqrt(cp.size))
        if metric < self.threshold:
            return None
        d_hz = (peak[0] - Q) * self.grid / FINE
        s0 = peak[1] * hop

        offs = np.arange(-4, 5) * max(S // 32, 1)
        mt = [self._costas_metric(z, s0 + o, f_nom, cp, cu, d_hz)
              for o in offs]
        s0 += offs[int(np.argmax(mt))]

        n_blocks = cp.size // 7
        times, freqs, wts = [], [], []
        dgrid = np.arange(-4, 4.5) * self.grid / 8
        for bb in range(n_blocks):
            blocks = range(7 * bb, 7 * bb + 7)
            mb = np.array([self._costas_metric(z, s0, f_nom, cp, cu,
                                               d_hz + dd, blocks)
                           for dd in dgrid])
            p = int(np.argmax(mb))
            df = dgrid[p]
            if 0 < p < mb.size - 1:
                den = mb[p - 1] - 2 * mb[p] + mb[p + 1]
                if den < 0:
                    df += 0.5 * (mb[p - 1] - mb[p + 1]) / den * (self.grid / 8)
            times.append((cp[7 * bb] + 3.5) * S / FS_DEC)
            freqs.append(d_hz + df)
            wts.append(mb[p])
        tt, ff, ww = map(np.asarray, (times, freqs, wts))
        A = np.stack([np.ones_like(tt), tt], axis=1) * np.sqrt(ww)[:, None]
        c0, c1 = np.linalg.lstsq(A, ff * np.sqrt(ww), rcond=None)[0]
        c1 = float(np.clip(c1, -DRIFT_MAX, DRIFT_MAX))
        return int(s0), float(c0), c1, metric


# -- the broadcast operating path ----------------------------------------------
def send_beacon(payload: BeaconPayload,
                gear: BeaconGear = BEACON_GEARS["beacon_deep"]) -> np.ndarray:
    return BeaconModem(gear).transmit(payload.pack())


def recv_beacon(samples: np.ndarray,
                gear: BeaconGear = BEACON_GEARS["beacon_deep"]
                ) -> tuple[BeaconPayload | None, dict]:
    block, stats = BeaconModem(gear).receive(samples, BEACON_BYTES)
    return (BeaconPayload.unpack(block) if block else None), stats


def receive_combining(captures, n_bytes: int = BEACON_BYTES,
                      gear: BeaconGear = BEACON_GEARS["beacon_deep"],
                      check=crc_ok) -> tuple[bytes | None, dict]:
    """One-to-many RX: soft-combine LLRs across repeated bursts, no ACKs.

    Each capture is synced independently; decode is retried after every
    combined copy, so the receiver finishes as soon as the accumulated
    energy suffices (Chase combining, ~+2.5 dB per doubling noncoherent).
    """
    modem = BeaconModem(gear)
    acc, used = None, 0
    for cap in captures:
        llr, _ = modem.demod(cap, n_bytes)
        if llr is None:
            continue
        acc = llr if acc is None else acc + llr
        used += 1
        block, dec = modem.tb.decode(acc, n_bytes, check)
        if block is not None:
            return block, {"combined": used, **dec}
    return None, {"combined": used, "ok": False}
