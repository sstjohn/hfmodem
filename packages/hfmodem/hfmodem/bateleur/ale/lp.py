# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Linking protection for 2G ALE: MIL-STD-188-141B appendix B, application
levels AL-1 and AL-2.

Every 24-bit ALE word is encrypted before the Golay coder with the Lattice
algorithm (B.5.6) under a 56-bit key and a 64-bit seed of date, time,
word number and frequency (B.5.2.3). No synchronisation words are sent: the
receiver finds the seed by trying the few the protocol allows (B.5.3.2). This
implementation covers AL-1 and AL-2. AL-3 and AL-4 are not implemented.

**Protection intervals.** AL-1's is 60 s (the seed's fine time all ones),
AL-2's 2 s. A seed's time part is kept here as a PI number, the whole PIs
since the Unix epoch, which carries the date, the minute and the second.

**Word numbers** (B.5.3.1): a scanning call, and a whole sound, alternate
w = 0 and 1, ending on 1, the time moving on only at a 0; the leading call
starts again at 0 with the time of its first word, and every later word
counts up to wmax (153 at a 60 s PI, 5 at 2 s), after which the PI moves on
and w goes back to 0. A response starts at 0 with its own time.

Word and DBM pair numbering, clear authentication and time exchange remain
unverified against a foreign implementation. ALE.md lists the recordings
needed to check these readings; the cipher examples alone do not settle them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from . import words as W

# Table B-I, the Lattice algorithm's substitution; its inverse is table B-II.
_F = bytes.fromhex(
    "9c f2 14 c1 8e cb b2 65 97 7a 60 17 92 f9 78 41 "
    "07 4c 67 6d 66 4a 30 7d 53 9d b5 bc c3 ca f1 04 "
    "03 ec d0 38 b0 ed ad c4 dd 56 42 bd a0 de 1b 81 "
    "55 44 5a e4 50 dc 43 63 09 5c 74 cf 0e ab 1d 3d "
    "6b 02 5d 28 e7 c6 ee b4 d9 7c 19 3e 5e 6c d6 6e "
    "2a 13 a5 08 b9 2d bb a2 d4 96 39 e0 ba d7 82 33 "
    "0d 5f 26 16 fe 22 af 00 11 c8 9e 88 8b a1 7b 87 "
    "27 e6 c7 94 d1 5b 9b f0 9f db e1 8d d2 1f 6a 90 "
    "f4 18 91 59 01 b1 fc 34 3c 37 47 29 e2 64 69 24 "
    "0a 2f 73 71 a9 84 8c a8 a3 3b e3 e9 58 80 a7 d3 "
    "b7 c2 1c 95 1e 4d 4f 4e fb 76 fd 99 c5 c9 e8 2e "
    "8a df f5 49 f3 6f 8f e5 eb f6 25 d5 31 c0 57 72 "
    "aa 46 68 0b 93 89 83 70 ef a4 85 f8 0f b3 ac 10 "
    "62 cc 61 40 f7 fa 52 7f ff 32 45 20 79 ce ea be "
    "cd 15 21 23 d8 b6 0c 3f 54 1a bf 98 48 3a 75 77 "
    "2b ae 36 da 7e 86 35 51 05 12 b8 a6 9a 2c 06 4b ")
_G = bytes(np.argsort(np.frombuffer(_F, dtype=np.uint8)).astype(np.uint8))

ROUNDS = 8


class Lattice:
    """B.5.6: three bytes, eight rounds, each byte mixed with its
    neighbours, a key byte and a seed byte and passed through the S-box.
    Every mixing is an exclusive-or ("+" in the text's equations)."""

    def __init__(self, key: bytes) -> None:
        if len(key) != 7:
            raise ValueError("a Lattice key is 56 bits, 7 bytes")
        self.key = bytes(key)

    def encrypt(self, word: int, seed: bytes) -> int:
        v, s = self.key, seed
        a, b, c = (word >> 16) & 0xFF, (word >> 8) & 0xFF, word & 0xFF
        k = 0
        for _ in range(ROUNDS):
            a = _F[a ^ b ^ v[k % 7] ^ s[k % 8]]
            k += 1
            c = _F[c ^ b ^ v[k % 7] ^ s[k % 8]]
            k += 1
            b = _F[a ^ b ^ c ^ v[k % 7] ^ s[k % 8]]
            k += 1
        return (a << 16) | (b << 8) | c

    def decrypt(self, word: int, seed: bytes) -> int:
        v, s = self.key, seed
        a, b, c = (word >> 16) & 0xFF, (word >> 8) & 0xFF, word & 0xFF
        k = 3 * ROUNDS - 1
        for _ in range(ROUNDS):
            b = _G[b] ^ a ^ c ^ v[k % 7] ^ s[k % 8]
            k -= 1
            c = _G[c] ^ b ^ v[k % 7] ^ s[k % 8]
            k -= 1
            a = _G[a] ^ b ^ v[k % 7] ^ s[k % 8]
            k -= 1
        return (a << 16) | (b << 8) | c


def seed(*, month: int, day: int, minute: int, fine: int, word: int, freq_hz: float) -> bytes:
    """Figure B-3: month (4 bits), day (5), coarse time in minutes since
    midnight (11), fine time in seconds (6; all ones when unknown, and always
    at AL-1), word number (8), two zeros, and the frequency as seven BCD
    digits from 100 MHz down to 100 Hz."""
    digits = f"{int(round(freq_hz / 100)):07d}"
    if len(digits) > 7:
        raise ValueError(f"{freq_hz} Hz does not fit the seed's frequency field")
    f = 0
    for d in digits:
        f = (f << 4) | int(d)
    v = month
    v = (v << 5) | day
    v = (v << 11) | minute
    v = (v << 6) | (fine & 0x3F)
    v = (v << 8) | (word & 0xFF)
    v = (v << 2)
    v = (v << 28) | f
    return v.to_bytes(8, "big")


#: B.4.1.1: each level's protection interval, and B.5.3.1's wmax for it.
PI_S = {1: 60, 2: 2}
WMAX = {60: 153, 2: 5, 1: 2}


FINE, COARSE, CLEAR = 0, 1, 2


@dataclass(frozen=True)
class State:
    """Where in the protocol a word was encrypted: the PI and word number,
    under the station's own seed (FINE), a coarse one (COARSE: minutes, the
    fine time all ones, for time acquisition, B.5.5.2), or none (CLEAR)."""

    pi: int
    w: int
    kind: int = FINE


class Protection:
    """One station's linking protection: its level, key and channels."""

    def __init__(self, level: int, key: bytes, channel_hz: list[float]) -> None:
        if level not in PI_S:
            raise ValueError("linking protection here is AL-1 or AL-2")
        self.level = level
        self.pi_s = PI_S[level]
        self.wmax = WMAX[self.pi_s]
        self.cipher = Lattice(key)
        self.channel_hz = list(channel_hz)

    # -- seeds -----------------------------------------------------------------

    def pi_at(self, utc: float, kind: int = FINE) -> int:
        return int(utc // (60 if kind == COARSE else self.pi_s))

    def seed_of(self, st: State, channel: int) -> bytes:
        t = datetime.fromtimestamp(st.pi * (60 if st.kind == COARSE else self.pi_s),
                                   tz=timezone.utc)
        fine = 0x3F if (self.level == 1 or st.kind == COARSE) else t.second
        return seed(month=t.month, day=t.day, minute=t.hour * 60 + t.minute, fine=fine,
                    word=st.w, freq_hz=self.channel_hz[channel])

    def encrypt(self, w: int, st: State, channel: int) -> int:
        if st.kind == CLEAR:
            return w & 0xFFFFFF
        return self.cipher.encrypt(w & 0xFFFFFF, self.seed_of(st, channel))

    def decrypt(self, w: int, st: State, channel: int) -> int:
        if st.kind == CLEAR:
            return w & 0xFFFFFF
        return self.cipher.decrypt(w & 0xFFFFFF, self.seed_of(st, channel))

    def next(self, st: State) -> State:
        """The word after ``st`` outside a scanning call."""
        if st.kind == CLEAR:
            return st
        wmax = WMAX[60] if st.kind == COARSE else self.wmax
        return State(st.pi + 1, 0, st.kind) if st.w >= wmax else State(st.pi, st.w + 1, st.kind)

    # -- sending (B.5.3.1) -------------------------------------------------------

    def protect(self, ws: list, utc: float, channel: int, *, scan: int = 0,
                sound: bool = False, coarse: bool = False) -> list:
        """A frame's words encrypted: ``scan`` words of scanning call first
        (all of them, for a sound), sent from ``utc``. A DBM block among the
        words is encrypted inside, pair by pair (B.5.3.4). ``coarse``: under
        the coarse seed, as a time request from a station without fine time
        is (B.5.5.2.1)."""
        from .fsk import BAUD
        kind = COARSE if coarse else FINE
        out: list = []
        t = utc
        tsc = len(ws) if sound else scan
        st: State | None = None
        for i, w in enumerate(ws):
            if i < tsc:
                wn = 1 if (tsc - 1 - i) % 2 == 0 else 0
                pi = self.pi_at(t, kind) if (wn == 0 or st is None) else st.pi
                st = State(pi, wn, kind)
            elif i == tsc:
                st = State(self.pi_at(t, kind), 0, kind)
            else:
                st = self.next(st)
            if isinstance(w, np.ndarray):
                blk, st = self._protect_dbm(w, st, channel)
                out.append(blk)
                t += w.size / 3 / BAUD
                continue
            out.append((w & ~0xFFFFFF) | self.encrypt(w, st, channel))
            t += W.TRIPLE_BITS / 3 / BAUD
        return out

    def dbm_states(self, st: State, depth: int) -> tuple[list[State], State]:
        """B.5.3.4: the states a DBM block's 12-bit pairs are encrypted under,
        after the CMD DBM word's ``st``: three pairs to a word number, and
        one step on after the last pair. Returns them and that last state."""
        pairs = 12 * depth // 24
        out = []
        cur = self.next(st)
        for k in range(pairs):
            if k and k % 3 == 0:
                cur = self.next(cur)
            out.append(cur)
        return out, cur

    def _protect_dbm(self, bits: np.ndarray, st: State, channel: int):
        depth = bits.size // 24
        rows = bits.reshape(24, depth).T
        fld = rows[:, :12].reshape(-1).astype(int)
        states, last = self.dbm_states(st, depth)
        fld = self.crypt_field(fld, states, channel, decrypt=False)
        return W.dbm_interleave(fld), last

    def crypt_field(self, fld, states: list[State], channel: int, *, decrypt: bool) -> np.ndarray:
        """A DBM data field's 12-bit words, encrypted (or decrypted) in pairs,
        the first the more significant; an odd last word is left alone."""
        fld = np.asarray(fld, dtype=int).copy()
        for k, st in enumerate(states):
            seg = fld[24 * k:24 * k + 24]
            v = W.bits_int(seg)
            v = self.decrypt(v, st, channel) if decrypt else self.encrypt(v, st, channel)
            fld[24 * k:24 * k + 24] = [(v >> (23 - i)) & 1 for i in range(24)]
        return fld

    # -- receiving (B.5.3.2) -----------------------------------------------------

    def initial(self, utc: float, *, coarse=(), clear: bool = False) -> list[State]:
        """The seeds a word heard while seeking word sync may be under: the
        present PI and those either side, with w = 0 and w = 1 (B.5.3.2);
        coarse seeds so many minutes on from the present one — a time
        server's the present (B.5.5.2.2), a station awaiting a time service
        response that and the next (B.5.5.2.4); and for clear frames, none."""
        n = self.pi_at(utc)
        out = [State(p, w) for p in (n, n - 1, n + 1) for w in (0, 1)]
        if self.pi_s != 60:
            m = self.pi_at(utc, COARSE)
            out += [State(m + k, w, COARSE) for k in coarse for w in (0, 1)]
        if clear:
            out.append(State(0, 0, CLEAR))
        return out

    def successors(self, st: State) -> list[State]:
        """What the next word may be under (figure B-4c, d). After a 1 the
        scanning call may go on (0, the PI as it was or moved on), or the
        two words were the leading call's first (2)."""
        if st.kind == CLEAR:
            return [st]
        k = st.kind
        if st.w == 0:
            return [State(st.pi, 1, k)]
        if st.w == 1:
            return [State(st.pi, 0, k), State(st.pi + 1, 0, k), State(st.pi, 2, k)]
        return [self.next(st)]

    def authenticate_clear(self, x: int, month: int, day: int, minute: int, channel: int) -> int:
        """B.5.5.3.2: the non-protected Time Service authenticator, the 24-bit
        exclusive-or encrypted under a seed of the coarse time sent and word
        number all ones; its 21 least significant bits."""
        sd = seed(month=month, day=day, minute=minute, fine=0x3F, word=0xFF,
                  freq_hz=self.channel_hz[channel])
        return self.cipher.encrypt(x & 0xFFFFFF, sd) & 0x1FFFFF


class View:
    """What a receiver needs of a station's protection: the seeds allowed
    now, on the channel it is tuned to. ``utc()`` is the station's time of
    day, ``channel()`` its channel."""

    def __init__(self, prot: Protection, utc, channel, *, coarse=lambda: (),
                 clear: bool = False) -> None:
        self.prot, self.utc, self.channel = prot, utc, channel
        self.coarse, self.clear = coarse, clear

    def initial(self) -> list[State]:
        return self.prot.initial(self.utc(), coarse=self.coarse(), clear=self.clear)

    def successors(self, st: State) -> list[State]:
        return self.prot.successors(st)

    def next(self, st: State) -> State:
        return self.prot.next(st)

    def decrypt(self, w: int, st: State) -> int:
        return self.prot.decrypt(w, st, self.channel())

    def dbm_states(self, st: State, depth: int):
        return self.prot.dbm_states(st, depth)

    def decrypt_field(self, fld, states: list[State]):
        return self.prot.crypt_field(fld, states, self.channel(), decrypt=True)
