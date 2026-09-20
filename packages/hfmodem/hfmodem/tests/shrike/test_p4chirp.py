# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The PACTOR-4 chirp entry: the spec's own numbers, read back off the render.

The demodulator here is the TEST'S, transcribed from [SCS-P4] section 11 rather
than imported from `p4chirp`: it de-chirps at 294.0 Hz/s from the instant 11.7.2
names, mixes at 550 and 1530 Hz, takes the lower carrier T/2 late as 11.7 says,
matched-filters with 11.8's printed taps and unwinds 11.6's printed pointer walk
-- so a render that got any of those wrong fails the CRC here rather than
agreeing with itself.

Two things here are no longer the document's, both off the sigidwiki recording
`Pactor_IV_chirps` (four chirp packets, one station, 2026-09-16): the header's
codeword-to-carrier assignment, crossed against PACTOR-2's, and the CRC's
`p4chirp.CHIRP_CRC_XOR`. `REAL_FIELDS` below carries those four packets' fields
as literals -- each is an exact codeword of `coding.CODE_K9` under 11.6's
pointer walk, so the coding chain under the field is a station's and not ours.

Two of the four carry marker codeword 8 and two carry 9, which is the same
codeword with P2's `flag` bit set, AND THE FLAG IS THE CARRIER SWAP: at k=8
channel rank 0 is the upper carrier, at k=9 the lower, four for four.
`pactor2.marker_index` records that the flag is not the swap in PACTOR-2, and
that measurement stands -- this is a PACTOR-4 chirp fact. The render is fixed at
k=8 and arranges its lanes accordingly.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.signal import resample_poly

from hfmodem.shrike import coding, onair, p4chirp, pactor1, pactor2, placement, spec
from hfmodem.shrike import arq as arq_mod
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_p3_offer import MESSAGE, Keyed, cs_event

FS = spec.SAMPLE_RATE

REAL_FIELDS = tuple(bytes.fromhex(h) for h in (
    "d10d19432d8a01c23424c563e89169fca2912c47257af59f8a",   # marker codeword 9
    "1af34e13ef95b5ff41f36189dce49d5c227101fd134cf67594",   # 9
    "d10c02fc28d325d44b18020b0915937e4789ef76daa2f76e75",   # 8
    "5dadcbb6a78bad08263df70d1d6a882b83da22639beaf56d87",   # 8
))
"""Four P4dragon chirp packets' fields, in transmission order, read off
`Pactor_IV_chirps` -- 22 payload bytes, the status byte, the CRC-16 last."""


def _spec_pointer_walk(packet_size: int, depth: int) -> np.ndarray:
    """11.6's loop as printed, with the 0-based `>=` the walk needs."""
    p, s = 0, 1
    out = np.empty(packet_size, dtype=np.int64)
    for i in range(packet_size):
        out[i] = p
        p += depth
        if p >= packet_size:
            p, s = s, s + 1
    return out


def _demod(audio: np.ndarray) -> tuple[bytes, float, float, float]:
    """(field, worst header step error, min |differential|, crossed-header score).

    The last is the header correlation against the CROSSED codeword pair --
    P2's upper-tone word read at 1530 Hz. The two words of one codeword are not
    orthogonal, so it does not go to zero; it goes to the same 0.55 a real
    packet scores the wrong way round, against 1.00 the right way, which is the
    margin the measurement itself rests on.
    """
    T, rate, sps8 = 0.015, 294.0, 8
    n_data, depth = 208, 16
    step90 = round(FS * T / sps8)                      # 90 at 48 kHz
    chirp_at = (12.5 * sps8 + (p4chirp.QUASI_RRC.size - 1) / 2 - sps8 / 2) * step90

    i = np.arange(audio.size)
    t_past = np.maximum(i - chirp_at, 0.0) / FS
    lanes, hdr_err = {}, 0.0
    dmin, crossed = 1.0, 0.0
    for tone, d8, rank in ((1530.0, 0, 0), (550.0, sps8 // 2, 1)):
        mix = audio * np.exp(-2j * np.pi * (tone * i / FS
                                            + rate / 2 * t_past ** 2))
        bb = resample_poly(mix, 1, step90)
        mf = np.convolve(bb, p4chirp.QUASI_RRC)
        y = mf[(p4chirp.QUASI_RRC.size - 1) + d8::sps8][:1 + 8 + n_data]
        d = y[1:] * np.conj(y[:-1])
        dmin = min(dmin, float(np.min(np.abs(d) / np.max(np.abs(d)))))
        # THE DRAGON CROSSES THE PAIR against PACTOR-2: `marker_steps`' first
        # word -- P2's LOWER tone -- is keyed at 1530 Hz, so rank indexes it
        # directly. Measured off the sigidwiki P4dragon recording
        # (Pactor_IV_chirps), four chirp packets, correlation 0.93-1.00 this way
        # against 0.55-0.70 the P2 way.
        steps = pactor2.marker_steps(p4chirp.MARKER_K)
        err = np.angle(np.exp(1j * (np.angle(d[:8]) - steps[rank])))
        hdr_err = max(hdr_err, float(np.max(np.abs(err))))
        crossed = max(crossed, float(abs(np.mean(
            np.exp(1j * (np.angle(d[:8]) - steps[1 - rank]))))))
        # measured P2 DBPSK diagonals: 0 steps 3*pi/4, 1 steps 7*pi/4
        lanes[rank] = (d[8:] * np.exp(-3j * np.pi / 4)).real
    soft_channel = np.stack([lanes[0], lanes[1]], axis=1).reshape(-1)
    fwd = _spec_pointer_walk(2 * n_data, depth)
    soft_code = np.empty(2 * n_data)
    soft_code[fwd] = soft_channel
    return pactor2.decode_frame(soft_code, p4chirp.PATH), hdr_err, dmin, crossed


def _granted(peer, ladder: tuple[str, ...]) -> PtcHost:
    host = PtcHost(peer=peer, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    host.on_rx_event(cs_event(pactor1.CS_SPEED))
    host.tick()
    host.arq.on_host_data(MESSAGE)
    host.arq.cfg.entry_ladder = ladder
    host.on_rx_event(onair.rxfront.Event(
        0.2, "unassigned", f"word {pactor1.CS_59A}", protocol="PACTOR-1",
        spare=pactor1.CS_59A, sense=0))
    return host


class _P4Keyed(Keyed):
    def __init__(self):
        super().__init__()
        self.p4_entries: list[tuple[bytes, int]] = []
        self.entries: list[tuple[int, bytes, int]] = []

    def send_entry_packet(self, sl, payload, status, *, acquire=False):
        self.entries.append((sl, bytes(payload), status))

    def send_p4_entry_packet(self, payload, status):
        self.p4_entries.append((bytes(payload), status))


def main() -> int:
    ok = True

    def check(claim: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {claim}"
              + (f" -- {detail}" if detail and not cond else ""), flush=True)

    print("\nthe geometry is 11.11/11.12's arithmetic:")
    path = p4chirp.PATH
    check("208 symbols x 2 carriers x 1 bit is the 416 transmitted bits",
          path.n_buf == 416 and path.n_symbols == 208
          and path.bits_per_cell == 1)
    check("...halved through R=1/2 to 26 bytes, 22 of them user bytes",
          path.n_pairs == 208 and path.crc_bytes == 25,
          f"{path.n_pairs} pairs, {path.crc_bytes} CRC bytes")
    check("depth 16 divides the packet, as 11.6 requires",
          path.stride == 16 and path.n_buf % path.stride == 0)
    check("the header is P2 speed level 1 / long's codeword",
          p4chirp.MARKER_K == pactor2.marker_index(0, long_frame=True))

    print("\nthe render, against the raster it has to fit:")
    payload = b"P4 CHIRP ENTRY W9SSJ *"
    assert len(payload) == 22
    audio = p4chirp.entry_packet(payload, 0x1A)
    check("3307.5 ms of audio, 11.11's own sum, to the sample",
          audio.size == round(p4chirp.PACKET_S * FS) == 158760,
          f"{audio.size} samples")
    check("THE FINDING: it does not fit one 1.25 s cycle -- it spans three",
          spec.CYCLE_SHORT_S < p4chirp.PACKET_S < 3 * spec.CYCLE_SHORT_S)
    room = 3 * spec.CYCLE_SHORT_S - p4chirp.PACKET_S
    check("...and 3.75 s is PACTOR-4 chirp mode's own cycle: the 442.5 ms left "
          "before the third boundary clears the 210 ms listen floor that "
          "grounds 'burst'",
          abs(room - 0.4425) < 1e-9 and room * FS >= onair.P3_CS_N,
          f"{room * 1e3:.1f} ms against {onair.P3_CS_N / FS * 1e3:.0f}")

    seg = audio[round(0.02 * FS):round(0.18 * FS)]
    S = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    f = np.fft.rfftfreq(seg.size, 1 / FS)
    lo = f[f < 1000][np.argmax(S[f < 1000])]
    hi = f[(f > 1000) & (f < 2200)][np.argmax(S[(f > 1000) & (f < 2200)])]
    check("before the chirp the carriers sit at 550 and 1530 Hz (11.7.1)",
          abs(lo - 550) < 25 and abs(hi - 1530) < 25, f"{lo:.0f}, {hi:.0f}")

    print("\nread back through the section-11 demodulator:")
    field, hdr_err, dmin, crossed = _demod(audio)
    want = placement.field_info(payload, 22, 0x1A)
    check("the header differentials are the P2 codeword's eight steps, on both "
          "carriers", hdr_err < 0.35, f"worst {hdr_err:.2f} rad")
    check("every differential carries energy -- the T/2 stagger and the "
          "de-chirp both hold", dmin > 0.5, f"min {dmin:.2f}")
    check("...on the DRAGON's carrier assignment, not PACTOR-2's: the crossed "
          "pair scores what it scores on a real packet keyed the other way",
          crossed < 0.7, f"crossed {crossed:.2f}")
    def crc_ok(f: bytes) -> bool:
        return (int.from_bytes(f[-2:], "little")
                == coding.crc16(f[:-2]) ^ p4chirp.CHIRP_CRC_XOR)

    check("the field comes back byte-exact: payload, status, and a CRC-16 "
          "that checks", field[:-2] == want and crc_ok(field), field.hex())

    empty, *_ = _demod(p4chirp.entry_packet(b"", 0x1A))
    check("an empty payload takes the template fill, the one convention the "
          "document does not carry",
          empty[:-3] == spec.field_fill(22), empty.hex())

    print("\nagainst four real P4dragon packets:")
    check("their stored CRCs are X-25 turned by CHIRP_CRC_XOR, every one",
          all(crc_ok(f) for f in REAL_FIELDS),
          " ".join(f"{int.from_bytes(f[-2:], 'little') ^ coding.crc16(f[:-2]):04x}"
                   for f in REAL_FIELDS))
    check("...and X-25 alone checks none of them, which is what the constant is",
          not any(int.from_bytes(f[-2:], "little") == coding.crc16(f[:-2])
                  for f in REAL_FIELDS))
    check("the status byte counts transmissions: f5, f6, f7, then f5 again "
          "where the wideband packet restarts the run",
          tuple(f[22] for f in REAL_FIELDS) == (0xF5, 0xF6, 0xF7, 0xF5),
          " ".join(f"{f[22]:02x}" for f in REAL_FIELDS))

    print("\nthe wiring, and the default it must not move:")
    check("the shipped ladder is still exactly ('template',)",
          arq_mod.ArqConfig().entry_ladder == ("template",),
          str(arq_mod.ArqConfig().entry_ladder))
    check("'p4chirp' is a rung the command line will build",
          onair._entry_ladder("template,p4chirp") == ("template", "p4chirp"))
    check("...and is not grounded",
          "p4chirp" not in arq_mod.ENTRY_RUNGS_GROUNDED)

    peer = _P4Keyed()
    host = _granted(peer, ("p4chirp",))
    check("a grant on the p4chirp rung keys the chirp entry, empty field",
          peer.p4_entries and peer.p4_entries[0][0] == b""
          and not peer.entries,
          f"{len(peer.p4_entries)} chirp, {len(peer.entries)} P3")

    peer = _P4Keyed()
    host = _granted(peer, arq_mod.ArqConfig().entry_ladder)
    check("the default ladder never reaches it, even on a peer that offers it "
          "-- the PACTOR-3 entry path is the control arm and renders as today",
          peer.entries and not peer.p4_entries,
          f"{len(peer.entries)} P3, {len(peer.p4_entries)} chirp")
    del host

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
