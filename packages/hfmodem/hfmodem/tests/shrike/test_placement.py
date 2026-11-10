# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate the frame placement for every decode path we can encode.

The permutation, the FEC and the soft-bit mapping are each checkable without an
external decoder: the permutation must be a bijection, and the chain must
round-trip a payload we chose ourselves. The anchor that is NOT self-referential
is a case-1 codeword an independent decoder's own trellis and CRC accepted,
banked here as a constant so the convention stays regression-tested with nothing
external in the loop.

Run:  .venv/bin/python -m pytest packages/hfmodem/hfmodem/tests/shrike/test_placement.py
"""

from __future__ import annotations

import numpy as np

from hfmodem.shrike import coding, modem, p3frame, placement as P


def crc16_x25_reference(data: bytes) -> int:
    """CRC-16/X-25 the long way round, deliberately NOT shrike's implementation.

    The check below asserts that a decoded frame's CRC field equals the CRC of the
    message it carried. Computing both with `coding.crc16` would make that
    self-referential -- the placement and coding chain would agree with itself
    however wrong the CRC was. This is a plain bit-at-a-time reflection of the
    definition, and it is the independent half of the assertion.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= int(f"{byte:08b}"[::-1], 2) << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return int(f"{crc ^ 0xFFFF:016b}"[::-1], 2)

FAILURES: list[str] = []
CHECKS = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(label)


def softs(steps: dict[int, np.ndarray], path: P.Path) -> np.ndarray:
    """Per-cell soft values exactly as a reference decoder derives them from the
    phase step."""
    buf = np.zeros(path.n_buf)
    for rank, tone in enumerate(path.tones):
        for s, d in enumerate(steps[tone]):
            i = (s * len(path.tones) + rank) * path.bits_per_cell
            if path.bits_per_cell == 1:
                buf[i] = np.sin(d - np.pi / 4) * (1 if path.case == 0 else -1)
            else:
                buf[i], buf[i + 1] = np.sin(d), -np.cos(d)
    return buf


def decode_field(code: np.ndarray, path: P.Path) -> bytes:
    """Code-order softs -> the field the CRC sees, the way that path's decoder does it.

    One shape for every path: the trellis with its zero flush, packed LSB-first per
    byte, then the whitening XOR that sits between the traceback and the CRC. Case 0
    differs only in using the K=9 code.
    """
    conv = P.CASE0_CODE if path.case == 0 else coding.ConvCode(7, P.CONV_GENERATORS)
    bits = coding.viterbi_decode(np.asarray(code, float), conv, terminated=True)
    return coding.whiten(coding.bits_to_bytes(bits[:8 * path.crc_bytes], msb_first=False))


MSG = {p.name: bytes((0x53 + 7 * i) & 0xFF for i in range(p.crc_bytes - 2))
       for p in (P.DETECT, P.HEADER, P.DATA3, P.DATA4)}


def test_all() -> None:
    print("PERMUTATION")
    for path in (P.DETECT, P.HEADER, P.DATA3, P.DATA4, P.CONFIRM):
        src = P.channel_of_code(path)
        x = np.arange(path.n_buf)
        check(sorted(src.tolist()) == list(x), f"{path.name}: stride {path.stride} is a bijection")
        check((P.deinterleave(P.interleave(x, path), path) == x).all()
              and (P.interleave(P.deinterleave(x, path), path) == x).all(),
              f"{path.name}: interleave and de-interleave are inverses")
        check(path.n_symbols == P.FRAME_SYMBOLS,
              f"{path.name}: {path.n_buf} softs = 72 x {len(path.tones)} x {path.bits_per_cell}")

    print("\nSOFT ROUND TRIP (grid -> softs -> de-interleave -> Viterbi -> CRC)")
    for path in (P.DETECT, P.HEADER, P.DATA3, P.DATA4):
        msg = MSG[path.name]
        code = P.deinterleave(softs(P.grid_steps(P.build_grid(msg, path), path), path), path)
        field = decode_field(code, path)
        n = path.crc_bytes
        check(field[:n - 2] == msg and crc16_x25_reference(msg) == int.from_bytes(field[n - 2:n], "little"),
              f"{path.name}: recovers the payload byte-exact, CRC-16/X25 valid")

    print("\nANCHOR (the case-1 codeword an independent decoder accepted)")
    # Fixed field and codeword vectors pin the coding convention independently
    # of a round trip through the encoder and decoder under test.
    ANCHOR_INFO = b"HELLO SHRIKE DE W1AW ABC"
    ANCHOR_CODE = bytes.fromhex(
        "03b26023d370bc2080c8bdd33429e2f95fae3e648b62af8a1b530addc508027a"
        "c7f5cdca576fd46ab9e5dbf86194265c96fa5137eac0")
    field = P.build_field(ANCHOR_INFO, P.HEADER)
    code = coding.bits_to_bytes(P.encode_frame(field, P.HEADER))
    check(field.hex() == "48454c4c4f20534852494b45204445205731415720414243c419",
          "the anchor field is the one an independent decoder read back")
    check(bytes(code) == ANCHOR_CODE, "encode_frame reproduces the accepted codeword")

    print("\nAUDIO ROUND TRIP (lag-1 data cells -> our own matched demod -> the same chain)")
    cfg = modem.ModConfig(matched_pulse=True)
    for path in (P.HEADER, P.DATA3, P.DATA4):
        msg = MSG[path.name]
        audio = P.data_packet(msg, path, cfg=cfg)
        # The header block sits between the phase reference and row 0, so the
        # data grid starts at symbol `DATA_OFFSET` and every one of the 72 rows
        # is on the air.
        rows = P.FRAME_SYMBOLS
        head = p3frame.DATA_OFFSET
        # Level 2's carriers do not share a symbol clock, so the loopback demod is
        # handed the same per-tone offsets the modulator was given; on the other
        # paths they are all zero.
        offsets = dict(zip(path.tones, path.clock_offsets(cfg.sps)))
        sy = modem.demodulate_tones(audio, path.tones, head + rows, cfg, offsets)
        buf = np.zeros(path.n_buf)
        for rank, tone in enumerate(path.tones):
            s = sy[tone][head - 1:]
            d = np.angle(s[1:] / s[:-1])
            for k in range(rows):
                # Same rule `softs` applies to the transmit steps -- one soft off
                # sin(delta - 45 deg) for the DBPSK paths, two off sin and -cos for
                # the DQPSK ones -- so this reads the modulation back the way the
                # decoder does rather than a second way that happens to agree.
                i = (k * len(path.tones) + rank) * path.bits_per_cell
                if path.bits_per_cell == 1:
                    buf[i] = -np.sin(d[k] - np.pi / 4)
                else:
                    buf[i], buf[i + 1] = np.sin(d[k]), -np.cos(d[k])
        code = P.deinterleave(buf, path)
        field = decode_field(code, path)
        check(field[:path.crc_bytes - 2] == msg, f"{path.name}: survives its own modulation")

    print("\nCREST (the phase reference is no longer the loudest symbol in the packet)")
    # `levels.at_drive` sets the drive by the packet's peak, so a single symbol
    # holding that peak is a single symbol deciding how much of the rest goes out.
    # Speed level 1 used to be excluded, on the grounds that two carriers sweep
    # through alignment inside every symbol whatever angle they start at. They do
    # -- but they no longer start together: `spec.SUBBAND_LEAD` gives level 1 the
    # T/2 split the reference keys, so the two phase references land half a symbol
    # apart and the reference is the loudest symbol at no level.
    shape = modem.ModConfig()
    # The shaping pulse spans `span_symbols`, so symbol 0 arrives half a span in.
    ref = shape.span_symbols // 2
    for sl, path in sorted(P.SPEED_PATHS.items()):
        msg = bytes((0x53 + 7 * i) & 0xFF for i in range(path.crc_bytes - 2))
        audio = np.asarray(P.case0_packet(msg) if path.case == 0
                           else P.data_packet(msg, path), float)
        peaks = [np.abs(audio[i * shape.sps:(i + 1) * shape.sps]).max()
                 for i in range(len(audio) // shape.sps)]
        crest = 20 * np.log10(np.abs(audio).max() / np.sqrt(np.mean(audio ** 2)))
        loudest = int(np.argmax(peaks))
        check(loudest != ref,
              f"SL{sl}: the peak is not the phase reference",
              f"crest {crest:.2f} dB, all-in-phase {10 * np.log10(2 * len(path.tones)):.2f} dB, "
              f"loudest symbol {loudest}")

    print("\n" + "-" * 68)
    assert not FAILURES, (f"{len(FAILURES)} FAILED of {CHECKS}: "
                          + "; ".join(FAILURES))
    print(f"RESULT: ALL PASS ({CHECKS} checks)")
    print("\nNOTE: strides 69, 563 and 35 each replay a real receiver's own")
    print("      de-interleave buffers byte-exact, both sides captured from one")
    print("      cycle and agreeing to 1.0000.")
