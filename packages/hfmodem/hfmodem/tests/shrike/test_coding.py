# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate the coding chain against synthetic ground truth.

No external decoder is involved here, and none is needed: a codec either
round-trips bit-exactly or it does not. Establishing that first is what makes a
later rejection by a real receiver diagnostic -- the fault is then in a
PACTOR-specific unknown rather than in this Viterbi.

Run:  .venv/bin/python -m pytest packages/hfmodem/hfmodem/tests/shrike/test_coding.py
"""

from __future__ import annotations

import numpy as np


from hfmodem.shrike import coding, spec, unknowns  # noqa: E402

rng = np.random.default_rng(20260713)
FAILURES: list[str] = []
CHECKS = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        FAILURES.append(label)


def to_soft(bits: np.ndarray) -> np.ndarray:
    """Noiseless soft values: +1 for 0, -1 for 1."""
    return np.where(np.asarray(bits) == 0, 1.0, -1.0)


def test_all() -> None:
    print("Coding-chain validation\n")

    # --- CRC ------------------------------------------------------------------
    print("0. data whitening (not in M.1798; its own inverse)")
    # the table must match the reflected CRC-16/CCITT table a real receiver
    # whitens with, byte-for-byte
    _t = coding.reflected_crc_ccitt_table()
    check(_t[:8] == [0x0000, 0x1189, 0x2312, 0x329b, 0x4624, 0x57ad, 0x6536, 0x74bf],
          "whitening table first 8 entries match an independent decoder")
    check(_t[255] == 0x0F78, "whitening table[255] == 0x0f78, likewise")
    for n in (7, 8, 62, 287):
        d = bytes(rng.integers(0, 256, n, dtype=np.uint8))
        check(coding.whiten(coding.whiten(d)) == d, f"whiten involution, len {n}")

    print("\n1. CRC-16 variants against published check values")
    # "123456789" is the standard check vector for every catalogued CRC.
    CHECK_VECTORS = {
        "ccitt-false": 0x29B1,
        "kermit":      0x2189,
        "x25":         0x906E,
    }
    for variant, expect in CHECK_VECTORS.items():
        got = coding.crc16(b"123456789", variant)
        check(got == expect, f"CRC-16/{variant}('123456789') == {expect:#06x}",
              f"got {got:#06x}")

    # --- bit packing ----------------------------------------------------------
    print("\n2. bit/byte round-trip")
    payload = bytes(rng.integers(0, 256, 64, dtype=np.uint8))
    for msb in (True, False):
        bits = coding.bytes_to_bits(payload, msb_first=msb)
        check(coding.bits_to_bytes(bits, msb_first=msb) == payload,
              f"bytes->bits->bytes (msb_first={msb})")

    # --- convolutional codes: noiseless round-trip ----------------------------
    print("\n3. convolutional encode -> soft Viterbi, noiseless (all speed levels)")
    for sl in sorted(spec.SPEED_LEVELS):
        code = coding.conv_code_for(sl)
        info = rng.integers(0, 2, 200, dtype=np.uint8)
        coded = code.encode(info, terminate=True)
        check(coded.size == (info.size + code.constraint_length - 1) * 2,
              f"SL{sl} K={code.constraint_length}: coded length is 2*(N+K-1)")
        decoded = coding.viterbi_decode(to_soft(coded), code, terminated=True)
        check(np.array_equal(decoded, info),
              f"SL{sl} K={code.constraint_length} "
              f"gen={tuple(oct(g) for g in code.generators)}: round-trips bit-exact")

    # --- puncturing -----------------------------------------------------------
    print("\n4. puncture -> depuncture round-trip and rate check")
    code7 = coding.ConvCode(7, unknowns.U1_GENERATORS.value[7])
    info = rng.integers(0, 2, 240, dtype=np.uint8)
    coded = code7.encode(info, terminate=True)

    for name, pat in coding.CANDIDATE_PUNCTURES.items():
        punctured = pat.apply(coded)
        n_pairs = coded.size // 2
        expected_len = int(round(coded.size * pat.rate[1] / (2 * pat.period)))
        check(abs(punctured.size - expected_len) <= 2,
              f"{name}: kept {punctured.size} of {coded.size} coded bits "
              f"(~rate {pat.period}/{pat.rate[1] * pat.period // pat.rate[1]})",
              f"expected ~{expected_len}")
        # depuncture must restore the original positions, with holes at 0.0
        restored = pat.depuncture(to_soft(punctured), n_pairs)
        check(restored.size == coded.size,
              f"{name}: depuncture restores full coded length")
        kept_mask = restored != 0.0
        check(np.array_equal(np.where(restored[kept_mask] > 0, 0, 1),
                             coded[kept_mask]),
              f"{name}: surviving soft values match the original coded bits")
        # and the punctured code must still decode noiselessly
        decoded = coding.viterbi_decode(restored, code7, terminated=True)
        check(np.array_equal(decoded, info),
              f"{name}: punctured code still decodes bit-exactly (noiseless)")

    # --- interleaver ----------------------------------------------------------
    print("\n5. interleaver is a permutation and inverts exactly")
    for rows, cols in [(8, 16), (16, 8), (12, 12), (7, 23)]:
        il = coding.BlockInterleaver(rows, cols)
        n = min(rows * cols, 100)
        bits = rng.integers(0, 2, n, dtype=np.uint8)
        woven = il.interleave(bits)
        check(sorted(woven.tolist()) == sorted(bits.tolist()),
              f"{rows}x{cols}: interleave is a permutation (multiset preserved)")
        check(np.array_equal(il.deinterleave(woven), bits),
              f"{rows}x{cols}: deinterleave(interleave(x)) == x")

    # --- full chain, bit-exact -------------------------------------------------
    print("\n6. full chain: payload+status+CRC -> conv -> puncture -> interleave -> back")
    for sl in sorted(spec.SPEED_LEVELS):
        s = spec.SPEED_LEVELS[sl]
        code = coding.conv_code_for(sl)
        pat = coding.RATE_1_2
        if s.code_rate == (3, 4):
            pat = coding.CANDIDATE_PUNCTURES["3/4-yasuda"]
        elif s.code_rate == (8, 9):
            pat = coding.CANDIDATE_PUNCTURES["8/9-yasuda"]

        user = bytes(rng.integers(0, 256, s.payload_short, dtype=np.uint8))
        info_packet = user + bytes([spec.status_byte(1)])
        info_packet += coding.crc16(info_packet).to_bytes(2, "big")

        bits = coding.bytes_to_bits(info_packet)
        coded = code.encode(bits, terminate=True)
        punctured = pat.apply(coded)

        n = punctured.size
        cols = s.n_tones * s.bits_per_symbol          # one column per symbol-slot
        rows = int(np.ceil(n / cols))
        il = coding.BlockInterleaver(rows, cols)
        woven = il.interleave(punctured)

        # ... and unwind
        unwoven = il.deinterleave(woven)
        restored = pat.depuncture(to_soft(unwoven), coded.size // 2)
        decoded = coding.viterbi_decode(restored, code, terminated=True)
        out = coding.bits_to_bytes(decoded)

        ok = out == info_packet
        check(ok, f"SL{sl}: {s.payload_short} B payload survives the full chain "
                  f"({s.modulation}, K={s.constraint_length}, "
                  f"rate {s.code_rate[0]}/{s.code_rate[1]})")
        if ok:
            body, crc = out[:-2], int.from_bytes(out[-2:], "big")
            check(coding.crc16(body) == crc, f"SL{sl}: CRC verifies after decode")

    # --- coding gain sanity ----------------------------------------------------
    # A soft Viterbi that does not actually correct errors would still pass every
    # test above. Prove it earns its keep: at a raw channel BER that would wreck an
    # uncoded link, the decoded BER must be dramatically better.
    print("\n7. soft-decision Viterbi actually corrects errors (K=7, rate 1/2)")
    info = rng.integers(0, 2, 2000, dtype=np.uint8)
    coded = code7.encode(info, terminate=True)
    tx = to_soft(coded)
    for esn0_db in (1.0, 3.0, 5.0):
        sigma = np.sqrt(1.0 / (2.0 * 10 ** (esn0_db / 10)))
        rx = tx + rng.normal(0, sigma, tx.size)
        raw_ber = np.mean((rx < 0).astype(np.uint8) != coded)
        dec = coding.viterbi_decode(rx, code7, terminated=True)
        dec_ber = np.mean(dec != info)
        check(dec_ber < raw_ber,
              f"Es/N0={esn0_db:>4} dB: raw BER {raw_ber:.4f} -> decoded BER {dec_ber:.5f}",
              "decoder made things WORSE")

    print("\n" + "-" * 68)
    assert not FAILURES, (f"{len(FAILURES)} FAILED of {CHECKS}: "
                          + "; ".join(FAILURES))
    print(f"RESULT: ALL PASS ({CHECKS} checks)")
    print("\nNOTE: this proves the CODEC is correct. It does NOT prove the")
    print("      PACTOR-specific parameters are right -- only a decode by an")
    print("      independent receiver can say that.")
    print("      Still open:", ", ".join(
        k for k, u in unknowns.REGISTER.items() if u.status == unknowns.OPEN))
