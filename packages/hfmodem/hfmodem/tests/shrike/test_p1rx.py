# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate shrike's PACTOR-1 receiver (shrike/p1rx.py).

Unforgeable gates for the connect-callsign decoder:

  1,2. Round-trip: decode shrike's OWN connect TX (pactor1.connect_signal) for
     several callsigns -> byte-exact recovery, Normal vs Longpath classified.
  3. Real capture: decode captures/real_connect_only.wav -> DL6MAA, the byte
     sequence an independent decoder reports as
     ###CONNECT: [Normal Call: DL6MAA].
  4. On-air bytes: the same capture, demodulated here rather than decoded, has to
     hand back what the ENCODER builds -- address section, redundancy section and
     bit phase. Gate 3 cannot do this: a decoder that hunts for the sync byte
     accepts a frame shifted a whole bit, so it passes on either encoding.
  5. A second station, and where the 200 Bd section falls. The three wrong
     encodings are measured alongside it, so the gate is known to have teeth.

Run:  python -m hfmodem.tests.shrike.test_p1rx
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from hfmodem.shrike import p1rx, pactor1
from hfmodem.tests.shrike import archive

ROOT = archive.ARCHIVE
REAL = "captures/real_connect_only.wav"
REAL_CALL = "DL6MAA"
SPS = p1rx.FS // 100          # 100 Bd address section
SPS_R = p1rx.FS // 200        # 200 Bd redundancy section


def _check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def gate_roundtrip() -> bool:
    print("Gate 1 - round-trip on shrike's own connect TX")
    ok = True
    # 1..7 chars: a 7-char callsign fills the address image up to its 0x0F
    # terminator at T[8]; an 8-char one leaves no room for the terminator in the
    # 9-byte 100-Bd address section, which the encoder cannot represent standalone.
    for call in ["W1AW", "DL6MAA", "K7ABC", "N0CALL", "VK2XYZ", "AB1CDEF"]:
        audio = pactor1.connect_signal(call)
        c = p1rx.decode_connect(audio)
        got = c.callsign if c else None
        v = c.variant if c else None
        ok &= _check(f"Normal {call}", got == call and v == "Normal",
                     f"recovered {got!r} as {v}")
    return ok


def gate_longpath() -> bool:
    print("Gate 2 - Longpath discriminator (own TX, both variants)")
    ok = True
    for call in ["W1AW", "K7ABC"]:
        audio = pactor1.connect_signal(call, variant="longpath")
        c = p1rx.decode_connect(audio)
        ok &= _check(f"Longpath {call}", c is not None and c.callsign == call
                     and c.variant == "Longpath",
                     f"recovered {c.callsign if c else None!r} as "
                     f"{c.variant if c else None}")
        # the Normal build of the same callsign must NOT read as Longpath
        cn = p1rx.decode_connect(pactor1.connect_signal(call, variant="normal"))
        ok &= _check(f"Normal  {call}", cn is not None and cn.callsign == call
                     and cn.variant == "Normal",
                     f"recovered {cn.callsign if cn else None!r} as "
                     f"{cn.variant if cn else None}")
    return ok


def gate_real() -> bool:
    print("Gate 3 - real off-air capture vs an independent decoder")
    c = p1rx.decode_connect(p1rx.load_wav(str(ROOT / REAL)))
    return _check("real_connect_only.wav -> DL6MAA",
                  c is not None and c.callsign == "DL6MAA" and c.variant == "Normal",
                  f"recovered {c.callsign if c else None!r} as "
                  f"{c.variant if c else None} "
                  f"(reference: ###CONNECT: [Normal Call: DL6MAA])")


def _image_start(tones: "p1rx._Tones", what: str = REAL) -> int:
    """Where the address image begins, found the way a receiver finds it.

    A sync byte on the 100 Bd grid followed by a run of parser-valid characters
    and the 0x0F fill to the end of the section -- structure only. Nothing here
    knows the callsign or what the encoder would have produced, so what the gate
    then compares is a measurement and not a restatement.
    """
    bits = tones.mark > tones.space
    n = bits.size - 7 * SPS
    ok = np.ones(n, bool)
    for k in range(8):
        ok &= bits[k * SPS: k * SPS + n] == bool((p1rx.SYNC[0] >> k) & 1)
    cand = np.flatnonzero(ok).astype(np.int64)
    cand = cand[tones.fits(cand, SPS, pactor1.ADDR_LEN * 8)]
    good = [s for s, row in zip(cand, tones.read(cand, SPS, pactor1.ADDR_LEN, False))
            if _is_image(row.tobytes())]
    if not good:
        raise AssertionError(f"no address image found in {what}")
    # The first lock, at the centre of its alignment plateau: every offset in the
    # plateau reads the same bytes, and the centre is the one that also reads the
    # bit before cleanly.
    frame = pactor1.ADDR_LEN * 8 * SPS
    return int(np.median([s for s in good if s - good[0] < frame]))


def _is_image(b: bytes) -> bool:
    body = b[1:]
    if pactor1.PAD not in body:
        return False
    term = body.index(pactor1.PAD)
    return (term >= 3 and all(x == pactor1.PAD for x in body[term:])
            and all(0x2C < x <= 0x5A for x in body[:term]))


def gate_onair_bytes() -> bool:
    """Gate 4 - the bytes and the bit phase a real station puts on the air.

    The address image can be transmitted as-is, or one bit late behind a leading
    zero (`pactor1._inverse_rotate`, which shrike once applied). The two are the
    same bitstream read a bit apart, so every byte-level check passes under both
    and only the RF can separate them: the disputed bit is either dead air, or a
    1600 Hz SPACE if the leading zero is really being sent.

    On this capture it is neither -- it is a MARK at full strength, the tail of
    the transmitter's idle carrier, which is a leading zero's opposite tone. So
    the on-air address section is the image itself, and reading the same audio
    one bit early yields 0xAB where the rotated encoding emits 0xAA.
    See analysis/connect/bit_phase.py for the measurement across the corpus.
    """
    print("Gate 4 - on-air bytes and bit phase, demodulated from the capture")
    audio = p1rx.load_wav(str(ROOT / REAL))
    tones = p1rx._Tones(audio, SPS)
    start = _image_start(tones)
    one = np.array([start], np.int64)
    addr = tones.read(one, SPS, pactor1.ADDR_LEN, False)[0].tobytes()
    red = p1rx._Tones(audio, SPS_R).read(
        one + pactor1.ADDR_LEN * 8 * SPS, SPS_R, pactor1.RED_LEN, False)[0].tobytes()
    want_a, want_r = pactor1.connect_frame_bytes(REAL_CALL)
    ok = _check("address section", addr == want_a,
                f"on air {addr.hex(' ')} vs encoder {want_a.hex(' ')}")
    ok &= _check("redundancy section", red == want_r,
                 f"on air {red.hex(' ')} vs encoder {want_r.hex(' ')}")
    mark, space = float(tones.mark[start - SPS]), float(tones.space[start - SPS])
    ok &= _check("no leading zero bit", mark > 4 * space,
                 f"the bit before the sync is {'MARK' if mark > space else 'SPACE'} "
                 f"by {mark / max(space, 1e-9):.0f}x, and a leading zero is SPACE")
    return ok


def _red_offset(audio: np.ndarray, start: int, want: bytes) -> int | None:
    """Where the 200 Bd section that reads back `want` begins, in 200 Bd bits.

    Measured from the end of the address section, so 0 means the two abut with
    nothing between them. Searched rather than assumed, because the offset is the
    quantity that separates the candidate encodings from each other.
    """
    tones = p1rx._Tones(audio, SPS_R)
    end = start + pactor1.ADDR_LEN * 8 * SPS
    for k in range(-6, 7):
        s = np.array([end + k * SPS_R], np.int64)
        if not tones.fits(s, SPS_R, pactor1.RED_LEN * 8)[0]:
            continue
        if tones.read(s, SPS_R, pactor1.RED_LEN, False)[0].tobytes() == want:
            return k
    return None


def _connect_audio(addr: bytes, red: bytes) -> np.ndarray:
    """A dual-rate connect burst carrying exactly these bytes, in clear air.

    The trailing 200 Bd alternation is probe headroom and NOT part of the frame:
    `_red_offset` slides its read up to +6 symbols past the nominal end, so an
    encoding whose redundancy sits late needs bits there or it reports "no offset
    found" rather than the offset it has -- and those bits land in the top of its
    last byte, so a run of one tone would decide the comparison. A real connect
    frame ends with its redundancy section, at 0.960 s.
    """
    freqs = np.concatenate([
        pactor1._fsk_freqs(pactor1._bits_lsb_first(addr), SPS),
        pactor1._fsk_freqs(pactor1._bits_lsb_first(red), SPS_R),
        pactor1._fsk_freqs((1, 0) * 3, SPS_R)])
    burst = 0.11 * np.cos(2 * np.pi * np.cumsum(freqs) / p1rx.FS)
    pad = np.zeros(p1rx.FS // 2)
    return np.concatenate([pad, burst, pad]).astype(np.float32)


# A third-party off-air call, tracked in this repo, demodulated once by hand and
# written down here as literals. The encoder is compared against THESE, not the
# other way round, so the gate cannot pass by agreeing with a broken encoder.
# captures/offair_KE5YTA_p3.wav, the burst at t = 1.760 s.
KE5YTA_ADDR = bytes.fromhex("55 4b 45 35 59 54 41 0f 0f")
KE5YTA_RED = bytes.fromhex("4b 45 35 59 54 41")


def gate_second_station() -> bool:
    """Gate 5 - a second station, and what the half-change actually did.

    `_inverse_rotate` inserts one leading zero, so a rotated address section runs
    one 100 Bd bit long at the front -- and that is TWO bit periods at 200 Bd.
    Correcting one half alone therefore does not half-fix the frame: it moves the
    redundancy two symbols away from the address section a decoder has just
    locked, and the cross-check secondary[i] == primary[i+1] fails on all six
    bytes rather than none. That is why a reference decoder rejected the change to
    `redundancy_bytes` alone, and the rejection was evidence FOR the plain form.

    What this gate does NOT do is separate the plain frame from the old
    double-rotated one. Both are self-consistent, so both put their redundancy at
    offset 0 and the measurement below reads 0 for each -- asserted here, because
    a gate that quietly could not tell them apart would be worse than one that
    says so. Gate 4 is what separates those two, on the tone of the disputed bit.
    """
    print("Gate 5 - a second station, and the 200 Bd section's offset")
    audio = p1rx.load_wav(str(ROOT / "captures/offair_KE5YTA_p3.wav"))
    start = _image_start(p1rx._Tones(audio, SPS), "offair_KE5YTA_p3.wav")
    addr = p1rx._Tones(audio, SPS).read(
        np.array([start], np.int64), SPS, pactor1.ADDR_LEN, False)[0].tobytes()
    ok = _check("off-air address section", addr == KE5YTA_ADDR,
                f"{addr.hex(' ')} (expected {KE5YTA_ADDR.hex(' ')})")
    ok &= _check("the encoder reproduces both halves",
                 pactor1.connect_frame_bytes("KE5YTA") == (KE5YTA_ADDR, KE5YTA_RED),
                 f"{[b.hex(' ') for b in pactor1.connect_frame_bytes('KE5YTA')]}")
    ok &= _check("the 200 Bd section abuts the 100 Bd section",
                 _red_offset(audio, start, KE5YTA_RED) == 0,
                 f"offset {_red_offset(audio, start, KE5YTA_RED)} bit periods")

    # The wrong encodings, on DL6MAA rather than on KE5YTA, and the reason is a
    # finding rather than a convenience: a rotated address section runs one bit
    # long, so its ninth byte borrows a bit from whatever follows. Under DL6MAA
    # that bit is a zero and the 0x0F fill still reads as fill, so all four
    # encodings lock and their offsets can be compared. Under KE5YTA the
    # redundancy section opens on a one, byte 8 reads 0x8F, and two of the four
    # do not present an address image at all. The offsets below are geometry --
    # one 100 Bd bit is two bit periods at 200 Bd -- and do not depend on the
    # callsign; whether a decoder gets far enough to see them does.
    T = pactor1.address_image("DL6MAA")
    rot = pactor1._inverse_rotate
    for name, a, r, want in [
        ("plain address + plain redundancy", T, T[1:7], 0),
        ("rotated address + double-rotated redundancy", rot(T), rot(rot(T)[1:7]), 0),
        ("rotated address + plain redundancy", rot(T), T[1:7], -2),
        ("plain address + double-rotated redundancy", T, rot(rot(T)[1:7]), 2),
    ]:
        synth = _connect_audio(a, r)
        got = _red_offset(synth, _image_start(p1rx._Tones(synth, SPS), name),
                          T[1:7])
        ok &= _check(f"{name} -> {want:+d}", got == want, f"offset {got}")
    return ok


def main() -> int:
    gates = [gate_roundtrip(), gate_longpath(), gate_real(), gate_onair_bytes(),
             gate_second_station()]
    print()
    if all(gates):
        print("ALL PASS")
        return 0
    print("FAILURES PRESENT")
    return 1


@pytest.mark.skipif(
    not (ROOT / REAL).exists(),
    reason=f"the off-air PACTOR-1 connect capture is not at {ROOT / REAL} — it "
           "lives in the working record, which does not cross the publication "
           "boundary")
def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
