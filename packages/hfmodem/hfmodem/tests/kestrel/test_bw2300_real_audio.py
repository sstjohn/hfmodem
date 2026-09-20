# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW2300 interop proof — kestrel decodes REAL captured VARA HF BW2300 audio.

The staged capture `analysis/caps/bw2300/AAAA1-BBBB2-bw2300-counter512__a2b.wav`
is real VARA HF, BW2300, DATA-over audio (station A's TX), known payload
`counter512` (byte at position p = p%256), 6 DATA overs of 89 bytes + 1 link-setup
burst. kestrel's BW2300 base receiver — built only from spec/01 §2300 (the promoted
bin-placement law) + spec/tables/bw2300 — must recover every DATA over CRC-clean
and reconstruct counter512 byte-exact. This proves kestrel's placement bit-matches
VARA's on-air layout.
"""
from __future__ import annotations

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.tests.kestrel.corpora import BW2300_CAPTURE, requires_bw2300_capture

_COUNTER = bytes([p % 256 for p in range(512)])


def _over_index(payload89: bytes):
    for o in range(6):
        exp = _COUNTER[89 * o:89 * o + 89] if o < 5 else _COUNTER[445:512]
        if payload89[:len(exp)] == exp:
            return o
    return None


@requires_bw2300_capture
def test_decode_real_vara_bw2300_counter512():
    frames = rx.decode_wav(str(BW2300_CAPTURE))
    clean = [f for f in frames if f.crc_ok]
    overs = {}
    for f in clean:
        o = _over_index(f.payload[:89])
        if o is not None:
            overs[o] = f.payload[:89]

    # all 6 DATA overs decode CRC-clean and byte-exact vs counter512
    assert set(overs) == set(range(6)), f"missing DATA overs: got {sorted(overs)}"
    for o in range(5):
        assert overs[o] == _COUNTER[89 * o:89 * o + 89], f"over {o} not byte-exact"
    assert overs[5][:67] == _COUNTER[445:512], "partial over 5 not byte-exact"

    # reconstruct the full 512-byte counter payload from the 6 overs
    recovered = b"".join(overs[o] for o in range(5)) + overs[5][:67]
    assert recovered == _COUNTER, "counter512 not reconstructed byte-exact"


if __name__ == "__main__":
    frames = rx.decode_wav(str(BW2300_CAPTURE))
    clean = sum(f.crc_ok for f in frames)
    overs = sorted(o for f in frames if f.crc_ok
                   for o in [_over_index(f.payload[:89])] if o is not None)
    print(f"segments={len(frames)} CRC-clean={clean} DATA-overs={overs} "
          f"(pass rate {len(overs)}/6 DATA overs)")
