# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-1 data frames: the bytes real stations sent, and the noise that is not one.

`decode_p1_packets` used to be a CRC-16 and nothing else, and a CRC-16 is not a
gate for a scan that reads thousands of candidate alignments per second. Measured:
240 s of white noise yielded 131 "packets", one every 1.8 s, and the 25 regression
fixtures -- FT8, VARA, PACTOR-II/III, an empty band -- yielded 83 more. None of it
was PACTOR-1. On a live link that is a session claiming data it never received.

It was not a coding bug and could not be fixed inside the CRC. The scan reads two
bauds x two polarities x 97 sub-symbol offsets per envelope rising edge, and 33161
raw accepts over 2,185,085,562 alignments of the 10.36 hour corpus against the
33340 that 2**-16 per trial predicts is combinatorics behaving exactly as it must.

Two gates fixed it, and neither weakens the CRC:

  * the header byte, which the CRC does not cover, must be one of the two the mode
    has (0x55/0xAA -- complements, so the test is blind to shift sense);
  * the FSK eye across the frame's own symbols must actually be open (`EYE_MIN`).

Both directions are asserted here, because a gate that silences the noise by also
silencing real frames is not a fix. The positives are byte-exact against what three
real stations transmitted -- NOT against our own encoder, which could agree with a
broken decoder -- and the negatives sweep the fixtures that are provably not
PACTOR-1 and require zero.

Run: python -m hfmodem.tests.shrike.test_p1data
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from hfmodem.shrike import arq, compress, p1rx, pactor1, ptc, spec
from hfmodem.tests.kestrel import corpora

# The recordings are not in the repo, and where they live is one fact with one
# home: `corpora` resolves `HFMODEM_CORPUS` or the shared corpus beside this tree.
CORPUS = corpora.REGRESS_FIXTURES
FS = p1rx.FS
ok = True
CHECKS = 0


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok, CHECKS
    ok &= passed
    CHECKS += 1
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


# What three independent stations put on the air, as recorded. These are the whole
# point of the file: our encoder cannot make them pass, because it did not write
# them. The status bytes disagree with each other in two places -- see below -- so
# they also pin that the decoder is not tuned to either station.
OFFAIR = {
    "pos_pactor1_local": [
        # W4DNA's link-setup announcement, retransmitted on five consecutive
        # cycles. The anchor the CRC variant and byte order were settled against.
        (100, 0xAA, 0x31, b"1w4dna\r"),
    ],
    "pos_p1_data_jn36lf": [
        # A European station on 14110 kHz, 100 Bd, three packets back to back at
        # t=14.28/15.25/16.23. The counter wraps 3 -> 0 and the header alternates
        # with it; the third packet repeats the second with the shift inverted.
        (100, 0x55, 0x03, b"|1O|KM17"),
        (100, 0xAA, 0x04, b"c\xa02\xf6\xa6\x9f\xff\xe1"),
    ],
    "oracle_pactor3_dl6maa": [
        # DL6MAA's link-setup announcement at t=3.400, and the only 200 Bd frame
        # any station but ours has put on record. Its first symbol lies 3.4 symbols
        # PAST the envelope onset (`p1rx.RISE_SYMBOLS`), which is why a search that
        # only widened ahead of the onset could not reach it anywhere in the
        # recording. The field is Huffman, not ASCII -- see DL6MAA_TEXT.
        (200, 0xAA, 0x35,
         b"\x148\x84.-\x04\xf3\xf0xx<<\x1e\x1e\x0f\x8f\x87\xc7\xc3\xe3"),
    ],
}

DL6MAA_TEXT = b"1dl6maa\r"
"""What the announcement's field decompresses to -- the system level number and
the master address, which is the whole reason the packet matters.

Two things it pins. An independent decoder reads the same nine characters off the
same audio, so this is a third-party payload rather than our own render coming
back. And the data mode is bits 2-3 ALONE: the field is plain Huffman (mode 1),
while the three-bit read of the same status byte says PMC German swapped and
produces `1DIT)   WS DUNZ0 AN`. Bit 4 is set here because the announcement's
capability declaration occupies bits 4-5, not because the mode needs it."""

# Provably not PACTOR-1 data. Every one of these produced 200 Bd "packets" from the
# unguarded decoder; `pos_p1_twosided_14110` is the sharpest of them, because its
# false accepts were a REAL 100 Bd signal read at 200 Bd (payloads of alternating
# 55/AA) rather than noise, and they scored the highest eye of anything in the
# corpus. A guard tuned only against silence would still let those through.
NEGATIVE = ("neg_noise_chatham", "neg_ft8_local", "neg_ft8_australia",
            "neg_ft8_weak_remote", "neg_narrow_200hz", "neg_wide_560hz",
            "edge_wspr_australia", "edge_winlink_oceania", "edge_clipped_9pct",
            "edge_long_80m", "edge_ardop_suspect", "pos_vara_cr",
            "pos_vara_dbpsk", "pos_vara_session", "pos_vara_netherlands",
            "qso_vara_ko2f", "qso_vara_ns0a", "qso_vara_kc9ghz",
            "oracle_pactor2_ref",
            "watch_pactor3_maryland", "pos_p1_twosided_14110")

CHUNK = int(30 * FS)
"""The scan is chunked at 30 s because `_burst_onsets` thresholds at 0.3 of the
buffer's PEAK envelope: one loud burst in a long recording raises the bar above
every quiet one, and a 172 s fixture read whole hides its own weak frames. This is
how the corpus sweep was run and how a streaming front end sees audio anyway."""


def _host_port(pkt) -> bytes:
    """What a station holding this link hands its host for one decoded packet.

    The seam the data mode's width actually reaches. `arq.PactorArq` is given the
    packet at `P1_SPEED_LEVEL` and under the protocol `rxfront` read it in, which
    is what tells the link layer which status-byte layout the field was coded
    under -- the level alone cannot, since the PACTOR-2 seam reports it too.
    """
    host = ptc.PtcHost(peer=None, mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.on_rx_connect("W9SSJ", "DL6MAA")
    host.arq.on_rx_packet(arq.P1_SPEED_LEVEL, pkt.payload, pkt.status, True,
                          protocol=spec.Protocol.PACTOR1)
    return bytes(host.channel(host.ptchn).rx)


def packets(audio: np.ndarray) -> list:
    return [p for i in range(0, max(1, audio.size), CHUNK)
            if audio[i:i + CHUNK].size >= FS
            for p in p1rx.decode_p1_packets(audio[i:i + CHUNK])]


def main() -> int:
    if not CORPUS.exists():
        print(f"  [SKIP] {CORPUS} absent -- set HFMODEM_CORPUS to the off-air "
              "recordings")
        return 2

    # -- the positives: exact bytes, off the air ----------------------------
    decoded: dict[str, list] = {}
    for name, want in OFFAIR.items():
        f = CORPUS / f"{name}.wav"
        if not f.exists():
            print(f"  [SKIP] {f.name} absent")
            continue
        got = packets(p1rx.load_wav(str(f)))
        decoded[name] = got
        seen = {(p.baud, p.header, p.status, p.payload) for p in got}
        for w in want:
            check(f"{name}: off-air packet {w[3]!r} decodes byte-exact",
                  w in seen,
                  f"got {sorted(seen)}" if w not in seen else "")
        check(f"{name}: ...and nothing else is accepted", len(got) == len(want),
              f"{len(got)} packets, expected {len(want)}")

    # The two stations disagree, and that is asserted rather than remembered,
    # because both disagreements are live temptations to add another guard.
    #
    # W4DNA sets the status byte's bits 4-5 ("noch nicht belegt" in the 1990
    # description); the JN36lf station leaves them clear. Gating on either reading
    # rejects the other station for every cycle of a whole session. The same goes
    # for the header/counter parity: W4DNA sends odd counts under 0xAA, and JN36lf
    # sends count 3 under 0x55 and count 0 under 0xAA -- the opposite convention.
    #
    # WHAT A RECEIVER MUST ACCEPT IS NOT WHAT A TRANSMITTER SHOULD SEND, and
    # reading these two lines as one fact is what stalled a whole month of links.
    # Only ONE of these two stations is being acknowledged: JN36lf's counter
    # advances 3 -> 0, W4DNA's is stuck at 1 across five cycles against a peer
    # repeating CS4. `pactor1.status_byte` copied W4DNA and announced PMC German
    # compression on plain ASCII to every gateway it called.
    w4dna = OFFAIR["pos_pactor1_local"][0]
    jn36 = OFFAIR["pos_p1_data_jn36lf"]
    check("the two stations disagree on status bits 4-5 -- so neither is a gate",
          (w4dna[2] & 0x30) == 0x30 and all((s & 0x30) == 0 for _, _, s, _ in jn36),
          f"W4DNA 0x{w4dna[2]:02x} vs JN36lf "
          + ", ".join(f"0x{s:02x}" for _, _, s, _ in jn36))
    check("...and on header/counter parity -- so that is not a gate either",
          (w4dna[2] & 1) == (w4dna[1] == pactor1.DATA_HEADER)
          and any((s & 1) != (h == pactor1.DATA_HEADER) for _, h, s, _ in jn36),
          "; ".join(f"hdr 0x{h:02x} count {s & 3}" for _, h, s, _ in jn36))
    # The status byte's declared data mode matches what its payload looks like:
    # the type-0 frame is printable ASCII end to end, the type-1 (Huffman) frame is
    # not. Two independent fields agreeing is what separates these from a lucky
    # checksum, and (95/256)**8 says the ASCII one is not chance.
    ascii_pkt = next(p for p in jn36 if (p[2] >> 2) & 3 == 0)
    huff_pkt = next(p for p in jn36 if (p[2] >> 2) & 3 == 1)
    check("the declared data mode matches the payload it carries",
          all(32 <= b < 127 for b in ascii_pkt[3])
          and not all(32 <= b < 127 for b in huff_pkt[3]),
          f"type 0 {ascii_pkt[3]!r}, type 1 {huff_pkt[3]!r}")

    # The 200 Bd announcement carries its address through the compressor, so the
    # bytes above are only half the frame: what the far end MEANT is the other
    # half, and it is what an independent decoder can be held against.
    dl6maa = OFFAIR["oracle_pactor3_dl6maa"][0]
    check("the 200 Bd announcement decompresses to its own callsign",
          compress.decompress(dl6maa[3], (dl6maa[2] >> 2) & 3) == DL6MAA_TEXT,
          repr(compress.decompress(dl6maa[3], (dl6maa[2] >> 2) & 3)))

    # AND THE LINK LAYER HAS TO READ IT THE SAME WAY, which is a separate claim
    # and the one an operator sees: `arq.PactorArq` feeds the decompressor from
    # the status byte itself, and it read three bits there while this file read
    # two, so DL6MAA's announcement reached the host data port as
    # `1DIT)   WS DUNZ0 AN`. Driven from the packets the decoder just took off
    # the recordings, and both stations, because the two disagree on bit 4 and a
    # reading that is right about one under three bits is wrong about the other.
    for name, want in (("oracle_pactor3_dl6maa", DL6MAA_TEXT),
                       ("pos_pactor1_local", b"1w4dna\r")):
        for pkt in decoded.get(name, ()):
            got = _host_port(pkt)
            check(f"{name}: its announcement reaches the host data port as "
                  f"{want.decode('latin-1')!r}", got == want, repr(got))

    # -- the negatives: swept, and required to be silent --------------------
    #
    # One hand-picked recording proves nothing. `decode_p1_packets` returns only
    # what it accepts, so the test is on the COUNT -- checking `is not None` here
    # would pass trivially and measure nothing.
    swept = accepted = 0.0
    bad = []
    for name in NEGATIVE:
        f = CORPUS / f"{name}.wav"
        if not f.exists():
            continue
        a = p1rx.load_wav(str(f))
        swept += a.size / FS
        got = packets(a)
        accepted += len(got)
        if got:
            bad.append(f"{name}: {[(p.baud, p.payload[:8]) for p in got]}")
    check("no foreign signal decodes as a PACTOR-1 data packet", not bad,
          "; ".join(bad))
    check("the sweep actually exercised the decoder", swept >= 500,
          f"{swept:.0f} s of negative audio")

    # White noise is the cleanest statement of the failure, because there is
    # nothing in it at all: 131 accepts in 240 s before the gates.
    rng = np.random.default_rng(0)
    noise_hits = sum(len(p1rx.decode_p1_packets(rng.normal(0, 0.05, 30 * FS)))
                     for _ in range(8))
    check("240 s of white noise yields no packets at all", noise_hits == 0,
          f"{noise_hits} accepted (was 131 before the header and eye gates)")

    # -- the other direction: the gates must not have deafened the decoder ---
    #
    # A guard that kills the noise by killing real frames passes everything above.
    # So inject a packet the corpus does not contain into REAL off-air noise and
    # require it back. Both bauds, because 200 Bd is the path with no off-air
    # anchor anywhere in the corpus and this synthetic round trip is the only
    # sensitivity figure it has.
    #
    # Levels are in the 400 Hz the signal occupies. Measured over 24 placements
    # per point, the shipped decoder finds the frame 24/24 down to +4 dB at 100 Bd
    # and +6 dB at 200 Bd, and the CRC itself collapses about two decibels below
    # that whatever the gates do (3 of 24 at +2 dB, unguarded). The floors asserted
    # sit at the measured cliff, not above it.
    nf = CORPUS / "neg_noise_chatham.wav"
    if nf.exists():
        from scipy.signal import butter, filtfilt
        noise = p1rx.load_wav(str(nf))
        b, a_ = butter(4, [1300 / (FS / 2), 1700 / (FS / 2)], btype="band")
        rng = np.random.default_rng(11)
        for baud, payload, snr in ((100, b"HELLO W9", 4.0),
                                   (200, b"DE W9SSJ QTC 1 MSG F", 6.0)):
            sig = pactor1.data_signal(payload, baud, repeats=1)
            s = sig / (np.abs(sig).max() or 1.0)
            found = spurious = 0
            for _ in range(12):
                off = int(rng.integers(0, noise.size - sig.size - FS))
                seg = noise[off:off + sig.size + FS].copy()
                amp = filtfilt(b, a_, seg).std() * 10 ** (snr / 20) / (s.std() or 1)
                seg[FS // 2:FS // 2 + sig.size] += amp * s
                got = p1rx.decode_p1_packets(seg)
                found += any(p.payload == payload for p in got)
                spurious += sum(1 for p in got if p.payload != payload)
            check(f"a {baud} Bd packet at {snr:+.0f} dB in real off-air noise is "
                  f"still found", found == 12, f"{found}/12")
            check(f"...and {baud} Bd finds nothing else in the same audio",
                  spurious == 0, f"{spurious} spurious")

        # The sensitivity check must be able to FAIL, or it is measuring nothing.
        # Far below the cliff the frame must NOT come back -- and if this ever
        # starts passing, the gate above has stopped meaning what it says.
        sig = pactor1.data_signal(b"HELLO W9", 100, repeats=1)
        s = sig / (np.abs(sig).max() or 1.0)
        weak = 0
        for _ in range(12):
            off = int(rng.integers(0, noise.size - sig.size - FS))
            seg = noise[off:off + sig.size + FS].copy()
            amp = filtfilt(b, a_, seg).std() * 10 ** (-8 / 20) / (s.std() or 1)
            seg[FS // 2:FS // 2 + sig.size] += amp * s
            weak += any(p.payload == b"HELLO W9"
                        for p in p1rx.decode_p1_packets(seg))
        check("-8 dB is NOT enough (the sensitivity check can still fail)",
              weak == 0, f"{weak}/12 decoded")

    # -- the threshold sits between two measured populations ----------------
    #
    # Not a tuned constant. Pure white noise tops out at 0.416; the weakest CORRECT
    # decode at an SNR where the frame still arrives 9 times in 10 scores 0.561.
    # Asserting the bracket keeps a future edit from drifting EYE_MIN into either
    # population without measuring again.
    #
    # The eye is NOT the only thing holding the far edge, and the constant's own
    # docstring says so: a CRC-valid flutter on 14105 kHz reaches 0.5056, above this
    # threshold, and it is the head gate that refuses it. Which is why the negative
    # sweep above is on the whole decoder and not on either gate alone.
    check("EYE_MIN sits above the pure-noise ceiling", p1rx.EYE_MIN > 0.416,
          f"EYE_MIN={p1rx.EYE_MIN}")
    check("...and below the weakest correct decode", p1rx.EYE_MIN < 0.561,
          f"EYE_MIN={p1rx.EYE_MIN}")
    # The header pair must stay complementary, or the gate acquires a preference
    # for one shift sense and drops every second packet a peer sends.
    check("the two legal headers are complements -- the gate is polarity-blind",
          p1rx.P1_DATA_HEADERS[0] ^ p1rx.P1_DATA_HEADERS[1] == 0xFF,
          f"{[hex(h) for h in p1rx.P1_DATA_HEADERS]}")

    print("\nALL PASS" if ok else "\nFAILED")
    # Nothing ran. Individual fixtures `continue` when absent, so a file
    # whose corpus directory exists but whose recordings do not would
    # otherwise report a green pass having asserted nothing — which is the
    # exact difference between "we checked" and "we could not".
    if CHECKS == 0:
        print("  [SKIP] nothing was checked — no fixture was present")
        return 2
    return 0 if ok else 1


def test_main() -> None:
    rc = main()
    if rc == 2:
        pytest.skip(f"no fixtures under {CORPUS} -- set HFMODEM_CORPUS to the "
                    "off-air recordings")
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
