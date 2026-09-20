# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Monitor mode: audio in, a timestamped text dump of everything shrike decodes.

A text renderer over `shrike.rxfront` -- the exact same decode path the live modem
receiver in `shrike.ptc` runs, so the monitor is a faithful dry-run of the session
receiver, plus the whole-file PACTOR-2 data pass (`_p2_data_events`) that only a
recording can afford. Every line says what was established and nothing above it.
CONNECT, CALL-B, CS and HEADER are decodes -- a callsign, a codeword at zero
errors, a frame whose CRC validates -- and a HEADER's TEXT lines are what the
field says, decoded by its own status byte. CALL-B is the Robust Call and the
two Free Signals, and its address is the station being CALLED, not the sender.
P1-FSK, P1-BURST and P2-PAIR are measurements of shape, and each carries the
numbers it was measured on so a reader can weigh it: they name no station, and
two of them have measured false-positive rates worth carrying. P1-BURST is produced by VARA and by any 500 Hz-class ARQ signal as
readily as by PACTOR-1. P1-FSK fired **103 times in 159 s of WWV** on
2026-08-14 -- a standards time broadcast is a carrier and a tick, and a strong
carrier near 1500 Hz sits between the two tones it looks at; the same detector
gave zero in 211 s of quiet band, so it is a carrier it answers, not noise. A
watched channel is where carriers live, so read P1-FSK on a busy frequency
accordingly.

    python -m hfmodem.shrike.monitor <capture.wav> [--hop 0.5] [--acquire]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import compress, p2rx, pactor2, rxfront, spec

_TAG = {"connect": "CONNECT", "cs": "CS", "packet": "HEADER",
        "detect": "P2-PAIR", "fsk": "P1-FSK", "p1reply": "P1-BURST",
        "unassigned": "SPARE-CS", "callb": "CALL-B"}
"""Display name per `rxfront.EVENT_KINDS`. Two of these deliberately do not match
their kind: `p1reply` reads P1-BURST and `detect` reads P2-PAIR, because the kinds
are a published interface that cannot be renamed cheaply while the tag is what an
operator reads at three in the morning, and neither kind name survives contact
with what its test can establish -- see `rxfront.EVENT_KINDS`.

Kept TOTAL, and gated as total by
tests/shrike/test_static.py: this map was missing `p1reply` and the monitor died with a
KeyError on its first one, discarding every later detection in that recording --
on 10 of the 23 corpus fixtures. The lookup below is also made forgiving, because
a monitor that stops reporting is worse than one that prints a name it does not
recognise."""


class _Text:
    """What each CRC-valid packet SAYS, decoded by its own declared data type.

    The packet line keeps the wire record -- status byte, length, the first
    payload bytes raw -- and is capped, which made a received message
    unreadable from the monitor even when every frame decoded. These lines
    are the message, in full. High bytes are the PTC terminal's codepage 437.

    One `compress.Decoder` for the whole capture, because the run-length seam
    crosses packet boundaries; a memory-ARQ repeat -- same counter, same
    payload, back to back -- is skipped so it cannot feed the stream twice.

    Only where it reads as text: an 8-bit-mode field can carry binary (a
    Winlink B2F transfer), and dumping that helps nobody. A pure idle fill
    decodes to nothing and prints nothing.
    """

    def __init__(self):
        self._dec = compress.Decoder()
        self._last = None

    def lines(self, ev) -> list[str]:
        if ev.kind != "packet" or ev.packet is None:
            return []
        _, status, payload, ok = ev.packet
        if not (ok and payload) or (status, payload) == self._last:
            return []
        self._last = (status, payload)
        dt = (status >> 2) & 0x7
        data = self._dec.feed(payload, dt)
        printable = sum(32 <= b < 127 or b in (9, 10, 13) or b >= 128
                        for b in data)
        if not data or printable < 0.9 * len(data):
            return []
        tag = compress.MODE_NAMES.get(dt, f"type {dt}")
        body = data.decode("cp437").replace("\r\n", "\n").replace("\r", "\n")
        return [f"{'':7s}  {'TEXT':8s} [{tag}] {line}"
                for line in body.split("\n")]


def _p2_data_events(audio) -> list[rxfront.Event]:
    """Every CRC-valid PACTOR-2 data burst in the capture, as packet events.

    `rxfront.decode_events` does not carry this pass: it is built for a stream,
    and `p2rx.decode_bursts` reads a RECORDING -- it fits one grid residue over
    every marker in the file at once, which is what covers the bursts whose own
    marker faded (29 of 32 on the HB9AK fixture against at most the armed
    markers one at a time). A session in a cycle has `decode_expected_burst`;
    a monitor holding the whole file has this.

    The field layout is PACTOR-3's -- data, status byte, CRC -- read from the
    END: the status byte is `field[crc_bytes - 3]`, the same place
    `onair._SessionRx._p2_packet` reads it. It is NOT byte 0; parsed that way
    every field decodes to soup, because byte 0 is the head of the compressed
    bit stream itself.
    """
    marks = [(k, bool((k >> 3) & 1)) for _t, _b, k, _s, sc, _sw
             in p2rx.find_markers(audio, rxfront.FS, 0.90)
             if sc >= p2rx.MARKER_ARM]
    out = []
    # BOTH FRAME LENGTHS, in two passes. `decode_bursts` fits ONE grid residue
    # over the file -- 1.25 s for the short frame, 3.75 for the long -- so a
    # recording that changes cycle length mid-link is two grids and asking for
    # the one that is not there returns nothing rather than a grid of the wrong
    # pitch. The long pass was simply missing, and a mailbox listing is exactly
    # what arrives in data mode ([SCS] s2), so the traffic worth reading was the
    # traffic this printed nothing for.
    for long_frame in (False, True):
        paths = pactor2.PATHS_LONG if long_frame else pactor2.PATHS
        for level in sorted({(k >> 1) & 3 for k, lf in marks if lf == long_frame}):
            path = paths[level]
            for t, field in p2rx.decode_bursts(audio, rxfront.FS, level,
                                               long_frame):
                st = field[path.crc_bytes - 3]
                data = spec.field_payload(field[:path.crc_bytes - 3])
                out.append(rxfront.Event(
                    t, "packet",
                    f"P2 {path.name} {len(data)}B status=0x{st:02x} "
                    f"({rxfront._status_str(st)}) {data[:20]!r}  [CRC-VALID]",
                    protocol="PACTOR-2", packet=(path.level + 1, st, data, True)))
    return sorted(out, key=lambda e: e.t)


def main() -> int:
    ap = argparse.ArgumentParser(description="shrike PACTOR monitor")
    ap.add_argument("wav")
    ap.add_argument("--hop", type=float, default=0.5)
    # Off by default because a monitor is not expecting anything, and the search
    # is only bounded by the caller knowing an answer is due; a session gets it
    # from `PtcHost` being in CONNECTING. It is here because the recordings this
    # reads are exactly the ones the connect-answer path is measured on, and the
    # difference is the whole finding: on captures/w6ids_silent/silent.wav, 0
    # control signals without it and 19 with, arriving on a 1.25 s grid.
    ap.add_argument("--acquire", action="store_true",
                    help="search for PACTOR-1 connect answers instead of reading "
                         "them at one point -- for a recording of a gateway "
                         "answering a call, where the cycle grid is unknown")
    args = ap.parse_args()

    audio = rxfront.load_wav(args.wav)
    print(f"# shrike monitor  |  {Path(args.wav).name}  |  "
          f"{len(audio)/rxfront.FS:.1f}s @ {rxfront.FS}Hz")
    print("#  t(s)   tag      decode")
    # Sorted, not streamed: `decode_events` yields its PACTOR-3 pass ahead of
    # the hop loop, and the P2 pass arrives as a block -- while `_Text`'s
    # run-length seam only joins correctly in the order the peer transmitted.
    # A file has no liveness to lose.
    events = [*rxfront.decode_events(audio, args.hop, acquiring=args.acquire),
              *_p2_data_events(audio)]
    events.sort(key=lambda ev: ev.t)
    seen = False
    text = _Text()
    for ev in events:
        seen = True
        print(f"{ev.t:7.2f}  {_TAG.get(ev.kind, ev.kind.upper()):8s} {ev.text}")
        for line in text.lines(ev):
            print(line)
    if not seen:
        print("#  (no PACTOR signal detected above the noise floor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
