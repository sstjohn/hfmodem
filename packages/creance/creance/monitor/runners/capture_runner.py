# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Live capture runner: one sound card in, raw s16le on stdout.

Its siblings here turn audio into detections; this one turns a card into audio.
It sits with them for the same three reasons they are here — it needs the modem
stack, it is spawned and never imported, and creance itself holds neither
PortAudio nor a device table.

ffmpeg used to open the card from inside creance and lost an eighth of every
pass: six live `creance monitor` passes on three days delivered 0.877, 0.870,
0.873, 0.875, 0.876 and 0.874 of real time in samples, and the shortfall is a
rate rather than a startup cost — a straight line through those six passes is
0.877 of the wall clock plus 0.4 s. PortAudio on the same card keeps 0.9979.

The card is resolved through the resolver every modem at this station binds
with, so the channel the sense reads and the channel the arm keys on are the
same one by construction, and an ambiguous name is refused here exactly as it is
refused on a transmit path.
"""

from __future__ import annotations

import queue
import signal
import sys
import threading

FS = 48000
DEVICE_HELP = "capture input: a name as `--list-devices` prints it, or its index"


def main(argv: list[str] | None = None) -> int:
    import sounddevice as sd
    from hfmodem.core.devices import find_device

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.exit(f"usage: capture_runner.py DEVICE\n  DEVICE is a {DEVICE_HELP}")
    device = find_device(args[0], "in", required=True)

    blocks: queue.Queue[bytes] = queue.Queue()
    stop = threading.Event()

    def on_block(pcm, frames, stamp, status) -> None:
        blocks.put(bytes(pcm))

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    out = sys.stdout.buffer
    with sd.RawInputStream(device=device, channels=1, samplerate=FS,
                           dtype="int16", callback=on_block):
        while not stop.is_set():
            try:
                pcm = blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                out.write(pcm)
                out.flush()
            except BrokenPipeError:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
