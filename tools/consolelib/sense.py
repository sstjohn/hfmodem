# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from hfmodem.core.busy import THRESHOLD_DB, POUNCE_WINDOW_S, WINDOW_S, dominant, scores
from hfmodem.core.devices import find_device
from hfmodem.core.occupied import occupied_hz
from hfmodem.core.levels import rx_verdict
from hfmodem.core.recording import RecordingSession, record_window, resolve_recording
from hfmodem.shrike.rxfront import call_b_frames

ROOT = Path(__file__).resolve().parents[2]


def sense_channel(dial, modem, bandwidth='', budget=0, *, device=None, root=ROOT,
                  opened=lambda: None, environ=None):
    environ = os.environ if environ is None else environ
    device = device or environ.get('AUDIO_IN', '')
    (Path(root) / 'logs/sense/host-free').unlink(missing_ok=True)
    resolved = resolve_recording(environ=environ)
    recording = RecordingSession(resolved.policy, Path(root) / 'logs/sense' / str(time.time_ns()),
                                 source=resolved.source, attached_children=False)
    try:
        _sense_channel(int(dial), modem, bandwidth, budget, device, root, opened, environ, recording)
    except SystemExit as exc:
        return exc.code
    finally:
        recording.close()


def _sense_channel(dial, modem, bandwidth, budget, device, root, opened, environ, recording):
    import sounddevice as sd
    FS = 48000
    sense_lane = recording.stream('sense', rate=FS, clock='card')
    FREE_SIGNAL = ("FreeSignalNormal", "FreeSignalEncrypted")
    emission = " ".join(str(a) for a in (modem, bandwidth) if a)
    band = occupied_hz(modem, bandwidth)
    DXCALL = environ.get("DXCALL", "").upper()
    # The launcher's `claim_pending sense`, said each window while a launch
    # awaits admission, so a long pounce is progress rather than silence.
    pending = environ.get("CLAIM_PENDING_LINE")
    try:
        dev = find_device(device, "in", required=True)
    except SystemExit as exc:
        print(f"  {exc}")
        raise SystemExit(2) from None

    def call_b_trains(a, ended):
        seen = {}
        for t, cn in call_b_frames(a):
            seen.setdefault(cn.callsign, []).append((ended - len(a) / FS + t, cn))
        out = {}
        for bursts in sorted(seen.values(), key=len):
            if len(bursts) >= 2:
                at, cn = bursts[-1]
                out["free" if cn.variant in FREE_SIGNAL else "robust"] = (cn, len(bursts), at)
        return out

    def window(stream, secs):
        if pending:
            print(pending, flush=True)
        a, over = stream.read(int(secs * FS))
        ended = time.time()
        if over:
            print("  !! input overflowed -- this window is missing samples", flush=True)
        a = np.asarray(a, dtype=float).ravel()
        record_window(sense_lane, a)
        s = scores(a, FS, band)
        trains = call_b_trains(a, ended)
        if "free" in trains:
            host_free(a, s, *trains["free"])
        robust = trains.get("robust")
        if s is None:
            return True, 0.0, "no verdict, the input was not live enough to judge", a, s
        name, margin = dominant(s)
        if name == "tone":
            name = f"tone at {s['tone_hz']:.0f} Hz"
        if robust is not None:
            cn, n, _at = robust
            mine = " -- the station this arm is calling" if cn.callsign == DXCALL else ""
            name = f"occupied: {n} robust calls to {cn.callsign} on {dial}{mine}"
            return True, margin, name, a, s
        return margin >= 0.0, margin, name, a, s

    # The verdict is kept under every policy; the audio is the sense lane's,
    # whose span for the deciding window the record names.
    def keep(audio, s, verdict, extra=None):
        d = recording.directory
        d.mkdir(parents=True, exist_ok=True)
        stem = d / "verdict.json"
        end = sense_lane.position
        stem.write_text(json.dumps(dict(
            dial=dial, emission=emission, band=list(band), verdict=verdict,
            thresholds=THRESHOLD_DB, scores=s, samplerate=FS,
            window=[end - len(audio), end], recording=recording.policy,
            **(extra or {})), indent=2) + "\n", encoding="utf-8")
        if recording.policy == "everything":
            print(f"  the window this rests on is kept: {d}, samples {end - len(audio)}-{end}"
                  " of the sense lane", flush=True)
        else:
            print(f"  the verdict this rests on is kept: {stem}; recording {recording.policy}"
                  " keeps no sense audio", flush=True)

    def host_free(audio, s, cn, n, at):
        print(f"  {cn.report()}  ({n} bursts)", flush=True)
        print(f"  HOST FREE -- {cn.callsign} has no link in progress on {dial}", flush=True)
        keep(audio, s, f"host {cn.callsign} free", dict(
            ident=cn.callsign, kind=cn.variant, bursts=n, last_burst_epoch=round(at, 3)))
        free = Path(root) / "logs/sense/host-free"
        free.parent.mkdir(parents=True, exist_ok=True)
        free.write_text(f"{cn.callsign} {cn.variant} {at:.3f}\n")
        raise SystemExit(5)

    def refuse(margin, what, s, audio, line, code=1):
        print(line, flush=True)
        if s is not None:
            print(f"  burst {s['burst']:.2f}/{THRESHOLD_DB['burst']}, "
                  f"shape {s['shape']:.2f}/{THRESHOLD_DB['shape']}, "
                  f"tone {s['tone']:.2f}/{THRESHOLD_DB['tone']} at "
                  f"{s['tone_hz']:.0f} Hz over {s['live_s']:.1f} s of live audio")
        keep(audio, s, f"{margin:+.1f} dB {what}")
        raise SystemExit(code)

    print(f"  judging {band[0]:.0f}-{band[1]:.0f} Hz -- what {emission} will occupy")
    with sd.InputStream(samplerate=FS, channels=1, device=dev) as stream:
        opened()
        if budget is None:
            for n in (1, 2):
                busy, margin, what, a, s = window(stream, WINDOW_S)
                line = (f"  channel sense {n}/2: {margin:+.1f} dB {what} -> "
                        f"{'OCCUPIED' if busy else 'clear'}")
                if busy:
                    refuse(margin, what, s, a, line)
                print(line)
            raise SystemExit(0)
        secs = POUNCE_WINDOW_S if budget else WINDOW_S
        clear = 0
        start = time.monotonic()
        while True:
            while clear < 2 and time.monotonic() - start < budget:
                busy, margin, what, a, s = window(stream, secs)
                clear = 0 if busy else clear + 1
                print(f"  {time.monotonic() - start:6.0f}s  {margin:+6.1f} dB  "
                      f"{'busy' if busy else 'CLEAR ' + str(clear)}  {what}", flush=True)
            if budget and clear < 2:
                refuse(margin, what, s, a, f"  still busy after "
                       f"{time.monotonic() - start:.0f}s of the {budget:g}s budget", code=3)
            if budget:
                print("  channel is free -- one more window, and it is the one that decides")
            busy, margin, what, a, s = window(stream, secs)
            line = (f"  channel sense: {margin:+.1f} dB {what} -> "
                    f"{'OCCUPIED' if busy else 'clear'}")
            if not busy:
                print(line)
                raise SystemExit(0)
            spent = time.monotonic() - start
            tail = "" if not budget else (
                f"\n  the channel was taken again on the window that decides, "
                f"{spent:.0f}s of the {budget:g}s budget spent")
            if spent >= budget:
                refuse(margin, what, s, a, line + tail)
            print(line + tail + " -- back to waiting", flush=True)
            clear = 0


def rx_level(*, device=None, opened=lambda: None):
    import sounddevice as sd
    try:
        dev = find_device(device or os.environ.get('AUDIO_IN', ''), 'in', required=True)
    except SystemExit as exc:
        print(f'  {exc}')
        return 2
    with sd.InputStream(samplerate=48000, channels=1, device=dev) as stream:
        opened()
        audio, overflow = stream.read(int(2.0 * 48000))
    state, line = rx_verdict(audio, 48000)
    print('  ' + line, flush=True)
    return {'live': 0, 'silent': 2}.get(state, 1)


def main(argv=None):
    from hfmodem.core import claim
    args = list(sys.argv[1:] if argv is None else argv)
    admission = claim.received(launch=True)
    try:
        if args[0] == 'rx-level':
            return rx_level(opened=admission.opened)
        dial, modem, bandwidth, *wait = args
        return sense_channel(dial, modem, bandwidth, float(wait[0]) if wait else None,
                             opened=admission.opened)
    finally:
        if admission.active:
            admission.failed()
            print('CLAIM NOT ADMITTED capture open failed', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
