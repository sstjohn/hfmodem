#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The keyer behind `console vara txcal --execute`, launched under the console claim.

Two real VARA waveforms, a BW500 CQ-frame body and a BW2300 data over, keyed at
stepped drives into the dial the rig sits on, with the rig's power and ALC meters
read while each is keyed. The codec volume and RFPOWER are read elsewhere and never
set here; the closing CW ID follows the fly path's.
"""
from __future__ import annotations

import argparse
import json
import queue
from pathlib import Path
import time

import numpy as np

FS = 48000
STEP_S, GAP_S = 2.0, 1.5
FADE_S = 0.005
#: Hamlib reads the FTDX10's ALC on Yaesu's default scale, raw RM4 0-64 as 0.0-1.0
#: (saturating above 64), and defines no compression range for it. At rest it reads
#: 0, so ALC is taken as not acting while it stays at or under this: raw 6 of 64.
ALC_QUIET = 0.1
ALC_RAW_FULL = 64


def drives(low=0.1, high=1.0, step_db=1.5):
    """Peaks from `low` to `high` in `step_db` steps, `high` last."""
    out, drive = [], low
    while drive < high * 10 ** (-step_db / 40):
        out.append(round(drive, 3))
        drive *= 10 ** (step_db / 20)
    return out + [high]


def trimmed(samples):
    out = np.array(samples[:round(STEP_S * FS)], dtype=np.float64)
    fade = round(FADE_S * FS)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


def waveforms(call):
    from hfmodem.kestrel.rx.cq import CQ, TRAINING
    from hfmodem.kestrel.tx import cq, varahf2300_tx
    body = (len(TRAINING) + CQ.columns) * CQ.symbol_samples
    return {'BW500 CQ body': trimmed(cq.synth_cq(call, 500)[-body:]),
            'BW2300 data': trimmed(varahf2300_tx.synth_burst(bytes(90)))}


def step(rig, send, name, samples, drive):
    send(samples, drive)
    try:
        report = rig.meter_reports.get(timeout=STEP_S)
    except queue.Empty:
        report = {}
    return dict(waveform=name, drive=drive, watts=report.get('peak_watts'),
                alc=report.get('peak_alc'), reads=report.get('reads', 0),
                unread=report.get('unread'))


def run(rig, send, waves, steps, *, pause=time.sleep, say=print):
    rows = []
    for drive in steps:
        for name, samples in waves.items():
            rows.append(step(rig, send, name, samples, drive))
            say(row_line(rows[-1]))
            pause(GAP_S)
    return rows


def recommend(rows):
    """The highest drive at which every waveform's ALC, and every lower drive's, read
    quiet; None when even the lowest moved the ALC or it did not read."""
    best = None
    for drive in sorted({r['drive'] for r in rows}):
        at = [r for r in rows if r['drive'] == drive]
        if not all(r['alc'] is not None and r['alc'] <= ALC_QUIET for r in at):
            break
        watts = [r['watts'] for r in at if r['watts'] is not None]
        best = dict(drive=drive, watts=max(watts) if watts else None)
    return best


def row_line(row):
    watts = 'unread' if row['watts'] is None else f"{row['watts']:5.1f} W"
    alc = ('unread' if row['alc'] is None
           else f"{row['alc']:.2f} (raw {round(row['alc'] * ALC_RAW_FULL)})")
    return f"  {row['drive']:5.3f}  {row['waveform']:<14} {watts:>8}  ALC {alc}"


def table(rows, best, rfpower):
    lines = [f'  drive  waveform        power     ALC (quiet at or under {ALC_QUIET:.2f}, '
             f'raw {round(ALC_QUIET * ALC_RAW_FULL)} of {ALC_RAW_FULL})',
             *map(row_line, rows)]
    if best is None:
        lines.append('  NO DRIVE KEPT THE ALC QUIET (or it did not read); nothing to recommend')
    else:
        lines.append(f"  RECOMMENDED tx_drive {best['drive']:.3f}: "
                     + ('power unread' if best['watts'] is None else f"{best['watts']:.1f} W peak")
                     + f' at RFPOWER {rfpower or "unread"}')
    return lines


def transmitter(io):
    """One keyed waveform through the VARA fly path's own transmit, at `drive`."""
    def send(samples, drive):
        io.key(True)
        try:
            io.tx(samples, drive)
        finally:
            io.key(False)
    return send


def identify(io, cw_call, data, drive):
    """The fly path's closing ID, sounding at the data over's level at `drive`
    rather than at the last and loudest step's."""
    from hfmodem.core import levels
    from kestrel_connect import identify as closing
    io.burst_level, before = (levels.sounding_rms(levels.at_drive(data, drive)), drive), io.bursts_sent
    closing(io, cw_call, '2300')
    if io.bursts_sent > before:
        print(f'  identified as {cw_call} in CW', flush=True)


def rfpower_reading(rig):
    try:
        reply = rig._cmd('l RFPOWER\n').split()
    except OSError:
        return None
    return reply[0] if reply and not reply[0].startswith('RPRT') else None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mycall', type=str.upper, required=True)
    p.add_argument('--idcall')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--rigctld', default='127.0.0.1:4532')
    p.add_argument('--expect-model', required=True)
    p.add_argument('--ptt-device', required=True)
    p.add_argument('--tx-device', required=True)
    p.add_argument('--rx-device', required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from hfmodem.core import identification
    from hfmodem.core.claim import inherited
    from hfmodem.core.ptt import arming_refusal
    from kestrel_connect import AudioVaraIO
    from vara_rig_bridge import Rig, _install_unkey_handlers

    identity = identification.resolve(args.mycall, args.idcall)
    identity.announce()
    if refusal := arming_refusal(args.ptt_device):
        raise ValueError(refusal)
    admission = inherited(9, launch=True)
    rig = io = None
    try:
        rig = Rig(args.rigctld, False, print, ptt_device=args.ptt_device, line_ptt=True)
        _install_unkey_handlers(rig)
        model, dial = rig.identify()
        if args.expect_model.lower() not in model.lower() or not dial.isdigit():
            raise RuntimeError(f'Rig identification failed: {model!r}, dial {dial!r}')
        admission.opened()
        rfpower = rfpower_reading(rig)
        print(f'  RFPOWER reads {rfpower or "unread"}; this never sets it', flush=True)
        waves, steps = waveforms(args.mycall), drives()
        io = AudioVaraIO(args.tx_device, args.rx_device, steps[0], rig,
                         record=args.session / 'receive.wav', warm=True)
        io.start()
        rig.armed = True
        rows = []
        try:
            rows = run(rig, transmitter(io), waves, steps, say=lambda line: print(line, flush=True))
        finally:
            best = recommend(rows)
            if rows:
                identify(io, identity.cw_call, waves['BW2300 data'], best['drive'] if best else steps[0])
            print('\n'.join(table(rows, best, rfpower)), flush=True)
            (args.session / 'txcal.json').write_text(json.dumps(
                dict(dial_hz=int(dial), rfpower=rfpower, alc_quiet=ALC_QUIET,
                     rows=rows, recommended=best), indent=2) + '\n')
        return 0
    finally:
        try:
            if rig is not None:
                rig.shutdown()
                print('PTT VERDICT: ' + ('keying line handed back LOW' if rig.released_low else
                      'CANNOT PROVE PTT IS DOWN'), flush=True)
            if io is not None:
                io.stop()
        finally:
            admission.failed()


if __name__ == '__main__':
    from kestrel_connect import exit_status
    raise SystemExit(exit_status(main))
