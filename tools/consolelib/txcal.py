# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""`console vara txcal`: the transmit drive, measured against the rig's own meters.

Real VARA waveforms, a BW500 CQ-frame body and a BW2300 data over, about 2 s each
with gaps, at drives stepped about 1.5 dB apart, on the dial the rig sits on. The
power and ALC meters are read while each is keyed, and the highest drive whose ALC
stays quiet is recommended with the power it gives at the RFPOWER set. Nothing keys
without --execute, which is the operator's go-ahead. The codec volume and RFPOWER are
read and reported, never set. --set records the recommended drive as the rig's one
tx_drive, together with the codec output volume it was measured at.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from hfmodem.contact.varac import CALLING_HZ
from . import arm, operator, rtty, station

HOLDS = SimpleNamespace(name='vara', lock=operator.mode_root('vara-txcal') / 'arm.lock',
                        arms=operator.mode_root('vara-txcal') / 'sessions')


def parser():
    p = argparse.ArgumentParser(prog='console vara txcal', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--band', type=int, choices=tuple(CALLING_HZ), default=40,
                   help='the band the rig is on; a dial off it is warned of (default 40)')
    p.add_argument('--idcall', help='CW call for the closing ID (default IDCALL, [station] idcall, then MYCALL)')
    keyed = p.add_mutually_exclusive_group()
    keyed.add_argument('--dry-run', action='store_true', help='print the steps and stop (the default)')
    keyed.add_argument('--execute', action='store_true', help='transmit the steps on the dial the rig sits on')
    p.add_argument('--set', action='store_true',
                   help='after --execute, record the recommended drive and the codec output volume '
                        "as the rig's tx_drive and codec_output in the station file")
    return p


def plan(args, now):
    from hfmodem.core import config
    import vara_txcal

    selection = station.resolve_selection()
    station.warn_receive_only(selection)
    stamp = now.strftime('%Y%m%dT%H%M%SZ')
    directory = HOLDS.arms / stamp
    command = [str(station.ONAIR), 'vara-txcal', '-', '--session', str(directory),
               *(['--idcall', args.idcall] if args.idcall is not None else [])]
    env = dict(CAT_PORT=selection.CAT_PORT, PTT_PORT=selection.PTT_PORT,
               HFMODEM_STATION=os.environ.get(config.STATION_ENV)
                              or str(station.ROOT / 'station.toml'))
    if selection.RIG_NAME != 'ft891':
        env['HFMODEM_RIG'] = selection.RIG_NAME
    steps = vara_txcal.drives()
    return dict(arm=f'vara-txcal-{stamp}', directory=str(directory), mode='txcal', call='',
                band_m=args.band, dial_hz=None, drives=steps,
                flow=[f'on the dial the rig sits on, at each drive {", ".join(f"{d:g}" for d in steps)}:',
                      f'key a {vara_txcal.STEP_S:g} s BW500 CQ-frame body, then a BW2300 data over, '
                      f'{vara_txcal.GAP_S:g} s apart, reading power and ALC while keyed',
                      f'recommend the highest drive whose ALC stays at or under {vara_txcal.ALC_QUIET:g}',
                      'close with our call in CW; the codec volume and RFPOWER are never set'],
                adjustments=dict(kiwi=None, witness_seconds=rtty.SESSION_S, retune=False),
                console_argv=getattr(args, 'console_argv', None), git=arm.git_head(),
                env=env, command=command)


def header(plan):
    readings = plan['readings']
    output = (readings.get('codec_levels') or {}).get('output')
    line = (f"vara txcal {plan['band_m']} m: dial {plan['dial_hz'] / 1e3:.3f} kHz, "
            f"RFPOWER {readings.get('rfpower') or 'unread'}, codec output "
            + (f'{output:.3f}' if isinstance(output, float) else str(output or 'unread')))
    on = station.band_m_of(dict(dial_hz=plan['dial_hz']))
    if on != plan['band_m']:
        line += f"\n  *** the dial is not on {plan['band_m']} m; keying where it sits ***"
    return line


def write_chain(path, chain, **values):
    """Set `values` in the station file's `[audio.chains.CHAIN]`, leaving every other line."""
    lines = path.read_text().splitlines() if path.is_file() else []
    head = f'[audio.chains.{chain}]'
    if head not in (line.strip() for line in lines):
        lines += ['', head]
    start = next(n for n, line in enumerate(lines) if line.strip() == head) + 1
    end = next((n for n in range(start, len(lines)) if lines[n].lstrip().startswith('[')), len(lines))
    for key, value in values.items():
        said = f'{key} = {value:.3f}'
        found = next((n for n in range(start, end) if re.match(rf'\s*{key}\s*=', lines[n])), None)
        if found is None:
            while end > start and not lines[end - 1].strip():
                end -= 1
            lines.insert(end, said)
            end += 1
        else:
            lines[found] = said
    path.write_text('\n'.join(lines) + '\n')


def record(plan):
    directory = Path(plan['directory'])
    try:
        best = json.loads((directory / 'txcal.json').read_text())['recommended']
        readings = json.loads((directory / 'provenance.json').read_text())['readings']
    except (OSError, ValueError, KeyError) as trouble:
        print(f'  nothing recorded: the session left no result ({trouble})', flush=True)
        return 1
    output = (readings.get('codec_levels') or {}).get('output')
    if best is None or not isinstance(output, float):
        print('  nothing recorded: ' + ('no drive kept the ALC quiet' if best is None
                                         else 'the codec output volume did not read'), flush=True)
        return 1
    selection = station.selection_for_plan(plan)
    chain = selection.RIG_PROFILE.get('audio_chain', '')
    path = Path(plan['env']['HFMODEM_STATION'])
    write_chain(path, chain, tx_drive=best['drive'], codec_output=output)
    print(f"  recorded tx_drive {best['drive']:.3f} at codec output {output:.3f} "
          f'in [audio.chains.{chain}] of {path}', flush=True)
    return 0


def show(plan, set_after):
    import vara_txcal
    try:
        result = json.loads((Path(plan['directory']) / 'txcal.json').read_text())
    except (OSError, ValueError) as trouble:
        print(f'  no table: the session left no result ({trouble})', flush=True)
        return
    print('\n'.join(vara_txcal.table(result.get('rows', []), result.get('recommended'), result.get('rfpower'))), flush=True)
    if result.get('recommended') is not None and not set_after:
        print('  add --set to record it as the rig\'s drive', flush=True)


def run(argv):
    args = parser().parse_args(argv)
    if args.set and not args.execute:
        print('  --set records what an --execute run measures; this run keys nothing.')
    args.console_argv = ['vara', 'txcal', *argv]
    built = plan(args, datetime.now(timezone.utc))
    if not args.execute:
        print(json.dumps(built, indent=2))
        print(arm.spelled(built))
        print('  nothing keyed: --execute transmits these steps.')
        return 0
    code = rtty.fly(built, holds=HOLDS, header=header)
    if code == 0:
        show(built, args.set)
    return record(built) if args.set and code == 0 else code
