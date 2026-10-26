# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""`console vara chat`: an attended Kestrel chat session using the VarAC protocol, and its record.

The session monitors the band's calling frequency and lists the CQs it hears.
`/answer N` QSYs to that CQ's slot at its bandwidth and connects; `/cq [SLOT]` sniffs
a slot, calls CQ twice on the calling frequency and waits on the slot, and with
`repeat` does so until a station connects or `/stop`; `/call CALL
[SLOT]` connects to a station. After a contact, or a CQ nobody answers, the session
is back on the calling frequency. Launching it is the attended start; each answer,
CQ or call is the go-ahead for that transmission. Incoming connects are answered,
as VarAC answers them, until `/auto off`. `/away [MESSAGE]` is VarAC's AWAY mode: a
caller gets `<AWAY>` and the message, may leave a VMail, is acknowledged with `<SMR>`
and the link is closed; `/back` says how many arrived. `/vmail` lists the VarAC mailbox under the operator-data root, `/vmail N` reads one, and `/vmail send SUBJECT` composes one
for the linked station. `/vmail pending` lists saved parked-mail notices;
`/vmail collect CALL` connects and QSYs to collect forwarded mail, and
`/vmail dismiss CALL` removes a notice without deleting mail.
`/band N` moves the session to another band's calling frequency
while nothing is linked or going out, and a KiwiSDR witness moves with it.
`/beacon hop 80 40 20 every 15` rotates through those calling channels;
`/beacon hop auto` selects from ready bands using locally heard activity and
periodic exploration. Verify each band's match and acknowledge `/band ready`
while on it before enabling hopping. Each arrival requests one clear-channel
beacon. `/beacon off` stops hopping; `/beacon hop status` shows readiness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from types import SimpleNamespace

from hfmodem.contact.varac import (AWAY_LINK_S, BANDWIDTHS, BEACON_MIN, CALLING_HZ, CLEAR_S,
                                   CQ_FRAMES, CQ_PERIOD_S, DEFAULT_BW, SLOT_WAIT_S,
                                   SNIFF_S, emit_tag)
from . import arm, operator, rtty, station

HOLDS = SimpleNamespace(name='vara', lock=operator.mode_root('vara-chat') / 'arm.lock',
                        arms=operator.mode_root('vara-chat') / 'sessions')
PROFILE = ('name', 'qth', 'loc', 'services')


def parser():
    p = argparse.ArgumentParser(prog='console vara chat', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--band', type=int, choices=tuple(CALLING_HZ), default=20,
                   help='the band whose calling frequency to monitor (default 20)')
    p.add_argument('--bw', type=int, choices=BANDWIDTHS, default=DEFAULT_BW,
                   help=f'VARA bandwidth for monitoring, /cq and /call (default {DEFAULT_BW}); '
                        '/answer uses the bandwidth the CQ advertised')
    helps = dict(loc='sent as <LOC:...> in the opening and in the beacon '
                     f'(default [station] grid, else {station.STATION_GRID})',
                 services='sent as <S:...> in the opening report')
    for field in PROFILE:
        p.add_argument('--' + field, help=helps.get(field, f'sent as <{field.upper()}:...> in the opening'))
    p.add_argument('--bbs-directory', help='serve this directory and advertise BBS; overrides station.toml')
    p.add_argument('--no-bbs', action='store_true', help='disable the configured BBS export')
    p.add_argument('--pskreporter', action='store_true',
                   help='opt in to public PSKReporter CQ/beacon reports (default off)')
    p.add_argument('--idcall', help='CW call for the closing ID (default IDCALL, [station] idcall, then MYCALL)')
    p.add_argument('--kiwi', nargs='?', const='', metavar='HOST',
                   help='place a KiwiSDR witness, preferring HOST when one is named')
    p.add_argument('--dry-run', action='store_true', help='print the plan and stop')
    return p


def plan(args, now):
    from hfmodem.core import config

    selection = station.resolve_selection()
    station.warn_receive_only(selection)
    stamp = now.strftime('%Y%m%dT%H%M%S%fZ')
    directory = HOLDS.arms / stamp
    dial = CALLING_HZ[args.band]
    if args.bw != DEFAULT_BW:
        print(f'warning: VarAC keeps its calling frequencies and slots at {DEFAULT_BW} Hz; '
              f'this session monitors and calls at {args.bw} Hz')
    command = [str(station.ONAIR), 'vara-chat', str(dial), '--band', str(args.band),
               '--bw', str(args.bw), '--session', str(directory)]
    if args.no_bbs:
        command += ['--no-bbs']
    elif args.bbs_directory is not None:
        command += ['--bbs-directory', args.bbs_directory]
    if getattr(args, 'pskreporter', False):
        command += ['--pskreporter']
    if args.idcall is not None:
        command += ['--idcall', args.idcall]
    env = dict(CAT_PORT=selection.CAT_PORT, PTT_PORT=selection.PTT_PORT,
               HFMODEM_STATION=os.environ.get(config.STATION_ENV)
                              or str(station.ROOT / 'station.toml'))
    if args.loc is None:
        args.loc = station.station_grid(dict(env=env))
    for field in PROFILE:
        value = getattr(args, field)
        if value is not None:
            emit_tag('S' if field == 'services' else field.upper(), value)
            command += ['--' + field, value]
    if selection.RIG_NAME != 'ft891':
        env['HFMODEM_RIG'] = selection.RIG_NAME
    return dict(arm=f'vara-chat-{stamp}', directory=str(directory), mode='chat', call='',
                band_m=args.band, dial_hz=dial, bandwidth_hz=args.bw, locator=args.loc,
                pskreporter=bool(getattr(args, 'pskreporter', False)),
                flow=['monitor the calling frequency; list heard CQs as they come',
                      "/answer N: QSY to that CQ's slot at its bandwidth and connect",
                      f'/cq [SLOT]: sniff the slot for {SNIFF_S} s, send {CQ_FRAMES} CQFRAMEs '
                      f'{CQ_PERIOD_S:g} s apart on the calling frequency, wait {SLOT_WAIT_S} s on the slot',
                      f'/cq [SLOT] repeat: the same, {SLOT_WAIT_S} s back on the calling frequency '
                      'between cycles, until a station connects or /stop',
                      '/call CALL [SLOT]: connect on the slot, or on the calling frequency',
                      'after a contact, or a CQ nobody answers, return to the calling frequency',
                      f'/beacon [MINUTES|now|force|off]: CALL-9 on the calling frequency every {BEACON_MIN} min '
                      f'(or whole MINUTES >= 1) while monitoring, once clear {CLEAR_S} s; '
                      'now sends once without changing the timer; force sends once despite QRM; '
                      'off resets the timer and cancels a pending one-shot',
                      '/beacon hop BANDS [every MINUTES]: rotate through calling channels; '
                      'default 15 min, minimum dwell 5 min; one clear-channel beacon on arrival',
                      '/beacon hop auto [BANDS] [every MINUTES]: choose using local heard activity; '
                      'explore every third choice; only ready bands are eligible',
                      '/band ready|unready: acknowledge the current band match for this session; '
                      '/beacon hop status shows readiness; /beacon off stops hopping',
                      'incoming connects are answered until /auto off',
                      '/band N: the calling frequency of band N, refused while linked or keying; '
                      'the filters are set again and read back, and the witness moves with it',
                      '/away [MESSAGE]: a caller gets <AWAY> and the message, may leave a VMail, is '
                      f'acknowledged with <SMR> and closed after it or at {AWAY_LINK_S // 60} min; '
                      '/back counts the new VMails',
                      '/vmail lists the VarAC mailbox, /vmail N reads one, /vmail send SUBJECT '
                      'composes one for the linked station; /vmail pending lists parked-mail notices, '
                      '/vmail collect CALL connects to collect, /vmail dismiss CALL removes a notice'],
                adjustments=dict(kiwi=args.kiwi, witness_seconds=rtty.SESSION_S, retune=True),
                console_argv=getattr(args, 'console_argv', None), git=arm.git_head(),
                env=env, command=command)


def fly(plan):
    return rtty.fly(plan, holds=HOLDS, header=header)


def header(plan):
    """The launch in one line; the rig readings and the command go to arm.log."""
    power = plan['readings'].get('rfpower')
    return (f"vara chat {plan['band_m']} m: calling frequency {plan['dial_hz'] / 1e3:.3f} kHz "
            f"at {plan['bandwidth_hz']} Hz, " + (f'RFPOWER {power}' if power else 'RFPOWER unread')
            + f", locator {plan['locator']}")


def run(argv):
    args = parser().parse_args(argv)
    args.console_argv = ['vara', 'chat', *argv]
    built = plan(args, datetime.now(timezone.utc))
    if args.dry_run:
        print(json.dumps(built, indent=2))
        print(arm.spelled(built))
        return 0
    return fly(built)
