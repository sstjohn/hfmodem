# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""FTDX10 tune readback and raw receive-filter CAT."""
from __future__ import annotations

import re
import sys

from .rigs import RIGS

DATA_WIDTHS = dict(enumerate((0, *RIGS['ftdx10']['data_widths'])))
ROOFS = {'6': 12000, '7': 3000, '9': 500, 'A': 300}
ROOF_SET = {12000: '1', 3000: '2', 500: '4', 300: '5'}


def filter_commands(profile=None):
    """The roofing, width and (where the profile names one) IF shift setters. Roofing
    goes first: the rig puts the width back to 3000 when the roofing filter changes.
    IS is written in the form the rig answers `IS0;` with, `IS00+0500;`."""
    profile = RIGS['ftdx10'] if profile is None else profile
    width, roof = profile['rx_width_hz'], profile['roofing_hz']
    index = next(i for i, hz in DATA_WIDTHS.items() if hz == width)
    commands = f'RF0{ROOF_SET[roof]};', f'SH00{index:02d};'
    if 'if_shift_hz' in profile:
        commands += f'IS00{profile["if_shift_hz"]:+05d};',
    return commands


def rigctld(*commands):
    """Commands as one rigctld connection in its extended framing, closed by `q`."""
    return ''.join(f'+\\{command}\n' for command in commands) + 'q\n'


def session_batch(dial):
    """`tune`'s setters on one connection: VFO-A, the dial and its readback, DATA-U, AGC
    off, IPO, then the filters. The dial goes first, since on another band the band stack
    brings back that band's own mode and filters; the filters belong to the mode."""
    return rigctld('set_vfo VFOA', f'send_cmd FA{dial:09d};', 'send_cmd FA;', 'send_cmd MD0C;',
                   'set_level AGC 0', 'get_level AGC', 'set_level PREAMP 0', 'get_level PREAMP',
                   *(f'send_cmd {command}' for command in filter_commands()), 'get_vfo')


#: Asked on a connection of its own once the setters have settled.
READBACK = rigctld(*(f'send_cmd {query}' for query in ('MD0;', 'SH0;', 'RF0;', 'IS0;')))
#: `AC P1 P2 P3;`, P3 0 off, 1 on, 2 tuning: the ATU's match memory is not on the CAT.
TUNER = rigctld('send_cmd AC;')


def tuner(reply):
    """True on, False off, None unread."""
    state = answer(r'\bAC00([0-2]);', reply)
    return None if state is None else state != '0'


def answer(pattern, reply):
    """The one answer to a query, never the setter's own echo on its header line."""
    found = re.findall(pattern, '\n'.join(line for line in reply.splitlines()
                                         if not line.startswith('send_cmd:')))
    return found[0] if len(found) == 1 else None


def read_filters(reply):
    width, roof = answer(r'\bSH00(\d{2});', reply), answer(r'\bRF0([0-9A]);', reply)
    return (None if width is None else DATA_WIDTHS.get(int(width)),
            None if roof is None else ROOFS.get(roof))


def read_shift(reply):
    shift = answer(r'\bIS00([+-]\d{4});', reply)
    return None if shift is None else int(shift)


def readback_report(reply, rc=0, daemon=False, say=print):
    """The rig's own answers to `MD0; SH0; RF0; IS0;`, asked once the setters have settled."""
    profile = RIGS['ftdx10']
    verified = True
    for label, query, read, expected in (
            ('mode', 'MD0;', lambda part: answer(r'\bMD0([0-9A-F]);', part), 'C'),
            ('width', 'SH0;', lambda part: read_filters(part)[0], profile['rx_width_hz']),
            ('roofing filter', 'RF0;', lambda part: read_filters(part)[1], profile['roofing_hz']),
            ('IF shift', 'IS0;', read_shift, profile['if_shift_hz'])):
        part = daemon_part(reply, f'send_cmd: {query}') if daemon else reply
        failed, reading = reply_failure(part, rc, daemon, count=1), read(part)
        verified = verified and not failed and reading == expected
        shown = f'{reading:+d} Hz' if label == 'IF shift' and reading is not None else reading
        if failed or reading is None:
            say(f'  !! {label} unverified ({failed or "missing or unknown CAT reply"})')
        elif reading == expected:
            say('  DATA-U verified' if label == 'mode' else f'  {label} {shown} verified')
        elif label == 'mode':
            say(f'  !! mode reads {reading} (not DATA-U)')
        else:
            say(f'  !! {label} reads {shown}')

    return bool(verified)


def daemon_part(reply, *headers):
    """The answers under these headers, each ending at its own RPRT line: the `RPRT 0`
    that answers the closing `q` has no header, and would otherwise join the last block."""
    blocks = re.split(r'(?m)^(?=(?:set_vfo|get_vfo|send_cmd|set_level|get_level):)', reply)
    return '\n'.join(re.sub(r'(?ms)^(RPRT -?\d+)$.*', r'\1', block) for block in blocks
                     if block.splitlines() and block.splitlines()[0] in headers)


def session_parts(reply, daemon=False):
    if daemon:
        def part(*headers):
            return daemon_part(reply, *headers)
        return dict(
            dial=part(*(f'send_cmd: {line}' for line in re.findall(r'(?m)^send_cmd: (FA[^\n]*)', reply))),
            vfo=part('set_vfo: VFOA', 'get_vfo:'),
            AGC=part('set_level: AGC 0', 'get_level: AGC'),
            PREAMP=part('set_level: PREAMP 0', 'get_level: PREAMP'))
    lines = reply.splitlines()
    dial_at = next((i for i, line in enumerate(lines) if re.fullmatch(r'\d{5,}', line)), len(lines))
    rest = lines[dial_at + 1:]
    level_end = [i for i, line in enumerate(rest) if re.fullmatch(r'-?\d+', line)]
    raw_at = next((i for i, line in enumerate(rest)
                   if re.search(r'\b(?:SH|RF|MD)\d|^(?:VFO|Main|Sub|MEM)', line)), len(rest))
    level_at = level_end[1] + 1 if len(level_end) >= 2 else raw_at
    levels = rest[:level_at]
    first = next((i + 1 for i, line in enumerate(levels) if re.fullmatch(r'-?\d+', line)), len(levels))
    vfo_at = next((i for i in range(level_at, len(rest))
                   if re.search(r'^(?:VFO|Main|Sub|MEM)', rest[i])), len(rest))
    return dict(dial='\n'.join(lines[dial_at:dial_at + 1]),
                vfo='\n'.join([line for line in lines[:dial_at] if 'error =' in line]
                              + rest[vfo_at:]),
                AGC='\n'.join(levels[:first]), PREAMP='\n'.join(levels[first:]))


def reply_failure(reply, rc=0, daemon=False, count=2):
    error = re.search(r'error =[^\n]*|RPRT -\d+', reply)
    if error:
        return error.group()
    if rc:
        return f'exit {rc}'
    if daemon and reply.splitlines().count('RPRT 0') != count:
        return 'incomplete rigctld reply'
    return None


def session_frequency(reply, rc=0, daemon=False):
    part = session_parts(reply, daemon)['dial']
    if daemon:
        if reply_failure(part, daemon=True):
            return ''
        values = re.findall(r'(?m)^(?:Reply: )?FA(\d{9});$', part)
        return str(int(values[0])) if len(values) == 1 else ''
    return part


def vfo_report(reply, rc=0, daemon=False, say=print):
    pattern = r'(?m)^VFO: (VFO[A-Z]+|Main|Sub|MEM)$' if daemon else (
        r'(?m)^(?:VFO: )?(VFO[A-Z]+|Main|Sub|MEM)$')
    vfos = re.findall(pattern, daemon_part(reply, 'get_vfo:') if daemon else reply)
    failed = reply_failure(reply, rc, daemon)
    if failed or len(vfos) != 1:
        say(f'  !! VFO unverified ({failed or "missing or unknown CAT reply"})')
    elif vfos[0] in ('VFOA', 'Main'):
        say('  VFO-A verified')
    else:
        say(f'  !! VFO reads {vfos[0]} (not A)')

    return not failed and len(vfos) == 1 and vfos[0] in ('VFOA', 'Main')


def session_report(reply, rc=0, daemon=False, say=print):
    parts = session_parts(reply, daemon)
    verified = vfo_report(parts['vfo'], rc, daemon, say)
    for level, description in (('AGC', 'AGC off'), ('PREAMP', 'IPO')):
        part = parts[level]
        failed = reply_failure(part, rc, daemon)
        values = re.findall(r'(?m)^(-?\d+)$', part)
        verified = verified and not failed and values == ['0']
        if failed or len(values) != 1:
            failed = failed or f'no {level} reading (exit {rc})'
            label = 'AGC' if level == 'AGC' else 'IPO'
            say(f'  !! {label} unverified (rigctl: {failed})')
        elif values[0] == '0':
            say(f'  {description} verified')
        else:
            expected = 'off' if level == 'AGC' else 'IPO'
            say(f'  !! {level} reads {values[0]} (not {expected})')

    return bool(verified)


def normal(reply):
    # Some rigctld builds end a send_cmd reply with NUL rather than a newline
    # and prefix its status with "Reply: "; read both as the plain form.
    return re.sub(r'(?m)^Reply: (RPRT -?\d+)$', r'\1', reply.replace('\0', '\n'))


if __name__ == '__main__':
    if sys.argv[1] == 'commands':
        print(' '.join(filter_commands()))
    elif sys.argv[1] == 'batch':
        sys.stdout.write(session_batch(int(sys.argv[2])))
    elif sys.argv[1] == 'readback-batch':
        sys.stdout.write(READBACK)
    elif sys.argv[1] == 'dial':
        print(session_frequency(normal(sys.stdin.read()), int(sys.argv[2]), bool(int(sys.argv[3]))))
    elif sys.argv[1] == 'session-report':
        session_report(normal(sys.stdin.read()), int(sys.argv[2]), bool(int(sys.argv[3])))
    elif sys.argv[1] == 'readback':
        readback_report(normal(sys.stdin.read()), int(sys.argv[2]), bool(int(sys.argv[3])))
