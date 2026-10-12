"""FTDX10 tune readback and raw receive-filter CAT."""
from __future__ import annotations

import re
import sys

from .rigs import RIGS

DATA_WIDTHS = dict(enumerate((0, *RIGS['ftdx10']['data_widths'])))
ROOFS = {'6': 12000, '7': 3000, '9': 500, 'A': 300}
ROOF_SET = {12000: '1', 3000: '2', 500: '4', 300: '5'}


def filter_commands(profile=None):
    profile = RIGS['ftdx10'] if profile is None else profile
    width, roof = profile['rx_width_hz'], profile['roofing_hz']
    index = next(i for i, hz in DATA_WIDTHS.items() if hz == width)
    return f'SH00{index:02d};', f'RF0{ROOF_SET[roof]};'


def read_filters(reply):
    reply = '\n'.join(line for line in reply.splitlines()
                      if not line.startswith('send_cmd:'))
    width = re.findall(r'\bSH00(\d{2});', reply)
    roof = re.findall(r'\bRF0([0-9A]);', reply)
    return (DATA_WIDTHS.get(int(width[0])) if len(width) == 1 else None,
            ROOFS.get(roof[0]) if len(roof) == 1 else None)


def tune_report(reply, rc=0, daemon=False):
    error = re.search(r'error =[^\n]*|RPRT -\d+', reply)
    failed = error.group() if error else (f'exit {rc}' if rc else None)
    if daemon and reply.splitlines().count('RPRT 0') != 4:
        failed = failed or 'incomplete rigctld reply'
    for label, reading, expected in zip(('width', 'roofing filter'), read_filters(reply),
            (RIGS['ftdx10']['rx_width_hz'], RIGS['ftdx10']['roofing_hz'])):
        if failed or reading is None:
            print(f'  !! {label} unverified ({failed or "missing or unknown CAT reply"})')
        elif reading == expected:
            print(f'  {label} {expected} verified')
        else:
            print(f'  !! {label} reads {reading}')


def daemon_part(reply, *headers):
    blocks = re.split(r'(?m)^(?=(?:set_vfo|get_vfo|send_cmd|set_level|get_level):)', reply)
    return '\n'.join(block for block in blocks
                     if block.splitlines() and block.splitlines()[0] in headers)


def session_parts(reply, daemon=False):
    if daemon:
        def part(*headers):
            return daemon_part(reply, *headers)
        return dict(
            dial=part(*(f'send_cmd: {line}' for line in re.findall(r'(?m)^send_cmd: (FA[^\n]*)', reply))),
            selection=part('set_vfo: VFOA', 'get_vfo:', 'send_cmd: MD0C;', 'send_cmd: MD0;'),
            AGC=part('set_level: AGC 0', 'get_level: AGC'),
            PREAMP=part('set_level: PREAMP 0', 'get_level: PREAMP'),
            filters=part(*(f'send_cmd: {line}' for line in re.findall(r'(?m)^send_cmd: ((?:SH|RF)[^\n]*)', reply))))
    lines = reply.splitlines()
    dial_at = next((i for i, line in enumerate(lines) if re.fullmatch(r'\d{5,}', line)), len(lines))
    rest = lines[dial_at + 1:]
    level_end = [i for i, line in enumerate(rest) if re.fullmatch(r'-?\d+', line)]
    raw_at = next((i for i, line in enumerate(rest)
                   if re.search(r'\b(?:SH|RF|MD)\d|^(?:VFO|Main|Sub|MEM)', line)), len(rest))
    level_at = level_end[1] + 1 if len(level_end) >= 2 else raw_at
    levels = rest[:level_at]
    first = next((i + 1 for i, line in enumerate(levels) if re.fullmatch(r'-?\d+', line)), len(levels))
    selection_at = next((i for i in range(level_at, len(rest))
                         if re.search(r'\bMD\d|^(?:VFO|Main|Sub|MEM)', rest[i])), len(rest))
    return dict(dial='\n'.join(lines[dial_at:dial_at + 1]),
                selection='\n'.join([line for line in lines[:dial_at] if 'error =' in line]
                                     + rest[selection_at:]),
                AGC='\n'.join(levels[:first]), PREAMP='\n'.join(levels[first:]),
                filters='\n'.join(rest[level_at:selection_at]))


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


def selection_report(reply, rc=0, daemon=False):
    data = '\n'.join(line for line in reply.splitlines()
                     if not line.startswith(('send_cmd:', 'set_vfo:')))
    vfos = re.findall(r'(?m)^(?:VFO: )?(VFO[A-Z]+|Main|Sub|MEM)$', data)
    modes = re.findall(r'\bMD0([0-9A-F]);', data)
    if daemon:
        vfos = re.findall(r'(?m)^VFO: (VFO[A-Z]+|Main|Sub|MEM)$', daemon_part(reply, 'get_vfo:'))
        modes = re.findall(r'\bMD0([0-9A-F]);', '\n'.join(
            line for line in daemon_part(reply, 'send_cmd: MD0;').splitlines()
            if not line.startswith('send_cmd:')))
    for label, readings, expected, accepted in (
            ('VFO', vfos, 'A', ('VFOA', 'Main')),
            ('mode', modes, 'DATA-U', ('C',))):
        part = reply
        if daemon:
            headers = (('set_vfo: VFOA', 'get_vfo:') if label == 'VFO'
                       else ('send_cmd: MD0C;', 'send_cmd: MD0;'))
            part = daemon_part(reply, *headers)
        failed = reply_failure(part, rc, daemon)
        verified = 'VFO-A' if label == 'VFO' else 'DATA-U'
        if failed or len(readings) != 1:
            print(f'  !! {label} unverified ({failed or "missing or unknown CAT reply"})')
        elif readings[0] in accepted:
            print(f'  {verified} verified')
        else:
            print(f'  !! {label} reads {readings[0]} (not {expected})')


def session_report(reply, rc=0, daemon=False):
    parts = session_parts(reply, daemon)
    selection_report(parts['selection'], rc, daemon)
    for level, verified in (('AGC', 'AGC off'), ('PREAMP', 'IPO')):
        part = parts[level]
        failed = reply_failure(part, rc, daemon)
        values = re.findall(r'(?m)^(-?\d+)$', part)
        if failed or len(values) != 1:
            failed = failed or f'no {level} reading (exit {rc})'
            label = 'AGC' if level == 'AGC' else 'IPO'
            print(f'  !! {label} unverified (rigctl: {failed})')
        elif values[0] == '0':
            print(f'  {verified} verified')
        else:
            expected = 'off' if level == 'AGC' else 'IPO'
            print(f'  !! {level} reads {values[0]} (not {expected})')
    tune_report(parts['filters'], rc, daemon)


def normal(reply):
    # Some rigctld builds end a send_cmd reply with NUL rather than a newline
    # and prefix its status with "Reply: "; read both as the plain form.
    return re.sub(r'(?m)^Reply: (RPRT -?\d+)$', r'\1', reply.replace('\0', '\n'))


if __name__ == '__main__':
    if sys.argv[1] == 'commands':
        print(' '.join(filter_commands()))
    elif sys.argv[1] == 'dial':
        print(session_frequency(normal(sys.stdin.read()), int(sys.argv[2]), bool(int(sys.argv[3]))))
    elif sys.argv[1] == 'session-report':
        session_report(normal(sys.stdin.read()), int(sys.argv[2]), bool(int(sys.argv[3])))
    else:
        tune_report(normal(sys.stdin.read()), int(sys.argv[2]), bool(int(sys.argv[3])))
