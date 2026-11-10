# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Readback of the radio path used by a supervised voice/CW over."""
import math
import re

from hfmodem.core.band import QSY_TOLERANCE_HZ
from hfmodem.core.ftdx10 import read_filters
from hfmodem.core.rig import Cat

# Hamlib AGC enum -> Yaesu GT value. SUPERFAST/USER are not FTDX10 settings.
FTDX10_AGC = {2: '1', 3: '3', 5: '2'}
FTDX10_ROUTE = {'010415': '1', '010416': '1', '010418': '1'}


def write_raw(cat, command):
    # Yaesu setters do not reply. Hamlib send_cmd nevertheless waits for the
    # serial timeout (2 s on FTDX10), exceeding our CAT socket's deadline.
    # send_raw 0 explicitly requests no reply; verify() reads the actual state.
    return cat.command('\\send_raw 0 ' + command)


def raw(cat, command, prefix):
    # Read exactly one semicolon-terminated answer rather than waiting for the
    # generic send_cmd reader to time out looking for further output.
    reply = cat.command('\\send_raw ; ' + command)
    # Ignore the echoed command header. Only an actual, unambiguous reply
    # containing a payload may establish the radio's state.
    body = '\n'.join(line for line in reply.replace('\0', '\n').splitlines()
                     if not line.startswith(('send_cmd:', 'send_raw:')))
    matches = re.findall(r'\b' + re.escape(prefix) + r'([^\s;]+);', body)
    if len(matches) != 1:
        raise ValueError(f'Radio did not give an unambiguous {command} readback')
    return matches[0]


def verify(session):
    rig, cfg = session.rig, session.cfg
    dial = rig.dial()
    if abs(dial - session.dial_hz) > QSY_TOLERANCE_HZ:
        raise ValueError(f'Dial changed: expected {session.dial_hz} Hz, read {dial}. Restore it before sending.')
    cat = rig.cat
    vfo = Cat._field(cat.command('v'), 'VFO')
    split = Cat._field(cat.command('s'), 'Split')
    if vfo not in ('VFOA', 'Main') or split != '0':
        raise ValueError(f'Voice needs VFO A with split off; read VFO={vfo!r}, split={split!r}')
    expected = '8' if session.dial_hz < 10_000_000 else 'C'
    result = dict(dial_hz=dial, vfo=vfo, split=False)
    if cfg.selected_rig == 'ftdx10':
        if raw(cat, 'VS;', 'VS') != '0' or raw(cat, 'MD0;', 'MD0') != expected:
            raise ValueError('FTDX10 VFO/mode changed; restore VFO A and the selected DATA sideband before sending')
        # RT/XT have no VFO-selector parameter: RT0/XT0 are setters, not queries.
        if raw(cat, 'RT;', 'RT') != '0' or raw(cat, 'XT;', 'XT') != '0':
            raise ValueError('FTDX10 RIT/XIT must be off for the planned voice dial')
        route = {item: raw(cat, f'EX{item};', 'EX' + item) for item in FTDX10_ROUTE}
        if route != FTDX10_ROUTE:
            raise ValueError(f'FTDX10 data audio route changed: {route}; expected REAR, USB and RTS')
        agc = raw(cat, 'GT0;', 'GT0')
        if agc != FTDX10_AGC.get(cfg.voice.voice_agc):
            raise ValueError(f'FTDX10 AGC changed: GT0 reads {agc}; restore the selected voice AGC')
        # Use raw width readback: Hamlib get_mode cannot decode all FTDX10 widths.
        width_reply = 'SH0' + raw(cat, 'SH0;', 'SH0') + ';'
        roof_reply = 'RF0' + raw(cat, 'RF0;', 'RF0') + ';'
        width, roof = read_filters(width_reply + roof_reply)
        if (width, roof) != (3000, 3000) or raw(cat, 'NA0;', 'NA0') != '0':
            raise ValueError(f'FTDX10 voice receive filter changed: width={width}, roof={roof}; need 3 kHz and NARROW off')
        power = raw(cat, 'PC;', 'PC')
        if not power.isdecimal() or not 5 <= int(power) <= 100:
            raise ValueError(f'Invalid FTDX10 RF power readback: {power!r}')
        power = int(power)
        configured = cfg.declared_power_w
        if configured is not None and abs(power - configured) > 1:
            raise ValueError(f'RF power changed: read {power} W, configured {configured:g} W')
        session.tx_power_w = power
        result.update(mode='DATA-L' if expected == '8' else 'DATA-U', route=route,
                      agc_raw=agc, width_hz=width, roofing_hz=roof, power_w=power)
    else:
        mode_reply = cat.command('m')
        mode = Cat._field(mode_reply, 'Mode')
        wanted = 'PKTLSB' if expected == '8' else 'PKTUSB'
        width = Cat._field(mode_reply, 'Passband')
        if mode != wanted or not width or not width.isdecimal() or int(width) < 2700:
            raise ValueError(f'Voice mode/filter changed: read {mode!r}, {width!r}; expected {wanted}, 3 kHz')
        result.update(mode=mode, width_hz=int(width))
    if not math.isfinite(dial):
        raise ValueError('Invalid radio frequency')
    return result
