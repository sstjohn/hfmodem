# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Receive-only console profiles, independent of the modes that can fly arms."""
from dataclasses import replace
import math

from creance.monitor.driver import MonitorSpec, RUNNERS, MODEM_ROOT, _interpreter

# None means that the preamble identifies the mode. Other defaults are explicit
# operator choices; 4285, 4529, the parallel tones and FSK cannot infer them.
WAVEFORMS = {
    'mil110': ('110', None), 'stanag4285': ('4285', '1200L'),
    'stanag4529': ('4529', '600L'), 'stanag4539': ('4539', None),
    'wbhf': ('WBHF', None), 'tone39': ('39', '600/33/A10.8'),
    'tone16': ('16', '1200'), 'fsk': ('FSK', '75W'),
}
NAMES = ('pactor', 'pactor4', 'vara', 'ardop', 'rtty', 'sitor-b', 'navtex',
         'sabir', 'rpr', 'ale', 'selcall', *WAVEFORMS, 'mod110', 'cw', 'stanag')
ALIASES = {
    'shrike': 'pactor', 'p4': 'pactor4', 'kestrel': 'vara', 'besra': 'ardop',
    'sitor': 'sitor-b', 'lanner': 'rtty', 'robust-packet': 'rpr', 'brubru': 'rpr',
    '110': 'mil110', 'mil-std-188-110': 'mil110', '4285': 'stanag4285',
    '4529': 'stanag4529', '4539': 'stanag4539', '39': 'tone39', '16': 'tone16',
    'bateleur': 'stanag', 'stang': 'stanag',
}


def canonical(name):
    name = name.lower()
    name = ALIASES.get(name, name)
    if name not in NAMES:
        raise ValueError(f'Unknown monitor {name!r}; choose {", ".join(NAMES)}.')
    return name


def selected(args, defaults):
    """Build only the requested decoders, validating every setting before CAT."""
    name = canonical(args.monitor_mode)
    waveform = getattr(args, 'mode', None)
    framing = getattr(args, 'framing', 'SYNC').upper()
    order = getattr(args, 'bit_order', 'lsb')
    centre = getattr(args, 'audio_centre', 700.0)
    if not math.isfinite(centre) or not 100 <= centre <= 4000:
        raise ValueError('--audio-centre must be between 100 and 4000 Hz.')
    if name != 'cw' and centre != 700:
        raise ValueError('--audio-centre is the CW tone, in Hz.')
    if name not in WAVEFORMS and name != 'stanag' and (waveform or framing != 'SYNC' or order != 'lsb'):
        raise ValueError('--mode, --framing and --bit-order apply to Bateleur waveforms.')
    if name in ('pactor', 'pactor4', 'vara', 'ardop'):
        runners = {'pactor': ('shrike', 'p4'), 'pactor4': ('p4',),
                   'vara': ('kestrel',), 'ardop': ('besra',)}[name]
        return [s for s in defaults if s.name in runners]
    if name in ('rtty', 'sitor-b', 'navtex'):
        lanner = next(s for s in defaults if s.name == 'lanner')
        return [replace(lanner, args=(*lanner.args, '--protocol', name.upper(),
                                     '--search-span', '2500'))]
    options = ['--protocol', name, '--audio-centre', str(centre)]
    if name in WAVEFORMS or name == 'stanag':
        from hfmodem.bateleur.modem import Mode, ASYNC_STANDARDS
        from hfmodem.bateleur.charframe import CharFormat
        if name == 'stanag':
            parts = (waveform or '4285 1200L').split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError("--mode for stanag needs a standard and mode, e.g. '4285 1200L'.")
            standard, waveform = parts
        else:
            standard, default = WAVEFORMS[name]
            waveform = waveform or default
        # Self-identifying receivers do not use a supplied rate restriction.
        if standard.upper() in ('110', '4539', 'WBHF') and waveform is not None:
            raise ValueError(f'{standard} identifies its rate in the preamble; omit --mode.')
        if waveform is not None:
            # Receive placement is independent of the station's TX passband.
            # These waveform parsers accept every supported sub-carrier;
            # Mode.parse additionally enforces the transmitter's 3 kHz span.
            if standard.upper() == 'FSK':
                from hfmodem.bateleur import fsk
                standard, waveform = 'FSK', fsk.mode(waveform).name
            elif standard == '4529':
                from hfmodem.bateleur import s4529
                waveform = s4529.mode(waveform).full_name
            else:
                mode = Mode.parse(standard, waveform)
                standard, waveform = mode.standard, mode.name
        elif standard.upper() not in ('110', '4539', 'WBHF'):
            raise ValueError(f'{standard} needs a receive mode.')
        standard = standard.upper()
        if framing != 'SYNC':
            if standard not in ASYNC_STANDARDS:
                raise ValueError(f'{standard} has its own character framing; omit --framing.')
            framing = CharFormat.parse(framing).name
        options += ['--standard', standard, '--bit-order', order, '--framing', framing]
        if waveform:
            options += ['--mode', waveform]
    return [MonitorSpec(name, _interpreter(), str(RUNNERS / 'single_runner.py'),
                        defaults[0].cwd if defaults else str(MODEM_ROOT), tuple(options))]


def peer_frequency(args):
    """A named channel is a startup dial shortcut, never a callsign filter."""
    from . import modes, roster
    name = getattr(args, 'peer', None)
    if not name:
        return
    if args.freq is not None or args.wav or args.arm:
        raise ValueError('A peer selects a receiver frequency; give a peer or --freq, not both, '
                         'and omit peers for recordings.')
    candidates = []
    profile = canonical(args.monitor_mode)
    for mode_name in ((profile,) if profile in modes.KNOWN else modes.KNOWN):
        try:
            peers = roster.peers(modes.resolve(mode_name))
        except FileNotFoundError:
            continue
        candidates.extend(p for p in peers if name.upper() in (p['id'].upper(), p['call'].upper())
                          and (args.band is None or p['band_m'] == args.band))
    frequencies = {p['dial_hz'] for p in candidates}
    if len(frequencies) != 1:
        choices = ', '.join(sorted({p['id'] for p in candidates})) or 'no matching saved channels'
        raise ValueError(f'Peer {name!r} must name one frequency: {choices}. '
                         'Use a full peer ID, --band, or --freq KHZ.')
    args.freq = frequencies.pop() / 1000
    args.monitor_peer = name
