# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The PACTOR profile: the assessed flight, and every flag an arm may vary from it.

`FLOWN` is the profile that flew every assessed arm and is not an adjustment.
What an arm may change is a flag, and what it changed is recorded in the arm's
own provenance beside the line that went out.

The roster and the arms directory stay where they were written: 57 flown arms,
the spacing figures and `status` all read out of
the configured per-mode operator data directory, and the lock lives there too, so the console
that supersedes the one in that directory cannot key over it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from hfmodem.shrike.compress import supervisor_block

from .modes import (ANSWERED, CLEARED, NOT_ATTEMPTED, NO_REPLY, Assessment, Mode, Readout,
                    highest, spoken)
from . import claim, station
from .station import ROOT

from .operator import mode_root
HERE = mode_root('pactor')
MAIL_PASSWORD_FILE = Path('~/.winlink-pw').expanduser()


# The operator authorised the PACTOR-3 phases at RFPOWER 1.0 and PACTOR-1 at the
# rig's current setting. A preset may not ask the radio for more than that.
AUTHORISED_WATTS = 100.0

LEVELS = {
    'current': dict(p1_watts=None, p3_watts=None, tx_drive=0.97, p1_drive=0.62,
                    note='the rig as it sits; the drives that flew every assessed arm'),
    'p3-100': dict(p1_watts=None, p3_watts=100.0, tx_drive=0.97, p1_drive=0.62,
                   note='PACTOR-1 on the rig\'s own level, PACTOR-3 phases at RFPOWER 1.0'),
    'split-25-60': dict(p1_watts=25.0, p3_watts=60.0, tx_drive=0.97, p1_drive=0.62,
                        note='the 09-14 phase-power experiment: 25 W drew the grant, 60 W entries went unread'),
    'p4-bench': dict(p1_watts=None, p3_watts=None, tx_drive=0.6, p1_drive=1.0,
                     hold=12,
                     note='PACTOR-4 chirp/entry bench: '
                          'drive split for the chirp waveform, a short hold'),
}

# The assessed profile, unchanged. Everything an arm may vary is a flag below.
FLOWN = ('--retries', '20', '--max-cycles', '32', '--hold', '96',
         '--p1-grant-only', '--p1-status-bits45', '3', '--p1-setup-phase', 'reply',
         '--p3-entry', 'template', '--announce-lower', '--p3-entry-stagger',
         '--p3-entry-rise', '--p3-speed-up', 'auto', '--over',
         '--p3-control-waveform', 'historical',
         '--p3-wideband-prekey')

# The mail lines unanchored, where the log's clock stamps some of them.
MILESTONES = (
    ('cleared', CLEARED),
    ('keyed', re.compile(r'TX\[\d+\][^\n]*-- keying|^keyed [1-9]\d* of', re.M)),
    ('answer', re.compile(r'^control signals decoded [1-9]|robust answer \(entry \d+\) -> link up', re.M)),
    ('connect', re.compile(r'\*\* CONNECTED to |state CONNECTING -> CONNECTED')),
    ('grant', re.compile(r'0x59A grant -> PACTOR-3 entry packet|'
                         r'0x6A9 experimental entry response -> PACTOR-2 SL1|'
                         r"the bridge's entry 6: PACTOR-2 -> PACTOR-3")),
    ('entry read', re.compile(r'the peer answered the entry packet')),
    ('packet acknowledged', re.compile(r'^verdict: ACKNOWLEDGED', re.M)),
    ('greeting', re.compile(r'\bmail: remote SID ')),
    ('login', re.compile(r'\bmail: stage (awaiting proposal answer|their turn|receiving|closed)')),
    ('mail', re.compile(r'\bmail: (received|wrote|accepted) ')),
)


def arguments(air):
    """What a PACTOR arm may vary. The defaults are what flew."""
    mail = air.add_argument_group('mail and presets')
    mail.add_argument('--mail-password-file', help='Winlink password file; otherwise prompt or use WINLINK_PASSWORD')
    mail.add_argument('--p4-mail', action='store_true',
                      help='the likely-working P4 mail preset: --p4-link with --p4-data body, ladder none, '
                           'normal-ack parity, chirp-ack parity, hold 16, entry timeout 20, long cycles and '
                           'the QRM guard off; a fetching mail client (plus --send delivery if --send is '
                           'absent); --level p4-bench unless given. Explicit flags and later --modem-arg words '
                           'override. Refused with --p4-chirp-counter')
    mail.add_argument('--p3-mail', action='store_true',
                      help='experimental mail starting at SL1 with the B +3.125 ms reply timing, continued '
                           'through turn changes; fetch, optionally upload with --send')
    mail.add_argument('--p2-mail', action='store_true',
                      help='experimental PTC-II-style mail: P1 setup/fallback, P2 maximum, 0x6A9 entry at SL1, '
                           '+18.75 ms P2 phase; never enters P3')
    mail.add_argument('--p1-mail', action='store_true',
                      help='the only configuration that has moved mail from this station: stay in P1, announce '
                           'nothing, open on the call phase. Forgoes P3 entirely')
    mail.add_argument('--no-mail', action='store_true',
                      help='omit every generated --mail-* flag, for an arm with no mail client (e.g. a '
                           '--p4-probe-* --modem-arg probe). Conflicts with --send and the mail presets')
    mail.add_argument('--no-wait-greeting', action='store_true',
                      help='take the turn rather than wait for a greeting the peer cannot send while we hold it')

    drive = air.add_argument_group('drive')
    drive.add_argument('--tx-drive', type=float, default=None,
                       help="PACTOR-3 phase drive (the modem's --tx-drive); default: the --level's own figure")
    drive.add_argument('--p1-drive', type=float, default=None,
                       help="PACTOR-1 phase drive (the modem's --p1-drive); default: the --level's own figure")

    connect = air.add_argument_group('connect and PACTOR-1')
    connect.add_argument('--call-hush', metavar='N,M',
                         help='go off the air for M cycles after every N unanswered calls (4,6 flew until '
                              '2026-09-23); default: call every cycle, as an SCS PTC does')
    connect.add_argument('--link-retries', type=int, metavar='CYCLES',
                         help='live-link retry limit (modem default 8); separate from connect --retries and '
                              'the hold budget')
    connect.add_argument('--robust', action='store_true',
                         help="key the SCS Robust Connect (the host's `C %%CALL`) rather than the ordinary call")
    connect.add_argument('--connect-fallback-after', type=int, metavar='N',
                         help='with --robust, key the remaining retries as an ordinary connect after N '
                              'unanswered robust calls')
    connect.add_argument('--after-connect', choices=('arq', 'listen', 'chirp', 'p1-robust'),
                         default='arq',
                         help="after the answered call: arq runs the link; listen keys nothing; chirp runs the "
                              "P4 entry probe from the answer; p1-robust follows with the robust short probe "
                              "(implied by --p4-link, the only one that can still carry mail)")
    connect.add_argument('--after-connect-cycles', type=int, default=6, metavar='N',
                         help='with --after-connect listen, peer cycles to read before teardown (default %(default)s)')
    connect.add_argument('--no-observe-after-link-down', dest='observe_after_link_down',
                         action='store_false',
                         help='stop when the ARQ gives up instead of listening out the hold (what assessed arms flew)')
    connect.add_argument('--pactor1-only', action='store_true',
                         help="never offer the upgrade; the session stays on 1400/1600 Hz and the channel sense "
                              "judges only 1200-1800 Hz. Drops the assessed profile's grant flag")
    connect.add_argument('--pactor2-only', action='store_true',
                         help="run as a PACTOR-2-capped station (the modem's --pactor2-only): announce bits45=1, "
                              "enter P2 only on 0x6A9 or a peer leading into it, never P3 or P4. "
                              "Drops the assessed profile's grant flag")
    connect.add_argument('--unannounced', action='store_true',
                         help='advertise nothing and go up anyway (--p3-uninvited in place of the grant flag). '
                              'The uninvited entry opens at SL3; --entry-sl 1 removes that confound')
    connect.add_argument('--setup-phase', choices=('reply', 'call'), default='reply',
                         help='which phase opens the link; call moved four messages at KB5LZK on 09-11 '
                              '(P1-only or P4-entry arms)')
    connect.add_argument('--p1-6a9', action='store_true',
                         help='announce P1 bits45=1 and select 0x6A9 as the entry response')
    connect.add_argument('--p1-level-digit', type=int, choices=(1, 2, 3, 4), default=1,
                         help="the announcement's leading digit, the 1990 level declaration (default %(default)s)")
    connect.add_argument('--p1-supervisor', nargs='+', default=[], metavar='BLOCK',
                         help="function blocks after the announcement, each a code and its parameters ('A 3' 'B')")
    connect.add_argument('--p1-supervisor-every', type=int, default=3, metavar='N',
                         help='acknowledged packets between --p1-supervisor blocks (default %(default)s)')
    connect.add_argument('--p1-idle-id-every', type=int, default=0, metavar='N',
                         help='function B with our call in every Nth idle P1 packet (try 10; default 0, off)')

    entry = air.add_argument_group('PACTOR-3 entry')
    entry.add_argument('--entry-sl', type=int, choices=range(1, 7), default=3, metavar='1..6',
                       help='speed level an UNINVITED P3 entry opens at (default %(default)s); a granted one '
                            'always opens at 1. All 27 uninvited entries on record opened at 3, which voids them')
    entry.add_argument('--changeover-p1-cs', action='store_true',
                       help="answer the peer's P3 changeover with a PACTOR-1 codeword in the P3 answer slot "
                            "(a P1 CS1 made a banner cross at KB5LZK, WS8EOC, VE1YZ)")
    entry.add_argument('--p3-fallback', action='store_true', help='allow the fall back to PACTOR-1')

    traffic = air.add_argument_group('PACTOR-3 traffic and speed')
    traffic.add_argument('--traffic-sl', type=int, choices=range(1, 7), default=1, metavar='1..6')
    traffic.add_argument('--speed-up', choices=('auto', 'hold'), default='auto',
                         help='auto requests a higher P3 speed after three consecutive cycles delivering new '
                              'payload; hold never asks (default %(default)s)')
    traffic.add_argument('--speed-policy', choices=('current', 'stock'), default='current',
                         help="the ISS's speed and cycle policy (--p3-speed-policy): current is the flown ladder; "
                              "stock follows the SCS manual's ISS and is unflown (default %(default)s)")
    traffic.add_argument('--p3-max-try', type=int, choices=range(1, 10), default=2, metavar='N',
                         help='SCS MAXTry: speed-up trial transmissions (1..9, default %(default)s)')
    traffic.add_argument('--p3-max-down', type=int, choices=range(2, 31), default=6, metavar='N',
                         help='SCS MAXDown: consecutive receive errors before asking for a lower speed '
                              '(2..30, default %(default)s)')
    traffic.add_argument('--repeat-gear', type=int, default=3)
    traffic.add_argument('--p3-cs4-parity', action='store_true',
                         help='unflown: a CS4 acknowledges only an odd packet counter; on an even one it asks '
                              'for the packet again (the IRS in PIII_Complete_1)')
    traffic.add_argument('--p3-tx-gear-hold', action='store_true',
                         help='outbound P3 trial: a repeated CS5 retains the pending packet')
    traffic.add_argument('--p3-irs-reply-offset-ms', type=float, default=0.0, metavar='N',
                         help='unflown, P3 mail: key IRS control signals N ms later than the reply clock '
                              '(default %(default)s)')
    traffic.add_argument('--p3-qrt-confirm', action='store_true',
                         help='on by --p3-mail: send the terminal marker after the QRT ACK and await opposite '
                              'ACK parity')
    traffic.add_argument('--long-cycle', '--long-cycles', action=argparse.BooleanOptionalAction,
                         default=None, help='enable long cycles (default with --p4-mail)')
    traffic.add_argument('--qrm-guard', action=argparse.BooleanOptionalAction, default=None,
                         help='keep the modem timing guard (default on; --p4-mail turns it off unless given); '
                              '--no-qrm-guard permits bursts crossing predicted peer reply slots')

    p4 = air.add_argument_group('PACTOR-4 link and ACK policy', 'experimental; needs --p4-link or --p4-mail')
    p4.add_argument('--p4-link', action='store_true',
                    help='run the experimental P4 turn machine after robust shorts. Implies --after-connect '
                         'p1-robust, --p1-status-bits45 2, robust shorts 7,38 on headers A,A (consecutive), '
                         '--p1-robust-stall 12 then goodbye and --no-observe-after-link-down; admits a mail '
                         'client (--send, or the default fetch). Explicit flags and later --modem-arg override')
    p4.add_argument('--p4-ladder', choices=('long', 'chirp', 'none'), default=None,
                    help='the P4 waveform ladder (default long; none with --p4-mail while data is body); '
                         'none sends a body then normal frames and needs --p4-data body')
    p4.add_argument('--p4-data', choices=('none', 'body', 'chirp'), default=None,
                    help='the data policy each turn keys (default none)')
    p4.add_argument('--p4-train-policy', choices=('repeat', 'passive'), default='repeat',
                    help='opening and first body response train policy (default %(default)s)')
    p4.add_argument('--p4-normal-ack', choices=('v27', 'x27', 'parity', 'x27-u27', 'alternate-packet'),
                    default=None, metavar='POLICY',
                    help='ack for peer normal frames: v27, x27, parity, x27-u27 or alternate-packet '
                         '(default v27; parity with --p4-mail)')
    p4.add_argument('--p4-word-render', choices=('off', 'stock'), default=None,
                    help='control-word carrier placement: off (legacy zero offsets), stock (measured offsets from body); default off.')
    p4.add_argument('--p4-long-ack', choices=('fixed-x27', 'alternate-packet', 'counter-odd-t27', 'counter-even-t27'),
                    default=None, metavar='POLICY', help='experimental ack policy for peer robust long frames (default fixed-x27)')
    p4.add_argument('--p4-text', metavar='TEXT',
                    help='Latin-1 text keyed after our SID; runs only with --no-mail')
    p4.add_argument('--p4-link-timeout', type=float, default=600., metavar='SECONDS',
                    help='session ceiling in seconds (default %(default)s)')
    p4.add_argument('--p4-hold-cycles', type=int, default=None, metavar='N',
                    help='idle cycles before the hold gives up (default 6; 16 with --p4-mail)')
    p4.add_argument('--p4-irs-silent-cycles', type=int, default=0, metavar='N',
                    help='key nothing for N peer cycles from the first IRS greeting body or ladder short '
                         '(default %(default)s, off)')
    p4.add_argument('--p4-answer-offset', type=float, default=.920, metavar='SECONDS',
                    help='our answer after a peer short, body or normal packet onset (default %(default)s)')
    p4.add_argument('--p4-breakin', choices=('body', 'y27'), default='body',
                    help='how we ask a sending peer for the turn: an idle 0x40 body, or a bare Y27 that waits '
                         'for its grant (default %(default)s)')

    probe = air.add_argument_group('PACTOR-4 entry probe', 'experimental')
    probe.add_argument('--p4-entry', action='store_true',
                       help='PACTOR-4 chirp entry after a grant; no mail or connected ARQ')
    probe.add_argument('--p4-chirp-counter', action='store_true',
                       help="P4 setup/ARQ counter challenge: three chirps (same counter twice, then advance); "
                            "not a mail transfer. Keys the modem's --p4-mail; refused with --p4-link")
    probe.add_argument('--p4-entry-attempts', type=int, choices=range(1, 5), default=2,
                       help='maximum chirps after a grant (default %(default)s)')
    probe.add_argument('--p4-entry-timeout', type=float, default=None,
                       help='post-grant probe deadline in seconds: at most 60 (default 60) for a probe-only '
                            'arm, 120 (default 120) with --p4-link')
    probe.add_argument('--p4-probe-packet', choices=('chirp', 'short', 'robust'),
                       default='chirp',
                       help="what the probe keys: chirp (SL1's 3.3 s entry), short (793 ms) or robust (the "
                            "setup tape's caller packet, 759 ms); default %(default)s")
    probe.add_argument('--p4-probe-statuses', metavar='BYTES',
                       help='status bytes to key, one chirp each: one to four, comma separated (e.g. '
                            '0x63,0x60,0x61); replaces --p4-entry-attempts')
    probe.add_argument('--p4-probe-payload', metavar='TEXT',
                       help='latin-1 text keyed in the chirp field; empty by default')
    probe.add_argument('--p4-probe-values', metavar='N[,N...]',
                       help='with --p4-probe-packet robust, datablock values 0..63, one packet each (one to '
                            'four); replaces --p4-entry-attempts')
    probe.add_argument('--p4-probe-headers', metavar='A|B[,...]',
                       help="header word per value; 'alt' alternates A,B (default all A). With --p4-link or "
                            "--p4-mail, a single A or B applies to every value")
    probe.add_argument('--p4-probe-cadence', choices=('spaced', 'consecutive'),
                       default=None,
                       help='spaced keys each packet after a full receive window (default); consecutive keys '
                            'them back to back, as the setup tape\'s caller did')

    robust = air.add_argument_group('Robust connect and the PACTOR-2 bridge')
    robust.add_argument('--after-robust', choices=('listen', 'p1', 'p2', 'p3', 'p4-robust'),
                        help="after the robust answer: listen keys nothing; p2 (default) sends our callsign in "
                             "P2 and waits for the P3 grant; p1 a P1 packet opening with our call; p3 the SL1 "
                             "entry and P3 ARQ; p4-robust the robust shorts (6,9,8 on header A unless "
                             "--p4-probe-values)")
    robust.add_argument('--robust-call-text', choices=('pmc', 'ascii'), default='pmc')
    robust.add_argument('--robust-p2-max', type=int, default=10)
    robust.add_argument('--robust-p2-marker-flag', choices=('parity', 'inverted'),
                        default='parity',
                        help="bit 0 of the bridge's P2 marker: the counter's parity, or its complement "
                             "(keys k=2 as K4MCK's did); default %(default)s")
    robust.add_argument('--robust-p2-opening', choices=('none', 'k9'), default='none',
                        help="k9 keys K4MCK's counter-0 k=9 frame first, counted as one of --robust-p2-max's "
                             "slots (default %(default)s)")
    robust.add_argument('--robust-p3-without-grant', type=int, metavar='K')

    keying = air.add_argument_group('PACTOR-3 keying experiments', 'each is a control or unflown')
    keying.add_argument('--follow-offset', choices=('none', 'control', 'all'), default='all')
    keying.add_argument('--keep-slots', choices=('all', 'controls', 'none'), default='all')
    keying.add_argument('--control-placement', choices=('pulse-center', 'audio-start'),
                        default='audio-start',
                        help="where the control burst sits in its slot: audio-start is the modem's default "
                             "(3 of 3 against 0 of 43 for pulse-center on the roster, confounded with the hour)")
    keying.add_argument('--control-tail', choices=('off', 'repeat'), default='off',
                        help='repeat adds the 22nd symbol every measured SCS control sends')
    keying.add_argument('--control-stagger', choices=('off', 'lead-5', 'lead-12'),
                        default='off',
                        help='5 ms half-symbol carrier stagger, alternating from this foot. The wrong foot is '
                             'worse than off: fly one arm on each')
    keying.add_argument('--crest-limit', default='off', metavar='off|DB',
                        help='hold every P3 data packet and control burst to this peak-to-average ratio in dB '
                             '(--p3-crest-limit); 6 is the reviewed point and raises average power: watch ALC '
                             'and PA temperature. Unflown')

    trials = air.add_argument_group('timing trials', 'bounded experiments, no mail')
    trials.add_argument('--timing-trial', choices=('A', 'B'),
                        help='P3 receive timing: A keeps current placement; B rotates the entry phase by 600 ms. '
                             '12 reply opportunities / 20 s')
    trials.add_argument('--entry-timing-trial', choices=('A', 'B'),
                        help='P3 entry acquisition: A uses zero delay; B the default 4.875 ms. Stops at P3 '
                             'acquisition or 20 s from first entry')
    trials.add_argument('--timing-reply-delay', type=float, choices=(0.0, 3.125),
                        help='with --timing-trial B, shift the reply grid by this many ms (3.125 targets '
                             'entry +603.125 ms)')
    trials.add_argument('--timing-unbounded', action='store_true',
                        help='with --timing-trial, receive without experiment or hold cutoffs until peer QRT, '
                             'turn change, link loss or operator stop')


def _without(words, *drop):
    """FLOWN minus named flags; a (flag, n) pair drops the flag and its n values."""
    taking = {d if isinstance(d, str) else d[0]: 0 if isinstance(d, str) else d[1] for d in drop}
    out, skip = [], 0
    for word in words:
        if skip:
            skip -= 1
        elif word in taking:
            skip = taking[word]
        else:
            out.append(word)
    return out


def _qrm_guard(args, default=True):
    """`--qrm-guard`, resolved: unset (None -- neither flag given) falls back to `default`."""
    value = getattr(args, 'qrm_guard', None)
    return default if value is None else value


def _said(args):
    """The operator's own launcher words, split as `arm.passthrough` splits them."""
    return [word for chunk in getattr(args, 'modem_arg', ()) for word in shlex.split(chunk)]


def _said_value(args, flag):
    """The value the operator's own words last give `flag`, as the modem reads it."""
    said, value = _said(args), None
    for i, word in enumerate(said):
        if word == flag and i + 1 < len(said):
            value = said[i + 1]
        elif word.startswith(flag + '='):
            value = word.split('=', 1)[1]
    return value


def flags(peer, level, args, directory, send):
    """The launcher words between the shared head and the shared witness/force tail."""
    # An explicit --tx-drive/--p1-drive wins; short of that, the chosen level's own
    # figure does, ahead of any hard-coded default.
    if getattr(args, 'selected_rig', station.RIG_NAME) == 'ft891' and not getattr(args, 'selected_audio', None):
        if args.tx_drive is None:
            args.tx_drive = level['tx_drive']
        if args.p1_drive is None:
            args.p1_drive = level['p1_drive']
    else:
        from hfmodem.core import config
        audio = getattr(args, 'selected_audio', None)
        preset = ('p4_drive' if level.get('name') == 'p4-bench' or
                  any(getattr(args, flag, False) for flag in
                      ('p4_mail', 'p4_link', 'p4_entry', 'p4_chirp_counter')) else 'p3_drive')
        sources = {}
        for field, chain_field in (('tx_drive', preset), ('p1_drive', 'p1_drive')):
            if getattr(args, field) is not None:
                sources[field] = 'operator'
            else:
                value = getattr(audio, chain_field, None) if audio else None
                if value is None:
                    value = getattr(audio, 'tx_drive', None) if audio else None
                sources[field] = 'station-toml' if value is not None else 'provisional-fallback'
                setattr(args, field, value if value is not None else config.ChainAudio().drive_for())
        args.drive_sources = sources
    after_robust = _after_robust(args)
    p4_chirp_counter = getattr(args, 'p4_chirp_counter', False)
    p4_mail_preset = getattr(args, 'p4_mail', False)
    if getattr(args, 'long_cycle', None) is None:
        args.long_cycle = p4_mail_preset
    if p4_chirp_counter and (getattr(args, 'p2_mail', False) or getattr(args, 'p3_mail', False)
                             or getattr(args, 'p4_entry', False)
                             or getattr(args, 'p4_link', False) or p4_mail_preset or send):
        raise ValueError('--p4-chirp-counter runs the counter experiment; choose it '
                         'separately from other mail/probe presets, --p4-link, '
                         '--p4-mail and --send')
    if p4_mail_preset:
        # The preset's own door into the P4 turn machine; everything else it
        # implies is resolved once `after_connect` is in scope, below.
        args.p4_link = True
    no_mail = getattr(args, 'no_mail', False)
    if no_mail and (send or p4_chirp_counter or p4_mail_preset or getattr(args, 'p2_mail', False)
                    or getattr(args, 'p3_mail', False) or args.p1_mail):
        raise ValueError('--no-mail drops the mail client; --send, --p1-mail, --p2-mail, '
                         '--p3-mail, --p4-mail and --p4-chirp-counter all need it')
    if (getattr(args, 'p4_probe_statuses', None) is not None
            or getattr(args, 'p4_probe_payload', None) is not None
            or getattr(args, 'p4_probe_values', None) is not None
            or getattr(args, 'p4_probe_headers', None) is not None
            or (getattr(args, 'p4_probe_cadence', None) or 'spaced') != 'spaced'
            or getattr(args, 'p4_probe_packet', 'chirp') != 'chirp'):
        if not (p4_chirp_counter or getattr(args, 'p4_entry', False)
                or getattr(args, 'p4_link', False)
                or getattr(args, 'after_connect', 'arq') in ('chirp', 'p1-robust')
                or after_robust == 'p4-robust'):
            raise ValueError('--p4-probe-statuses, --p4-probe-payload and '
                             '--p4-probe-packet are the chirp probe\'s query; '
                             'the probe is off')
    p4_link = getattr(args, 'p4_link', False)
    p4_ladder = getattr(args, 'p4_ladder', None)
    p4_data = getattr(args, 'p4_data', None) or 'none'
    p4_train_policy = getattr(args, 'p4_train_policy', 'repeat')
    p4_normal_ack = (getattr(args, 'p4_normal_ack', None)
                     or ('parity' if p4_mail_preset else 'v27'))
    p4_word_render = getattr(args, 'p4_word_render', None)
    if p4_word_render == 'stock' and not p4_link:
        raise ValueError('--p4-word-render stock requires --p4-link or --p4-mail')
    p4_long_ack = getattr(args, 'p4_long_ack', None)
    if p4_long_ack is not None and not p4_link:
        raise ValueError('--p4-long-ack requires --p4-link or --p4-mail')
    p4_link_timeout = getattr(args, 'p4_link_timeout', 600.)
    p4_hold_cycles = getattr(args, 'p4_hold_cycles', None)
    if p4_hold_cycles is None:
        p4_hold_cycles = 16 if p4_mail_preset else 6
    p4_answer_offset = getattr(args, 'p4_answer_offset', .920)
    p4_irs_silent_cycles = getattr(args, 'p4_irs_silent_cycles', 0)
    if p4_irs_silent_cycles < 0:
        raise ValueError('--p4-irs-silent-cycles must be zero or positive')
    p4_breakin = getattr(args, 'p4_breakin', 'body')
    p4_text = getattr(args, 'p4_text', None)
    if not p4_link and (p4_ladder not in (None, 'long') or p4_data != 'none' or p4_link_timeout != 600.
                        or p4_hold_cycles != 6 or p4_answer_offset != .920 or p4_breakin != 'body'
                        or p4_irs_silent_cycles or p4_train_policy != 'repeat'):
        raise ValueError('--p4-ladder, --p4-data, --p4-link-timeout, --p4-hold-cycles, '
                         '--p4-answer-offset, --p4-irs-silent-cycles, --p4-train-policy and --p4-breakin describe the --p4-link turn '
                         'machine, which is off')
    if p4_text is not None:
        if not p4_link:
            raise ValueError('--p4-text runs within --p4-link, which is off')
        if not no_mail:
            raise ValueError('--p4-text only runs with --no-mail (no fetch or send)')
        p4_text.encode('latin-1')
    supervisor = _supervisor_channel(args, send)
    if getattr(args, 'p2_mail', False):
        words, adjustments = _p2_mail_flags(level, args, directory, send)
        return words + _call_hush(args, adjustments), adjustments
    words = list(FLOWN)
    hold = level.get('hold')
    if hold is not None:
        words[words.index('--hold') + 1] = str(hold)
    # Both spellings would reach the parser and the last would win, but the line
    # is the record of what flew: one waveform per line or the tape is scored
    # against a guess.
    if '--p3-control-waveform' in _said(args):
        words = _without(words, ('--p3-control-waveform', 1))
    speed_up = getattr(args, 'speed_up', 'auto')
    words[words.index('--p3-speed-up') + 1] = speed_up
    for name, default, low, high in (('p3_max_try', 2, 1, 9), ('p3_max_down', 6, 2, 30)):
        value = getattr(args, name, default)
        flag = '--' + name.replace('_', '-')
        if not low <= value <= high:
            raise ValueError(f'{flag} must be {low}..{high}')
        words += [flag, str(value)]
    if getattr(args, 'p3_tx_gear_hold', False):
        words += ['--p3-tx-gear-hold']
    if getattr(args, 'p3_cs4_parity', False):
        words += ['--p3-cs4-parity']
    if getattr(args, 'p3_irs_reply_offset_ms', 0.0):
        words += ['--p3-irs-reply-offset-ms', f'{args.p3_irs_reply_offset_ms:g}']
    speed_policy = getattr(args, 'speed_policy', 'current')
    if speed_policy != 'current':
        words += ['--p3-speed-policy', speed_policy]
    link_retries = getattr(args, 'link_retries', None)
    if link_retries is not None:
        if link_retries < 1:
            raise ValueError('--link-retries must be a positive cycle count')
        words += ['--link-retries', str(link_retries)]
    timing_trial = getattr(args, 'timing_trial', None)
    entry_trial = getattr(args, 'entry_timing_trial', None)
    unbounded = getattr(args, 'timing_unbounded', False)
    reply_delay = getattr(args, 'timing_reply_delay', None)
    p3_mail = getattr(args, 'p3_mail', False)
    qrt_confirm = p3_mail or getattr(args, 'p3_qrt_confirm', False)
    if qrt_confirm:
        if not p3_mail:
            raise ValueError('--p3-qrt-confirm requires --p3-mail')
        words += ['--p3-qrt-confirm']
    if p3_mail:
        if (timing_trial or entry_trial or unbounded or reply_delay is not None
                or getattr(args, 'p4_entry', False) or args.unannounced
                or args.p1_mail or args.pactor1_only
                or args.changeover_p1_cs or args.p3_fallback
                or args.setup_phase != 'reply'
                or args.control_placement != 'audio-start'):
            raise ValueError('--p3-mail requires the granted P3 profile '
                             'without timing trials or forced modes')
        words += ['--p3-mail']
        args.repeat_gear = 0
        args.entry_sl = args.traffic_sl = 1
        args.observe_after_link_down = False
    if reply_delay is not None and (timing_trial != 'B' or entry_trial):
        raise ValueError('--timing-reply-delay requires --timing-trial B')
    if unbounded and (not timing_trial or entry_trial):
        raise ValueError('--timing-unbounded requires --timing-trial A or B')
    if p4_link:
        # The working arm's own settings: a robust short probe after the
        # ordinary connect, then the P4 turn machine. Each piece only fills a
        # default still at its own -- an explicit flag or a later --modem-arg
        # word always wins.
        named_after_connect = getattr(args, 'after_connect', 'arq')
        if named_after_connect not in ('arq', 'p1-robust'):
            raise ValueError(f'--p4-link keys after a robust short probe; '
                             f'--after-connect {named_after_connect} does not carry one')
        args.after_connect = 'p1-robust'
        named_probe_packet = getattr(args, 'p4_probe_packet', 'chirp')
        if named_probe_packet not in ('chirp', 'robust'):
            raise ValueError('--p4-link keys the robust short packet, not '
                             f'--p4-probe-packet {named_probe_packet}')
        args.p4_probe_packet = 'robust'
        if getattr(args, 'p4_probe_values', None) is None:
            args.p4_probe_values = '7,38'
        if getattr(args, 'p4_probe_cadence', None) is None:
            args.p4_probe_cadence = 'consecutive'
        if not 0 < p4_link_timeout < float('inf'):
            raise ValueError('--p4-link-timeout must be positive')
        if p4_hold_cycles < 0:
            raise ValueError('--p4-hold-cycles must be zero or positive')
        if p4_mail_preset:
            if getattr(args, 'p4_data', None) is None:
                p4_data = args.p4_data = 'body'
            if getattr(args, 'qrm_guard', None) is None:
                args.qrm_guard = False
    if p4_ladder is None:
        p4_ladder = ('none' if p4_mail_preset and p4_data == 'body'
                     and _said_value(args, '--p4-data') in (None, 'body') else 'long')
    after_connect = getattr(args, 'after_connect', 'arq')
    p4_entry = (getattr(args, 'p4_entry', False) or p4_chirp_counter
                or after_connect in ('chirp', 'p1-robust') or after_robust == 'p4-robust')
    qrm_guard = _qrm_guard(args)
    if p4_entry:
        if (timing_trial or entry_trial or args.unannounced or args.p1_mail
                or args.pactor1_only or (send and not p4_link)
                or after_connect == 'listen'):
            raise ValueError('--p4-entry requires a granted entry probe without '
                             'mail, forced modes, P3 timing trials or a '
                             'listening arm')
        early = ('--after-connect chirp' if after_connect == 'chirp'
                 else '--after-robust p4-robust' if after_robust == 'p4-robust' else None)
        if args.setup_phase == 'call' and early:
            raise ValueError(f'--setup-phase call holds until the first P1 payload ACK, '
                             f'and {early} leaves PACTOR-1 before one')
        words = _without(words, ('--p3-entry', 1), ('--p3-speed-up', 1), '--over')
        if p4_link:
            words = _without(words, ('--p1-status-bits45', 1))
            stall_then = (_said_value(args, '--p1-robust-stall-then')
                          or ('shorts' if p4_mail_preset else 'goodbye'))
            words += ['--p1-status-bits45', '2', '--p1-robust-stall', '12',
                      '--p1-robust-stall-then', stall_then]
        attempts = 3 if p4_chirp_counter else getattr(args, 'p4_entry_attempts', 2)
        timeout_cap = 120 if p4_link else 60
        timeout = getattr(args, 'p4_entry_timeout', None)
        if timeout is None:
            timeout = 20.0 if p4_mail_preset else 120.0 if p4_link else 60.0
        if not 0 < timeout <= timeout_cap:
            raise ValueError(f'--p4-entry-timeout must be positive and at most '
                             f'{timeout_cap} seconds')
        words = _without(words, ('--hold', 1))
        schedule = _status_schedule(getattr(args, 'p4_probe_statuses', None), p4_chirp_counter)
        field = getattr(args, 'p4_probe_payload', None)
        packet = getattr(args, 'p4_probe_packet', 'chirp')
        if field is not None:
            if p4_chirp_counter:
                raise ValueError('--p4-probe-payload cannot change the '
                                 '--p4-chirp-counter field')
            field.encode('latin-1')
        if packet != 'chirp' and p4_chirp_counter:
            raise ValueError('--p4-probe-packet cannot change the --p4-chirp-counter waveform')
        if packet == 'robust' and (schedule or field is not None):
            raise ValueError('--p4-probe-packet robust keys a datablock value, '
                             'which carries neither our status byte nor our field')
        values, heads = _datablock(args, packet)
        if schedule:
            attempts = len(schedule)
        if values:
            attempts = len(values)
        words += ['--p4-mail' if p4_chirp_counter else '--p4-entry', '--p3-speed-up', 'hold',
                  '--hold', '12', '--p4-entry-attempts', str(attempts),
                  '--p4-entry-timeout', str(timeout)]
        if schedule:
            words += ['--p4-probe-statuses', ','.join(f'0x{v:02X}' for v in schedule)]
        if field is not None:
            words += ['--p4-probe-payload', field]
        if packet != 'chirp':
            words += ['--p4-probe-packet', packet]
        if values:
            words += ['--p4-probe-values', ','.join(str(v) for v in values),
                      '--p4-probe-headers', ','.join(heads)]
        cadence = getattr(args, 'p4_probe_cadence', None) or 'spaced'
        if cadence != 'spaced':
            words += ['--p4-probe-cadence', cadence]
        if p4_link:
            words += ['--p4-link', '--p4-ladder', p4_ladder, '--p4-data', p4_data,
                      '--p4-normal-ack', p4_normal_ack,
                      '--p4-link-timeout', str(p4_link_timeout), '--p4-hold-cycles', str(p4_hold_cycles),
                      '--p4-irs-silent-cycles', str(p4_irs_silent_cycles),
                      '--p4-answer-offset', str(p4_answer_offset), '--p4-breakin', p4_breakin]
            if p4_long_ack is not None:
                words += ['--p4-long-ack', p4_long_ack]
            if p4_train_policy != 'repeat':
                words += ['--p4-train-policy', p4_train_policy]
            if p4_mail_preset:
                words += ['--p4-chirp-ack', 'parity']
            if p4_text is not None:
                words += ['--p4-text', p4_text]
        args.repeat_gear = 0
    if p4_word_render is not None:
        words += ['--p4-word-render', p4_word_render]
    robust = getattr(args, 'robust', False)
    fallback = getattr(args, 'connect_fallback_after', None)
    if fallback is not None and not robust:
        raise ValueError('--connect-fallback-after falls back FROM --robust')
    if robust:
        words += ['--connect', 'robust', '--after-robust', after_robust]
        if after_robust == 'p2':
            words += ['--robust-call-text', getattr(args, 'robust_call_text', 'pmc'),
                      '--robust-p2-max', str(max(1, getattr(args, 'robust_p2_max', 10)))]
            forced = getattr(args, 'robust_p3_without_grant', None)
            if forced is not None:
                words += ['--robust-p3-without-grant', str(max(1, forced))]
            if getattr(args, 'robust_p2_marker_flag', 'parity') != 'parity':
                words += ['--robust-p2-marker-flag', args.robust_p2_marker_flag]
            if getattr(args, 'robust_p2_opening', 'none') != 'none':
                words += ['--robust-p2-opening', args.robust_p2_opening]
        if fallback is not None:
            if fallback < 1:
                raise ValueError('--connect-fallback-after counts unanswered calls')
            words += ['--connect-fallback-after', str(fallback)]
    if after_connect != 'arq':
        if send and not (after_connect == 'p1-robust' and p4_link):
            raise ValueError(f'--after-connect {after_connect} keys no data packet, '
                             'so it can carry no --send')
        words += ['--after-connect', after_connect]
        if after_connect == 'listen':
            cycles = getattr(args, 'after_connect_cycles', 6)
            if cycles < 1:
                raise ValueError('--after-connect-cycles reads at least one peer cycle')
            words += ['--after-connect-cycles', str(cycles)]
    if not qrm_guard:
        words.append('--no-qrm-guard')
    if timing_trial and entry_trial:
        raise ValueError('--entry-timing-trial and --timing-trial test separate phases')
    if timing_trial or entry_trial:
        if (args.unannounced or args.p1_mail or args.pactor1_only or send
                or args.long_cycle or args.changeover_p1_cs or args.p3_fallback
                or args.control_placement != 'audio-start'
                or args.control_tail != 'off' or args.control_stagger != 'off'):
            raise ValueError('--timing-trial requires the granted P3 historical '
                             'audio-start profile without mail, alternate controls '
                             'or long cycles')
        args.repeat_gear = 0
        args.observe_after_link_down = False
        words = _without(words, ('--p3-speed-up', 1), ('--hold', 1))
        words += ['--p3-speed-up', 'hold', '--hold', '32',
                  '--p3-entry-timing-trial' if entry_trial else '--p3-timing-trial',
                  entry_trial or timing_trial]
        if unbounded:
            words.append('--p3-timing-unbounded')
        if reply_delay is not None:
            words += ['--p3-timing-reply-delay', str(reply_delay)]
    if args.unannounced and (args.p1_mail or args.pactor1_only):
        raise SystemExit('--unannounced goes up without asking; --p1-mail and '
                         '--pactor1-only forgo going up at all. Pick one.')
    if args.unannounced:
        # Dropping the grant flag alone leaves a mail arm with no door named, and
        # `onair._arm_defaults` then holds it in PACTOR-1: three arms flew that way
        # on 2026-09-17 and one keyed no entry packet at all.
        words = _without(words, '--p1-grant-only', ('--p1-status-bits45', 1))
        words.append('--p3-uninvited')
    if args.p1_mail:
        args.pactor1_only, args.setup_phase = True, 'call'
    if args.pactor1_only:
        # The modem zeroes the announce bits itself for a PACTOR-1 mail arm, but only
        # where the flag was left unset. Hard-coding them here advertised an upgrade
        # this arm then declines, and WS8EOC answered the advertisement with a grant.
        words = _without(words, '--p1-grant-only', ('--p1-status-bits45', 1))
        words.append('--pactor1-only')
    if getattr(args, 'pactor2_only', False):
        if args.unannounced or args.pactor1_only or getattr(args, 'p1_6a9', False):
            raise ValueError('--pactor2-only sets its own announcement and ceiling; '
                             'drop --unannounced, --pactor1-only and --p1-6a9')
        words = _without(words, '--p1-grant-only', ('--p1-status-bits45', 1))
        words.append('--pactor2-only')
    if getattr(args, 'p1_6a9', False):
        if args.unannounced:
            raise ValueError('--p1-6a9 conflicts with --unannounced')
        words = _without(words, ('--p1-status-bits45', 1))
        words += ['--p1-6a9', '--p1-status-bits45', '1']
    if args.setup_phase != 'reply':
        words = _without(words, ('--p1-setup-phase', 1)) + ['--p1-setup-phase', args.setup_phase]
    words += ['--tx-drive', f'{args.tx_drive:g}', '--p1-drive', f'{args.p1_drive:g}']
    for flag, watts in (('--p1-watts', level['p1_watts']), ('--p3-watts', level['p3_watts'])):
        if watts is not None:
            words += [flag, f'{watts:g}']
    words += ['--p3-follow-offset', args.follow_offset,
              '--p3-entry-sl', str(args.entry_sl),
              '--p3-traffic-sl', str(args.traffic_sl),
              '--p3-repeat-gear', str(args.repeat_gear),
              '--p3-keep-slots', args.keep_slots,
              '--p3-control-placement', args.control_placement,
              '--long-cycle' if args.long_cycle else '--no-long-cycle']
    for flag, value in (('--p3-control-tail', args.control_tail),
                        ('--p3-control-stagger', args.control_stagger)):
        if value != 'off':
            words += [flag, value]
    crest_limit = getattr(args, 'crest_limit', 'off')
    if crest_limit != 'off':
        try:
            db = float(crest_limit)
        except ValueError:
            db = 0.0
        if db <= 0:
            raise ValueError('--crest-limit takes off or a positive dB target')
        words += ['--p3-crest-limit', f'{db:g}']
    if args.changeover_p1_cs:
        words.append('--p3-changeover-p1-cs')
    if not args.p3_fallback and not getattr(args, 'pactor2_only', False):
        words += ['--no-p3-fallback']
    # The flag defers our break-in until the mail client has a reply, and the mail
    # client cannot have a reply until we break in: VE3KPG 2026-09-18 keyed 19
    # empty SL1 packets at a gateway that acknowledged nine of them while the
    # banner it was holding had nowhere to go. Forward the explicit override
    # so the P3 mail preset cannot restore its default greeting wait.
    mail_eligible = ((after_connect == 'arq' and not p4_entry)
                     or (after_connect == 'p1-robust' and p4_link))
    if not timing_trial and not entry_trial and mail_eligible and not no_mail:
        words += ['--mail-fetch', '--no-mail-wait-greeting' if args.no_wait_greeting
                  else '--mail-wait-greeting']
        words += ['--mail-sid', args.client_sid,
                  '--mail-out', str(directory / 'mail')]
    if send:
        words += ['--mail-send', str(send)]
    if '--mail-fetch' in words:
        words += _password_flags(args)
    words += supervisor
    idle_id = getattr(args, 'p1_idle_id_every', 0)
    if idle_id < 0:
        raise ValueError('--p1-idle-id-every counts idle packets: 0 (off) or more')
    if idle_id:
        words += ['--p1-idle-id-every', str(idle_id)]
    if args.observe_after_link_down and not p4_entry:
        words += ['--observe-after-link-down']
    adjustments = dict(follow_offset=args.follow_offset, entry_sl=args.entry_sl,
                       traffic_sl=args.traffic_sl,
                       repeat_gear=args.repeat_gear, keep_slots=args.keep_slots,
                       control_placement=args.control_placement,
                       control_tail=args.control_tail,
                       control_stagger=args.control_stagger,
                       crest_limit=crest_limit,
                       changeover_p1_cs=args.changeover_p1_cs,
                       long_cycle=args.long_cycle, p3_fallback=args.p3_fallback,
                       pactor1_only=args.pactor1_only, setup_phase=args.setup_phase,
                       p1_mail=args.p1_mail, unannounced=args.unannounced,
                       tx_drive=args.tx_drive, p1_drive=args.p1_drive,
                       mail='none' if no_mail else ('send' if send else 'fetch-only'),
                       send_file=str(send) if send else None,
                       observe_after_link_down=args.observe_after_link_down)
    if link_retries is not None:
        adjustments['link_retries'] = link_retries
    words += _call_hush(args, adjustments)
    adjustments['p3_max_try'] = getattr(args, 'p3_max_try', 2)
    adjustments['p3_max_down'] = getattr(args, 'p3_max_down', 6)
    adjustments['speed_policy'] = speed_policy
    if qrt_confirm:
        adjustments['p3_qrt_confirm'] = True
    if getattr(args, 'p3_tx_gear_hold', False):
        adjustments['p3_tx_gear_hold'] = True
    if getattr(args, 'p3_cs4_parity', False):
        adjustments['p3_cs4_parity'] = True
    if getattr(args, 'p3_irs_reply_offset_ms', 0.0):
        adjustments['p3_irs_reply_offset_ms'] = args.p3_irs_reply_offset_ms
    if idle_id:
        adjustments['p1_idle_id_every'] = idle_id
    if supervisor:
        adjustments.update(p1_level_digit=args.p1_level_digit,
                           p1_supervisor=list(args.p1_supervisor),
                           p1_supervisor_every=args.p1_supervisor_every)
    if timing_trial:
        adjustments.update(timing_trial=timing_trial, mail='none', speed_up='hold',
                           timing_unbounded=unbounded,
                           trial_reply_opportunities=None if unbounded else 12,
                           trial_seconds=None if unbounded else 20,
                           trial_ordinary_fields=None if unbounded else 3,
                           hold_unbounded=unbounded)
        if reply_delay is not None:
            adjustments.update(timing_reply_delay_ms=reply_delay,
                               target_entry_rotation_ms=600+reply_delay)
    if p3_mail:
        adjustments.update(p3_mail=True, speed_up=speed_up,
                           initial_reply_rotation_ms=603.125,
                           mail_wait_greeting=not args.no_wait_greeting)
    if entry_trial:
        adjustments.update(entry_timing_trial=entry_trial, mail='none', speed_up='hold',
                           entry_delay_ms=0 if entry_trial == 'A' else 4.875,
                           trial_seconds_from_first_entry=20)
    if p4_entry:
        adjustments.update(p4_entry=True, entry_ladder=['p4chirp'],
                           speed_up='hold', qrm_guard=qrm_guard,
                           p4_entry_attempts=attempts, p4_entry_timeout=timeout,
                           receive_seconds=4.0, observe_after_link_down=False)
        if not p4_link:
            adjustments['mail'] = 'none'
        if schedule:
            adjustments['status_sequence'] = list(schedule)
        if field is not None:
            adjustments['p4_probe_payload'] = field
        adjustments['p4_probe_packet'] = packet
        adjustments['p4_probe_cadence'] = cadence
        if values:
            adjustments.update(p4_probe_values=list(values),
                               p4_probe_headers=list(heads))
        if p4_chirp_counter:
            adjustments.update(p4_chirp_counter=True, experiment='grant-chirp-counter-v2',
                               transition_guard='selected grant releases future P1 slots',
                               status_sequence=[0x61, 0x61, 0x62],
                               mail='experimental-setup-only')
        if p4_link:
            adjustments.update(p4_link=True, after_connect='p1-robust',
                               p1_status_bits45=2, p1_robust_stall=12,
                               p1_robust_stall_then=stall_then,
                               p4_ladder=p4_ladder, p4_data=p4_data,
                               p4_normal_ack=p4_normal_ack,
                               p4_link_timeout=p4_link_timeout, p4_hold_cycles=p4_hold_cycles,
                               p4_irs_silent_cycles=p4_irs_silent_cycles,
                               p4_answer_offset=p4_answer_offset, p4_breakin=p4_breakin)
            if p4_long_ack is not None:
                adjustments['p4_long_ack'] = p4_long_ack
            if p4_text is not None:
                adjustments['p4_text'] = p4_text
            if p4_mail_preset:
                adjustments['p4_mail'] = True
    elif not qrm_guard:
        adjustments['qrm_guard'] = False
    if p4_word_render is not None:
        adjustments['p4_word_render'] = p4_word_render
    if robust:
        adjustments.update(connect='robust', after_robust=after_robust,
                           robust_calls=int(words[words.index("--retries")+1]), call_hush=getattr(args, 'call_hush', None))
        if after_robust == 'p2':
            adjustments.update(robust_call_text=getattr(args, 'robust_call_text', 'pmc'),
                               robust_p2_max=max(1, getattr(args, 'robust_p2_max', 10)))
            if getattr(args, 'robust_p3_without_grant', None) is not None:
                adjustments['robust_p3_forced'] = max(1, args.robust_p3_without_grant)
            if getattr(args, 'robust_p2_marker_flag', 'parity') != 'parity':
                adjustments['robust_p2_marker_flag'] = args.robust_p2_marker_flag
            if getattr(args, 'robust_p2_opening', 'none') != 'none':
                adjustments['robust_p2_opening'] = args.robust_p2_opening
        if fallback is not None:
            adjustments['connect_fallback_after'] = fallback
    if after_connect != 'arq' and not (after_connect == 'p1-robust' and p4_link):
        adjustments.update(after_connect=after_connect, mail='none')
        if after_connect == 'listen':
            adjustments['after_connect_cycles'] = cycles
    if getattr(args, 'p1_6a9', False):
        adjustments.update(p1_6a9=True, p1_entry_response='0x6A9', p1_status_bits45=1)
    return words, adjustments


def _call_hush(args, adjustments):
    hush = getattr(args, 'call_hush', None)
    if hush is None:
        return []
    try:
        calls, quiet = (int(v) for v in hush.split(','))
    except ValueError:
        raise ValueError('--call-hush takes N,M, e.g. 4,6') from None
    if calls < 1 or quiet < 1:
        raise ValueError('--call-hush takes two positive counts')
    adjustments['call_hush'] = f'{calls},{quiet}'
    return ['--call-hush', f'{calls},{quiet}']


def _supervisor_channel(args, send):
    """The supervisor-channel words: a sense arm's, so nothing that carries mail
    or keys PACTOR-4 rides with them."""
    digit = getattr(args, 'p1_level_digit', 1)
    blocks = getattr(args, 'p1_supervisor', None) or []
    every = getattr(args, 'p1_supervisor_every', 3)
    if digit == 1 and not blocks:
        return []
    beside = [flag for flag, on in (
        ('--send', send),
        ('--p1-mail', getattr(args, 'p1_mail', False)),
        ('--p2-mail', getattr(args, 'p2_mail', False)),
        ('--p3-mail', getattr(args, 'p3_mail', False)),
        ('--p4-mail', getattr(args, 'p4_mail', False)),
        ('--p4-chirp-counter', getattr(args, 'p4_chirp_counter', False)),
        ('--p4-link', getattr(args, 'p4_link', False)),
        ('--p4-entry', getattr(args, 'p4_entry', False)),
        ('--after-connect chirp', getattr(args, 'after_connect', 'arq') == 'chirp'),
    ) if on]
    if beside:
        raise ValueError('--p1-level-digit and --p1-supervisor sound the supervisor '
                         f'channel on a sense arm; not with {", ".join(beside)}')
    if every < 1:
        raise ValueError('--p1-supervisor-every counts acknowledged packets: 1 or more')
    for block in blocks:
        supervisor_block(block)
    words = ['--p1-level-digit', str(digit)] if digit != 1 else []
    if blocks:
        words += ['--p1-supervisor', *blocks, '--p1-supervisor-every', str(every)]
    return words


def _after_robust(args):
    """`--after-robust` resolved, with the probe it implies set up on `args`."""
    after = getattr(args, 'after_robust', None)
    if after is None:
        after = 'p2' if getattr(args, 'robust', False) else None
        args.after_robust = after
        return after
    if not getattr(args, 'robust', False):
        raise ValueError(f'--after-robust {after} follows a robust answer; --robust draws one')
    if after == 'p3':
        closed = next((flag for flag in ('pactor1_only', 'p1_mail', 'p2_mail')
                       if getattr(args, flag, False)), None)
        if closed:
            raise ValueError(f"--after-robust p3 keys a PACTOR-3 entry packet, which "
                             f"--{closed.replace('_', '-')} rules out")
    if after == 'p4-robust':
        if getattr(args, 'p4_probe_packet', 'chirp') not in ('chirp', 'robust'):
            raise ValueError('--after-robust p4-robust keys the robust short packet, '
                             f'not --p4-probe-packet {args.p4_probe_packet}')
        args.p4_probe_packet = 'robust'
        if getattr(args, 'p4_probe_values', None) is None:
            args.p4_probe_values = '6,9,8'
    return after


def _datablock(args, packet):
    """(values, headers) for a robust probe, or why they are not a query."""
    text = getattr(args, 'p4_probe_values', None)
    named = getattr(args, 'p4_probe_headers', None)
    if text is None and named is None:
        return (), ()
    if packet != 'robust':
        raise ValueError('--p4-probe-values and --p4-probe-headers describe the '
                         'robust datablock; --p4-probe-packet robust keys it')
    if text is None:
        raise ValueError('--p4-probe-headers names a header per value, and '
                         '--p4-probe-values names none')
    try:
        values = tuple(int(v, 0) for v in text.split(',') if v.strip())
    except ValueError:
        raise ValueError('--p4-probe-values takes comma-separated numbers, '
                         'e.g. 6,9,8') from None
    if not 1 <= len(values) <= 4 or any(not 0 <= v <= 63 for v in values):
        raise ValueError('--p4-probe-values takes one to four values in 0..63')
    if named is None:
        return values, ('A',) * len(values)
    if named.strip().lower() == 'alt':
        return values, ('A', 'B', 'A', 'B')[:len(values)]
    words = tuple(w.strip().upper() for w in named.split(',') if w.strip())
    if any(w not in ('A', 'B') for w in words):
        raise ValueError("--p4-probe-headers takes 'alt' or a list of A,B")
    if len(words) == 1 and getattr(args, 'p4_link', False):
        words *= len(values)
    if len(words) != len(values):
        raise ValueError('--p4-probe-headers takes one header per '
                         '--p4-probe-values entry')
    return values, words


def _status_schedule(text, p4_chirp_counter):
    """The probe's query schedule off the command line, or why it is not one."""
    if text is None:
        return ()
    if p4_chirp_counter:
        raise ValueError('--p4-probe-statuses cannot rewrite the --p4-chirp-counter schedule')
    try:
        schedule = tuple(int(v, 0) for v in text.split(',') if v.strip())
    except ValueError:
        raise ValueError('--p4-probe-statuses takes comma-separated byte '
                         'literals, e.g. 0x63,0x60') from None
    if not 1 <= len(schedule) <= 4 or any(not 0 <= v <= 255 for v in schedule):
        raise ValueError('--p4-probe-statuses takes one to four bytes in 0..255')
    return schedule


def _p2_mail_flags(level, args, directory, send):
    """A level-2 mail arm does not inherit the assessed P3 entry experiments."""
    incompatible = ('p1_mail', 'p3_mail', 'pactor1_only', 'unannounced',
                    'p4_entry', 'timing_trial', 'entry_timing_trial',
                    'timing_unbounded', 'p3_qrt_confirm', 'p3_tx_gear_hold',
                    'changeover_p1_cs')
    if (any(getattr(args, name, False) for name in incompatible)
            or getattr(args, 'timing_reply_delay', None) is not None
            or args.setup_phase != 'reply'
            or args.control_placement != 'audio-start'
            or args.control_tail != 'off' or args.control_stagger != 'off'):
        raise ValueError('--p2-mail requires the P2 mail profile without P1/P3/P4 '
                         'presets, timing trials or alternate controls')
    words = ['--retries', '20', '--max-cycles', '32', '--hold', '96',
             '--p2-mail', '--announce-lower',
             '--p3-speed-up', args.speed_up,
             '--tx-drive', f'{args.tx_drive:g}', '--p1-drive', f'{args.p1_drive:g}',
             '--long-cycle' if args.long_cycle else '--no-long-cycle',
             '--mail-fetch', '--no-mail-wait-greeting' if args.no_wait_greeting
             else '--mail-wait-greeting', '--mail-sid', args.client_sid,
             '--mail-out', str(directory / 'mail')]
    words += _password_flags(args)
    for flag, watts in (('--p1-watts', level['p1_watts']), ('--p3-watts', level['p3_watts'])):
        if watts is not None:
            words += [flag, f'{watts:g}']
    retries = getattr(args, 'link_retries', None)
    if retries is not None:
        if retries < 1:
            raise ValueError('--link-retries must be a positive cycle count')
        words += ['--link-retries', str(retries)]
    if not _qrm_guard(args):
        words.append('--no-qrm-guard')
    if send:
        words += ['--mail-send', str(send)]
    return words, dict(p2_mail=True, max_pactor_level=2,
                       p2_phase_delay_ms=18.75,
                       p1_status_bits45=1, p1_entry_response='0x6A9',
                       entry_sl=1, max_speed_level=3, setup_phase='reply',
                       speed_up=args.speed_up,
                       long_cycle=args.long_cycle, link_retries=retries,
                       tx_drive=args.tx_drive, p1_drive=args.p1_drive,
                       qrm_guard=_qrm_guard(args),
                       mail='send' if send else 'fetch-only',
                       send_file=str(send) if send else None,
                       mail_wait_greeting=not args.no_wait_greeting,
                       observe_after_link_down=False)


# WHAT IS ON THE CHANNEL, WITH TODAY'S INSTRUMENT, WHICH IS THE WRONG ONE.
#
# `sense_channel` measures energy and shape in the band the emission would fill.
# It reads an empty window as occupancy -- one with nothing in it has measured
# 2258 Hz against a real PACTOR-3 reception at 1344 Hz -- and on a good night it
# pegs on traffic nowhere near the station being called. What it cannot do at all
# is say WHO is there, which is the only question a campaign has: a gateway in a
# session will not answer, and a free one answers under other traffic.
#
# So it answers `is_target` with None and `confidence` with 0.0, and says so in
# its own detail line. The one exception is real: a Free Signal carries an
# eight-character ident behind a CRC-16 and the sense demands two bursts agreeing
# before it reports one, so exit 5 names a station and means it.
#
# rig.sh's exit codes: 0 clear, 1 busy, 2 deaf, 3 the wait ran out, 5 a host on
# the channel with no link in progress.
INSTRUMENT = 'energy sense, tools/lib/rig.sh'
# `core.busy.WINDOW_S`. The dwell a scan asks for cannot be honoured below this:
# the shell sense reads one fixed window, and its only other setting is a pounce
# that wants three clear 3 s windows in a row. A 2 s read at each stop is one of
# the things the decode assessor behind this same seam buys.
WINDOW_S = 8.0
CANNOT_SAY = 'energy and shape only; it cannot name an occupant'
HOST_FREE = ROOT / 'logs/sense/host-free'
SENSE_MEANING = {
    0: (False, 'nothing above the threshold in the band this emission would fill'),
    1: (True, 'something is in the band; ' + CANNOT_SAY),
    2: (None, 'the receiver delivered nothing to judge -- the station is deaf, '
              'and the launcher refuses a deaf receiver whatever this says'),
    3: (True, 'still occupied when the wait ran out; ' + CANNOT_SAY),
}


def assess(mode, peer, dwell_s=WINDOW_S):
    """Tune to the channel, read one window, and report what little it can tell.

    Never a wait. A scan spends its time across frequency rather than sitting on
    one dial, and the launcher does its own sensing at key-down, so a budget spent
    here would be spent twice.
    """
    script = (f'. "{ROOT}/tools/lib/ports.sh"; . "{ROOT}/tools/lib/rig.sh"; '
              f'sense_channel {peer["dial_hz"]} pactor "" 0')
    short = f' (a {dwell_s:g}s dwell was asked for; this instrument reads {WINDOW_S:g}s)' \
        if dwell_s < WINDOW_S else ''
    try:
        heard = claim.assess('assessment', script, timeout=WINDOW_S + 180,
                             env=dict(CAT_PORT=station.CAT_PORT, DXCALL=peer['call']))
    except (subprocess.SubprocessError, OSError) as trouble:
        return Assessment(occupied=None, instrument=INSTRUMENT,
                          detail=f'the sense did not run ({type(trouble).__name__}); '
                                 'the launcher gate still decides at key-down')
    said = [line.strip() for line in heard.stdout.splitlines() if line.strip()]
    detail = said[-1] if said else heard.stderr.strip()
    if heard.returncode == 5 and HOST_FREE.is_file():
        ident = HOST_FREE.read_text().split()[0]
        return Assessment(occupied=True, occupant=f'{ident}, no link in progress',
                          is_target=ident.upper() == peer['call'].upper(), inviting=True,
                          confidence=0.9, instrument=INSTRUMENT, spent_s=WINDOW_S,
                          detail=f'{ident} is inviting a call ({detail})')
    occupied, meaning = SENSE_MEANING.get(heard.returncode,
                                          (None, f'unexpected exit {heard.returncode}'))
    return Assessment(occupied=occupied, instrument=INSTRUMENT, spent_s=WINDOW_S,
                      detail=meaning + short + (f' -- {detail}' if detail else ''))


# DID WE KEY, AND DID ANYTHING COME BACK. Two readings out of the arm's own
# transcript, and neither is an inference.
#
# `TRANSMIT ARMED` is printed where the transmitter is armed, which is past the
# occupancy gate: across the 57 arms flown by the console this one supersedes, no
# transcript refused at the gate carries it and every transcript that keyed does.
# So its absence is "never attempted" and nothing else has to be enumerated.
#
# `control signals decoded N` is the count the arm itself prints at teardown. The
# milestone reader is not used for this: it reads the transcript for phrases and
# has been shown to under-report, and a station dropped from an elimination
# campaign on a missed phrase is a station that never gets called again.
KEYED = re.compile(r'TRANSMIT ARMED')
DECODED = re.compile(r'^control signals decoded (\d+)', re.M)


def robust_observations(directory):
    """Read only the modem's final observations, independent of capture paths."""
    path = Path(directory) / 'arm.log'
    text = path.read_text(errors='replace') if path.is_file() else ''
    rows = re.findall(r'^\[robust result\] (.+)$', text, re.M)
    if not rows:
        return {}
    try:
        data = json.loads(rows[-1])
    except json.JSONDecodeError:
        return {}  # An interrupted modem may leave its final line incomplete.
    return dict(robust_answer=data.get('answer'), robust_bridge=data,
                robust_calls_sent=data.get('robust_calls', 0))


def outcome(mode, record, code):
    """Which of the three this arm was. The transcript decides; the exit code does not."""
    log = Path(record['directory']) / 'arm.log'
    text = log.read_text(errors='replace') if log.is_file() else ''
    if not KEYED.search(text):
        return NOT_ATTEMPTED
    if re.search(r'robust answer \(entry \d+\) -> link up', text):
        return ANSWERED
    decoded = DECODED.search(text)
    # `experiment`, not `p4_chirp_counter`: every arm this ever flew, before and
    # after that flag was named, recorded this same string.
    if (record.get('adjustments') or {}).get('experiment') == 'grant-chirp-counter-v2':
        decoded = re.search(r'session ended: [^\n]*; (\d+) controls decoded before probe takeover', text)
    return ANSWERED if decoded and int(decoded.group(1)) else NO_REPLY


# WHAT THE ARM ITSELF SAID IT DID, past the rung it reached. Every pattern here
# reads a line the arm prints at teardown or a line it prints where the thing
# happened; none of them measures anything and none infers. The phrases handed
# back quote the transcript, so an operator reading the report and the log side
# by side sees the same words in both.
HELD = re.compile(r'the peer answered \d+ held cycles \([^)]+\)'
                  r'(?: and never alternated|, alternating \d+ times)')
ALTERNATED = re.compile(r'the peer never alternated its acknowledgement \([^)]*\)')
PACKETS = re.compile(r'^packet counters sent: (.+)$', re.M)
COUNTER = re.compile(r'#(\d+)')
DECODED_FULL = re.compile(r'^control signals decoded (\d+) \(([^)]*)\)', re.M)
NULL = re.compile(r'^connect candidates: (.+?) -- ', re.M)
UPGRADED = re.compile(r'link upgraded to PACTOR-3')
GRANT = re.compile(r'0x59A')
MAIL_MOVED = re.compile(r'^mail: (?:received|wrote|accepted) .+$', re.M)
MAIL_STAGE = re.compile(r'^mail: stage (.+)$', re.M)
# The one refusal a gateway states outright, read out of the gateway's own words
# rather than out of a waveform. WS8EOC and VE3KPG both hung up on 09-17 with
# "This RMS channel (<call>) is set to only accept Pactor levels above 2 -
# Disconnecting", and it names the CHANNEL rather than the station or the
# software, so it is a sysop setting on that frequency and not a property of a
# Trimode build. Matched against what `spoken` gathered, so a refusal that came
# after the banner still finds it.
LEVEL_POLICY = re.compile(r'accept Pactor levels above (\d)', re.I)

# THE OBSERVATIONS THAT BECOME FINDINGS. An occupied answer slot, a receiver
# that cannot carry the arm, a transmit slot lost to a missed deadline, and an
# onset the arm could not attribute -- the operator has said that unexplained
# behaviour meets their bar, so a report that drops it is the wrong report.
#
# PTT LEAD SHORT is deliberately not here. It fired on 374 lines of this corpus
# and is a property of this rig rather than of any contact; `view.py` already
# states it once and counts it, and a campaign report that led with it would
# bury the three that matter.
ANOMALIES = (
    ('an answer slot stood above the session\'s own floor and no reader took any of it',
     re.compile(r'!! ANSWER SLOT OCCUPIED')),
    ('a transmit slot was lost to a missed deadline', re.compile(r'SLOT \d+ IS GONE')),
    ('a changeover the arm would not place', re.compile(r'!! CHANGEOVER NOT PLACED')),
    ('a PACTOR-3 reply held back for want of a fresh packet clock',
     re.compile(r'!! P3 REPLY CLOCK')),
    ('an answer the arm could not attribute -- onset by energy alone',
     re.compile(r'onset by energy .*unattributed')),
    ('the receiver was clipping; an answer would have arrived undecodable',
     re.compile(r'-> CLIPPING')),
    ('the receive level could not carry the arm', re.compile(r'receive level cannot carry')),
    ('samples the converter timestamped and the arm never received',
     re.compile(r'CAPTURE LOSS')),
    ('the witness could not be registered, so silence in it is not the peer\'s',
     re.compile(r'WITNESS UNALIGNED|WITNESS UNREGISTERED|WITNESS DEAF|!! NO WITNESS')),
)


def counters(said):
    """The distinct packet counters that went out, and whether more than one did.

    "did we get more than one data packet" is this line and nothing else: the arm
    prints what it sent against what was acknowledged, and a counter that never
    left #1 is a session that repeated one packet however many times it keyed.
    """
    seen = list(dict.fromkeys(COUNTER.findall(said)))
    if not seen:
        return ''
    return (f' -- the counter reached #{max(seen, key=int)}' if len(seen) > 1 else
            f' -- the counter never advanced past #{seen[0]}')


def readout(mode, record, text):
    """What one PACTOR arm did, in the arm's own words, with the limits on each figure."""
    reached = (record.get('verdict') or {}).get('reached') or []
    said = record.get('adjustments') or {}
    did, caveats = [], []
    if 'P2 call packet never taken:' in text:
        did.append('robust call answered; P2 call packet never taken')
    elif 'PACTOR-2 -> PACTOR-3' in text:
        did.append('P2 bridge reached P3 entry')
    for pattern in (ALTERNATED, HELD):
        if found := pattern.search(text):
            did.append(found.group(0))
            break
    if packets := PACKETS.search(text):
        did.append(f'packet counters sent: {packets.group(1)}{counters(packets.group(1))}')
    if decoded := DECODED_FULL.search(text):
        did.append(f'control signals decoded {decoded.group(1)} ({decoded.group(2)})')
        # THE NULL GOES BESIDE THE COUNT, not into a footnote. A bare count reads
        # as evidence about the channel, and the arm has already worked out what
        # chance alone gives on the same windows.
        if (null := NULL.search(text)) and int(decoded.group(1)):
            did.append(f'and the arm\'s own null beside it: {null.group(1)}')
            caveats.append(('a control-signal count on its own',
                            'it is not evidence about the channel until the null beside it '
                            'is read; the arm computes that on the same windows and prints '
                            'it in every transcript'))
        if not UPGRADED.search(text):
            caveats.append((f'{decoded.group(2).split(", ")[-1]} in that count',
                            'the PACTOR-3 control reader only runs inside a PACTOR-3 link and '
                            'this arm never entered one, so the zero is the link state rather '
                            'than a reading of the channel'))
    if UPGRADED.search(text):
        did.append('the link upgraded to PACTOR-3 on the peer\'s grant')
    if 'link upgraded to PACTOR-2' in text:
        did.append('the link entered PACTOR-2')
    if said.get('p2_mail'):
        caveats.append(('PACTOR-2 mail is experimental',
                        'P1-to-P2 negotiation and P2 control signals still need '
                        'real-peer validation; transmit speed is capped at SL3'))
    crossed = ''
    if moved := MAIL_MOVED.search(text):
        crossed = moved.group(0)
    elif stage := MAIL_STAGE.search(text):
        crossed = f'nothing moved; the mail client stopped at {stage.group(1)}'
    spoke = spoken(text)
    if said.get('experiment') == 'grant-chirp-counter-v2':
        if result := re.search(r'^P4 experiment: .+$', text, re.M):
            did.append(result.group(0))
        crossed = 'no mail transferred; P4 setup/counter experiment only'
        caveats.append(('the counter changed deliberately',
                        'it was an experimental stimulus, not evidence of an ACK; '
                        'header candidates do not establish P4 control meanings'))
    suggests = ''
    if refused := LEVEL_POLICY.search(spoke):
        suggests = (f'the gateway refused in its own words -- it takes only Pactor levels '
                    f'above {refused.group(1)} -- so a PACTOR-1 session is not served on this '
                    'channel however it is flown, and only a pass that is already upgraded '
                    'when the mail client speaks can be')
    elif said.get('pactor1_only') and GRANT.search(text):
        suggests = ('this arm flew --pactor1-only and the peer sent 0x59A; an arm without '
                    'it is what puts that grant to the test.')
    elif any(pattern.search(text) for phrase, pattern in ANOMALIES[-3:-1]):
        suggests = 'set the codec gain before another arm here; the receive level was the fault.'
    return Readout(
        rung=highest(mode, reached), did=tuple(did), crossed=crossed, suggests=suggests,
        said=spoke,
        anomalies=tuple((many, what) for what, pattern in ANOMALIES
                        if (many := len(pattern.findall(text)))),
        caveats=tuple(caveats))


def _password_flags(args):
    named = getattr(args, 'mail_password_file', None)
    if not named and not os.environ.get('WINLINK_PASSWORD') and not os.environ.get('WINLINK_PASSWORD_FILE') and MAIL_PASSWORD_FILE.is_file():
        named = str(MAIL_PASSWORD_FILE)
    return ['--mail-password-file', str(Path(named).expanduser())] if named else []


def ready(mode, plan):
    from contextlib import nullcontext
    from .station import winlink_password
    words = plan['command']
    if '--mail-fetch' not in words:
        return nullcontext()
    named = words[words.index('--mail-password-file') + 1] if '--mail-password-file' in words else None
    return winlink_password(named)


PROFILE = Mode(
    name='pactor',
    launcher='pactor',
    # `onair.sh pactor` runs `python -m hfmodem.shrike.onair`, and that import is
    # the whole of what this arm resolves out of the candidate.
    entry=('hfmodem.shrike.onair',),
    roster=HERE / 'peers.json',
    arms=HERE / 'arms',
    lock=HERE / 'arm.lock',
    bands=(160, 80, 60, 40, 30, 20, 17, 15, 12, 10),
    levels=LEVELS,
    phases=('p1_watts', 'p3_watts'),
    drives=('tx_drive', 'p1_drive'),
    authorised_watts=AUTHORISED_WATTS,
    flags=flags,
    ready=ready,
    arguments=arguments,
    milestones=MILESTONES,
    assess=assess,
    dwell_floor_s=WINDOW_S,
    outcome=outcome,
    readout=readout,
)
