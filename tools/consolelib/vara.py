# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The VARA profile: the bounded level experiment, and every flag an arm may vary.

`FLOWN` is the line that flew every arm of the original bounded on-air runner,
and a test holds this profile's line against that runner's, word for word, across
the spread of arguments the two of them share. What an arm may change is a flag,
and what it changed is recorded in the arm's own provenance beside the line.

The roster, the fixed-MID messages, the shipped transcript parser and operating constraints stay
where they were written, and the arms directory stays with them, so this console
and the one in that directory read the same 82 flown arms and cannot key over
each other.

Five guards of the runner's are not moved, and each is named here rather than
quietly missing:

  * The supervisor set RFPOWER over CAT and verified the readback. The presets
    carry the drive and not the watts; `--power` and the station clamp set RFPOWER
    at arm start, and the arm's own `readings` record what the rig was actually on.
  * The supervisor's no-progress clock ended a link that was keying and moving
    nothing. `--timeout` and `--mail-timeout` still bound the launcher from
    inside; no watchdog sits outside it.
  * The runner refused to start a new arm while any earlier one had an
    unresolved cleanup, and the supervisor checked that nothing else held the CAT
    and PTT ports and that PTT read low before it keyed. Neither is repeated
    here: `onair.sh` claims the rig and requires the ports before it opens a
    device, and proves the key is down when the verb ends, and the console's own
    arm lock is the other half of the claim.
  * The slot is recorded and never enforced. `minimum_slot_remaining_s` says what
    this arm needs and the operator weighs it; nothing here stands between their
    word and the carrier going up.

Three things this console does that the runner did not. `--witness` is on, so a
third-party KiwiSDR records the arm; no VARA arm has ever flown with one.
High speed capable is on by default, so the link-setup carries the high speed
marker and the gear-shift climbs the full ladder; `--winlink-cartel-mode --no-witness`
makes the launcher line byte-identical to what flew. And a
credential the operator does not name as a file is read into this arm's
environment rather than passed as a path, so it is in no transcript that quotes
the line.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess

from .modes import (ANSWERED, CLEARED, NOT_ATTEMPTED, NO_REPLY, Assessment, Mode, Readout,
                    highest, spoken)
from .roster import read_json
from . import claim, station
from .station import ONAIR, ROOT, winlink_password

from .operator import mode_root
HERE = mode_root('vara')

RETRY_FORMS = ('full', 'stock')
BANDWIDTHS = (500, 2300, 2750)
BANDS = ((3525000, 3600000), (7025000, 7125000),
         (10100000, 10150000), (14025000, 14150000))

# The operator authorised these arms at 50 W. No preset asks the radio for watts,
# so the figure stands as the ceiling a preset may name.
AUTHORISED_WATTS = 50.0

#: The soundcard peak every assessed arm flew. The retry form and the drive are
#: the two levers a below-offer signal report leaves; everything else about the
#: transmitter was fixed while those arms were flown.
FLOWN_PEAK = 0.50

LEVELS = {
    'current': dict(amplitude=FLOWN_PEAK,
                    note='the peak every assessed arm flew; RFPOWER as the rig sits, '
                         'read back into the arm rather than set by it'),
}

# The assessed profile, unchanged and in order. Everything an arm may vary is
# built around it below.
FLOWN = ('--timeout', '75', '--max-cr', '10', '--cr-interval', '3',
         '--max-link-setups', '3')

# Unanchored where a line can carry a log stamp or kestrel_connect's indent. The
# gateway's confirmed connect response is as much an answer as a lattice frame.
MILESTONES = (
    ('cleared', CLEARED),
    ('keyed', re.compile(r'PTT ON|^originating: ', re.M)),
    ('answer', re.compile(r'lattice frame \d+|rx connect-response confirmed for ')),
    ('connect', re.compile(r'— CONNECTED|— the link is up|\bCONNECTED\s+[A-Z0-9]')),
    ('greeting', re.compile(r'\bmail: remote SID ')),
    ('mail', re.compile(r'\bmail: exchange complete')),
)

ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
# Below-offer answers: the gateway heard our request and named how much of it
# survived, without offering a connection. This grades our transmitter, not
# theirs, and the two levers are `--cr-retry-form` and `--amplitude`.
SIGNAL_REPORT = re.compile(
    r'could not read it — its answer sits at lattice frame \d+'
    r'|answered with BW\d+ lattice frame \d+')
# The launcher's own refusals: a dial that did not take or a passband that clips
# the emission, a deaf or unusable receiver, arguments it would not read. A pass
# ends on these rather than moving to the next peer.
LAUNCHER_STOP = re.compile(
    r'TUNE DID NOT TAKE|RECEIVER DEAF|RECEIVER INTERMITTENT|cannot hear the channel'
    r'|the receive level cannot carry this arm|NOTHING WAS TRANSMITTED')

POUNCE_EXPIRED, CHANNEL_BUSY, PREKEY_REFUSED = 3, 4, 7
CONTINUE_REASON = {0: 'the arm finished its own bounded run',
                   1: 'no answer, or a captured protocol failure',
                   POUNCE_EXPIRED: 'occupancy: the pounce budget expired without a clear channel',
                   CHANNEL_BUSY: 'occupancy: the gate refused the channel',
                   PREKEY_REFUSED: 'nothing keyed -- a rig or channel refusal ahead of '
                                   'key-down, not an attempt at the gateway'}

NOTE = ('Level is a coverage target, not a numeric modem command. '
        '500/2300 have known session defects; this is a bounded diagnostic profile.')

# THE LEVEL LADDER, READ OFF THE TRANSCRIPT. Our over prints its record; the
# peer's continue after it is the acknowledgement. The peer's over prints its
# level, and the DATA line that follows says whether we answered it. The older
# record-suffix forms still read: at BW2300 record L-1 is level L and the base
# level prints no suffix at all.
KEYED = re.compile(r'\btx DATA over #\d+(?: at record (\d+))? \(')
HEARD = re.compile(r'\brx over-continue from |solicited pending over #\d+ acknowledged'
                   r'| continued at frame \d+ — the link goes to level')
PEER_OVER = re.compile(r"\brx \S+'s over at level (\d+) \(asked")
WIDE = re.compile(r'\brx wideband frame:.*?reference columns(?: at record (\d+))?, CRC clean')
NARROW = re.compile(r'\brx BW500 burst: \d+ frame\(s\) at level (\d+), CRC clean')
OVER_ANSWERED = re.compile(r'\brx DATA over #\d+ .*— answering')
BW = re.compile(r'\bBW\s*(500|2300|2750)\b')


def host_level(bw, record):
    if record is None:
        return 4
    if bw == 500:
        return {0: 1, 1: 2, 2: 3, 3: 4}.get(record, record)
    return record + 1 - (100 if bw == 2750 else 0)


def ladder(text, bw):
    """Every level each direction reached. Ours count as decoded when keyed, and
    as acknowledged on the peer's continue; theirs when read here, and when we
    answered the over."""
    keyed, acknowledged, decoded, answered = set(), set(), set(), set()
    ours = theirs = None
    for line in text.splitlines():
        if match := KEYED.search(line):
            ours = host_level(bw, int(match[1]) if match[1] else None)
            keyed.add(ours)
        elif ours is not None and HEARD.search(line):
            acknowledged.add(ours)
            ours = None
        elif match := PEER_OVER.search(line) or NARROW.search(line):
            theirs = int(match[1])
            decoded.add(theirs)
        elif match := WIDE.search(line):
            theirs = host_level(bw, int(match[1]) if match[1] else None)
            decoded.add(theirs)
        elif theirs is not None and OVER_ANSWERED.search(line):
            answered.add(theirs)
            theirs = None
    return dict(tx=dict(decoded=sorted(keyed), acknowledged=sorted(acknowledged)),
                rx=dict(decoded=sorted(decoded), acknowledged=sorted(answered)))


def drive_peak(value):
    try:
        peak = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Expected a soundcard peak above 0 and at most 1.') from None
    if not 0 < peak <= 1:
        raise argparse.ArgumentTypeError('Expected a soundcard peak above 0 and at most 1.')
    return peak


def widths(args):
    """Every width on the line, in the order it was named. A campaign may name several."""
    asked = args.bw if isinstance(args.bw, list) else [args.bw]
    return list(dict.fromkeys(asked))


def width_rank(peer, bw):
    """How well one listing fits a call at one width, or None if it may not take it.

    A gateway listed at BW2750 will take a BW2300 call, so a BW2300 arm may use
    one; a gateway that advertises both is wanted at the width it advertises
    exactly, which is why this ranks rather than filters.
    """
    if peer['bw_hz'] == bw:
        return 0
    return 1 if bw == 2300 and peer['bw_hz'] == 2750 else None


def channel_rank(peer, args):
    """Which listings this line may call at all, at the best of the widths it names."""
    fits = [fit for fit in (width_rank(peer, bw) for bw in widths(args)) if fit is not None]
    return min(fits, default=None)


def channel_arms(args, peer):
    """One arm for each width this channel may be called at, in the order they were named.

    A width change costs nothing -- no retune, no ATU memory, no different
    propagation picture -- so the widths sit innermost, and a channel is asked at
    every one of them it may be asked at before the pass moves on to the next.
    """
    at = [argparse.Namespace(**dict(vars(args), bw=bw)) for bw in widths(args)]
    return [(one, f'BW{one.bw}') for one in at if width_rank(peer, one.bw) is not None]


def arguments(air):
    """What a VARA arm may vary. The defaults are what flew."""
    link = air.add_argument_group('VARA link')
    link.add_argument('--bw', type=int, choices=BANDWIDTHS, required=True,
                      help='the channel width this arm calls at (required)')
    varied(air, link)


def campaign_arguments(many):
    """The same, except that a campaign names widths the way it names bands.

    One arm calls at one width. A campaign may name several, and every channel
    it finds is called at each of the named widths that channel may take.
    """
    link = many.add_argument_group('VARA link')
    link.add_argument('--bw', type=int, nargs='+', choices=BANDWIDTHS, required=True,
                      help='the channel width(s) this campaign may call at (required); a channel is called '
                           'at each it may take before the pass moves on')
    varied(many, link)


def varied(air, link):
    """Everything past the width that an arm may vary. The defaults are what flew."""
    link.add_argument('--direction', choices=('tx', 'rx'), default='tx')
    link.add_argument('--mail-timeout', type=int, default=240, choices=range(60, 601),
                      metavar='60..600')
    link.add_argument('--winlink-cartel-mode', action='store_true',
                      help='key the stock link-setup and hold the L1-L4 ladder; the default is the high speed '
                           'capable form every user of this modem gets')
    link.add_argument('--cr-retry-form', choices=RETRY_FORMS, default=None,
                      help="what a re-keyed connect-request goes out as: this station's 41-symbol request "
                           "(one arm's default) or the 32-symbol form a stock caller re-keys (a campaign's)")
    link.add_argument('--data-retries', action='store_true',
                      help='repeat an unacknowledged DATA rather than close with delivery unconfirmed (off '
                           'since 2026-09-12). At BW2750 this also drops --probe-intermediate-query, which the '
                           'launcher refuses without --no-data-retries')

    transmit = air.add_argument_group('transmit and credentials')
    transmit.add_argument('--amplitude', type=drive_peak, default=None, metavar='PEAK',
                          help=f'soundcard peak for transmit (default {FLOWN_PEAK:.2f})')
    transmit.add_argument('--mail-password-file', metavar='PATH',
                          help="a credential to put on the line; otherwise it goes into the arm's "
                               'environment and never into the transcript')


def retry_form(args):
    """The re-keyed request's form, which one arm and a campaign default differently.

    One arm asks with this station's own 41-symbol request, which is what the
    assessed arms flew. A campaign asks with the 32-symbol form a stock caller
    re-keys, because what a campaign is measuring is which gateways answer at all
    and the stock form is the one they were built against. Naming the flag says
    it for both.
    """
    return args.cr_retry_form or ('stock' if getattr(args, 'action', None) == 'campaign'
                                 else 'full')


def head(mode, peer, args, name):
    """`vara-mail` reads its bandwidth as a bare word where `--band` would stand.

    So this verb's head is its own: the width goes in as a positional and there
    is no band flag on the line at all. `--log` is taken off the line by
    `onair.sh` before any verb reads a positional, so it stands here harmlessly.
    """
    return [str(ONAIR), mode.launcher, peer['call'], str(peer['centre_hz']), str(args.bw),
            '--log', name]


def flags(peer, level, args, directory, send):
    """The launcher words between the shared head and the shared witness/force tail."""
    if send and args.direction != 'tx':
        raise ValueError('`--direction rx` sends nothing, so there is no arm for a delivery '
                         'test to ride. Fly it at `--direction tx`.')
    if args.amplitude is None:
        from hfmodem.core import config
        audio = getattr(args, 'selected_audio', None)
        if getattr(args, 'selected_rig', station.RIG_NAME) == 'ft891' and not audio:
            args.amplitude = FLOWN_PEAK
        else:
            args.amplitude = (audio.tx_drive if audio and audio.tx_drive is not None
                              else config.ChainAudio().drive_for())
            args.drive_sources = {'amplitude': 'station-toml' if audio and audio.tx_drive is not None else 'provisional-fallback'}
    form = retry_form(args)
    words = list(FLOWN)
    words += ['--cr-retry-form', form, '--amplitude', str(args.amplitude),
              '--mail-timeout', str(args.mail_timeout), '--mail-fetch',
              '--mail-sid', args.client_sid, '--mail-out', str(mode_root('vara') / 'mail'),
              '--trace-timing']
    if not args.data_retries:
        words += ['--no-data-retries']
    if send:
        words += ['--mail-send', str(send)]
    if args.bw in (2300, 2750):
        words += ['--over-continue', 'short']
    # The pending-over query holds the original DATA until a response resolves it,
    # which is the same job the retry does and the launcher will not take both.
    probe = args.bw == 2750 and not args.data_retries
    if probe:
        words += ['--probe-intermediate-query']
    if args.winlink_cartel_mode:
        words += ['--winlink-cartel-mode']
    password = (str(Path(args.mail_password_file).expanduser().resolve())
                if args.mail_password_file else None)
    if password:
        words += ['--mail-password-file', password]

    # Source-checked lengths, not synthesized DSP: 9 further CRs * (1.75 + 3)s,
    # and three <=4.5s setups separated by 8s.
    budget = dict(timeout_s=75, max_cr=10, cr_train_s=42.75,
                  max_link_setups=3, setup_train_s=29.5, required_s=72.25,
                  minimum_timeout_s=73, fits=True,
                  basis='4.5s upper bound per link setup; constants checked locally')
    adjustments = dict(row=f'{args.bw}-{args.direction}',
                       bw_hz=args.bw, direction=args.direction,
                       cr_retry_form=form, amplitude=args.amplitude,
                       high_speed=not args.winlink_cartel_mode, mail_timeout_s=args.mail_timeout,
                       data_retries=args.data_retries, probe_intermediate_query=probe,
                       advertised_hours_utc=peer.get('hours'),
                       roster_retrieved_utc=read_json(mode_root('vara') / 'peers.json').get('retrieved_utc'),
                       minimum_slot_remaining_s=75 + args.mail_timeout + args.pounce + 120,
                       connect_budget=budget, message=None,
                       mail_password_file=password,
                       diagnostic_only=args.bw != 2750, note=NOTE)
    return words, adjustments


def ready(mode, plan):
    """A credential in this arm's environment, and only for as long as it flies."""
    return winlink_password(plan['adjustments']['mail_password_file'])


# WHAT IS ON THE CHANNEL. The same instrument the launcher's own gate uses at
# key-down, asked early and without a wait -- `sense_channel <dial> vara <bw>`,
# which judges the passband a VARA emission of that width actually fills rather
# than the whole audio band. That narrowing is real and it is the only thing this
# assessor has over PACTOR's.
#
# It still cannot say WHO is there, which is the only question a campaign has: a
# gateway already in a session will not answer, and a free one answers under
# other traffic. Only a demodulator separates those, and VARA's own reading needs
# the codec open, which a scan cannot afford at every stop. So `is_target` is None
# and `confidence` is 0.0, and the detail line says as much.
#
# The width is the channel's own listing rather than the arm's `--bw`: an
# assessor is handed the peer and not the arm's arguments, so a BW2300 call to a
# BW2750 listing is sensed over the wider band. Wider is the conservative
# direction and it is the only one available here.
#
# ARDOP's channels are listed the same way and read by the same gate, so its
# profile asks this with its own emission name.
WINDOW_S = 8.0          # `core.busy.WINDOW_S`; the shell sense reads one fixed window
CANNOT_SAY = 'energy and shape only; it cannot name an occupant'
HOST_FREE = ROOT / 'logs/sense/host-free'
SENSE_MEANING = {
    0: (False, 'nothing above the threshold in the band this emission would fill'),
    1: (True, 'something is in the band; ' + CANNOT_SAY),
    2: (None, 'the receiver delivered nothing to judge -- the station is deaf, '
              'and the launcher refuses a deaf receiver whatever this says'),
    3: (True, 'still occupied when the wait ran out; ' + CANNOT_SAY),
}


def sense(emission, peer, dwell_s):
    """Tune to the channel, read one window, and report what little it can tell."""
    instrument = f'energy sense over the {emission.upper()} passband, tools/lib/rig.sh'
    script = (f'. "{ROOT}/tools/lib/ports.sh"; . "{ROOT}/tools/lib/rig.sh"; '
              f'sense_channel {peer["dial_hz"]} {emission} {peer["bw_hz"]} 0')
    short = f' (a {dwell_s:g}s dwell was asked for; this instrument reads {WINDOW_S:g}s)' \
        if dwell_s < WINDOW_S else ''
    try:
        heard = claim.assess('assessment', script, timeout=WINDOW_S + 180,
                             env=dict(CAT_PORT=station.CAT_PORT, DXCALL=peer['call']))
    except (subprocess.SubprocessError, OSError) as trouble:
        return Assessment(occupied=None, instrument=instrument,
                          detail=f'the sense did not run ({type(trouble).__name__}); '
                                 'the launcher gate still decides at key-down')
    said = [line.strip() for line in heard.stdout.splitlines() if line.strip()]
    detail = said[-1] if said else heard.stderr.strip()
    # A Free Signal is PACTOR's, so the ident it names is never the gateway we are
    # calling in this mode. The gate reads a 5 as a go-ahead and calls; so does this.
    if heard.returncode == 5 and HOST_FREE.is_file():
        ident = HOST_FREE.read_text().split()[0]
        return Assessment(occupied=True, occupant=f'{ident}, no link in progress',
                          is_target=ident.upper() == peer['call'].upper(), inviting=True,
                          confidence=0.9, instrument=instrument, spent_s=WINDOW_S,
                          detail=f'{ident} is inviting a call, on a PACTOR Free Signal ({detail})')
    occupied, meaning = SENSE_MEANING.get(heard.returncode,
                                          (None, f'unexpected exit {heard.returncode}'))
    return Assessment(occupied=occupied, instrument=instrument, spent_s=WINDOW_S,
                      detail=meaning + short + (f' -- {detail}' if detail else ''))


def assess(mode, peer, dwell_s=WINDOW_S):
    return sense('vara', peer, dwell_s)


def analyzer():
    """The shipped transcript reader, independent of historical session storage."""
    from . import vara_evidence
    return vara_evidence


def spelled_levels(levels):
    return ', '.join(f'L{level}' for level in levels)


def account(report, rungs, keyed, reports):
    """What this arm did, in the evidence's own terms and never the peer's."""
    said = []
    if report['connected']:
        said.append('connected')
    if rungs['tx']['decoded']:
        said.append('keyed overs at ' + spelled_levels(rungs['tx']['decoded']))
    elif keyed:
        said.append('keyed connect requests and no over')
    if rungs['rx']['decoded']:
        said.append('decoded overs at ' + spelled_levels(rungs['rx']['decoded']))
    if rungs['tx']['acknowledged']:
        said.append('acknowledgement read for ' + spelled_levels(rungs['tx']['acknowledged']))
    elif report['connected']:
        said.append('no acknowledgement read')
    if reports:
        said.append(f'{len(reports)} below-offer signal report(s): the gateway heard the '
                    'request and could not read it')
    said.append({'complete': 'mail exchange complete',
                 'failed_or_incomplete': 'mail did not complete'}.get(
                     report['mail_status'], 'no mail-completion line in the evidence'))
    return '; '.join(said) + '.'


def verdict(mode, directory):
    """What the arm's own transcript says, read by `analyze.py` and kept to its verdicts.

    The pieces `outcome` and `disposition` decide on are written down here, once,
    at teardown -- so a campaign classifies a channel out of the record rather
    than out of a phrase it re-reads later.
    """
    directory = Path(directory)
    log = directory / 'arm.log'
    if not log.is_file():
        return dict(reached=[], line='no transcript', keyed=False, connected=False,
                    mail='unknown', answered=False, signal_reports=[], stops=[])
    text = ANSI.sub('', log.read_text(errors='replace'))
    written = directory / 'provenance.json'
    said = read_json(written)['adjustments'] if written.is_file() else {}
    report = analyzer().analyze(log, bw=said.get('bw_hz'),
                                direction=said.get('direction') or 'either')
    heard = BW.search(text)
    rungs = ladder(text, said.get('bw_hz') or (int(heard[1]) if heard else None))
    reached = [name for name, pattern in MILESTONES if pattern.search(text)]
    reports = [line.strip() for line in text.splitlines() if SIGNAL_REPORT.search(line)]
    return dict(reached=reached,
                line=account(report, rungs, 'keyed' in reached, reports),
                keyed='keyed' in reached, connected=report['connected'],
                mail=report['mail_status'],
                answered=bool(rungs['rx']['decoded'] or rungs['tx']['acknowledged']),
                levels={direction: {f'{rung}_max': max(seen, default=None)
                                    for rung, seen in reached.items()}
                        for direction, reached in rungs.items()},
                signal_reports=reports,
                stops=[line.strip() for line in text.splitlines() if LAUNCHER_STOP.search(line)],
                failures=[failure['text'] for failure in report['failures'][:4]])


# DID WE KEY, AND DID ANYTHING COME BACK. Both out of the arm's own record, which
# `verdict` wrote at teardown.
#
# `originating:` is printed where the connect-request ladder starts, which is past
# the occupancy gate and one statement before the first request goes on the air.
# Every refusal -- the gate, the pounce budget, a deaf receiver, arguments the
# launcher would not read -- happens before it, so its absence is "never
# attempted" and nothing else has to be enumerated.
#
# A reply is the analyzer's own `connected`, or an over it decoded, or a
# below-offer signal report, which is the gateway saying it heard the request and
# could not read it. That last one is the answer a campaign would otherwise drop:
# the channel replied, and what it replied is a verdict on our transmitter.
def outcome(mode, record, code):
    """Which of the three this arm was."""
    read = record.get('verdict') or {}
    if read.get('stops') or not read.get('keyed'):
        return NOT_ATTEMPTED
    if read.get('connected') or read.get('answered') or read.get('signal_reports'):
        return ANSWERED
    return NO_REPLY


def disposition(code, record):
    """Whether the pass goes on, read off what the arm and the launcher recorded.

    Move on for a channel that answered nothing, refused us, or failed a protocol
    we captured. Stop for wrong tuning, a clipped emission, or a receiver that
    cannot carry the arm: another channel will not fix any of those.
    """
    if record is None:
        return 'stop', 'The arm wrote no record; the launcher never started.'
    stops = (record.get('verdict') or {}).get('stops') or []
    if stops:
        return 'stop', 'The launcher refused this channel: ' + stops[0]
    return 'continue', CONTINUE_REASON.get(code, f'launcher exit {code}')


# THE 82 ARMS THE SUPERVISOR FLEW, IN THE SPELLING THE LOGBOOK READS.
#
# They are flat JSON beside the console's own arm directories -- `label` where the
# core says `arm`, `gateway` where it says `call`, a compact stamp where it says
# an ISO instant, and no directory at all, because the transcript went to the
# session tree under `the optional historical on-air transcript folders`. This maps them; it does not rewrite them.
# `logbook.py` is a reader and there is no second write path.
#
# THE EXIT CODE CANNOT SAY WHETHER WE KEYED. Six of these arms exited 1 without
# ever reaching the connect-request ladder -- four because the launcher refused
# the channel outright and two before `originating:` was printed -- and an exit
# of 1 covers those and an ordinary NOT-connected alike. So the same evidence
# `outcome` reads for an arm of this console's own is read here: the arm's
# transcript, once, for the milestones it reached and the refusals it carries.
# Reading the code alone called those six "no reply", which in an elimination
# campaign is a station that keeps being called for an answer it was never asked
# for -- and called the six arms that CONNECTED that as well.
#
# It costs one pass over each transcript, 1.3 MB across all 82. A full reading of
# one of them is still `analyze.py` over the same file.
def band_metres(centre_hz):
    from .gateways import band_for
    return band_for(centre_hz) if centre_hz else None


def started_at(path, record):
    """The stamp the supervisor gave the arm, or the file's own time if it has none."""
    try:
        return datetime.strptime(record['utc'], '%Y%m%dT%H%M%S.%fZ').replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


def transcript(record):
    """The arm's own log, where the supervisor left it. Absent reads as no evidence."""
    path = Path(record['log_path']) if record.get('log_path') else None
    return ANSI.sub('', path.read_text(errors='replace')) if path and path.is_file() else ''


def from_supervisor(path, record):
    text = transcript(record)
    reached = [name for name, pattern in MILESTONES if pattern.search(text)]
    stops = [line.strip() for line in text.splitlines() if LAUNCHER_STOP.search(line)]
    reports = record.get('signal_reports') or []
    said = CONTINUE_REASON.get(record.get('returncode'),
                               f'launcher exit {record.get("returncode")}')
    if record.get('status') != 'complete':
        said = f'{said}; recorded status {record.get("status")}'
    if stops:
        said = 'the launcher refused this channel: ' + stops[0]
    return dict(record, arm=record.get('label') or path.stem,
                directory=str(Path(record['log_path']).parent) if record.get('log_path') else None,
                call=record.get('gateway'), band_m=band_metres(record.get('centre_hz')),
                dial_hz=record.get('usb_dial_hz'),
                mode=('force' if record.get('force_once') else
                      'pounce' if record.get('pounce_s') else 'sense'),
                level=dict(name=record.get('row')),
                started_utc=started_at(path, record).isoformat(), ended_utc=None,
                verdict=dict(reached=reached,
                             line=f'{said} ({len(reports)} below-offer signal report(s); '
                                  f'analyze.py reads the transcript for the rest)',
                             keyed='keyed' in reached, connected='connect' in reached,
                             answered='answer' in reached,
                             mail='complete' if 'mail' in reached else 'unknown',
                             signal_reports=reports, stops=stops))


def records(mode):
    """Every VARA arm: this console's, and the supervisor's, in one spelling."""
    read = [read_json(path) for path in mode.arms.glob('*/provenance.json')]
    for path in mode.arms.glob('*.json'):
        read.append(from_supervisor(path, read_json(path)))
    return read


# THE REPORT'S RUNGS, WHICH ARE NOT QUITE THE MILESTONES. `answer` matches every
# lattice frame, and the frames are two opposite facts wearing one pattern: frame
# 17 is the gateway offering a link, and 14, 15 and 16 are it saying it heard the
# request and could not read it. Bench-settled 09-09. A campaign line that said
# "answer" for both would be exactly as uninformative as the tally this report
# replaces, so a below-offer answer is its own rung between keying and an answer.
LADDER = ('cleared', 'keyed', 'below offer', 'answer', 'connect', 'greeting', 'mail')
FRAME = re.compile(r'lattice frame (\d+)')
OFFER = 17

# The exit code is not read here and must not be: the port established that one
# VARA code covers a launcher refusal, a silent call and a connected session
# alike, which is why `outcome` reads the transcript. A report that filed a
# refusal as silence would be wrong about where the fault was.
EXIT_CANNOT_SAY = ('one launcher exit code covers a refused channel, a call nothing '
                   'answered and a connected session alike; nothing in this report '
                   'reads it, and every line above comes off the transcript')


def readout(mode, record, text):
    """What one VARA arm did, out of the verdict `analyze.py` wrote at teardown.

    Nothing here re-reads the transcript for meaning. `verdict` already carries
    the connection, the mail status, the below-offer reports and the launcher's
    own refusals, written once where the evidence was fresh.
    """
    said = record.get('verdict') or {}
    reached, reports, stops = (said.get('reached') or [], said.get('signal_reports') or [],
                               said.get('stops') or [])
    # The gateway's own words, which this launcher prints on the same line the
    # PACTOR one does -- both drive the same B2F mail client. `signal_reports`
    # and `stops` are OUR readings of the far end and are not it.
    spoke = spoken(ANSI.sub('', text))
    frames = sorted({int(number) for report in reports for number in FRAME.findall(report)})
    # Every lattice frame it saw was a below-offer one, so `answer` overstates it.
    below = reports and not (said.get('connected') or said.get('answered'))
    rung = highest(mode, reached)
    if rung == 'answer' and below:
        rung = 'below offer'
    did, caveats = [], [('the launcher exit code', EXIT_CANNOT_SAY)]
    if said.get('line'):
        did.append(said['line'])
    if frames:
        did.append('below-offer lattice frame(s) ' + ', '.join(str(one) for one in frames)
                   + f' -- the gateway heard the request and could not read it; {OFFER} is '
                     'the only offer')
        caveats.append((f'lattice frame {frames[-1]}',
                        'it grades our request\'s legibility at the gateway, not the link; '
                        'the levers are the retry form and the drive'))
    return Readout(
        rung=rung, did=tuple(did), said=spoke,
        crossed='mail exchange complete' if said.get('mail') == 'complete' else '',
        anomalies=tuple([(1, f'the launcher refused this channel: {stop}') for stop in stops]
                        + [(1, failure) for failure in said.get('failures') or []]),
        caveats=tuple(caveats),
        suggests=('the two levers a below-offer report leaves are --cr-retry-form and '
                  '--amplitude.' if below else ''))


PROFILE = Mode(
    name='vara',
    launcher='vara-mail',
    # `onair.sh vara-mail` runs tools/kestrel_connect.py as a script, so the
    # residency probe reads it the same way -- its directory first on the path.
    entry=(str(ROOT / 'tools/kestrel_connect.py'),),
    roster=HERE / 'peers.json',
    arms=HERE / 'arms',
    lock=HERE / 'arm.lock',
    bands=(160, 80, 60, 40, 30, 20, 17, 15, 12, 10),
    levels=LEVELS,
    phases=(),
    drives=('amplitude',),
    authorised_watts=AUTHORISED_WATTS,
    flags=flags,
    arguments=arguments,
    campaign_arguments=campaign_arguments,
    milestones=MILESTONES,
    head=head,
    ready=ready,
    channel_rank=channel_rank,
    channel_arms=channel_arms,
    verdict=verdict,
    disposition=disposition,
    assess=assess,
    dwell_floor_s=WINDOW_S,
    outcome=outcome,
    records=records,
    readout=readout,
    ladder=LADDER,
)
