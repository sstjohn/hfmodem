"""The station console: one tool for every mode this station flies.

    console <mode> <verb> [...]        pactor, vara and ardop

    console pactor peers               the channels this station has worked
    console pactor levels              the per-phase power presets
    console pactor status              recent arms and their verdicts
    console pactor collect --preview   rank receive-only corpus channels
    console pactor collect --execute   hop idle channels and retain PACTOR exchanges
    console pactor rig                 the same readback an arm takes, off the air
    console pactor gain                what the codec gain should be for the band
    console pactor fly KB5LZK --band 40 --level p3-100 --sense --dry-run
    console pactor campaign --band 80 --cycles 3 --sense --dry-run

    console vara fly K0SI-40m-7104000-2300 --bw 2300 --sense --dry-run
    console vara campaign --band 40 --bw 2750 2300 --cycles 3 --sense --dry-run

    console ardop peers [--refresh]
    console ardop fly W6IDS-40m-7061500-500 --sense --dry-run
    console ardop campaign --band 80 40 --bw 500 --cycles 2 --sense --dry-run

    console campaign --band 80 --cycles 3 --sense --dry-run \
        --with pactor --level p3-100 \
        --with vara --bw 2300

    console log --station KB5LZK --band 40     every arm, every mode, one view
    console fingerprints                       what the transcripts measure of each peer
    console sabir prepare OUT --callsign CALL   render a one-way proof offline
    console sabir proof MANIFEST [...]          witness a bounded transmission
    console sabir verify MANIFEST CAPTURE       decode a recording offline
    console power status|on|off|cycle          mains to the rig, on the station's smartplug
    console hflink station --help             automatic ALE, calls and native text
    console selcall station --help            held-channel calls and text pages

    console monitor                            listen on the station's receive device
    console monitor --freq 7104                after setting the rig's dial, once
    console monitor --kiwi HOST --freq 7104 --span 12000   a remote receiver, any width
    console monitor --minutes 30               and stop
    console monitor --wav FILE                 replay a recording, as fast as it decodes
    console monitor --arm NAME                 rehear one of our arms, our keyings labelled
    console monitor --dial 7104000             label a run whose CAT is down
    console monitor --no-record                events only, no audio to disk
    console monitor --no-rtty-usos             keep RTTY in figures across spaces
    console monitor --wav FILE --whole-band    only the decoders unshifted over the whole band
    console monitor --wav FILE --per-track     only each track's own decoders

    console pactor monitor                    PACTOR only, on the current dial
    console rtty monitor --freq 14080          tune once (kHz), then listen
    console rtty qso --freq 14080              the RTTY terminal; you key every over
    console vara monitor NS0A --band 40        listen on a saved peer channel
    console stanag4285 monitor --mode 1200L    choose the receive rate/interleaver
    console monitor --list-modes              every single-mode monitor and alias

`monitor` runs the whole-band and the per-track decoders together by default.
`--per-track` alone is a comparison mode: on this station's own recordings it
misses frames the whole-band decoders find, our keyings under the mute and the
answers at the unmute. Under `--arm` a PACTOR-1 connect naming the station the
arm called is ours, which would mislabel a third station calling that gateway.

One campaign over a band calls every station in the mode that is right for it.
The words before the first `--with` are the ones every mode reads; the words
after each `--with` are that mode's own, and are exactly what `console <mode>
campaign` would take.

A flight requires `--dry-run` or `--execute`. Sabir's dry run reads CAT when
the channel is omitted; it never tunes, records remotely or transmits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys

from consolelib import (arm, campaign, fingerprints, logbook, modes, multimode,
                        power, roster, station)

INTERRUPTED = 130


def fly_arguments(parser, mode, own, flight):
    """Everything that shapes one arm: the substrate's flags, then the mode's own."""
    flight.add_argument('--level', default='current', choices=tuple(mode.levels))
    choice = flight.add_mutually_exclusive_group(required=True)
    choice.add_argument('--sense', action='store_true',
                        help='check occupancy once; refuse if busy')
    choice.add_argument('--pounce', type=int, metavar='SECONDS', default=0,
                        help='wait up to this long for three clear windows')
    choice.add_argument('--force', action='store_true',
                        help='bypass the measured busy status; every other guard stands')
    flight.add_argument('--execute', action='store_true', help='key the rig')
    flight.add_argument('--dry-run', action='store_true', help='print the command and stop')
    flight.add_argument('--prepare-radio', action='store_true',
                        help='apply and verify supported Yaesu data-audio settings and configured RF power before calling')
    flight.add_argument('--cwid-optional', action='store_true',
                        help='skip the closing CW ID only when digital and CW callsigns are equal')
    if mode.name in ('pactor', 'vara', 'ardop'):
        flight.add_argument('--idcall', help='CW station callsign, independent of the digital caller; '
                            'default: IDCALL, [station] idcall, then digital mycall')
    flight.add_argument('--send', metavar='FILE|delivery',
                        help='a message to send; `delivery` sends a test message addressed out of Winlink, '
                             'its subject naming this gateway and mode. Default is fetch-only')

    recording = parser.add_argument_group('witness and recording')
    recording.add_argument('--no-witness', dest='witness', action='store_false')
    recording.add_argument('--witness-seconds', type=int, default=arm.WITNESS_SECONDS)
    recording.add_argument('--kiwi', metavar='HOST',
                           help='place this receiver rather than the nearest free one')
    recording.add_argument('--no-retune', dest='retune', action='store_false',
                           help='leave the dial alone; set it yourself')
    recording.add_argument('--tap-tail', type=float, default=arm.TAP_TAIL_S, metavar='S',
                           help='keep the tap recording this long after teardown (default %(default)s)')

    own(parser)

    launcher = parser.add_argument_group('launcher')
    launcher.add_argument('--qualify', metavar='MANIFEST',
                          help='fly the snapshot this frozen manifest describes; refuse the arm, before the rig '
                               'is claimed, if the interpreter resolves anything else')
    launcher.add_argument('--modem-arg', action='append', default=[], metavar='WORDS',
                          help='hand this to the launcher unchanged, after every generated word, so a later word '
                               'overrides (repeatable, unchecked). A flag and its value are one word '
                               "(--modem-arg '--over-continue long'); a lone flag needs the equals form "
                               '(--modem-arg=--no-mail-fetch)')


def sections_help(prog):
    return argparse.HelpFormatter(prog, max_help_position=34, width=100)


def parser_for(mode):
    parser = argparse.ArgumentParser(prog=f'console {mode.name}', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('monitor', parents=[monitor_parser(mode.name)], add_help=False,
                        help='receive only this protocol, on the current dial or a named channel')
    if mode.name == 'pactor':
        import corpus_watch
        commands.add_parser('collect', parents=[corpus_watch.parser()], add_help=False,
                            help='receive-only corpus watch ranked by Winlink sessions')
    listed = commands.add_parser('peers', help=f'the {mode.name} channels this station has worked')
    if mode.refresh:
        listed.add_argument('--refresh', action='store_true',
                            help='rebuild the roster from the live gateway listing and say '
                                 'what was added, dropped or changed')
    commands.add_parser('levels', help='the per-phase power presets')
    level = commands.add_parser('gain', help='what the codec gain should be for what is on the band')
    level.add_argument('--seconds', type=float, default=20.0, help='how long to listen')
    level.add_argument('--device', help=f'default {station.AUDIO_DEVICE}; name the tap to do the tap')
    level.add_argument('--set', dest='apply', action='store_true',
                       help='write the scalar it recommends; without this it only says')
    dump = commands.add_parser('rig', help='the same readback an arm takes, off the air')
    dump.add_argument('--json', dest='as_json', action='store_true', help='the raw record')
    dump.add_argument('--against', metavar='ARM',
                      help='an arm name under arms/, or a path to a provenance.json')
    dump.add_argument('--watch', type=float, default=0, metavar='S',
                      help='re-read every S seconds and print only what moved')
    dump.add_argument('--band', type=int, metavar='M',
                      help="name a band and the codec line reads exactly as an arm's does")
    recent = commands.add_parser('status', help='recent arms and their verdicts')
    recent.add_argument('--limit', type=int, default=12)

    air = commands.add_parser('fly', help='one arm; --dry-run prints it and opens nothing',
                              usage=f'console {mode.name} fly PEER (--sense | --pounce SECONDS | --force) '
                                    '[--dry-run | --execute] [options]',
                              formatter_class=sections_help)
    air.add_argument('peer', help=f'a call or a full ID from `console {mode.name} peers`')
    flight = air.add_argument_group('flight')
    flight.add_argument('--band', type=int, choices=tuple(mode.bands))
    fly_arguments(air, mode, mode.arguments, flight)

    many = commands.add_parser(
        'campaign', help='one arm at each open channel on a band, again until it answers',
        usage=f'console {mode.name} campaign --band BAND [BAND ...] (--sense | --pounce SECONDS | --force) '
              '[--dry-run | --execute] [options]',
        formatter_class=sections_help)
    pick = many.add_argument_group('campaign selection')
    pick.add_argument('--band', type=int, nargs='+', choices=tuple(mode.bands), default=[],
                      help='the band(s) this campaign may call; never called by omission')
    pick.add_argument('--skip', nargs='*', default=[], metavar='CALL',
                      help='calls or IDs this campaign leaves alone')
    pick.add_argument('--only', nargs='+', default=[], metavar='CALL',
                      help='restrict this campaign to these calls or full peer IDs')
    pick.add_argument('--limit', type=int, default=0, help='call at most this many channels')
    pick.add_argument('--order', choices=tuple(roster.ORDERS), default='roster')
    pick.add_argument('--prefer', choices=('near', 'far', 'dx'),
                      help='try this distance group first: near <1000 km, far 1000–4000 km, '
                           'dx >=4000 km; other stations remain eligible')
    pick.add_argument('--from-grid', metavar='GRID',
                      help='origin for distances; defaults to the configured station grid')
    pick.add_argument('--min-distance-km', type=float, metavar='KM',
                      help='exclude nearer stations and stations with unknown distance')
    pick.add_argument('--max-distance-km', type=float, metavar='KM',
                      help='exclude farther stations and stations with unknown distance')
    pick.add_argument('--cycles', type=int, default=1, metavar='N',
                      help='go through the list N times; a station that answers is dropped and the '
                           'remaining passes go where the answer is still unknown')
    pick.add_argument('--dwell', type=float, default=campaign.DWELL_S, metavar='S',
                      help='how long the scan asks the assessor to spend at each stop')
    pick.add_argument('--no-assess', dest='assess', action='store_false',
                      help='do not ask the mode what is on a channel before keying at it')
    flight = many.add_argument_group('flight')
    fly_arguments(many, mode, mode.campaign_arguments or mode.arguments, flight)
    return parser


def sections(words):
    """A multimode campaign's words: the shared ones, then each `--with MODE`'s own.

    There is no parser of its own here on purpose. The shared words are handed to
    every mode's own campaign parser, so a multimode pass reads exactly the flags
    the per-mode campaigns read, refuses what they refuse, and defaults to what
    they default to -- and a mode's required flags are named by that mode.
    """
    shared, named, own = [], [], None
    words = iter(words)
    for word in words:
        if word != '--with':
            (shared if own is None else own).append(word)
            continue
        name = next(words, None)
        if name is None:
            raise ValueError('--with wants a mode name after it, and there is none. '
                             f'This console flies {", ".join(modes.KNOWN)}.')
        if any(name == taken for taken, _ in named):
            raise ValueError(f'--with {name} twice: give that mode all its words once.')
        own = []
        named.append((name, own))
    return shared, named or [(name, []) for name in modes.KNOWN]


def multimode_modes(words):
    """Each named mode, and the campaign arguments it reads out of this one line."""
    shared, named = sections(words)
    chosen = []
    for name, own in named:
        mode = modes.resolve(name)
        chosen.append((mode, parser_for(mode).parse_args(['campaign'] + shared + own)))
    return chosen


def power_parser():
    parser = argparse.ArgumentParser(
        prog='console power',
        description="Mains to the transceiver. `cycle` is what brings the rig's CAT "
                    'interface back after the ATU has hunted a match and taken it down '
                    'with its RFI; the tuner keeps its stored matches across the cut. It '
                    'refuses to switch a rig that is drawing more than idle receive.')
    parser.add_argument('action', choices=('status', 'on', 'off', 'cycle'))
    return parser


def monitor_parser(mode=None):
    parser = argparse.ArgumentParser(
        prog=f'console {mode} monitor' if mode else 'console monitor',
        description='What the receiver is hearing, one line a detection, and every record '
                    'in events.jsonl beside provenance.json. Receive-only: the rig is '
                    'tuned once when --freq asks, and never keyed.')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--kiwi', metavar='HOST[:PORT]', help='a remote KiwiSDR, in IQ')
    source.add_argument('--wav', metavar='FILE', help='replay a recording, as fast as it decodes')
    source.add_argument('--arm', metavar='NAME', help='rehear one of our arms off its tap')
    parser.add_argument('--freq', type=float, metavar='KHZ',
                        help="the rig's dial, set once before listening; a Kiwi's dial")
    parser.add_argument('--span', type=int, nargs='+', metavar='HZ',
                        help="a Kiwi's width, 12000 by default; or LO HI, the band of a "
                             "recording to hear in place of the station's filter")
    parser.add_argument('--dial', type=int, metavar='HZ', help='label a run whose dial is unknown')
    parser.add_argument('--minutes', type=float, metavar='N', help='stop after N minutes of audio')
    parser.add_argument('--no-record', action='store_true', help='events only, no audio to disk')
    parser.add_argument('--no-rtty-usos', action='store_true',
                        help='keep RTTY in figures across spaces instead of returning to letters')
    parser.add_argument('--device', help=f'the receive input; default [audio] input, '
                                          f'else {station.AUDIO_DEVICE}')
    parser.add_argument('--cap', type=int, default=12, metavar='N',
                        help='decoders running at once across all tracks, loudest '
                             'tracks first (default 12)')
    alone = parser.add_mutually_exclusive_group()
    alone.add_argument('--whole-band', dest='dispatch', action='store_const',
                       const='whole-band', help='only one of each decoder over the whole '
                                                'band, unshifted, as `creance monitor` runs them')
    alone.add_argument('--per-track', dest='dispatch', action='store_const',
                       const='per-track', help="only each track's own decoders")
    parser.set_defaults(dispatch='both')
    parser.add_argument('--list-modes', action='store_true', help='list passive monitor commands')
    if mode:
        parser.add_argument('peer', nargs='?', help='saved callsign or full peer channel ID')
        parser.add_argument('--peer', dest='peer_option', help='same as the optional peer argument')
        parser.add_argument('--band', type=int, metavar='M', help='select a peer channel on this band')
        parser.add_argument('--mode', metavar='NAME', help='receive waveform setting; see --list-modes')
        parser.add_argument('--framing', default='SYNC', metavar='SYNC|8N1|7E1',
                            help='Bateleur character framing (default: raw synchronous bytes)')
        parser.add_argument('--bit-order', choices=('lsb', 'msb'), default='lsb')
        parser.add_argument('--audio-centre', type=float, default=700, metavar='HZ',
                            help='CW audio tone (default: 700 Hz)')
        parser.set_defaults(monitor_mode=mode, dispatch='whole-band')
    return parser


def log_parser():
    parser = argparse.ArgumentParser(
        prog='console log',
        description='Every arm this station has flown, across modes, read out of the '
                    'provenance each one wrote. Nothing here writes anything.')
    parser.add_argument('--mode', nargs='*', choices=modes.KNOWN, metavar='NAME')
    parser.add_argument('--station', metavar='CALL')
    parser.add_argument('--band', type=int, metavar='M')
    parser.add_argument('--hours', metavar='HH-HH',
                        help='UTC hours, inclusive, and they may wrap midnight')
    parser.add_argument('--since', metavar='WHEN', help='a date, 2026-09-16, or a timestamp')
    parser.add_argument('--until', metavar='WHEN')
    parser.add_argument('--reached', action='store_true',
                        help='only the arms that reached a milestone')
    parser.add_argument('--limit', type=int, default=0)
    return parser


def log(args):
    rows, waiting = logbook.entries(
        names=args.mode or None, station=args.station, band=args.band, hours=args.hours,
        since=args.since, until=args.until, reached=args.reached, limit=args.limit)
    for line in logbook.spell(rows, waiting):
        print(line)
    return 0


def fingerprint_parser():
    parser = argparse.ArgumentParser(
        prog='console fingerprints',
        description='Where each gateway answers on the raster, how its cycle runs against '
                    'our own stream clock, and which codeword arrives beside which -- read '
                    'off the transcripts the arms already wrote. Derived, and regenerable.')
    parser.add_argument('--write', nargs='?', const=str(fingerprints.FOUND), metavar='FILE',
                        help='also save the whole census as JSON')
    return parser


def fingerprint(args):
    found = (fingerprints.write(args.write) if args.write else fingerprints.fingerprints())
    for line in fingerprints.spell(found):
        print(line)
    if args.write:
        print(f'\n{args.write}')
    return 0


def dispatch(mode, args):
    if args.action == 'peers':
        for line in (mode.refresh(mode) if getattr(args, 'refresh', False)
                     else roster.spell_peers(mode)):
            print(line)
        return 0
    if args.action == 'levels':
        for line in modes.spell_levels(mode):
            print(line)
        return 0
    if args.action == 'status':
        return arm.status(mode, args.limit)
    if args.action == 'gain':
        return station.gain_report(args.seconds, args.apply, args.device)
    if args.action == 'rig':
        return station.rig_report(mode, args.as_json, args.against, args.watch, args.band)
    if args.execute == args.dry_run:
        raise ValueError('Choose --dry-run or --execute; an arm keys the rig and a dry run '
                         'opens nothing.')
    now = datetime.now(timezone.utc)
    if args.action == 'campaign':
        return campaign.run(mode, args, now)
    plan = arm.plan(mode, args, now)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print(arm.spelled(plan))
        if args.qualify:
            print(f'A flight would put the snapshot {args.qualify} names on this arm\'s '
                  'PYTHONPATH and refuse, before the lock, anything the interpreter '
                  'resolved elsewhere. Nothing is imported in a dry run.')
        return 0
    return arm.fly(mode, plan, args.qualify)


def run(argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__.strip())
        return 0 if argv else 2
    try:
        if argv[0] == 'setup':
            from consolelib import onboarding
            return onboarding.setup(argv[1:])
        if argv[0] == 'winlink':
            from consolelib import onboarding
            return onboarding.winlink(argv[1:])
        if len(argv) >= 2 and argv[1] == 'monitor':
            from consolelib import monitor, monitor_modes
            mode = monitor_modes.canonical(argv[0])
            args = monitor_parser(mode).parse_args(argv[2:])
            if args.peer and args.peer_option:
                raise ValueError('Give a peer once, as an argument or --peer.')
            args.peer = args.peer or args.peer_option
            args.console_argv = argv
            return monitor.report(args)
        if argv[:2] == ['rtty', 'qso']:
            from consolelib import rtty
            return rtty.run(argv[2:])
        if argv[:2] == ['pactor', 'collect']:
            import corpus_watch
            return corpus_watch.main(argv[2:])
        if argv[0] == 'data':
            from consolelib import operator
            return operator.run(argv[1:])
        if argv[0] == 'log':
            return log(log_parser().parse_args(argv[1:]))
        if argv[0] == 'monitor':
            args = monitor_parser().parse_args(argv[1:])
            args.console_argv = argv
            from consolelib import monitor
            return monitor.report(args)
        if argv[0] == 'power':
            return power.report(power_parser().parse_args(argv[1:]).action)
        if argv[0] == 'fingerprints':
            return fingerprint(fingerprint_parser().parse_args(argv[1:]))
        if argv[0] == 'sabir':
            from consolelib import sabir
            return sabir.run(argv[1:])
        if argv[0] in ('hflink', 'ale', 'selcall'):
            from consolelib import hflink
            return hflink.run(argv[1:], family=argv[0])
        if argv[0] == 'campaign':
            chosen = multimode_modes(argv[1:])
            for _, args in chosen:
                args.console_argv = argv
            first = chosen[0][1]
            if first.execute == first.dry_run:
                raise ValueError('Choose --dry-run or --execute; an arm keys the rig and a '
                                 'dry run opens nothing.')
            return multimode.run(chosen, datetime.now(timezone.utc))
        mode = modes.resolve(argv[0])
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f'REFUSED: {error}', file=sys.stderr)
        return 2
    args = parser_for(mode).parse_args(argv[1:])
    args.console_argv = argv
    try:
        return dispatch(mode, args)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f'REFUSED: {error}', file=sys.stderr)
        return 2


def main(argv=None):
    """A ^C is the operator saying stop now, and it is answered, never traced back.

    What brought the arm down and whether the keying line was proved low is said
    by `arm.fly` on its way out; this is the one line that covers every other
    place an interrupt can land, and the status a shell reads as interrupted.
    """
    try:
        return run(argv)
    except KeyboardInterrupt:
        print('\ninterrupted.', file=sys.stderr)
        return INTERRUPTED


if __name__ == '__main__':
    raise SystemExit(main())
