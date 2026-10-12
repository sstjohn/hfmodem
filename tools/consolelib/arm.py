"""CALL A HOST: build one arm, fly it, tail it, and write down what it did.

An arm is a plan before it is a flight. The plan is the whole launcher line, the
environment it runs in, the peer and the preset, the runtime digest and git HEAD;
a dry run prints it and opens nothing. A flight adds the lock, the rig readback,
the tap, the witness, the transcript and the verdict, and writes the lot to
`provenance.json` at the start and again at teardown. It also asks the arm's own
interpreter what it loaded, twice, because the digest above is a statement about
files and `residency` is the statement about imports.

Three things can stop a flight and none is a judgement about operating: the arm
lock, a dial that will not read, and `--qualify` against a snapshot this arm
would not have flown. Everything else is reported.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from . import operator, claim, logbook, mail, modes, residency, roster, station, view

PACKAGE = station.ROOT / 'packages/hfmodem'
STAGING = Path(os.environ['HFMODEM_SOURCE_BASELINE']) if os.environ.get('HFMODEM_SOURCE_BASELINE') else None

WITNESS_SECONDS = 720
# The old console stopped the tap the instant the arm exited and lost ~15 s of a
# peer the operator could still hear after teardown. The recorder closes its own
# wav, so the tail costs nothing but the wait.
TAP_TAIL_S = 30.0
# How long a hand-stopped launcher is given to finish unwinding. The mail
# transcript, stream.wav and every rx window are written on that path and nowhere
# else, so this is the difference between an interrupted arm and a lost one.
INTERRUPT_GRACE_S = 45.0
DOWN_TIMEOUT_S = 120.0


def source_evidence():
    """Hash shipped runtime files; an explicitly supplied baseline is optional."""
    files = [p for parent in ('hfmodem', 'hfhost', 'creance')
             for p in (station.ROOT / 'packages' / parent).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.c', '.h', '.npz', '.npy', '.i32', '.json', '.z')
             and '__pycache__' not in p.parts and 'tests' not in p.parts]
    files += [p for parent in ('consolelib', 'logbook_view')
              for p in (station.ROOT / 'tools' / parent).rglob('*')
              if p.is_file() and (p.suffix in ('.py', '.sh', '.mjs', '.js', '.json') or p.name == 'console')
              and '__pycache__' not in p.parts and not p.name.startswith('test_')]
    files += [station.ROOT / 'tools' / name for name in (
        'console', 'console.py', 'onair.sh', 'capture_identity.py', 'ardop_call.py',
        'kestrel_connect.py', 'kiwi_witness.py', 'txwitness.py', 'codec_gain.py',
        'lib/ports.sh', 'lib/rig.sh', 'lib/attempts.sh')
        if (station.ROOT / 'tools' / name).is_file()]
    actual = {p.relative_to(station.ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(files)}
    if os.environ.get('HFMODEM_OPERATOR_INSTALLED') == '1':
        import importlib
        for package in ('hfmodem', 'hfhost', 'creance'):
            location = Path(importlib.import_module(package).__file__).parent
            for path in location.rglob('*'):
                if (path.is_file() and path.suffix in ('.py', '.c', '.h', '.npz', '.z')
                        and 'tests' not in path.parts and '__pycache__' not in path.parts
                        and '_operator' not in path.parts):
                    actual[f'packages/{package}/{package}/' + path.relative_to(location).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    result = dict(working_tree_sha256=digest, runtime_files=len(actual), baseline_available=False,
                  v32_staging_sha256=None, is_v32_staging=False, missing=[], differs_from_v32_staging=[])
    if STAGING is not None:
        staged = roster.read_json(STAGING)
        baseline_files = staged['files']
        compared = {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
                    for name in baseline_files if (PACKAGE / name).is_file()}
        result.update(baseline_available=True, v32_staging_sha256=staged['source_set_sha256'],
                      is_v32_staging=compared == baseline_files,
                      missing=sorted(set(baseline_files) - set(compared)),
                      differs_from_v32_staging=sorted(name for name, sha in compared.items()
                                                    if sha != baseline_files[name]))
    return result


def git_head():
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=station.ROOT, timeout=10,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=station.ROOT, timeout=20,
                               capture_output=True, text=True, check=True).stdout
        return dict(head=head, dirty=bool(dirty.strip()))
    except (OSError, subprocess.SubprocessError):
        return dict(head='unknown', dirty=None)


def arm_name(mode, peer, level, fly_mode, now):
    stamp = now.strftime('%Y%m%dT%H%M%SZ')
    return f'{mode.name}-{level["name"]}-{peer["id"].lower()}-{fly_mode}-{stamp}'


def passthrough(args):
    """The operator's own words for the launcher, in the order they were named.

    Checked against nothing on purpose: the console cannot know which flag the
    modem grew this hour, and a whitelist here is what turns a flyable arm into a
    hand-pasted launcher line. A word the modem will not read is refused by the
    modem, in its own words, and `onair.sh` already tells that apart from a band
    result. Each occurrence is split as a shell would, so a flag and its value go
    in together.
    """
    return [word for said in args.modem_arg for word in shlex.split(said)]


def passthrough_note(plan):
    """What a non-standard arm says about itself, printed and written down alike."""
    said = plan['adjustments']['passthrough']
    return (f'  passthrough: {shlex.join(said)} -- the operator\'s own words, appended '
            'to the launcher line and checked by nothing here. This arm did not fly the '
            'line the console generates.') if said else ''


def plan(mode, args, now):
    """The whole arm, decided before anything opens. Nothing here touches hardware."""
    if args.pounce < 0 or (args.pounce == 0 and not args.sense and not args.force):
        raise ValueError('--pounce takes a positive number of seconds to wait.')
    if args.kiwi and not args.witness:
        raise ValueError('--kiwi names a receiver to place and --no-witness places none.')
    from hfmodem.core import config
    extra = passthrough(args)
    requested = None
    for i, word in enumerate(extra):
        if word in ('--rig', '--radio') and i + 1 < len(extra):
            requested = extra[i + 1]
        elif word.startswith(('--rig=', '--radio=')):
            requested = word.split('=', 1)[1]
    selection = station.resolve_selection(requested)
    station.warn_receive_only(selection)
    selected_name = selection.RIG_NAME
    selected_cfg = selection._STATION_CONFIG
    args.selected_rig = selected_name
    args.selected_audio = (selected_cfg.audio if selected_cfg and
                           selected_cfg.selected_rig == selected_name else None)
    if selected_cfg or selected_name != 'ft891':
        for field in ('tx_drive', 'p1_drive', 'amplitude'):
            flag = '--' + field.replace('_', '-')
            for i, word in enumerate(extra):
                if word == flag and i + 1 < len(extra):
                    setattr(args, field, float(extra[i + 1]))
                elif word.startswith(flag + '='):
                    setattr(args, field, float(word.split('=', 1)[1]))
    peer = roster.peer_for(mode, args.peer, args.band, args)
    level_name = args.level
    if mode.name == 'pactor' and getattr(args, 'p4_mail', False) and level_name == 'current':
        level_name = 'p4-bench'
    level = modes.level_for(mode, level_name)
    fly_mode = 'pounce' if args.pounce else 'force' if args.force else 'sense'
    name = arm_name(mode, peer, level, fly_mode, now)
    if mode.name == 'pactor' and getattr(args, 'p4_chirp_counter', False):
        name += '-p4-chirp-counter'
    elif mode.name == 'pactor' and getattr(args, 'p4_entry', False):
        name += '-p4-entry'
    if mode.name == 'pactor' and getattr(args, 'p4_mail', False):
        name += '-p4-mail'
    elif mode.name == 'pactor' and getattr(args, 'p4_link', False):
        name += '-p4-link'
    if mode.name == 'pactor' and getattr(args, 'robust', False):
        name += '-robust'
    if mode.name == 'pactor' and getattr(args, 'after_connect', 'arq') != 'arq':
        name += f'-{args.after_connect}'
    if mode.name == 'pactor' and getattr(args, 'after_robust', None) not in (None, 'listen'):
        name += f'-then-{args.after_robust}'
    if mode.name == 'pactor' and getattr(args, 'p4_probe_packet', 'chirp') != 'chirp':
        # `-robust` already names the connect kind, so the robust packet gets
        # its own: a clone of the tape, or our own datablock values.
        named = '-rv' if getattr(args, 'p4_probe_values', None) else '-rclone'
        name += {'robust': named}.get(args.p4_probe_packet,
                                      f'-{args.p4_probe_packet}')
    if mode.name == 'pactor' and getattr(args, 'p4_probe_cadence', None) not in (None, 'spaced'):
        name += f'-{args.p4_probe_cadence}'
    if mode.name == 'pactor' and getattr(args, 'p1_6a9', False):
        name += '-p1-6a9'
    if mode.name == 'pactor' and getattr(args, 'p1_level_digit', 1) != 1:
        name += f'-lvl{args.p1_level_digit}'
    if mode.name == 'pactor' and getattr(args, 'p1_supervisor', None):
        name += f'-sup{len(args.p1_supervisor)}'
    if mode.name == 'pactor' and getattr(args, 'p1_idle_id_every', 0):
        name += '-idid'
    if mode.name == 'pactor' and getattr(args, 'timing_trial', None):
        name += f'-timing-{args.timing_trial.lower()}'
    if mode.name == 'pactor' and getattr(args, 'entry_timing_trial', None):
        name += f'-entry-timing-{args.entry_timing_trial.lower()}'
    directory = mode.arms / name
    named_send = args.send
    if mode.name == 'vara' and args.direction == 'tx' and not named_send:
        named_send = mail.DELIVERY
    if mode.name == 'pactor' and getattr(args, 'p4_mail', False) and not named_send:
        named_send = mail.DELIVERY
    profile = dict(bw_hz=args.bw) if mode.name == 'vara' else None
    source, git = source_evidence(), git_head()
    metadata = dict(centre_hz=peer['centre_hz'], dial_hz=peer['dial_hz'],
                    console_argv=getattr(args, 'console_argv', None),
                    source=source, git=git)
    station_file = os.environ.get(config.STATION_ENV) or str(station.ROOT / 'station.toml')
    identity = None
    if mode.name in ('pactor', 'vara', 'ardop'):
        from hfmodem.core import identification
        identity_env = dict(os.environ)
        if Path(station_file).is_file():
            identity_env[config.STATION_ENV] = station_file
        effective = identification.options(
            extra, mycall=config.mycall(path=station_file if Path(station_file).is_file() else None),
            idcall=getattr(args, 'idcall', None))
        identity = identification.resolve(effective.mycall, effective.idcall, environ=identity_env)
    send, delivery = mail.outbound(mode.name, peer, named_send, directory, now, profile,
                                   metadata=metadata, station=station_file,
                                   sender=identity.digital_call if identity is not None else None)

    command = (mode.head(mode, peer, args, name) if mode.head else
               [str(station.ONAIR), mode.launcher, peer['call'], str(peer['centre_hz']),
                '--log', name, '--band', str(peer['band_m'])])
    words, adjustments = mode.flags(peer, level, args, directory, send)
    command += words
    if selected_cfg and mode.name == 'pactor':
        level = dict(level, tx_drive=args.tx_drive, p1_drive=args.p1_drive,
                     note='selected station chain; calibration remains advisory')
    if mode.name == 'pactor':
        command_path = ('p2-mail' if getattr(args, 'p2_mail', False) else
                        'p4-bench' if level_name == 'p4-bench' else 'flown')
        command += ['--tx-command-path', command_path]
    if args.witness and not args.kiwi:
        command += ['--witness']
    if args.force:
        command += ['--force']
    if getattr(args, 'cwid_optional', False):
        command += ['--cwid-optional']
    command += extra

    env = dict(CAT_PORT=selection.CAT_PORT, PTT_PORT=selection.PTT_PORT,
               POUNCE_WAIT=str(args.pounce), WITNESS_MAX_S=str(args.witness_seconds),
               HFMODEM_STATION=station_file)
    if identity is not None:
        env['MYCALL'] = identity.digital_call
        env['IDCALL'] = identity.cw_call
        if not os.environ.get(config.STATION_ENV) and not Path(station_file).is_file():
            del env[config.STATION_ENV]
    if selected_name != 'ft891' or requested:
        env['HFMODEM_RIG'] = selected_name
    timing = None
    if mode.name == 'pactor':
        from hfmodem.core.rigs import resolve_tx_timing

        override = None
        selected_rig = None
        for i, word in enumerate(extra):
            if word == '--tx-latency-ms' and i + 1 < len(extra):
                override = float(extra[i + 1])
            elif word.startswith('--tx-latency-ms='):
                override = float(word.split('=', 1)[1])
            elif word == '--rig' and i + 1 < len(extra):
                selected_rig = extra[i + 1]
            elif word.startswith('--rig='):
                selected_rig = word.split('=', 1)[1]
        timing_env = dict(os.environ)
        timing_path = env.get(config.STATION_ENV)
        if timing_path and Path(timing_path).is_file():
            timing_env[config.STATION_ENV] = timing_path
        timing = asdict(resolve_tx_timing(override, rig=selected_rig,
                                         command_path=command_path, environ=timing_env))
    # `mode` has meant the sense/pounce/force choice in provenance since the first
    # flown arm; the radio mode is the arm name's own prefix.
    return dict(arm=name, directory=str(directory), mode=fly_mode, pounce_s=args.pounce,
                call=peer['call'], band_m=peer['band_m'], centre_hz=peer['centre_hz'],
                dial_hz=peer['dial_hz'], peer=peer, level=level, delivery=delivery,
                adjustments=dict(adjustments,
                                 witness=args.witness, witness_seconds=args.witness_seconds,
                                 kiwi=args.kiwi, retune=args.retune,
                                 tap_tail_s=args.tap_tail, passthrough=extra,
                                 prepare_radio=getattr(args, 'prepare_radio', False),
                                 **({"drive_sources": args.drive_sources}
                                    if hasattr(args, "drive_sources") else {})),
                codec_gain_working_point=(roster.gain_hint(mode, peer['band_m'], now)
                                          if selected_name == 'ft891' else None),
                source=source, git=git, env=env, command=command, tx_timing_plan=timing,
                **({'identity': dict(asdict(identity), child_cw_source=(
                    '--idcall' if identification.options(extra, mycall='').idcall is not None
                    else 'IDCALL'))}
                   if identity is not None else {}))


def spelled(plan):
    return shlex.join([f'{k}={v}' for k, v in plan['env'].items()] + plan['command'])


def milestones(mode, text):
    return [name for name, pattern in mode.milestones if pattern.search(text)]


def verdict(mode, directory, code=0):
    if mode.verdict:
        return mode.verdict(mode, directory)
    log = Path(directory) / 'arm.log'
    if not log.is_file():
        return dict(reached=[], line='no transcript')
    text = log.read_text(errors='replace')
    reached = milestones(mode, text)
    said = [line.strip() for line in text.splitlines() if line.startswith('verdict:')]
    line = said[-1][len('verdict:'):].strip() if said else unsaid(reached)
    if code and not said and 'keyed' not in reached:
        line += f' ({refusal(text) or f"the modem exited {code}"})'
    return dict(reached=reached, line=line)


def refusal(text):
    """The last warning the modem printed after the argv it was launched with, which
    is where an argument it would not take is said."""
    _, _, after = text.rpartition('\n  argv: ')
    warned = [line[2:].strip() for line in after.splitlines()[1:] if line.startswith('!! ')]
    return warned[-1] if warned else None


def unsaid(reached):
    """The closing line for a transcript that printed no verdict of its own. Our own
    sense and keying are not an answer, and an arm that never keyed was not flown."""
    if set(reached) - {'cleared', 'keyed'}:
        return 'answered, no verdict line'
    if 'keyed' in reached:
        return 'nothing answered'
    return 'not keyed; nothing was transmitted'


def ready(mode, plan):
    """What a mode must have in place before its launcher starts, and only then.

    The credential is the whole of it today: a Winlink password belongs in the
    environment of one arm and never in a plan that is written to disk, so it is
    put there around the flight rather than carried in the record.
    """
    return mode.ready(mode, plan) if mode.ready else nullcontext()


def qualified(mode, plan, named):
    """The residency this arm flies, refused before the rig is claimed if it must be.

    A named snapshot is flown as well as checked: it goes on the arm's PYTHONPATH,
    which is where the editable install's appended finder loses to it, and the
    probe then reads back what that actually resolved. Naming one and importing
    another is the failure this exists to make impossible, so the console does not
    leave the two to be spelled separately.
    """
    if named:
        manifest = roster.read_json(Path(named))
        root = residency.snapshot_root(manifest, named)
        plan = dict(plan, qualify=str(named),
                    env=dict(plan['env'], PYTHONPATH=str(root)))
    found = residency.evidence(mode, plan, named)
    if named and found['refusal']:
        raise RuntimeError(f'--qualify named {Path(named).name} and this arm is not it: '
                           + found['refusal'] + '. Nothing was claimed and nothing keyed.')
    if named and plan.get('identity'):
        from hfmodem.core.identification import CallIdentity
        identity = plan['identity']
        separate = CallIdentity(identity['digital_call'], identity['cw_call'],
                                identity['source']).separate
        if separate and 'hfmodem.core.identification' not in (found['modules'] or {}):
            raise RuntimeError('--qualify snapshot lacks separate CW identity support: '
                               'its modem did not import hfmodem.core.identification. '
                               'Nothing was claimed and nothing keyed.')
    return plan, found


def pump(launcher, sink, screen):
    """Every line to the transcript, the contact to the screen. Returns the exit code."""
    for line in launcher.stdout:
        sink.write(line)
        for shown in screen.feed(line):
            sys.stdout.write(shown)
            sys.stdout.flush()
    return launcher.wait()


def settle(launcher, sink, screen):
    """Read the launcher's own teardown rather than exiting out from under it.

    A ^C at the terminal reaches the whole foreground group, so the modem is
    already unkeying itself; the console's part is not to exit first. When it
    did, the launcher took SIGPIPE at the next line it printed -- before the mail
    summary, before the recorders closed, before `prove_unkeyed` -- so a hand
    stop cost the evidence AND the keying verdict rather than only the arm.

    The clock is for a console interrupted on its own, whose launcher never saw
    anything; a second ^C ends the wait early. Either way the return is None when
    the launcher did not finish, which is what names the artifacts at risk.
    """
    def out_of_time(_signum, _frame):
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, out_of_time)
    signal.setitimer(signal.ITIMER_REAL, INTERRUPT_GRACE_S)
    try:
        return pump(launcher, sink, screen)
    except (KeyboardInterrupt, TimeoutError):
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def bring_down(directory, *, env=None):
    """`tools/onair.sh down`, whose `prove_down` is the only reading of the keying line.

    In a session of its own, and with ^C ignored for the seconds it takes: the
    operator is hammering the key by now, and the one reading that says whether
    the transmitter is still up is the one thing here that may not be skipped.
    Everything before this point honours the second interrupt; this does not, and
    `subprocess.run` would kill the check on its way out if it did.

    Its exit code is not read -- `down` comes back non-zero with nothing left to
    stop, and reading that as failure has already put two arms on the air at once.
    """
    deferred = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        done = subprocess.run([str(station.ONAIR), 'down'], cwd=station.ROOT, text=True,
                              env=None if env is None else dict(os.environ, **env),
                              capture_output=True, timeout=DOWN_TIMEOUT_S,
                              start_new_session=True)
        said = done.stdout + done.stderr
    except (OSError, subprocess.SubprocessError) as trouble:
        said = f'`onair.sh down` did not run: {type(trouble).__name__}: {trouble}'
    finally:
        signal.signal(signal.SIGINT, deferred)
    (directory / 'interrupt.log').write_text(said + '\n')
    return next((line.strip() for line in reversed(said.splitlines())
                 if 'PTT VERDICT' in line), None)


def hand_stop(plan, launcher, sink, screen):
    """A ^C mid-arm: let the launcher close its own artifacts, then prove the line down."""
    print('\n  stopping the arm. It closes its recordings as it goes; ^C again to '
          'stop it harder.', flush=True)
    code = settle(launcher, sink, screen)
    if code is None:
        print('  the launcher did not finish unwinding: the mail transcript, stream.wav '
              'and the rx windows are written at teardown and will be short or missing. '
              'Everything it printed up to here is in arm.log.', flush=True)
    keying = bring_down(Path(plan['directory']))
    print(f'  {keying}' if keying else
          '  NO PTT VERDICT -- `onair.sh down` never reached the keying line, so this '
          'console cannot say the transmitter is unkeyed. Check it by hand.', flush=True)
    return code


def capture_plan(mode, plan):
    """Allocate/rebind PACTOR evidence before any readiness or rig context."""
    if mode.name != 'pactor':
        return plan, None
    from hfmodem.core import capture_identity

    old = Path(plan['directory'])
    capture = capture_identity.allocate_arm(
        operator.data_root(), transcripts=operator.data_root() / 'working',
        log_label=plan['arm'], native_root=old,
        kind='launch-group' if plan['pounce_s'] else 'arm',
        arguments=plan['command'], source={'git': plan.get('git'), 'files': plan.get('source')})
    directory = capture.root('native')
    command = list(plan['command'])
    delivery = plan['delivery']
    if delivery and Path(delivery['path']).is_relative_to(old):
        replacement = str(directory / Path(delivery['path']).relative_to(old))
        command = [replacement if word == delivery['path'] else word for word in command]
        delivery = dict(delivery, path=replacement)
    command.extend(['--capture-id', capture.capture_id,
                    '--capture-root', str(capture.root('capture'))])
    return dict(plan, requested_arm=plan['arm'], arm=directory.name, directory=str(directory),
                capture_id=capture.capture_id, capture_root=str(capture.root('capture')),
                transcript=str(capture.root('transcript') / capture.record['transcript']['path']),
                command=command, delivery=delivery), capture


@contextmanager
def _preserve_primary_failure():
    original, trace = sys.exc_info()[1:]
    try:
        yield
    except BaseException:
        if original is not None:
            raise original.with_traceback(trace)
        raise


def fly(mode, plan, qualify=None):
    from .operator import station_file
    from hfmodem.core.config import load
    cfg = load(plan.get('env', {}).get('HFMODEM_STATION') or station_file())
    if not cfg.station.transmit:
        raise ValueError('Station transmit=false; no flight hardware was opened.')
    allocated, capture = capture_plan(mode, plan)
    # Callers retain this plan across normal return and KeyboardInterrupt.
    # Publish the allocated output identity before any arm work can interrupt.
    plan.update(allocated)
    state = {'writers': [], 'returned': False}
    try:
        result = _fly(mode, plan, qualify, capture=capture, capture_state=state)
        state['returned'] = True
        return result
    finally:
        if capture is not None:
            primary = sys.exc_info()[0] is not None
            trouble = None
            try:
                capture.end()
            except BaseException as error:
                trouble = error
            for writer in state['writers']:
                if not state['returned']:
                    writer._status['failed'] = True
                try:
                    writer.update_status(
                        state='finalized' if state['returned'] else 'partial',
                        written=1 if state['returned'] else 0,
                        partial=0 if state['returned'] else 1,
                        component='native-session')
                except BaseException as error:
                    writer._status['journal_error'] = type(error).__name__
                    if trouble is None:
                        trouble = error
            if trouble is not None and not primary:
                raise trouble


def _fly(mode, plan, qualify=None, *, capture=None, capture_state=None):
    directory = Path(plan['directory'])
    plan, resided = qualified(mode, plan, qualify)
    now = datetime.now(timezone.utc)
    writer = capture.claim('native-metadata') if capture else None
    if writer is not None:
        capture_state['writers'].append(writer)
        writer.update_status(state='ongoing', pending=1, component='native-session')
    provenance = writer.reserve('provenance.json').versions() if writer else None
    transcript_writer = capture.claim('native-transcript') if capture else None
    if transcript_writer is not None:
        capture_state['writers'].append(transcript_writer)
        transcript_writer.update_status(state='ongoing', pending=1, component='native-session')
    transcript = transcript_writer.reserve('arm.log') if transcript_writer else None
    with claim.lease(mode), ready(mode, plan):
        note = claim.spacing_note(mode, plan['call'], now)
        if note:
            print(note, flush=True)
        if plan['adjustments'].get('prepare_radio'):
            station.prepare_radio(plan)
        plan = dict(plan, started_utc=now.isoformat(), residency=resided,
                    readings=station.check_rig(plan))
        if capture is None:
            directory.mkdir(parents=True)
        if plan['delivery']:
            mail.write(plan['delivery'])
        if provenance is None:
            (directory / 'provenance.json').write_text(json.dumps(plan, indent=2) + '\n')
        else:
            provenance.update(plan | {'capture_state': 'ongoing'})
        for line in station.spell_environment(plan['readings'], plan['band_m'],
                                              plan['codec_gain_working_point'],
                                              selection=station.selection_for_plan(plan)):
            print(line, flush=True)
        print('  the launcher loads ' + residency.label(resided['modules'])
              + (f' -- {resided["refusal"]}' if resided['refusal'] else ''), flush=True)
        flown_with = passthrough_note(plan)
        if flown_with:
            print(flown_with, flush=True)
        tap = station.start_tap(plan, directory, writer=writer) if writer else station.start_tap(plan, directory)
        witness = ((station.place_kiwi(plan, directory, writer=writer) if writer
                    else station.place_kiwi(plan, directory)) if plan['adjustments']['kiwi'] else None)
        print(spelled(plan), flush=True)
        code = None
        # The sink takes the whole stream and the screen takes the contact: this
        # arm.log is what every sweep and tools/rehear read, so the filter is on
        # the terminal write and nowhere else. The screen's tally is printed in
        # the teardown, because an operator who interrupts an arm still wants it.
        screen = view.View()
        interrupted = False
        try:
            with (io.TextIOWrapper(transcript.open()) if transcript
                  else (directory / 'arm.log').open('w')) as sink:
                # Into the transcript as well as onto the screen: the terminal is
                # gone by the time anyone asks what a surprising arm actually keyed.
                if flown_with:
                    sink.write(flown_with + '\n')
                launcher = subprocess.Popen(
                    plan['command'], cwd=station.ROOT, env=dict(os.environ, **plan['env']),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                try:
                    code = pump(launcher, sink, screen)
                except KeyboardInterrupt:
                    interrupted = True
                    code = hand_stop(plan, launcher, sink, screen)
        finally:
            with _preserve_primary_failure():
                sys.stdout.writelines(screen.close())
                try:
                    if witness and witness.poll() is None:
                        witness.terminate()         # it closes its wav in a finally
                        witness.wait(timeout=60)
                    # No tail on a hand stop: the operator asked for it to be over.
                    tail = 0 if interrupted else plan['adjustments']['tap_tail_s']
                    if tap and tap.poll() is None and tail:
                        print(f'  the tap records {tail:g}s past teardown; a peer that is still '
                              'audible belongs in the recording. Interrupt to end it sooner.',
                              flush=True)
                        try:
                            time.sleep(tail)
                        except KeyboardInterrupt:   # the record is written below either way
                            pass
                    if tap and tap.poll() is None:
                        tap.terminate()             # likewise
                        tap.wait(timeout=60)
                    try:
                        plan['readings_at_end'] = station.read_environment(
                            selection=station.selection_for_plan(plan))
                    except Exception as trouble:
                        plan['readings_at_end'] = dict(
                            error=f'{type(trouble).__name__}: {trouble}')
                    # Read again, because the runtime this console flies has moved eight
                    # times inside one day. A pair that disagrees names an arm whose
                    # source set is not single-valued, whatever either reading says.
                    ended = residency.evidence(mode, plan, qualify)
                    plan['residency_at_end'] = dict(
                        ended, same_as_start=ended['modules'] == resided['modules'])
                except KeyboardInterrupt:
                    # Honoured, but not at the cost of the record: an operator who
                    # hammers ^C through the teardown still gets the arm written down.
                    interrupted = True
                    for unread in ('readings_at_end', 'residency_at_end'):
                        plan.setdefault(unread, dict(error='the teardown was interrupted'))
                plan.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                            returncode=code, verdict=verdict(mode, directory, code))
                if mode.name == 'pactor' and plan.get('adjustments', {}).get('connect') == 'robust':
                    from .pactor import robust_observations
                    plan['adjustments'].update(robust_observations(directory))
                if interrupted:
                    # Only on the arms it is true of: this record is what every sweep
                    # reads, and a verdict of "nothing answered" means one thing after
                    # a spent budget and another after a hand stop.
                    plan['hand_stopped'] = True
                if provenance is None:
                    (directory / 'provenance.json').write_text(json.dumps(plan, indent=2) + '\n')
                else:
                    provenance.update(plan | {'capture_state': (
                    'finalized' if code is not None and sys.exc_info()[0] is None
                    else 'partial')})
        print(f'  {plan["arm"]}: {", ".join(plan["verdict"]["reached"]) or "no milestone"} '
              f'-- {plan["verdict"]["line"]}', flush=True)
        if interrupted:
            raise KeyboardInterrupt
        return code


def status(mode, limit):
    for record in logbook.records(mode)[:limit]:
        reached = ', '.join(record.get('verdict', {}).get('reached', [])) or 'no milestone'
        print(f'{record["started_utc"][:19]}Z {record["call"]:8} {record["band_m"]:>3} m  '
              f'{record["level"]["name"]:12} {record["mode"]:6}  {reached}')
        print(f'{"":22} {record.get("verdict", {}).get("line", "running")}')
    return 0
