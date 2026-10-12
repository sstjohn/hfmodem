"""Prepare, witness and verify one-way Sabir waveform proofs.

Preparation and verification are offline. Plans read the rig dial if no channel
is supplied. Only --execute starts a remote receiver or transmitter launcher.
"""
from __future__ import annotations

from . import operator

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import tomllib

from hfmodem.sabir.proof import PROOF_PROFILES

from . import arm, station


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_evidence():
    root = station.ROOT
    package = root / 'packages/hfmodem/hfmodem'
    paths = [*sorted((package / 'sabir').rglob('*.py')),
             *sorted((package / 'core').rglob('*.py')),
             root / 'tools/console.py', root / 'tools/consolelib/sabir.py',
             station.ONAIR, station.KIWI]
    files = {str(path.relative_to(root)): sha256(path) for path in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, files=files)


def read_rig_dial():
    """Query the existing daemon, or direct CAT if no daemon is listening.

    Send only a frequency query: no daemon startup, tuning or keying. Do not
    contend for serial CAT if a running daemon fails to answer correctly.
    """
    try:
        connection = socket.create_connection(('127.0.0.1', 4532), timeout=2)
    except ConnectionRefusedError:
        value = station.cat('f')
        source = 'serial CAT'
    except OSError as exc:
        raise RuntimeError(f'cannot read rig dial from rigctld: {exc}') from exc
    else:
        source = 'rigctld 127.0.0.1:4532'
        with connection:
            connection.sendall(b'+f\n')
            reply = b''
            deadline = time.monotonic() + 2
            while not re.search(rb'(?:^|\n)RPRT -?\d+\r?\n', reply):
                left = deadline - time.monotonic()
                if left <= 0 or len(reply) >= 4096:
                    raise RuntimeError('rigctld did not finish its frequency reply')
                connection.settimeout(left)
                chunk = connection.recv(4096 - len(reply))
                if not chunk:
                    raise RuntimeError('rigctld closed before reporting the dial')
                reply += chunk
        text = reply.decode('ascii', errors='replace')
        if not re.search(r'(?:^|\n)RPRT 0\r?\n', text):
            raise RuntimeError('rigctld refused the frequency query')
        field = re.search(r'(?:^|\n)Frequency: ([0-9]+(?:\.0+)?)\r?\n', text)
        value = field[1].split('.')[0] if field else None
    if value is None or not value.isdigit() or int(value) <= 0:
        raise RuntimeError('cannot read the tuned rig frequency; supply --channel or restore CAT')
    return dict(dial_hz=int(value), source=source, read_utc=utc())


def station_settings():
    """Read this test's transmit enable and drive without requiring an emission profile."""
    from hfmodem.core import config, levels
    explicit = os.environ.get(config.STATION_ENV)
    path = Path(explicit or station.ROOT / 'station.toml').expanduser().resolve()
    if not path.is_file() and not explicit:
        return dict(source=None, transmit=True, tx_drive=levels.TX_DRIVE)
    content = path.read_bytes()
    configured = tomllib.loads(content.decode('utf-8'))
    enabled = configured.get('station', {}).get('transmit', False)
    if not isinstance(enabled, bool):
        raise ValueError('[station] transmit must be true or false')
    return dict(source=str(path), sha256=hashlib.sha256(content).hexdigest(), transmit=enabled,
                tx_drive=configured.get('audio', {}).get('tx_drive', levels.TX_DRIVE))


def parser():
    result = argparse.ArgumentParser(prog='console sabir', description=__doc__)
    actions = result.add_subparsers(dest='action', required=True)
    prepare = actions.add_parser('prepare', help='render a proof and manifest offline')
    prepare.add_argument('directory', type=Path)
    prepare.add_argument('--callsign', required=True)
    prepare.add_argument('--gears', '--profiles', dest='profiles', nargs='+',
                         choices=PROOF_PROFILES, default=['workhorse'])
    prepare.add_argument('--bytes', type=int, default=256, dest='payload_bytes')
    prepare.add_argument('--seed', type=int, help='deterministic rehearsal; omit for fresh proof IDs')
    verify = actions.add_parser('verify', help='decode an existing remote recording offline')
    verify.add_argument('manifest', type=Path)
    verify.add_argument('capture', type=Path)
    verify.add_argument('--report', type=Path)
    proof = actions.add_parser('proof', help='record a remote receiver around one bounded transmission')
    proof.add_argument('manifest', type=Path)
    proof.add_argument('--gears', nargs='+', choices=PROOF_PROFILES,
                       help='prepare fresh packets in these gears, in order, using the manifest callsign')
    proof.add_argument('--repeat', type=int, default=1,
                       help='repeat the gear sequence this many times (with --gears; at most 8 packets)')
    proof.add_argument('--bytes', type=int, dest='payload_bytes',
                       help='payload bytes per fresh packet (with --gears; default: 256)')
    proof.add_argument('--channel', type=int, help='RF center in Hz; default: read rig dial and add 1500 Hz')
    proof.add_argument('--kiwi', help='require this receiver; default: automatically place a nearby available receiver')
    proof.add_argument('--passband', nargs=2, type=int, default=[0, 3000],
                       metavar=('LOW', 'HIGH'), help='requested remote USB passband, Hz')
    proof.add_argument('--gain', type=float, help='transmit audio drive override, 0 < gain <= 1; default: station TX drive')
    proof.add_argument('--power', type=float, help='optional declared PEP in watts; does not set rig power')
    proof.add_argument('--out', type=Path, help='new directory for this attempt and its evidence')
    proof.add_argument('--timeout', type=float, default=300, help='maximum launcher wall time, seconds')
    proof.add_argument('--ready-timeout', type=float, default=30, help='maximum remote-receiver startup, seconds')
    proof.add_argument('--tail', type=float, default=3, help='remote recording after the transmitter exits, seconds')
    occupancy = proof.add_mutually_exclusive_group(required=True)
    occupancy.add_argument('--sense', action='store_true')
    occupancy.add_argument('--force', action='store_true')
    execution = proof.add_mutually_exclusive_group(required=True)
    execution.add_argument('--dry-run', action='store_true')
    execution.add_argument('--execute', action='store_true')
    return result


def plan(args):
    """Resolve inputs and return commands; only an omitted channel queries CAT."""
    from hfmodem.sabir import proof
    settings = station_settings()
    gain = settings['tx_drive'] if args.gain is None else args.gain
    if type(gain) not in (int, float) or not 0 < gain <= 1:
        raise ValueError('--gain must be greater than zero and at most one')
    if args.channel is not None and args.channel <= 1500:
        raise ValueError('--channel must exceed the 1500 Hz audio offset')
    if (not all(math.isfinite(v) for v in (args.timeout, args.ready_timeout, args.tail))
            or args.timeout <= 0 or args.ready_timeout <= 0 or args.tail < 0):
        raise ValueError('timeouts must be positive and tail must be nonnegative')
    if not 0 <= args.passband[0] < args.passband[1] <= 6000:
        raise ValueError('--passband requires 0 <= LOW < HIGH <= 6000 Hz')
    if args.execute and settings['transmit'] is False:
        raise ValueError('transmit is disabled in the station configuration')
    if not args.gears and (args.repeat != 1 or args.payload_bytes is not None):
        raise ValueError('--repeat and --bytes require --gears')
    if args.gears and (args.repeat < 1 or len(args.gears) * args.repeat > 8):
        raise ValueError('choose 1..8 packets per burst with --gears and --repeat')
    source_manifest = args.manifest.resolve()
    contents = proof.validate_manifest(source_manifest, require_current_source=not bool(args.gears))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.out or source_manifest.parent / 'runs' / stamp).resolve()
    if output.exists():
        raise ValueError(f'proof output already exists: {output}; choose a new directory')
    manifest = source_manifest
    if args.gears:
        say('Preparing fresh packets: ' + ', '.join(args.gears * args.repeat) + '…')
        manifest = proof.prepare(source_manifest.parent / 'waveforms' / stamp,
                                 profiles=tuple(args.gears * args.repeat), callsign=contents['callsign'],
                                 payload_bytes=args.payload_bytes if args.payload_bytes is not None else 256)
        contents = proof.validate_manifest(manifest)
        say(f'Manifest: {manifest}')
    if args.timeout <= contents['transmit_seconds']:
        raise ValueError('--timeout must exceed the identified waveform duration')
    callsign = contents['callsign']
    if args.channel is None:
        say('Reading the rig’s current dial…')
    rig_readback = read_rig_dial() if args.channel is None else None
    channel = args.channel if args.channel is not None else rig_readback['dial_hz'] + 1500
    name = 'sabir-proof-' + stamp
    command = [str(station.ONAIR), 'sabir', str(channel), '--log', name,
               '--proof-manifest', str(manifest), '--mycall', callsign,
               '--gain', str(gain), '--regulatory', 'unregulated',
               '--because', 'operator-controlled Sabir RF test']
    if args.power is not None:
        command += ['--power', str(args.power)]
    if args.force:
        command.append('--force')
    capture = output / 'witness.wav'
    recorder = [str(station.VENV), str(station.KIWI),
                '--dial', f'{(channel - 1500) / 1000:.3f}',
                '--seconds', str(args.timeout + args.ready_timeout + args.tail + 30),
                '--out', str(capture), '--candidates',
                str(operator.witness_inventory()),
                '--passband', *map(str, args.passband), '--agc-gain', '50']
    if args.kiwi:
        recorder += ['--prefer', args.kiwi]
    return dict(kind='sabir-one-way-proof', created_utc=utc(), directory=str(output),
                manifest=str(manifest), manifest_sha256=sha256(manifest),
                source_manifest=str(source_manifest),
                gears=[trial['profile'] for trial in contents.get('trials', [])],
                waveform=str(manifest.parent / contents['waveform']),
                waveform_sha256=contents['waveform_sha256'],
                transmit_seconds=contents['transmit_seconds'],
                callsign=callsign, centre_hz=channel, dial_hz=channel - 1500,
                channel_source='rig' if rig_readback else 'explicit', rig_readback=rig_readback,
                station_settings=settings,
                regulatory_checks=False,
                gain=gain, gain_source='explicit' if args.gain is not None else 'station TX drive',
                receiver_selection='explicit' if args.kiwi else 'nearby available receiver with usable audio',
                receiver_requested=args.kiwi, capture=str(capture), command=command,
                receiver_settings=dict(passband=args.passband, agc_gain=50),
                recorder_command=recorder, source=source_evidence(),
                timeout_s=args.timeout, ready_timeout_s=args.ready_timeout, tail_s=args.tail,
                env=dict(CAT_PORT=station.CAT_PORT, PTT_PORT=station.PTT_PORT,
                         POUNCE_WAIT='0', PYTHONUNBUFFERED='1'),
                scope='one-way remote reception; does not establish an ARQ session')


def say(message):
    print(message, flush=True)


def show_plan(prepared, *, preview=False):
    say('Sabir test preview' if preview else 'Sabir on-air test')
    source = 'read from rig' if prepared['channel_source'] == 'rig' else 'specified'
    say(f"  Dial: {prepared['dial_hz'] / 1e6:.6f} MHz USB ({source})")
    say(f"  Audio drive: {prepared['gain']:.2f} peak; transmission: {prepared['transmit_seconds']:.1f} s")
    if prepared.get('gears'):
        say('  Gears: ' + ', '.join(prepared['gears']))
    say(f"  Receiver: {prepared['receiver_requested'] or 'automatic selection'}")
    say('  Channel check: operator override' if '--force' in prepared['command'] else
        '  Channel check: stop if occupied')
    say(f"  Evidence: {prepared['directory']}")
    if preview:
        say('Preview only. Use --execute in place of --dry-run to run this test.')


def show_decode(report):
    trials = report.get('trials', [])
    if trials:
        count = sum(bool(trial['ok']) for trial in trials)
        say(f'Decoded {count}/{len(trials)} packets with exact payloads.')
        gears = dict.fromkeys(trial['profile'] for trial in trials if 'profile' in trial)
        for gear in gears:
            selected = [trial for trial in trials if trial.get('profile') == gear]
            say(f"  {gear}: {sum(bool(trial['ok']) for trial in selected)}/{len(selected)} exact")
    else:
        say('Payload verification passed.' if report.get('ok') else 'Payload verification failed.')
    if report.get('capture_is_reference'):
        say('This is the local reference waveform; it does not prove radio reception.')


def show_result(evidence):
    if evidence.get('error'):
        error = evidence['error']
        if evidence.get('launcher_returncode') == 4:
            say('Not transmitted: channel sensing reported OCCUPIED.')
            say('Retry on a clear channel, or use --force to override the channel check.')
        else:
            say(f'Stopped: {error}')
    if evidence.get('verification'):
        show_decode(evidence['verification'])
    if evidence.get('radio_proof'):
        say('PASS: remote recording proves reception of every test packet.')
    elif not evidence.get('error'):
        say('Radio reception not proven.')
        if not evidence.get('receiver_provenance_valid'):
            say('Receiver evidence is incomplete or inconsistent; see provenance.json.')
        elif evidence.get('receiver_metadata', {}).get('complete') is not True:
            say('The remote recording did not finish cleanly.')
        if not evidence.get('transmitted'):
            say('Transmission completion was not confirmed in the launcher log.')
    if 'ptt_verdict' in evidence:
        say(evidence['ptt_verdict'] or 'PTT state could not be confirmed; check the rig.')
    for error in evidence.get('cleanup_errors', []):
        say(f'Cleanup failed: {error}')
    say(f"Evidence saved: {evidence['directory']}")


def show_launcher_progress(stream):
    """Forward operator milestones, leaving diagnostics in the full log."""
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line.endswith('\n'):
            stream.seek(position)
            return
        line = line.strip()
        if line.startswith(('channel sense:', 'TX power:', 'PTT ON', 'PTT OFF', '!!',
                            'NOT KEYING:', 'QSY FAILED:', 'nothing transmitted')):
            say(line)
        elif line.startswith('transmitted '):
            say(line.split(' Decode the far end', 1)[0])


def _receiver_matches(actual, requested):
    return bool(actual) and (requested is None or actual == requested
                            or (':' not in requested and actual.rsplit(':', 1)[0] == requested))


def wait_ready(process, log, requested, timeout):
    """Require live audio; an explicit receiver overrides automatic placement."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('remote recorder exited before ready; no transmission started')
        text = log.read_text(errors='replace')
        selected = re.search(r'recording [\d.]+s from (\S+) at dial', text)
        side = log.with_suffix('.json')
        if selected and 'WITNESS PLACED' in text and side.exists():
            try:
                metadata = json.loads(side.read_text())
            except ValueError:
                time.sleep(.2)
                continue
            if not _receiver_matches(selected[1], requested):
                raise RuntimeError(f'recorder selected {selected[1]}, not requested {requested}; no transmission started')
            if metadata.get('receiver') == selected[1] and metadata.get('seconds', 0) > 0:
                return dict(receiver=selected[1], ready_utc=utc(), metadata=metadata)
        time.sleep(.2)
    raise RuntimeError('remote recorder readiness timed out; no transmission started')


def _stop(process, *, group=False):
    if process is None or process.poll() is not None:
        return
    try:
        if group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if group:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def _artifacts(directory):
    result = {}
    for name in ('witness.wav', 'witness.json', 'witness.log', 'transmit.log'):
        path = directory / name
        if path.exists():
            result[name] = dict(sha256=sha256(path), bytes=path.stat().st_size)
    return result


def execute(prepared):
    """Record first, launch the existing guarded TX path, then decode the recording."""
    from hfmodem.sabir import proof
    directory = Path(prepared['directory'])
    directory.mkdir(parents=True, exist_ok=False)
    evidence = dict(prepared, started_utc=utc(), transmitted=False)
    record = directory / 'provenance.json'
    recorder = launcher = None
    aborted = False
    code = 2
    show_plan(prepared)
    try:
        proof.validate_manifest(prepared['manifest'])
        (directory / 'transmit-manifest.json').write_bytes(Path(prepared['manifest']).read_bytes())
        record.write_text(json.dumps(evidence, indent=2) + '\n')
        with (directory / 'witness.log').open('w') as remote_log:
            say('Finding a remote receiver and waiting for live audio…')
            recorder = subprocess.Popen(prepared['recorder_command'], cwd=station.ROOT,
                                        stdout=remote_log, stderr=subprocess.STDOUT)
            evidence['receiver_ready'] = wait_ready(recorder, directory / 'witness.log',
                                                    prepared['receiver_requested'], prepared['ready_timeout_s'])
            say(f"Recording from {evidence['receiver_ready']['receiver']}.")
            if evidence['receiver_ready']['metadata'].get('requested_settings') != prepared['receiver_settings']:
                raise RuntimeError('remote recorder settings differ from the prepared plan')
            if sha256(prepared['manifest']) != prepared['manifest_sha256']:
                raise RuntimeError('proof manifest changed after the plan was prepared')
            if source_evidence() != prepared['source']:
                raise RuntimeError('proof sources changed after the plan was prepared')
            settings = prepared['station_settings']
            if 'sha256' in settings and sha256(settings['source']) != settings['sha256']:
                raise RuntimeError('station settings changed after the plan was prepared')
            if prepared['channel_source'] == 'rig':
                evidence['rig_before_launch'] = read_rig_dial()
                if abs(evidence['rig_before_launch']['dial_hz'] - prepared['dial_hz']) > station.DIAL_TOLERANCE_HZ:
                    raise RuntimeError('rig dial changed during receiver placement; retry on the current channel')
            evidence['launcher_started_utc'] = utc()
            say('Checking the rig and channel before transmission…')
            with (directory / 'transmit.log').open('w') as tx_log, \
                    (directory / 'transmit.log').open() as progress:
                launcher = subprocess.Popen(prepared['command'], cwd=station.ROOT,
                                            env=dict(os.environ, **prepared['env']),
                                            stdout=tx_log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                deadline = time.monotonic() + prepared['timeout_s']
                next_update = time.monotonic() + 15
                while launcher.poll() is None:
                    show_launcher_progress(progress)
                    if recorder.poll() is not None:
                        raise RuntimeError('remote recorder ended while launcher was active')
                    if time.monotonic() >= deadline:
                        raise RuntimeError('transmit launcher exceeded its wall-time limit')
                    if time.monotonic() >= next_update:
                        say('Test in progress; remote recording continues…')
                        next_update = time.monotonic() + 15
                    time.sleep(.2)
                evidence['launcher_returncode'] = launcher.wait()
                evidence['launcher_ended_utc'] = utc()
                tx_log.flush()
                show_launcher_progress(progress)
            if evidence['launcher_returncode']:
                lines = (directory / 'transmit.log').read_text(errors='replace').splitlines()
                refusal = next((line.strip().removeprefix('!!').strip()
                                for line in reversed(lines) if line.strip().startswith('!!')), None)
                reason = f': {refusal}' if refusal else '; see transmit.log'
                raise RuntimeError(f'transmitter launcher returned {evidence["launcher_returncode"]}{reason}')
            if prepared['tail_s']:
                say(f"Finishing remote recording ({prepared['tail_s']:g} s tail)…")
                time.sleep(prepared['tail_s'])
            _stop(recorder)
        capture = Path(prepared['capture'])
        if not capture.is_file():
            raise RuntimeError('remote recorder produced no WAV')
        say('Decoding the remote recording and checking exact payloads…')
        report = proof.verify(prepared['manifest'], capture)
        (directory / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
        evidence['verification'] = report
        code = 0 if report['ok'] else 1
    except KeyboardInterrupt:
        aborted = True
        evidence['error'] = 'interrupted'
        code = 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        aborted = True
        evidence['error'] = str(exc)
    finally:
        def cleanup(process, *, group=False):
            try:
                _stop(process, group=group)
            except (OSError, subprocess.SubprocessError) as exc:
                evidence.setdefault('cleanup_errors', []).append(str(exc))

        cleanup(launcher, group=True)
        if (aborted or evidence.get('cleanup_errors')) and launcher is not None:
            say('Stopping the test and checking that PTT is down…')
            try:
                evidence['ptt_verdict'] = arm.bring_down(directory, env=prepared['env'])
            except (OSError, subprocess.SubprocessError) as exc:
                evidence.setdefault('cleanup_errors', []).append(str(exc))
        cleanup(recorder)
        tx_path = directory / 'transmit.log'
        evidence['transmitted'] = tx_path.exists() and any(
            line.startswith('transmitted ') for line in tx_path.read_text(errors='replace').splitlines())
        evidence['ended_utc'] = utc()
        evidence['artifacts'] = _artifacts(directory)
        sidecar = directory / 'witness.json'
        if sidecar.exists():
            try:
                evidence['receiver_metadata'] = json.loads(sidecar.read_text())
                if not isinstance(evidence['receiver_metadata'], dict):
                    evidence['receiver_metadata'] = {}
                    raise ValueError('receiver metadata is not an object')
            except (ValueError, OSError) as exc:
                evidence['receiver_metadata_error'] = str(exc)
        metadata = evidence.get('receiver_metadata', {})
        ready_receiver = evidence.get('receiver_ready', {}).get('receiver')
        valid_receiver = bool(
            not evidence.get('receiver_metadata_error')
            and metadata.get('receiver') == ready_receiver
            and ready_receiver is not None
            and _receiver_matches(ready_receiver, prepared['receiver_requested'])
            and metadata.get('requested_settings') == prepared['receiver_settings']
            and isinstance(metadata.get('seconds'), (int, float))
            and math.isfinite(metadata['seconds']) and metadata['seconds'] > 0
            and evidence['artifacts'].get('witness.wav', {}).get('bytes', 0) > 44)
        evidence['receiver_provenance_valid'] = valid_receiver
        evidence['radio_proof'] = bool(
            code == 0 and evidence['transmitted'] and not evidence.get('cleanup_errors')
            and valid_receiver and metadata.get('complete') is True
            and evidence.get('verification', {}).get('byte_exact')
            and not evidence.get('verification', {}).get('capture_is_reference', False))
        if code == 0 and not evidence['radio_proof']:
            code = 1
        record.write_text(json.dumps(evidence, indent=2) + '\n')
    show_result(evidence)
    return code


def run(argv):
    args = parser().parse_args(argv)
    from hfmodem.sabir import proof
    if args.action == 'prepare':
        say('Preparing the test waveform…')
        path = proof.prepare(args.directory, profiles=tuple(args.profiles),
                             callsign=args.callsign, seed=args.seed,
                             payload_bytes=args.payload_bytes)
        manifest = json.loads(Path(path).read_text())
        say(f"Prepared {len(manifest['trials'])} packets; {manifest['transmit_seconds']:.1f} s transmission.")
        say(f'Manifest: {path}')
        return 0
    if args.action == 'verify':
        say('Decoding the recording and checking exact payloads…')
        result = proof.verify(args.manifest, args.capture)
        text = json.dumps(result, indent=2) + '\n'
        if args.report:
            args.report.write_text(text)
        show_decode(result)
        if args.report:
            say(f'Report saved: {args.report}')
        return 0 if result['ok'] else 1
    prepared = plan(args)
    if args.dry_run:
        show_plan(prepared, preview=True)
        return 0
    return execute(prepared)
