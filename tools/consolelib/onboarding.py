"""One guided path from a connected radio to a Winlink mail session."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from . import gateways, operator


def ask(label, default=None):
    if not sys.stdin.isatty():
        raise ValueError(f'{label} is needed; run interactively or supply the setup option.')
    return input(label + (f' [{default}]' if default is not None else '') + ': ').strip() or default or ''


def choose(label, choices, default=0):
    if not choices:
        raise ValueError(f'No {label.lower()} found. Check the USB cable and power, then try again.')
    if len(choices) == 1:
        print(f'{label}: {choices[0][0]}')
        return choices[0][1]
    print(label)
    for i, (text, _) in enumerate(choices, 1):
        print(f'  {i}. {text}')
    while True:
        value = ask('Number', str(default + 1))
        if value.isdigit() and 1 <= int(value) <= len(choices):
            return choices[int(value) - 1][1]
        print('Choose one of the numbers above.')


def dependencies(install=False):
    missing = [name for name in ('rigctl', 'rigctld', 'flac', 'nc') if not shutil.which(name)]
    if not missing:
        return
    command = (['brew', 'install', 'hamlib', 'flac'] + ([] if shutil.which('nc') else ['netcat']) if shutil.which('brew') else
               ['sudo', 'apt-get', 'install', '-y', 'libhamlib-utils', 'libportaudio2', 'flac', 'netcat-openbsd']
               if shutil.which('apt-get') else None)
    if command and not install and sys.stdin.isatty():
        print('Radio tools are needed: ' + ' '.join(command))
        install = ask('Install it now? (y/n)', 'y').lower() in ('y', 'yes')
    if command and install:
        print('Installing radio control and recording tools. This may take longer than a minute.')
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise ValueError('The system installer did not finish. Complete the command above, then run hfmodem winlink again.') from exc
        if all(shutil.which(name) for name in missing):
            return
    instruction = ' '.join(command) if command else 'Install Hamlib using your system package manager.'
    raise ValueError('Radio tools are missing (' + ', '.join(missing) + '). Run: ' + instruction +
                     '\nThen run hfmodem winlink again, or use hfmodem console setup --install-system-deps.')


def serial_choices():
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    ports.sort(key=lambda p: ('standard' in (getattr(p, 'interface', '') or '').lower(), p.device))
    return [(f'{p.description} ({p.device})', p.device) for p in ports]


def audio_choices(kind, devices=None):
    if devices is None:
        try:
            import sounddevice
            devices = sounddevice.query_devices()
        except (OSError, RuntimeError) as exc:
            raise ValueError('Cannot read audio devices. On Linux install PortAudio (for apt: sudo apt-get install libportaudio2), then reconnect the sound card.') from exc
    count = 'max_input_channels' if kind == 'input' else 'max_output_channels'
    usable = [d for d in devices if d[count] > 0]
    usb = [d for d in usable if any(word in d['name'].lower() for word in ('usb', 'signalink', 'digirig'))]
    # A built-in microphone/speaker is never an automatic radio choice.
    if not usb:
        raise ValueError('No USB radio sound card found. Check its cable and driver, then try again. '
                         'Use setup --audio-in and --audio-out for a differently named interface.')
    selected = usb
    names = list(dict.fromkeys(d['name'] for d in selected))
    return [(name, name) for name in names]


def write_station(path, *, rig, call, grid, cat, ptt, audio_in, audio_out, licence,
                  regulatory, because, drive, power, transmit):
    from hfmodem.core import config
    quote = json.dumps
    body = ('schema = 1\n[station]\n' + f'mycall = {quote(call)}\ngrid = {quote(grid)}\n'
            'control = "local"\n' + f'regulatory = {quote(regulatory)}\n'
            + (f'licence = {quote(licence)}\n' if regulatory == 'part97' else f'because = {quote(because)}\n')
            + f'transmit = {str(transmit).lower()}\n\n[rig]\nmodel = {quote(rig)}\n'
            'centre_hz = 28150000\nmode = "PKTUSB"\n' + f'power_w = {power:g}\n'
            '[rig.ptt]\nbackend = "rts"\n' + f'port = {quote(ptt)}\n'
            f'\n[rig.profiles.{rig}]\ncat_port = {quote(cat)}\nptt_port = {quote(ptt)}\n'
            'calibration_status = "PROVISIONAL"\n'
            f'\n[audio]\ninput = {quote(audio_in)}\noutput = {quote(audio_out)}\n'
            f'rate = 48000\ntx_drive = {drive:g}\n'
            + (f'\n[audio.chains.ftdx10_usb]\ninput = {quote(audio_in)}\noutput = {quote(audio_out)}\n'
               f'rate = 48000\ntx_drive = {drive:g}\ncalibration_status = "PROVISIONAL"\n' if rig == 'ftdx10' else ''))
    config.parse(__import__('tomllib').loads(body))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.station-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            out.write(body)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def setup(argv=None):
    parser = argparse.ArgumentParser(prog='hfmodem console setup', description='Set up a connected FTDX10 or FT-891. Nothing here transmits.')
    parser.add_argument('--rig', choices=('ftdx10', 'ft891'))
    for name in ('mycall', 'grid', 'cat-port', 'ptt-port', 'audio-in', 'audio-out'):
        parser.add_argument('--' + name)
    parser.add_argument('--licence', choices=('technician', 'general', 'advanced', 'extra'))
    parser.add_argument('--regulatory', choices=('part97', 'unregulated'))
    parser.add_argument('--because', help='required explanation when limits are applied outside hfmodem')
    parser.add_argument('--drive', type=float, default=0.25, help='initial audio drive, 0–1; verify ALC on your radio')
    parser.add_argument('--power', type=float, default=20, help='actual radio RF power in watts, for emission checks')
    parser.add_argument('--transmit', action='store_true', help='enable later explicit calls; setup itself never keys')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--install-system-deps', action='store_true', help='install missing Hamlib with the system package manager')
    args = parser.parse_args(argv)
    path = operator.station_file()
    if path.is_file() and not args.replace:
        print(f'Station already configured: {path}\nUse --replace to run setup again.')
        return 0
    if not 0 < args.drive <= 1 or not 0 < args.power <= 100:
        raise ValueError('Drive must be greater than zero and at most 1; radio power must be 1–100 W.')
    dependencies(args.install_system_deps)
    rig = args.rig or choose('Your radio', [('Yaesu FTDX10 (USB audio)', 'ftdx10'),
                                          ('Yaesu FT-891 (external sound card)', 'ft891')])
    call = (args.mycall or ask('Your callsign')).upper()
    grid = (args.grid if args.grid is not None else ask('Your grid square, if known (nearby gateways)', '')).upper()
    if not re.fullmatch(r'[A-Z0-9]+(?:/[A-Z0-9]+)?', call) or (grid and not re.fullmatch(r'[A-R]{2}\d{2}(?:[A-X]{2})?', grid)):
        raise ValueError('Enter your callsign; an optional grid square has four or six characters.')
    regulatory = args.regulatory or choose('Operating profile', [('United States amateur licence', 'part97'),
                                                                 ('Other country / limits applied separately', 'unregulated')])
    licence = args.licence or (choose('Your licence', [(name.title(), name) for name in
                             ('technician', 'general', 'advanced', 'extra')]) if regulatory == 'part97' else None)
    because = args.because or (ask('How are your operating limits applied?') if regulatory == 'unregulated' else '')
    cat = args.cat_port or choose('Radio USB control connection', serial_choices())
    ptt = args.ptt_port
    if not ptt:
        from hfmodem.core.ptt import derive_ptt_port
        try:
            candidate = derive_ptt_port(cat)
        except ValueError:
            candidate = None
        ptt = candidate if candidate and Path(candidate).exists() else choose('Radio PTT connection',
              [(label, port) for label, port in serial_choices() if port != cat])
    if ptt == cat:
        raise ValueError('Choose the separate PTT interface; the CAT connection cannot also be RTS PTT.')
    audio_in = args.audio_in or choose('Radio sound card input', audio_choices('input'))
    audio_out = args.audio_out or choose('Radio sound card output', audio_choices('output'))
    print(f'{rig.upper()} · {call} · {grid}\nControl: {cat} · PTT: {ptt}\nAudio: {audio_in} → {audio_out}\n'
          f'Later mail calls will prepare the radio for DATA USB at {args.power:g} W.\n'
          'Initial audio drive is provisional; keep ALC out of compression during the first call.')
    transmit = args.transmit
    if not transmit and sys.stdin.isatty():
        transmit = ask('Enable later mail calls from this station? (y/n)', 'y').lower() in ('y', 'yes')
    write_station(path, rig=rig, call=call, grid=grid, cat=cat, ptt=ptt, audio_in=audio_in,
                  audio_out=audio_out, licence=licence, regulatory=regulatory, because=because,
                  drive=args.drive, power=args.power, transmit=transmit)
    print(f'Saved {path}. Setup has not transmitted.')
    return 0


def eligible(peers, cfg, mode, band=None):
    from hfmodem.core.occupied import keyed_hz
    from hfmodem.core.regulatory import Emission, NotPermitted
    from . import roster
    result = []
    for peer in peers:
        if mode == 'pactor' and peer.get('advertised_mode') and not re.search(r'\b[34]\b', peer['advertised_mode']):
            continue
        if band and peer['band_m'] != band:
            continue
        if not roster.hours_open(peer.get('hours', ''), datetime.now(timezone.utc).hour)[0]:
            continue
        lo, hi = keyed_hz(mode, str(peer.get('bw_hz', '')))
        try:
            cfg.profile.check(Emission(peer['dial_hz'], lo, hi, power_w=cfg.rig.power_w), control=cfg.control)
        except NotPermitted:
            continue
        result.append(peer)
    if cfg.station.grid and result:
        from hfcapture import grid_to_latlon, distance_km
        home = grid_to_latlon(cfg.station.grid)
        def distance(peer):
            try:
                return distance_km(home, grid_to_latlon(peer['grid']))
            except (ValueError, KeyError, IndexError):
                return float('inf')
        result.sort(key=lambda peer: (distance(peer), peer['centre_hz']))
    return result


def compose_mail(cfg, recipient, subject, body):
    """Save a normal text message; no B2F knowledge is needed by the operator."""
    from hfmodem.winlink.message import compose
    if not recipient or any(c.isspace() for c in recipient) or not subject or any(c in subject for c in '\r\n'):
        raise ValueError('Enter one callsign or email address and a subject on one line.')
    message = compose(cfg.station.mycall, recipient, subject, body)
    directory = operator.data_root() / 'outbox'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (message.mid + '.b2f')
    with path.open('xb') as stream:
        stream.write(message.render())
    print(f'Message saved: {subject} → {recipient}')
    return path


def inbox():
    """Show received mail directly, without a logbook server or file format tools."""
    from hfmodem.winlink.message import Message, MessageError
    files = sorted((operator.data_root() / 'working').rglob('*.b2f'), key=lambda p: p.stat().st_mtime)
    messages = []
    for path in files:
        if 'mail' not in path.parts:
            continue
        try:
            messages.append((path, Message.parse(path.read_bytes())))
        except MessageError:
            continue
    if not messages:
        print('No received mail saved yet.')
        return 0
    for path, message in messages[-10:]:
        print(f'\nFrom: {message.sender}\nSubject: {message.subject}\nDate: {message.get("Date")}')
        print(message.body.decode('latin-1'))
        if message.attachments:
            print('Attachments: ' + ', '.join(a.name for a in message.attachments))
        print(f'Saved: {path}')
    return 0


def winlink(argv=None):
    from hfmodem.core import config
    parser = argparse.ArgumentParser(prog='hfmodem winlink', description='Set up, choose a gateway and exchange Winlink mail. Ctrl-C stops the session.')
    parser.add_argument('mode', nargs='?', choices=('pactor', 'vara'))
    parser.add_argument('peer', nargs='?', help='gateway callsign or full channel ID')
    parser.add_argument('--band', type=int)
    parser.add_argument('--refresh', action='store_true', help='reload current gateway channels')
    parser.add_argument('--csv', type=Path, help='load saved Winlink gateway CSV')
    parser.add_argument('--dry-run', action='store_true', help='preview the mail call without opening hardware')
    parser.add_argument('--execute', action='store_true', help='start the reviewed call without another prompt')
    parser.add_argument('--install-system-deps', action='store_true')
    parser.add_argument('--send', type=Path, help='advanced: send a saved B2F message and also fetch mail')
    parser.add_argument('--to', help='send a text message to this callsign or email address')
    parser.add_argument('--subject', help='subject for --to')
    parser.add_argument('--body', help='text for --to; otherwise ask interactively')
    parser.add_argument('--inbox', action='store_true', help='read saved received mail; no radio or network needed')
    args = parser.parse_args(argv)
    if args.inbox:
        return inbox()
    if args.send and args.to:
        raise ValueError('Choose a saved message or --to, not both.')
    if (args.subject or args.body) and not args.to:
        raise ValueError('--subject and --body need --to.')
    if args.dry_run and args.execute:
        raise ValueError('Choose --dry-run or --execute.')
    if not operator.station_file().is_file():
        setup(['--install-system-deps'] if args.install_system_deps else [])
    cfg = config.load(operator.station_file())
    mode = args.mode or choose('Mail modem', [('PACTOR-3', 'pactor'), ('VARA HF', 'vara')])
    from . import roster, modes
    profile = modes.resolve(mode)
    if args.refresh or args.csv or not roster.peers(profile):
        print('Loading public Winlink gateways…')
        gateways.refresh(mode, csv_path=args.csv)
    candidates = eligible(roster.peers(profile), cfg, mode, args.band)
    if not args.band and not args.peer and sys.stdin.isatty() and candidates:
        bands = sorted({p['band_m'] for p in candidates}, reverse=True)
        band = choose('Band your antenna is ready for', [(f'{b} metres', b) for b in bands], default=bands.index(20) if 20 in bands else 0)
        candidates = [p for p in candidates if p['band_m'] == band]
    if args.peer:
        wanted = args.peer.upper()
        candidates = [p for p in candidates if wanted in (p['id'].upper(), p['call'].upper())]
    if not candidates:
        raise ValueError('No listed gateways fit your selected band, licence and operating hours. Try another band or refresh. '
                         'The Technician profile offers HF data only on 10 m; propagation and gateway availability still matter.')
    peer = choose('Choose a gateway' + (' (nearest first)' if cfg.station.grid else ''), [(f'{p["call"]} · {p["band_m"]}m · '
                  f'{p["dial_hz"]/1000:.3f}kHz USB · {p["qth"]}', p) for p in candidates[:12]])
    send = args.send
    recipient = args.to
    if not send and not recipient and sys.stdin.isatty():
        action = choose('Mail task', [('Receive mail', 'receive'), ('Send a text message and receive mail', 'send')])
        if action == 'send':
            recipient = ask('To (callsign or email address)')
    if recipient:
        subject = args.subject or ask('Subject')
        body = args.body if args.body is not None else ask('Message text')
        send = compose_mail(cfg, recipient, subject, body)
    words = [mode, 'fly', peer['id'], '--sense', '--no-witness', '--prepare-radio']
    if mode == 'pactor':
        words += ['--p3-mail']
    else:
        words += ['--bw', str(peer['bw_hz']), '--direction', 'tx' if send else 'rx']
    if send:
        words += ['--send', str(send)]
    print(f'{mode.upper()} mail via {peer["call"]}, {peer["dial_hz"]/1000:.3f}kHz USB. '
          + ('Send and receive.' if send else 'Receive mail.'))
    if args.dry_run:
        from hfmodem.operator.bootstrap import console
        return console(words + ['--dry-run'])
    dependencies(args.install_system_deps)
    print(f'The call will set DATA USB, the radio data-audio route, RTS PTT and {cfg.rig.power_w:g} W RF power.')
    if not args.execute and ask('Start this mail call? (y/n)', 'y').lower() not in ('y', 'yes'):
        print('Cancelled; nothing transmitted.')
        return 0
    if not cfg.station.transmit:
        # Explicitly enabling the saved station is a separate operator choice.
        raise ValueError('Station is receive-only. Run hfmodem console setup --replace --transmit to enable explicit calls.')
    prompted = False
    if not os.environ.get(config.PASSWORD_ENV) and not os.environ.get('WINLINK_PASSWORD_FILE'):
        if not sys.stdin.isatty():
            raise ValueError('A Winlink password is needed. Run interactively for the hidden prompt, or supply WINLINK_PASSWORD_FILE. Nothing was opened.')
        password = getpass.getpass('Winlink password (hidden, not saved): ').strip()
        if not password:
            raise ValueError('No password entered; nothing was opened.')
        os.environ[config.PASSWORD_ENV] = password
        del password
        prompted = True
    try:
        from hfmodem.operator.bootstrap import console
        result = console(words + ['--execute'])
        print('Read saved mail with: hfmodem winlink --inbox')
        return result
    finally:
        if prompted:
            os.environ.pop(config.PASSWORD_ENV, None)
