# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Supervised voice or CW contacts, with a computer-only automatic test loop."""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from . import operator, monitor, claim

HOLDS = SimpleNamespace(name='voice', lock=operator.mode_root('voice') / 'arm.lock',
                        arms=operator.mode_root('voice') / 'sessions')


def parser():
    p = argparse.ArgumentParser(prog='console voice', description=__doc__)
    verbs = p.add_subparsers(dest='action', required=True)
    for name in ('fly', 'test', 'monitor', 'levels', 'channel'):
        sub = verbs.add_parser(name, help={'fly': 'supervised RTS-keyed voice or CW',
            'test': 'speakers and microphone, no CAT/PTT', 'monitor': 'receive only',
            'levels': 'render CQ through the TX filter to a WAV',
            'channel': 'offline loaded-word packet experiment, no devices or rig'}[name])
        sub.add_argument('--station', help='station TOML (default HFMODEM_STATION or station.toml)')
        sub.add_argument('--rig', choices=('ft891', 'ftdx10', 'g90', 'x6100'))
        sub.add_argument('--dial', type=float, metavar='MHZ', help='radio dial in MHz, e.g. 14.285')
        sub.add_argument('--input', help='microphone in test mode; rig receive device otherwise')
        sub.add_argument('--output', help='speakers in test mode; rig codec otherwise')
        sub.add_argument('--preview-output', help='local monitor speakers (never the rig output by default)')
        sub.add_argument('--models', metavar='MANIFEST', help='local manifest written by voice_setup.py')
        sub.add_argument('--worker-python', help='isolated speech runtime Python')
        sub.add_argument('--drive', type=float, help='TX peak drive override, 0..1')
        sub.add_argument('--compression', choices=('off', 'soft', 'medium', 'strong'),
                         help='speech compression; soft preserves the existing default')
        sub.add_argument('--phonetics', choices=('nato', 'alternate', 'varied'),
                         help='callsign pronunciation: NATO, niner/sugar, or a random mix')
        sub.add_argument('--voice', help='stock Kokoro voice name (default am_michael)')
        sub.add_argument('--waveform', choices=('voice', 'cw'), default='voice',
                         help='speech or keyed CW audio using the same contact loop')
        sub.add_argument('--cw-wpm', type=float, default=20.)
        sub.add_argument('--cw-tone', type=float, default=700., metavar='HZ')
        sub.add_argument('--log', help='session directory (contains WAVs, scripts and provenance)')
        if name in ('test', 'fly'):
            sub.add_argument('--cq', action='store_true', help='start with an operator-approved CQ draft')
        if name in ('test', 'levels'):
            marking = sub.add_mutually_exclusive_group()
            marking.add_argument('--mark-hfmodem', action='store_true', help='implicit hfmodem mark in every over, voice or CW')
            marking.add_argument('--mark-code', metavar='NNNN', help='implicit four-digit code in every over, preserving leading zeros')
            sub.add_argument('--mark-key-file', help='optional shared 32-byte selection key for the implicit mark')
        if name == 'test':
            sub.add_argument('--watermark', metavar='CODEBOOK', help='experimental hfmodem signature, e.g. hfmodem-v1')
            sub.add_argument('--watermark-key-file', help='optional shared 32-byte raw key; requires --watermark')
            sub.add_argument('--auto', action='store_true', help='automatic simulated QSO through computer speakers only')
            sub.add_argument('--partner', help='preconfirmed simulated partner callsign; otherwise decode it from audio')
            sub.add_argument('--report', default='', help='explicit simulated report to send; omitted by default')
            sub.add_argument('--max-turns', type=int, default=8, help='finish after this many replies (default 8)')
            sub.add_argument('--reply-timeout', type=float, default=30., metavar='S')
            sub.add_argument('--cq-interval', type=float, default=30., metavar='S')
        if name == 'fly':
            choice = sub.add_mutually_exclusive_group(required=True)
            choice.add_argument('--execute', action='store_true', help='enable Enter to key each approved over')
            choice.add_argument('--dry-run', action='store_true', help='print the plan; no models/audio/CAT/PTT')
        if name == 'monitor':
            sub.add_argument('--wav', help='decode a recording; never open the rig')
        if name == 'levels':
            sub.add_argument('--wav-out', required=True, help='conditioned 48 kHz CQ WAV destination')
        if name == 'channel':
            data = sub.add_mutually_exclusive_group(required=True)
            data.add_argument('--payload', help='UTF-8 packet payload')
            data.add_argument('--payload-file', help='binary packet payload, at most 255 bytes')
            data.add_argument('--decode-run', help='independently decode recognized speech from a saved offline experiment')
            sub.add_argument('--codebook', default='hfmodem-v1')
            sub.add_argument('--symbol-bits', type=int, choices=(1, 2, 4), default=4)
            sub.add_argument('--method', choices=('sentence', 'logit'), default='sentence')
            sub.add_argument('--cover', choices=('qso', 'free'), default='qso')
            sub.add_argument('--topic', choices=('antenna', 'amplifier', 'home', 'weather', 'appointments', 'solar', 'customs'))
            key = sub.add_mutually_exclusive_group()
            key.add_argument('--key-file', help='shared 32-byte raw selection key; never logged')
            key.add_argument('--new-key-file', help='create a new shared selection key with private file permissions')
            sub.add_argument('--bias', type=float, default=3., help='token probability bias for the logit method')
            sub.add_argument('--temperature', type=float, default=.7)
            sub.add_argument('--min-words', type=int, default=6)
            sub.add_argument('--min-fraction', type=float, default=.65)
            sub.add_argument('--min-margin', type=int, default=3)
            sub.add_argument('--max-words', type=int, default=24)
            sub.add_argument('--attempts', type=int, default=12)
            sub.add_argument('--repetitions', type=int, default=1)
            sub.add_argument('--rounds', type=int, default=2)
            sub.add_argument('--controls', type=int, default=4)
            sub.add_argument('--noise', type=float, default=0., help='seeded added Gaussian noise RMS')
            sub.add_argument('--seed', type=int, default=1)
            sub.add_argument('--text-only', action='store_true', help='model/text experiment; skips TTS and ASR')
    return p


def settings(args):
    from hfmodem.core.config import load
    cfg = load(args.station or operator.station_file(), rig_name=args.rig)
    overrides = {}
    if args.compression is not None:
        if args.waveform != 'voice':
            raise ValueError('--compression applies to synthesized voice only')
        overrides['compression'] = args.compression
    if args.phonetics is not None:
        if args.waveform != 'voice':
            raise ValueError('--phonetics applies to synthesized voice only')
        overrides['phonetics'] = args.phonetics
    if args.models:
        data = json.loads(Path(args.models).read_text())
        overrides['language_model'] = data['language']['path']
        if args.waveform == 'voice':
            overrides.update(synthesis_model=data['synthesis']['path'],
                             recognition_model=data['recognition']['path'])
            synth = data['synthesis']
            overrides.update(synthesis_backend=synth.get('backend', 'kokoro'),
                             synthesis_voice=synth.get('voice', cfg.voice.synthesis_voice),
                             synthesis_speaker=synth.get('speaker', 5))
    if args.worker_python:
        overrides['worker_python'] = args.worker_python
    if args.voice:
        if not args.voice.isascii() or not args.voice.replace('_', '').isalnum():
            raise ValueError('--voice is a stock Kokoro voice name, e.g. am_michael')
        overrides['synthesis_voice'] = args.voice
    if args.drive is not None:
        if not math.isfinite(args.drive) or not 0 <= args.drive <= 1:
            raise ValueError('--drive must be finite and between 0 and 1')
        overrides['drive'] = args.drive
    return replace(cfg, voice=replace(cfg.voice, **overrides))


def read_key(path):
    data = Path(path).expanduser().read_bytes()
    if len(data) != 32:
        raise ValueError('shared selection key must be exactly 32 raw bytes')
    return data


@contextmanager
def collector_claim(cfg):
    claim.hardware.inherited(9)
    yield


def radio(session, dial):
    from hfmodem.core.rig import Cat, Rig
    from hfmodem.core.ptt import RtsPtt
    from hfmodem.core.ftdx10 import filter_commands
    from hfmodem.core.regulatory import Control
    from hfmodem.contact.radio import FTDX10_AGC, FTDX10_ROUTE, verify, write_raw
    cfg = session.cfg
    if not cfg.rig.ptt.port:
        raise ValueError('[rig.ptt] port is unset; select the configured rig or set its RTS port.')
    if cfg.selected_rig == 'ftdx10' and cfg.voice.voice_agc not in FTDX10_AGC:
        raise ValueError('FTDX10 voice_agc supports Hamlib 2 (FAST), 3 (SLOW), or 5 (MID).')
    cat = Cat(cfg.rig.host, cfg.rig.port)
    line = RtsPtt(cfg.rig.ptt.port)
    session.rig = Rig(model=cfg.selected_rig, cat=cat, ptt=line, profile=cfg.profile,
        control=Control.LOCAL, mycall=cfg.station.mycall, transmit=cfg.station.transmit,
        max_key_s=cfg.rig.max_key_s)
    # No unapproved proof keying: identify/arm without asserting the line.
    cat.open()
    report = session.rig.arm(prove_ptt=False)
    notes = []
    if not session.rig._check_ptt_owner(notes):
        raise ValueError('Cannot verify that rigctld leaves RTS keying to this session: ' + '; '.join(notes))
    cat.command('\\set_vfo VFOA')
    # Tune before setting mode/filter/routing: crossing a band can recall the
    # radio's saved settings. Rig.tune takes a centre; this CLI takes the dial.
    session.rig.tune(dial + 1500)
    if cfg.selected_rig == 'ftdx10':
        write_raw(cat, 'MD08;' if dial < 10_000_000 else 'MD0C;')
        width, roof = filter_commands(dict(rx_width_hz=3000, roofing_hz=3000))
        write_raw(cat, width)
        write_raw(cat, roof)
        write_raw(cat, 'NA00;')
        for item, value in FTDX10_ROUTE.items():
            write_raw(cat, f'EX{item}{value};')
        write_raw(cat, 'RT0;')
        write_raw(cat, 'XT0;')
    else:
        cat.set_mode('PKTLSB' if dial < 10_000_000 else 'PKTUSB', 3000)
    cat.command(f'L AGC {cfg.voice.voice_agc}')
    if cfg.rig.power_w is not None:
        cat.command(f'L RFPOWER {cfg.rig.power_w / 100:g}')
    session.radio_check = lambda: verify(session)
    readback = session.radio_check()
    line.open()
    session.event('rig_ready', report=str(report), notes=notes, readback=readback)


def run(argv):
    from hfmodem.contact.runtime import Session, model_hashes, write_wav
    from hfmodem.contact import terminal
    from hfmodem.contact.waveforms import select_waveform
    from hfmodem.contact.facts import facts_from
    from hfmodem.core.band import AMATEUR_HF_BANDS
    from hfmodem.core.rig import unkey_on_signal
    args = parser().parse_args(argv)
    cfg = settings(args)
    if args.action == 'channel':
        if args.waveform != 'voice' or args.dial is not None or args.input or args.output or args.preview_output:
            raise ValueError('channel is an offline voice experiment; device, dial and CW options are unavailable')
        from hfmodem.contact.channel import simulate, replay
        from hfmodem.contact.watermark import Options, pack
        if args.decode_run:
            if args.new_key_file:
                raise ValueError('--decode-run accepts an existing key, not --new-key-file')
            result = replay(args.decode_run, read_key(args.key_file) if args.key_file else None)
            print(json.dumps(result, indent=2))
            return 0 if result['status'] == 'crc_pass' else 1
        payload = args.payload.encode('utf-8') if args.payload is not None else Path(args.payload_file).read_bytes()
        pack(payload)  # Validate before opening a model or creating a session.
        options = Options(codebook=args.codebook, symbol_bits=args.symbol_bits, bias=args.bias,
            temperature=args.temperature, min_words=args.min_words, min_fraction=args.min_fraction,
            min_margin=args.min_margin, max_words=args.max_words, attempts=args.attempts,
            method=args.method, cover_mode=args.cover)
        key = read_key(args.key_file) if args.key_file else None
        if args.new_key_file:
            import secrets
            key = secrets.token_bytes(32)
            descriptor = os.open(Path(args.new_key_file).expanduser(), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(key)
        directory = Path(args.log).expanduser().resolve() if args.log else HOLDS.arms / (
            'channel-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        directory.mkdir(parents=True, exist_ok=False)
        result = simulate(cfg, directory, payload, options, repetitions=args.repetitions, rounds=args.rounds,
            controls=args.controls, noise=args.noise, seed=args.seed, text_only=args.text_only,
            key=key, topic=args.topic)
        print(json.dumps({k: v for k, v in result.items() if k != 'decisions'}, indent=2))
        print(f'Channel experiment: {directory}', flush=True)
        return 0 if result['payload_match'] else 1
    waveform = select_waveform(args.waveform, args.cw_wpm, args.cw_tone)
    script = waveform.codec
    own_call = cfg.station.mycall.upper()
    script.phonetic(own_call)  # Validate before a device or model opens.
    if args.dial is not None and (not math.isfinite(args.dial) or not 1 <= args.dial <= 60):
        raise ValueError('--dial is in MHz, e.g. 14.285')
    dial = round(args.dial * 1e6) if args.dial is not None else None
    test = args.action in ('test', 'levels')
    from hfmodem.contact.txmark import Tag
    tx_mark = Tag('hfmodem') if getattr(args, 'mark_hfmodem', False) else (
        Tag('code', args.mark_code) if getattr(args, 'mark_code', None) is not None else Tag())
    watermark = getattr(args, 'watermark', None)
    watermark_key_path = getattr(args, 'watermark_key_file', None)
    mark_key_path = getattr(args, 'mark_key_file', None)
    if mark_key_path and tx_mark.mode == 'off':
        raise ValueError('--mark-key-file requires --mark-hfmodem or --mark-code')
    if watermark_key_path and mark_key_path:
        raise ValueError('Choose the legacy watermark or integrated mark key')
    if watermark and tx_mark.mode != 'off':
        raise ValueError('Choose legacy --watermark or --mark-hfmodem/--mark-code')
    if watermark_key_path and not watermark:
        raise ValueError('--watermark-key-file requires --watermark')
    watermark_key_path = watermark_key_path or mark_key_path
    watermark_key = read_key(watermark_key_path) if watermark_key_path else None
    if watermark and args.waveform != 'voice':
        raise ValueError('--watermark currently supports voice computer tests only')
    simulation = None
    if args.action == 'test' and args.auto:
        from hfmodem.contact.calls import checked
        from hfmodem.contact.simulation import TestOptions
        simulation = TestOptions(report=args.report, partner=checked(args.partner) if args.partner else '',
            max_turns=args.max_turns, reply_timeout_s=args.reply_timeout, cq_interval_s=args.cq_interval)
        if args.waveform == 'voice' and len(args.report) == 3:
            raise ValueError('Voice uses a two-digit report; three-digit RST reports are for --waveform cw.')
    live = args.action == 'fly'
    replay = getattr(args, 'wav', None)
    if live and dial is None:
        raise ValueError('voice fly requires --dial in MHz')
    if live and not cfg.station.transmit:
        raise ValueError('[station] transmit = false; voice fly cannot key')
    if live and not args.dry_run and not sys.stdin.isatty():
        raise ValueError('voice fly needs an interactive terminal; Enter approves each over')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = Path(args.log).expanduser().resolve() if args.log else HOLDS.arms / stamp
    plan = dict(arm=f'{args.waveform}-{stamp}', directory=str(directory), call=own_call,
        band_m=next((m for lo, hi, m in AMATEUR_HF_BANDS
                     if dial and lo * 1e6 <= dial <= hi * 1e6), None),
        centre_hz=(dial + (1500 if dial >= 10_000_000 else -1500)) if dial else None,
        dial_hz=dial, mode=args.waveform, waveform=args.waveform,
        cw_wpm=args.cw_wpm if args.waveform == 'cw' else None,
        cw_tone_hz=args.cw_tone if args.waveform == 'cw' else None,
        level=dict(name=args.waveform, drive=cfg.voice.drive,
                   compression=cfg.voice.compression if args.waveform == 'voice' else None),
        test=test, action=args.action, simulation=asdict(simulation) if simulation else None,
        watermark=watermark,
        tx_mark=tx_mark.record(),
        watermark_keyed=watermark_key is not None,
        started_utc=datetime.now(timezone.utc).isoformat(),
        settings=asdict(cfg.voice), station_facts=facts_from(cfg),
        input=args.input if test or replay else args.input or cfg.audio.input,
        output=args.output if test else args.output or cfg.audio.output,
        cq=script.cq(own_call))
    if live and args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not cfg.voice.language_model or (args.waveform == 'voice' and (not cfg.voice.recognition_model or not cfg.voice.synthesis_model)):
        raise ValueError('Voice models are unset. Pass --models MANIFEST and --worker-python PYTHON, or set [voice] paths.')
    if not replay and args.action != 'levels':
        if not test and not plan['input']:
            raise ValueError('Name the station receive device with [audio] input or --input.')
        if live and not plan['output']:
            raise ValueError('Name the rig codec with [audio] output or --output.')
    directory.mkdir(parents=True, exist_ok=False)
    record = directory / 'provenance.json'
    code = 1
    session = Session(cfg, directory, test=test, replay=replay, input_device=plan['input'],
        output_device=plan['output'], preview_device=args.preview_output,
        dial_hz=dial, monitor=args.action == 'monitor', simulation=simulation,
        waveform=args.waveform, cw_wpm=args.cw_wpm, cw_tone=args.cw_tone, watermark=watermark,
        watermark_key=watermark_key, tx_mark=tx_mark)
    record.write_text(json.dumps(plan, indent=2) + '\n')
    try:
        with ExitStack() as held:
            if args.action in ('fly', 'monitor') and not replay:
                held.enter_context(monitor.every_lock(env={'HFMODEM_STATION': args.station} if args.station else None))
                held.enter_context(collector_claim(cfg))
            held.enter_context(unkey_on_signal(lambda: session.rig))
            held.callback(session.close)
            print(f'Loading local {args.waveform} contact stack…', flush=True)
            session.start_models()
            plan['model_hashes'] = model_hashes(cfg.voice, args.waveform)
            record.write_text(json.dumps(plan, indent=2) + '\n')
            if args.action == 'levels':
                session.make_draft('cq', plan['cq'])
                session.pending.result(timeout=60)
                draft = session.poll_draft()
                write_wav(args.wav_out, draft['audio'])
                print(f'{args.wav_out}: {draft["duration_s"]:.1f}s, peak drive {cfg.voice.drive:g}, mark {session.tx_mark.label}')
            else:
                if live:
                    session.prepare_audio()
                    radio(session, dial)
                elif args.action == 'monitor' and dial and not replay:
                    from . import rigsource
                    rigsource.tune(dial)
                session.start_audio()
                terminal.run(session)
            code = 0
    finally:
        # Keep signal protection and locks through all teardown, including PTT.
        session.close()
        plan.update(ended_utc=datetime.now(timezone.utc).isoformat(), returncode=code,
                    tx_mark_final=session.tx_mark.record(),
                    verdict=dict(line='voice session finished' if code == 0 else 'voice session interrupted or failed'))
        record.write_text(json.dumps(plan, indent=2) + '\n')
        print(f'Voice session: {directory}', flush=True)
    return code
