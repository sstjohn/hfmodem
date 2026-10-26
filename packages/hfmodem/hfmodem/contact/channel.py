# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Offline packet rehearsal through the shared local voice model/audio stack."""
from dataclasses import asdict
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from .language import Language
from .runtime import Session, write_wav
from .watermark import Encoder, GenerationFailure, carrier_text, detector, frame_from_symbols, pack, symbols, unpack
from . import cover

TOPICS = (
    'Enjoying friendly conversations and sharing ideas about amateur radio.',
    'The pleasure of listening and learning through a friendly radio conversation.',
    'Appreciating the opportunity to talk and exchange interesting thoughts.',
    'Taking time to listen carefully and enjoy a pleasant conversation.',
)


def cover_statistics(marked, controls):
    """Exploratory distribution distances, not a calibrated detection test."""
    from .watermark import normalized
    left = Counter(w for block in marked for w in normalized(block['recognized']).split())
    right = Counter(w for block in controls for w in normalized(block['recognized']).split())
    n_left, n_right = sum(left.values()), sum(right.values())
    if not n_left or not n_right:
        return dict(marked_words=n_left, control_words=n_right, total_variation=None, jensen_shannon_bits=None)
    rows, js = [], 0.
    for word in left.keys() | right.keys():
        p, q = left[word] / n_left, right[word] / n_right
        mean = (p + q) / 2
        js += .5 * ((p * math.log2(p / mean) if p else 0) + (q * math.log2(q / mean) if q else 0))
        rows.append(dict(word=word, marked=p, control=q, difference=p-q))
    return dict(marked_words=n_left, control_words=n_right,
                total_variation=.5 * sum(abs(r['difference']) for r in rows), jensen_shannon_bits=js,
                largest_frequency_changes=sorted(rows, key=lambda r: (-abs(r['difference']), r['word']))[:10],
                interpretation='finite-sample descriptive distances; no deniability or detection guarantee')


def replay(directory, key=None):
    """Independent receiver: recognized text and shared configuration only.

    Never read generated wording, requested symbols, or the manifest payload.
    Block positions come from the offline transport container's filenames.
    """
    from .watermark import Options
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    options = Options(**manifest['options'])
    receiver = detector(options, manifest['active_vocabulary'], key=key)
    observations = {}
    blocks, baseline = [], []
    for path in sorted(directory.glob('block-*-round-*-repeat-*.json')):
        match = re.fullmatch(r'block-(\d+)-round-\d+-repeat-\d+\.json', path.name)
        if not match:
            continue
        record = json.loads(path.read_text())
        text = carrier_text(record['recognition']['text']) if 'recognition' in record else record['recognized']
        observations.setdefault(int(match[1]), []).append(receiver.score(text))
        blocks.append(dict(recognized=text))
    for path in sorted(directory.glob('control-*.json')):
        record = json.loads(path.read_text())
        text = carrier_text(record['recognition']['text']) if 'recognition' in record else record['recognized']
        baseline.append(dict(recognized=text))
    decisions = [receiver.combine(observations.get(i, [])) for i in range(max(observations, default=-1) + 1)]
    status, payload, sequence = 'incomplete', None, None
    try:
        frame = frame_from_symbols([d['symbol'] for d in decisions], options.symbol_bits)
        sequence, payload = unpack(frame)
        status = 'crc_pass'
    except ValueError:
        status = 'crc_fail' if decisions and all(d['symbol'] is not None for d in decisions) else 'incomplete'
    return dict(status=status, recovered_hex=payload.hex() if payload is not None else None,
                received_sequence=sequence, method=options.method, keyed=key is not None,
                symbols=len(decisions), symbol_erasures=sum(d['symbol'] is None for d in decisions),
                cover_statistics=cover_statistics(blocks, baseline),
                decoded_from='recognized text only; block positions from offline container')


def simulate(cfg, directory, payload, options, *, repetitions=1, rounds=2, controls=4, noise=0., seed=1, text_only=False, key=None, topic=None):
    if not 1 <= repetitions <= 5 or not 1 <= rounds <= 5 or not 0 <= controls <= 100:
        raise ValueError('repetitions/rounds must be 1–5 and controls 0–100')
    if not np.isfinite(noise) or not 0 <= noise <= .2:
        raise ValueError('noise must be finite and between 0 and .2 RMS')
    frame = pack(payload)
    wanted = symbols(frame, options.symbol_bits)
    # This entry point has no hardware routing and never calls play/start_audio.
    session = Session(cfg, directory, test=True)
    observations = [[] for _ in wanted]
    blocks, baseline, wrong = [], [], []
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    seconds = 0.
    status, recovered, received_sequence = 'incomplete', None, None
    try:
        if text_only:
            session.language = Language(cfg.voice.language_model, cfg.voice.llama_server, directory)
        else:
            session.start_models()
        encoder = Encoder(session.language, options, key=key)
        receiver = detector(options, encoder.codebook.words, key=key)
        wrong_options = dict(asdict(options), codebook=options.codebook if key is not None else options.codebook + '-wrong')
        from .watermark import Options
        wrong_detector = detector(Options(**wrong_options), encoder.codebook.words,
                                  key=hashlib.sha256(key).digest() if key is not None else None)
        with open(session.language.model, 'rb') as stream:
            model_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest = dict(version='voice-channel-lab-v1', offline=True, text_only=text_only,
            options=asdict(options), active_vocabulary=encoder.codebook.words,
            keyed=key is not None, topic=topic,
            vocabulary_sha256=encoder.codebook.digest,
            model_sha256=model_hash,
            payload_hex=payload.hex(), repetitions=repetitions, rounds=rounds,
            noise_rms=noise, seed=seed, boundary_source='offline block container')
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

        def receive(generated, label):
            nonlocal seconds
            if text_only:
                text, duration = generated['text'], 0.
                paths = {}
            else:
                session.make_draft('channel', generated['text'])
                session.pending.result(timeout=60)
                draft = session.poll_draft()
                duration = draft['duration_s']
                seconds += duration
                audio = draft['audio'].copy()
                if noise:
                    audio += rng.normal(0, noise, len(audio)).astype(np.float32)
                rx_path = directory / f'{label}-rx.wav'
                write_wav(rx_path, audio)
                conditioned = session.waveform.receive_chain().block(audio)
                asr_path = directory / f'{label}-16k.wav'
                write_wav(asr_path, conditioned, 16000)
                recognition = session.speech.request('transcribe', path=str(asr_path), own_call=session.contact.own_call)
                # Exclude the trusted station identifier from the carrier score.
                text = carrier_text(recognition['text'])
                paths = dict(tx=draft['wav'], rx=str(rx_path), asr=str(asr_path), recognition=recognition)
            record = dict(label=label, generated=generated, recognized=text, duration_s=duration,
                          received=receiver.score(text), wrong_codebook=wrong_detector.score(text), **paths)
            (directory / f'{label}.json').write_text(json.dumps(record, indent=2) + '\n')
            return record

        pending = list(range(len(wanted)))
        for round_index in range(rounds):
            for position in pending:
                for repeat in range(repetitions):
                    label = f'block-{position:03d}-round-{round_index}-repeat-{repeat}'
                    try:
                        cover_topic = topic or (tuple(cover.TOPICS)[position % len(cover.TOPICS)]
                            if options.cover_mode == 'qso' else TOPICS[position % len(TOPICS)])
                        generated = encoder.generate(wanted[position], cover_topic,
                            seed=seed + position * 1000 + round_index * 100 + repeat * 20)
                    except GenerationFailure as exc:
                        (directory / f'{label}-generation-failure.json').write_text(json.dumps(
                            dict(position=position, requested_symbol=wanted[position], attempts=exc.attempts), indent=2) + '\n')
                        print(f'{label}: generation failed; retain an erasure and retry in the next round', flush=True)
                        continue
                    record = receive(generated, label)
                    record.update(position=position, round=round_index, requested_symbol=wanted[position])
                    (directory / f'{label}.json').write_text(json.dumps(record, indent=2) + '\n')
                    blocks.append(record)
                    observations[position].append(record['received'])
                    wrong.append(record['wrong_codebook'])
                    print(f'{label}: requested {wanted[position]}, copied {record["received"]["symbol"]}: '
                          f'{record["recognized"]}', flush=True)
            decisions = [receiver.combine(items) for items in observations]
            pending = [i for i, result in enumerate(decisions) if result['symbol'] is None]
            if pending:
                continue
            try:
                candidate = frame_from_symbols([result['symbol'] for result in decisions], options.symbol_bits)
                received_sequence, recovered = unpack(candidate)
                status = 'crc_pass'
                break
            except ValueError:
                status = 'crc_fail'
                pending = list(range(len(wanted)))
        payload_audio_s = seconds
        for index in range(controls):
            control_topic = topic or (tuple(cover.TOPICS)[index % len(cover.TOPICS)]
                if options.cover_mode == 'qso' else TOPICS[index % len(TOPICS)])
            generated = encoder.generate(0, control_topic, seed=seed + 100000 + index, marked=False)
            record = receive(generated, f'control-{index:03d}')
            baseline.append(record)
        decisions = [receiver.combine(items) for items in observations]
        correct = sum(result['symbol'] == expected for result, expected in zip(decisions, wanted))
        generation_failures = [json.loads(path.read_text())
                               for path in directory.glob('block-*-generation-failure.json')]
        wrong_decisions = [wrong_detector.combine([b['wrong_codebook'] for b in blocks if b['position'] == i])
                           for i in range(len(wanted))]
        wrong_crc_pass = False
        try:
            unpack(frame_from_symbols([d['symbol'] for d in wrong_decisions], options.symbol_bits))
            wrong_crc_pass = True
        except ValueError:
            pass
        result = dict(status=status, payload_match=recovered == payload if recovered is not None else False,
            recovered_hex=recovered.hex() if recovered is not None else None, received_sequence=received_sequence,
            symbol_bits=options.symbol_bits, symbols=len(wanted), symbol_correct=correct,
            symbol_errors=sum(result['symbol'] is not None and result['symbol'] != expected
                              for result, expected in zip(decisions, wanted)),
            symbol_erasures=sum(result['symbol'] is None for result in decisions),
            transmitted_blocks=len(blocks), generation_failures=len(generation_failures),
            generation_attempts=(sum(len(b['generated']['attempts']) for b in blocks)
                + sum(len(failure['attempts']) for failure in generation_failures)),
            payload_audio_s=payload_audio_s, total_audio_s=seconds,
            delivered_payload_bits_per_audio_s=(len(payload) * 8 / payload_audio_s
                if recovered == payload and payload_audio_s else None),
            unmarked_controls=len(baseline), unmarked_accepted=sum(b['received']['accepted'] for b in baseline),
            wrong_codebook_samples=len(wrong), wrong_codebook_accepted=sum(b['accepted'] for b in wrong),
            wrong_key_test=key is not None, wrong_decoder_packet_crc_pass=wrong_crc_pass,
            method=options.method, keyed=key is not None,
            symbol_decisions_are_presence_detections=False,
            elapsed_s=time.monotonic() - started, text_only=text_only,
            threshold_note='engineering thresholds, not calibrated statistical significance',
            cover_statistics=cover_statistics(blocks, baseline),
            decisions=decisions)
        (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    finally:
        session.close()
