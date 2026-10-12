# SPDX-License-Identifier: AGPL-3.0-only
"""Post-capture analysis of a fixed P4 setup/counter challenge.

Observations never drive acknowledgements. Header scores are hypotheses, not
decoded controls; silence cannot distinguish a wrong setup from a lost signal.
"""
import hashlib
import json

import numpy as np

from . import p1rx, p4sig, pactor1, rxfront

EXPERIMENT = 'grant-chirp-counter-v2'


def observe(audio, start, fs=48000):
    if fs != 48000:
        raise ValueError('experiment analysis requires 48000-Hz receive audio')
    x = np.asarray(audio, dtype=np.float32)
    result = dict(start_sample=start, end_sample=start + len(x),
                  audio_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                  rms=float(np.sqrt(np.mean(x.astype(float)**2))) if x.size else 0.,
                  p1_controls=[], p4_candidate=None)
    if len(x) < fs // 2 or not np.any(x):
        return result
    for seconds, duration in rxfront.p1_bursts(x):
        if .07 <= duration <= .26:
            control = p1rx.decode_control_signal(x, seconds, duration, fs)
            if control is not None and control.errors == 0:
                result['p1_controls'].append(dict(sample=start + round(seconds * fs),
                    index=control.index, word=f'0x{pactor1.CS_WORDS[control.index]:03X}',
                    sense=control.sense, errors=control.errors))
    # Fixed, overlapping half-second windows; bounded to the four-second RX.
    # Full Chu search runs once per receive interval, after radio teardown.
    candidates = [(p4sig.spread_score(x[i:i + fs // 2], fs), i)
                  for i in range(0, len(x) - fs // 2 + 1, fs // 4)]
    score, offset = max(candidates)
    result['spread_peak'] = float(score)
    if score >= p4sig.SPREAD_KNEE:
        header, root, shift, cfo = p4sig.header_score(x[offset:offset + fs // 2], fs)
        result['p4_candidate'] = dict(window_start=start + offset,
            window_end=start + offset + fs // 2, spread_score=float(score),
            header_score=float(header), root=int(root), shift=int(shift), cfo_hz=float(cfo))
    return result


def verdicts(observations, complete):
    """Compare preregistered predictions, without promoting correlations to ACKs."""
    headers = [o.get('p4_candidate') for o in observations]
    setup = 'inconclusive'
    if complete and any(headers):
        setup = 'P4-signature-observed; transition-candidate-only'
    elif complete and sum(len(o['p1_controls']) for o in observations) >= 2:
        setup = 'P1-replies-persist; direct-upgrade-prediction-not-observed'
    counter = 'inconclusive'
    if complete and len(headers) == 3 and all(h and h['header_score'] >= .35 for h in headers):
        keys = [(h['root'], h['shift']) for h in headers]
        if keys[0] == keys[1] and keys[2] != keys[1]:
            counter = 'candidate-pattern-consistent-with-counter-sensitive-replies'
        elif keys[0] != keys[1]:
            counter = 'repeatability-prediction-not-observed'
        else:
            counter = 'counter-change-prediction-not-observed'
    return setup, counter


def write_report(probe, directory, fs=48000, mode='live', context=None, *, capture_writer=None):
    if capture_writer is None:
        return _write_report(probe, directory, fs, mode, context)
    component = 'p4-mail-experiment'
    written = 0
    try:
        report = capture_writer.reserve((directory / 'p4-mail-experiment.json').absolute().relative_to(capture_writer.root))
        receive = capture_writer.reserve((directory / 'p4-mail-rx.npz').absolute().relative_to(capture_writer.root))
        capture_writer.update_status(state='ongoing', pending=2, component=component)
        def published():
            nonlocal written
            written += 1
        def progress():
            capture_writer.update_status(state='ongoing', pending=2-written,
                                         written=written, component=component)
        result = _write_report(probe, directory, fs, mode, context,
                               receive=receive, report=report, published=published, progress=progress)
        capture_writer.update_status(state='finalized', written=written, component=component)
        return result
    except BaseException:
        capture_writer._status['failed'] = True
        try:
            capture_writer.update_status(state='partial', pending=2-written,
                                         written=written, partial=1, component=component)
        except BaseException as error:
            capture_writer._status['journal_error'] = type(error).__name__
        raise


def _write_report(probe, directory, fs, mode, context, *, receive=None, report=None,
                  published=None, progress=None):
    # Save bounded raw windows before analysis, so errors leave replayable input.
    path = directory / 'p4-mail-rx.npz'
    arrays = {f'probe{i}': x for i, _, x in probe.received}
    if receive is None:
        np.savez_compressed(path, **arrays)
    else:
        temporary, stream = receive.temporary()
        with stream:
            np.savez_compressed(stream, **arrays)
        receive.publish_file(temporary, published=published)
        progress()
    observations = []
    errors = []
    for attempt, start, audio in probe.received:
        try:
            row = observe(audio, start, fs)
            row['attempt'] = attempt
            observations.append(row)
        except Exception as error:
            errors.append(dict(attempt=attempt, error=f'{type(error).__name__}: {error}'))
    complete = (len(probe.emissions) == 3 and len(observations) == 3 and not errors
                and [o['attempt'] for o in observations] == [0, 1, 2]
                and all(o['end_sample'] - o['start_sample'] == round(probe.listen_seconds * fs)
                        for o in observations))
    setup, counter = verdicts(observations, complete)
    result = dict(experiment=EXPERIMENT, mode=mode, sample_rate=fs, context=context or {},
        setup_hypothesis='selected P1 grant permits a direct P4 SL1 chirp transition',
        counter_hypothesis='identical chirps repeat a reply signature; changing only counter changes it',
        planned_statuses=[0x61, 0x61, 0x62], trigger_word=probe.trigger_word,
        grant_received=probe.requested, grant_sample=probe.grant_sample if probe.requested else None,
        payload_sha256=hashlib.sha256(probe.payload).hexdigest(),
        inherited_status=probe.status, emissions=probe.emissions,
        guard_exceptions=probe.guard_exceptions,
        receive_windows=probe.windows, observations=observations,
        complete=complete, stop_reason=probe.reason, analysis_errors=errors,
        setup_verdict=setup, counter_verdict=counter,
        mail_delivered=False, arq_acknowledgements_sent=0,
        limitations=['Header root/shift candidates do not establish ACK meanings; timing/CFO can confound them.',
                     'Four-second listening gaps test response behavior, not continuous 3.75-second P4 ARQ.',
                     'A failed prediction may reflect channel loss, wrong timing, fill, or waveform.',
                     'Counter advancement is a deliberate challenge, not an inferred ACK.'],
        receive_artifact=path.name,
        receive_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    raw = json.dumps(result, indent=2) + '\n'
    if report is None:
        (directory / 'p4-mail-experiment.json').write_text(raw)
    else:
        report.publish(raw.encode('utf-8'), published=published)
        progress()
    return result
