# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Recorded LONG contract; real receiver, no radio or prerecorded decoder output.

Run this module as a script for an all-failures JSON report, optionally selecting
one tape with --tape. Pytest's semantic gate is deliberately not wall_clock.
Only SL2--5 exist in this census; this is not evidence for recorded SL6--10.
"""

import argparse
from collections import Counter
import hashlib
import json
from math import gcd
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import hilbert, resample_poly

from hfmodem.core import wav
from hfmodem.shrike import compress, onair, p4frame, p4host, p4longnative, p4turn
from hfmodem.shrike.p4budget import LongDeadline
from hfmodem.shrike.p4receiver import Clock, Expect, Kind, P4Receiver, Registered
from hfmodem.tests.evidence import RECORD, require_input
from hfmodem.tests.shrike.test_p4_bare_y27_0929 import K0003
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx

MANIFEST = Path(__file__).with_name('fixtures') / 'p4-long-ledger-1111/frames.json'
FRAMES = json.loads(require_input(MANIFEST, allow_module_level=True).read_text())['frames']
TAPES = sorted({row['tape'] for row in FRAMES})
# The census read this stock SL5 with free trellis boundaries and unclamped
# LLRs; the terminated, clamped engine stalls nine bits short of it. See
# working/p4-long-rx-1111/k0003-investigation.txt. It must stay unread until
# a signal-based change reads it, and then leave this set.
K0003_TAPE = K0003.relative_to(K0003.parents[5]).as_posix()
KNOWN_UNREAD = {f'stock:{K0003_TAPE}|1510.255'}


def _tape_path(relative):
    return require_input(RECORD / relative)


def _capture_audio(tape):
    """Read immutable census tapes directly, without capture-registry lookup."""
    with _tape_path(tape).open('rb') as stream, wav.decoded(stream) as source:
        source_fs, pcm = wavfile.read(source)
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    if np.issubdtype(pcm.dtype, np.unsignedinteger):
        half = (np.iinfo(pcm.dtype).max + 1) / 2
        audio = (pcm.astype(np.float64) - half) / half
    elif np.issubdtype(pcm.dtype, np.integer):
        audio = pcm.astype(np.float64) / -np.iinfo(pcm.dtype).min
    else:
        audio = pcm.astype(np.float64)
    if source_fs != 48000:
        divisor = gcd(source_fs, 48000)
        audio = resample_poly(audio, 48000 // divisor, source_fs // divisor)
    return audio


def _stock_clip(row, *, last=None):
    """Match the census Kiwi loader, keeping its true rate and analytic shift."""
    prep = row['preprocess']
    _, pcm = wavfile.read(_tape_path(row['tape']), mmap=True)
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    rate, fs = prep['source_rate_hz'], row['sample_rate']
    # Keep pre/post support for the Hilbert transform and resampler edge effects.
    lo = max(0, int((row['onset_sample'] / fs - .6) * rate))
    hi = int(((last or row)['onset_sample'] / fs + 4) * rate)
    x = np.asarray(pcm[lo:hi], dtype=np.float64)
    if pcm.dtype == np.int16:
        x /= 32768
    x = resample_poly(x, prep['upsample'], 1)
    x = (hilbert(x) * np.exp(-2j * np.pi * prep['downshift_hz'] * np.arange(len(x)) / fs)).real
    return x, round(lo / rate * fs)


def _fresh(at, cfo, fs):
    rx = P4Receiver(fs, admitted=True, cfo_hz=cfo, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .33, 3.75, cfo)
    return rx


def replay_tape(tape, *, budgeted=False):
    """Replay chains in order and return every outcome, including exceptions.

    As in census/live_replay.py, persistent state spans consecutive copies and
    resets after a gap of 1.5 slots. No offline decoder or fixture payload is fed
    to the receiver. Durations are diagnostics, not a key-margin qualification.
    """
    rows = sorted((r for r in FRAMES if r['tape'] == tape), key=lambda r: r['onset_sample'])
    _tape_path(tape)  # Missing recordings must fail, even outside pytest.
    audio = None if 'preprocess' in rows[0] else _capture_audio(tape)
    rx, previous = None, None
    results = []
    for row in rows:
        fs, at, cfo = row['sample_rate'], row['onset_sample'], row['cfo_hz']
        result = dict(id=row['id'], tape=tape, onset_sample=at, form=row['form'],
                      speedlevel=row['speedlevel'], source_stage=row['stage'],
                      prior_replay=row['tip_live_replay'], passed=False)
        started = monotonic()
        try:
            if rx is None or previous is None or at - previous > round(1.5 * 3.75 * fs):
                rx = _fresh(at, cfo, fs)
            previous = at
            rx.clock = Clock(Expect.PACKET, at, .33, 3.75, cfo)
            clip, base = _stock_clip(row) if 'preprocess' in row else (audio, 0)
            origin = at - round(.5 * fs)
            end = at + round(3.300 * fs)
            if origin < base or end > base + len(clip):
                raise ValueError('required replay support outside tape')
            def through(seconds):
                return clip[origin - base:at + round(seconds * fs) - base]
            event_start = len(rx.live_events)
            rx.read_partial(through(.845), origin, deadline=at + round(.845 * fs))
            rx.prepare_sf8(through(3.230), origin)
            final_started = monotonic()
            if budgeted:
                limits = {}
                def deadline(onset, *, sf8=False):
                    key = (onset, sf8)
                    if key not in limits:
                        limits[key] = LongDeadline(fs, onset,
                            lambda: at + round(3.300 * fs) + round((monotonic() - final_started) * fs),
                            lead_s=.044, sf8=sf8)
                    return limits[key]
                rx.long_deadline = deadline
                for job in rx._long_jobs.values():
                    if abs(job['obs'].stream_sample - at) < fs // 10:
                        job['limits'] = deadline(job['obs'].stream_sample, sf8=job['obs'].sf8)
            records = rx.read_window(through(3.300), origin, whole=True)
            result['finalize_seconds'] = monotonic() - final_started
            result['events'] = rx.live_events[event_start:]
            result['engine_events'] = [e for e in result['events'] if e.get('event') == 'long-engine']
            result['work_budget_seconds'] = .051 - (.010 if row['form'] == 'SF8' else 0)
            result['publication_budget_seconds'] = .095
            result['prekey_margin_seconds'] = .100 - .044 - result['finalize_seconds']
            if budgeted:
                rx.long_deadline = None
            found = [r.observation for r in records if r.outcome != Registered.EXPIRED
                     and r.observation.kind == Kind.LONG and r.observation.placed
                     and r.observation.frame is not None]
            result['frames'] = [dict(status=o.frame.status.byte,
                data_sha256=hashlib.sha256(o.frame.data).hexdigest(),
                speedlevel=str(o.speedlevel)) for o in found]
            expected_level = str(row['speedlevel']) if row['form'] == 'SF16' else f"SL{row['speedlevel']}-long"
            result['passed'] = any(f['status'] == row['status']
                and f['data_sha256'] == row['data_sha256']
                and f['speedlevel'] == expected_level for f in result['frames'])
            if not result['passed']:
                result['error'] = 'real receiver did not register the expected LONG identity and level'
            rx.finish_cycle(at)
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
            rx = None
        result['seconds'] = monotonic() - started
        results.append(result)
    return results


def replay_report(results):
    """Stable JSON-serializable census counts and measured elapsed durations."""
    timings = {}
    for form in sorted({f"{r['form']} SL{r['speedlevel']}" for r in results}):
        values = [r['finalize_seconds'] for r in results
                  if f"{r['form']} SL{r['speedlevel']}" == form and 'finalize_seconds' in r]
        if values:
            timings[form] = dict(count=len(values), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=max(values))
    baseline = dict(total=len(FRAMES),
        by_form=dict(Counter(f"{r['form']} SL{r['speedlevel']}" for r in FRAMES)),
        prior_replay_passed=sum(bool((r['tip_live_replay'] or {}).get('frame')) for r in FRAMES),
        offline_exact_c2=sum(r['stage'].startswith('exact-C2') for r in FRAMES),
        prior_replay_by_form=dict(Counter(f"{r['form']} SL{r['speedlevel']}"
            for r in FRAMES if (r['tip_live_replay'] or {}).get('frame'))),
        live_reader_by_form=dict(Counter(f"{r['form']} SL{r['speedlevel']}"
            for r in FRAMES if r['live_reader_pass'])))
    return dict(baseline=baseline, finalize_seconds_by_form=timings, total=len(results), passed=sum(r['passed'] for r in results),
        failed=sum(not r['passed'] for r in results),
        by_form=dict(Counter(f"{r['form']} SL{r['speedlevel']}" for r in results)),
        passed_by_form=dict(Counter(f"{r['form']} SL{r['speedlevel']}" for r in results if r['passed'])),
        by_source_stage=dict(Counter(r['source_stage'] for r in results)),
        prior_replay_passed=sum(bool((r.get('prior_replay') or {}).get('frame')) for r in results),
        recovered_prior_replay_failure=sum(r['passed'] and r.get('prior_replay') is not None
            and not r['prior_replay'].get('frame') for r in results),
        regressed_prior_replay_pass=sum(not r['passed']
            and bool((r.get('prior_replay') or {}).get('frame')) for r in results),
        exact_c2_source_passed=sum(r['passed'] and r['source_stage'].startswith('exact-C2')
            for r in results),
        seconds=sum(r['seconds'] for r in results), results=results)


@pytest.mark.parametrize('tape', TAPES)
def test_recorded_long_identities_through_real_receiver(tape):
    results = replay_tape(tape)
    failures = [r for r in results if r['passed'] == (r['id'] in KNOWN_UNREAD)]
    assert not failures, json.dumps(replay_report(results), indent=2)


@pytest.mark.wall_clock
@pytest.mark.parametrize('tape', TAPES)
def test_warm_recorded_final_callback_publication_budget(tape):
    if p4longnative.backend is None:
        pytest.skip('pure-Python LONG decode measured over the publication budget; the native backend meets it')
    # Warm imports/native tables and the actual frontend with this same tape.
    # A fresh replay still supplies a full causal prefix for every measured row.
    replay_tape(tape)
    results = replay_tape(tape, budgeted=True)
    failures = [r for r in results if r.get('finalize_seconds', float('inf')) > .095
                or r['id'] not in KNOWN_UNREAD and (not r['passed'] or not any(
                    e.get('crc_valid') and e.get('timely') for e in r.get('engine_events', [])))]
    assert not failures, json.dumps(replay_report(results), indent=2)


DOWNLOAD_ARMS = (
    'captures/onair-0929-1024/stream.flac',
    'captures/onair-20261010T233501182327Z-54a54140-3f95-46d2-856b-daed7e6f24f7/stream.flac',
    pytest.param(K0003_TAPE,
                 marks=pytest.mark.xfail(strict=True, reason='its only SL5 is KNOWN_UNREAD')),
)


def run_driver(row, *, following=(), monkeypatch, stall_ms=0, key_notice=0, lead_samples=2131):
    """Run real driver/receiver on a crop, retaining a source-coordinate map.

    IRS initialization mirrors the existing recorded LONG driver test. The
    sample-clock stall is deterministic; physical PTT is outside this harness.
    """
    from hfmodem.shrike import p4longdecode
    # Decoder CPU is outside this replay's declared sample-clock charges.
    monkeypatch.setattr(p4longdecode, '_python_step_seconds', lambda *args: 0.)
    monkeypatch.setattr(p4longdecode, '_python_check_seconds', lambda _: 0.)
    rows = [row, *following]
    fs = 48000
    if 'preprocess' in row:
        audio, source_origin = _stock_clip(row, last=rows[-1])
        source_fs = row['sample_rate']
        divisor = gcd(fs, source_fs)
        audio = resample_poly(audio, fs // divisor, source_fs // divisor)
    else:
        source_fs = fs
        source_origin = max(0, row['onset_sample'] - round(.6 * fs))
        audio = _capture_audio(row['tape'])[source_origin:rows[-1]['onset_sample'] + 4 * fs]
    onsets = {r['id']: round((r['onset_sample'] - source_origin) * fs / source_fs) for r in rows}
    at = onsets[row['id']]
    stream = Stream(audio)
    stream.pos = max(0, at - round(.6 * fs))
    stream.census_onsets = onsets
    stream.census_origin_sample = source_origin
    stream.census_sample_rate = source_fs
    stream.wall = stream.pos
    stream.sample_now = lambda: max(stream.pos, stream.wall)
    stream.key_notice = key_notice
    stream.transmit = True
    stream.clamp_late = lambda when: max(0, stream.sample_now() + key_notice - when)
    tx = Tx(stream)
    tx.settle = lead_samples / fs
    tx.sessrx.p4 = _fresh(at, row['cfo_hz'], fs)
    tx.p4_probe.emissions.clear()
    tx.p4_probe.handoff_audio = 0, audio[:stream.pos].copy()
    tx.p4_link_options.p4_link_timeout = len(audio) / fs
    tx.p4_link_options.p4_data = 'body'
    received = []
    tx.p4_link_host.deliver = received.append
    read = tx.sessrx.p4.read_window
    def timed(*args, **kwargs):
        records = read(*args, **kwargs)
        if any(r.observation.kind == Kind.LONG for r in records):
            stream.wall = max(stream.wall, stream.pos) + round(stall_ms / 1000 * fs)
        return records
    monkeypatch.setattr(tx.sessrx.p4, 'read_window', timed)
    monkeypatch.setattr(onair, '_prekey_lead', lambda *args: lead_samples)
    turn = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *args, **kwargs:
                        turn(*args, **{**kwargs, 'role': 'irs'}))
    original_key = tx._tx
    def flushing_key(audio, what):
        original_key(audio, what)
        stream.pos = tx.tx_end
    tx._tx = flushing_key
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    return tx, stream, received


@pytest.mark.parametrize('tape', DOWNLOAD_ARMS)
def test_recorded_downloads_reach_real_driver(monkeypatch, tape):
    rows = sorted((r for r in FRAMES if r['tape'] == tape), key=lambda r: r['onset_sample'])[:2]
    if 'kiwi/' in tape:
        rows = [r for r in FRAMES if r['tape'] == tape and r['speedlevel'] == 5]
        assert len(rows) == 1
    else:
        assert len(rows) == 2
    assert len({(r['status'], r['data_sha256']) for r in rows}) == 1
    tx, stream, received = run_driver(rows[0], following=rows[1:], monkeypatch=monkeypatch)
    heard = [r for r in tx.p4_probe.heard if r['kind'] == Kind.LONG
             and not r['provisional'] and 'status' in r and 'data' in r]
    supervisor = compress.Supervisor()
    identities, expected_delivery = set(), []
    for row in rows:
        matches = [r for r in heard if abs(r['onset_sample'] - stream.census_onsets[row['id']]) < 4800]
        verified = [r for r in matches if r['status'] == row['status']
                    and hashlib.sha256(bytes.fromhex(r['data'])).hexdigest() == row['data_sha256']]
        assert verified, (row['id'], heard, tx.p4_probe.link_events)
        replies = [e for e in tx.p4_probe.emissions
                   if any(abs(e['audio_start'] - r['onset_sample'] - round(3.4 * 48000)) <= 1
                          for r in verified)]
        assert len(replies) == 1, tx.p4_probe.link_events
        # Each crop starts a fresh chain and contains two copies of one
        # identity: toggle starts at X27 and a repeat keeps that same word.
        # Its robust LONG partner would be V27 (normal LONG uses T27),
        # independently of the starting counter's parity.
        assert replies[0]['word'] == 'X27', ('ACK mismatch', row['id'], replies)
        identity = row['status'], row['data_sha256']
        if identity not in identities:
            identities.add(identity)
            payload = supervisor.feed(p4host.source_payload(
                p4frame.Status.from_byte(row['status']), bytes.fromhex(verified[0]['data'])))
            if payload:
                expected_delivery.append(payload)
    assert expected_delivery
    assert received == expected_delivery, ('host delivery mismatch', received, expected_delivery, tx.p4_probe.link_events)
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'Refused']


@pytest.mark.parametrize('damage', ['wrong-bytes', 'duplicate-bytes', 'wrong-ack'])
def test_recorded_download_contract_rejects_corruption(monkeypatch, damage):
    import sys
    module = sys.modules[__name__]
    def corrupted(row, *, following, **kwargs):
        # Isolate the assertions from receiver/deadline failures in the real replay.
        rows = [row, *following]
        payload = b'RMS Trimode 1.4.4.0\r\nW9SSJ has 103 daily'
        heard = [dict(kind=Kind.LONG, provisional=False, status=r['status'],
                      data=payload.hex(), onset_sample=r['onset_sample']) for r in rows]
        emissions = [dict(audio_start=r['onset_sample']+round(3.4*48000), word='X27') for r in rows]
        tx = SimpleNamespace(p4_probe=SimpleNamespace(heard=heard, emissions=emissions, link_events=[]))
        stream = SimpleNamespace(census_onsets={r['id']: r['onset_sample'] for r in rows})
        received = [payload]
        if damage == 'wrong-bytes':
            received[:] = [b'wrong host bytes']
        elif damage == 'duplicate-bytes':
            received.extend(received[:])
        else:
            for emission in tx.p4_probe.emissions:
                emission['word'] = 'U27'
        return tx, stream, received
    monkeypatch.setattr(module, 'run_driver', corrupted)
    with pytest.raises(AssertionError, match='ACK mismatch' if damage == 'wrong-ack' else 'host delivery mismatch'):
        test_recorded_downloads_reach_real_driver(monkeypatch, DOWNLOAD_ARMS[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tape', help='substring selecting tape paths')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for tape in TAPES:
        if args.tape is None or args.tape in tape:
            try:
                results.extend(replay_tape(tape))
            except Exception as error:
                for row in FRAMES:
                    if row['tape'] == tape:
                        results.append(dict(id=row['id'], tape=tape, form=row['form'],
                            speedlevel=row['speedlevel'], source_stage=row['stage'],
                            passed=False, seconds=0, error=f'{type(error).__name__}: {error}'))
    if not results:
        parser.error('no matching tapes')
    report = replay_report(results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))
    raise SystemExit(bool(report['failed']))
