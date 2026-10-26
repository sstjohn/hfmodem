# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""D2(b)/D3: recorded IRS chirps, host extension and explicit decode charges."""
from pathlib import Path
import runpy

import pytest

from hfmodem.shrike import p4chirprx, session

ROOT = Path(__file__).resolve().parents[5]
REPLAY = ROOT / 'working/p4-plan-0930/chirp-rx/replay.py'


def require_local(path):
    if not path.is_file():
        pytest.skip(f'local chirp replay input absent: {path}')
    return path


def replay(tmp_path, arm, start, end, cost=35):
    require_local(REPLAY)
    require_local(ROOT / 'working/p4-plan-0930/stall-bisect/drive.py')
    for name in ('stream.flac', 'p4-probe.json'):
        require_local(ROOT / f'captures/onair-0930-{arm}' / name)
    return runpy.run_path(str(REPLAY))['replay'](
        arm, tmp_path/'replay.json', start, end, cost)


def has_word(result, word, at):
    return any(e.get('word') == word and abs(e['audio_start']/48000-at) < .006
               for e in result['emissions'][1:])


def test_2226_alternates_and_delivers_extended_greeting_before_deadline(tmp_path):
    result = replay(tmp_path, '2226', 55, 79)
    assert has_word(result, 'X27', 65.65)
    assert has_word(result, 'V27', 69.40)
    assert bytes.fromhex('7465732072656d61696e696e67207769746820') in [
        bytes.fromhex(s) for s in result['delivered']]
    assert has_word(result, 'X27', 76.90)


def test_2206_counter_two_changes_word_in_its_own_slot(tmp_path):
    result = replay(tmp_path, '2206', 46, 59)
    assert has_word(result, 'V27', 54.40)


def test_2206_extended_counter_three_delivers_and_acks(tmp_path):
    result = replay(tmp_path, '2206', 110, 120)
    assert b' Gateway at Murray ' in [bytes.fromhex(s) for s in result['delivered']]
    assert has_word(result, 'X27', 118.15)


def test_2206_recorded_costs_reduce_seven_deferrals(tmp_path):
    result = replay(tmp_path, '2206', None, 119, -1)
    summary = next(e for e in result['events'] if e['event'] == 'DecodeBudgetSummary')
    assert summary['decisions'] == 29
    # B: the last late chirp is ACKed in its own slot instead of deferred.
    assert summary['deferred_decisions'] == 0


def test_over_budget_chirp_logs_decode_duration_and_deferral(tmp_path, capsys):
    result = replay(tmp_path, '2226', 55, 69, 80)
    # B: over budget yet keyable, the chirp's answer keys at its own due.
    keyed = {e['at'] for e in result['events'] if e['event'] == 'Keyed'}
    assert any(e['event'] == 'Decision' and e['reason'] == 'proposal'
               and 'chirp_decode' in e['work_ms'] and e['scheduled_at'] in keyed
               for e in result['events'])
    log = capsys.readouterr().out
    assert 'reason=proposal' in log and 'chirp_decode=' in log


def test_envelope_without_acquired_marker_does_not_enter_fast_decode(tmp_path, monkeypatch):
    from hfmodem.shrike.p4observation import Kind, Observation
    from hfmodem.shrike.p4receiver import P4Receiver

    partial = P4Receiver.read_partial

    def envelope(self, audio, origin, **kwargs):
        records = partial(self, audio, origin, **kwargs)
        at = self.clock.next_due
        if 58 < at/48000 < 59:
            # Supply only the provisional envelope hint, without admitting a
            # marker. The complete waveform still comes from the recording.
            self._emit(Observation(Kind.CHIRP, 48000, origin, at-origin, at,
                .81, self.clock.cfo_hz, 0., provisional=True), records)
        return records

    monkeypatch.setattr(P4Receiver, 'read_partial', envelope)
    monkeypatch.setattr(p4chirprx, 'acquire_header', lambda *a, **kw: None)
    monkeypatch.setattr(p4chirprx, 'decode_packet',
                        lambda *a, **kw: pytest.fail('decoded an unacquired chirp'))
    result = replay(tmp_path, '2226', 55, 62.2, 0)
    assert any(h['kind'] == 'chirp' and h['provisional'] for h in result['heard'])


def test_recorded_chirp_mixes_at_reduced_rate(monkeypatch):
    # A source-work bound, not a wall-clock assertion: both lane mixers together
    # must process 52,920 samples, rather than 317,520 at the capture rate.
    path = require_local(ROOT / 'captures/onair-0930-2206/stream.flac')
    audio = session.load_wav(str(path), 48000)
    audio = audio[1908010:1908010+158760]
    sizes = []
    exp = p4chirprx.np.exp

    def counted(x, *a, **kw):
        sizes.append(p4chirprx.np.size(x))
        return exp(x, *a, **kw)

    monkeypatch.setattr(p4chirprx.np, 'exp', counted)
    result = p4chirprx.decode_packet(audio, frequency_offset=.444286032449214)
    assert result.frame.status.byte == 0x61
    assert result.frame.data == b'RMS Trimode 1.4.4.0 We'
    assert sum(s for s in sizes if s > 1000) == 52920


@pytest.mark.parametrize('fast_first', [False, True])
def test_recorded_chirp_read_paths_share_decode_and_registration(monkeypatch, fast_first):
    import numpy as np
    from hfmodem.shrike.p4observation import Kind
    from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver

    with np.load(Path(__file__).parent / 'fixtures/sigidwiki/p4-chirps/audio.npz') as fixture:
        fs = int(fixture['fs'])
        audio = np.r_[np.zeros(fs//3), fixture['audio'][0], np.zeros(fs//3)]
    rx = P4Receiver(fs, admitted=True, chirp=True)
    rx.expected = True
    rx.clock = Clock(Expect.PACKET, fs//3, .33, 3.75, -242.4)
    calls = []
    decode = p4chirprx.decode_packet

    def counted(*args, **kwargs):
        calls.append(kwargs['start_sample'])
        return decode(*args, **kwargs)

    monkeypatch.setattr(p4chirprx, 'decode_packet', counted)
    assert rx.read_chirp(audio, 0, onset=fs//3) is None
    early = fs//3+round(.81*fs)
    records = rx.read_partial(audio[:early], 0, deadline=early)
    onset = next(r.observation.stream_sample for r in records
                 if r.observation.kind == Kind.CHIRP)
    assert rx.has_chirp_header(onset)
    assert rx.read_chirp(audio[:early], 0, onset=onset) is None
    assert calls == []

    def fast():
        return rx.read_chirp(audio, 0, onset=onset)

    def ordinary():
        return rx.read_window(audio, 0, whole=True)

    first, second = (fast, ordinary) if fast_first else (ordinary, fast)
    records = first()
    assert len(records) == 1
    assert records[0].observation.frame is not None
    assert second() == []
    assert calls == [onset]
    rx.clock.cfo_hz += 1
    assert not rx.has_chirp_header(onset)
    assert rx.read_chirp(audio, 0, onset=onset) is None
    assert calls == [onset]
