# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The station's own transmit latency, on the index every burst is keyed at.

`_LiveInput` schedules a burst so that output frame `at - _lat` reaches the DAC
as capture index `at` reaches the ADC, where `_lat` is the converter's own
`outputBufferDacTime - inputBufferAdcTime`. That is not the whole path: the
codec's in-and-out delay is not in the driver's number, and what is left over
went out on every burst this station has ever keyed.

`arm-v23-A-40-ws8eoc` measured the residue twice on one arm and got the same
answer both ways -- 19.8 ms from where WS8EOC's PACTOR-1 answer sat against our
960 ms packet, and about 20 ms from what the PACTOR-3 witness chain needs to
close against an SCS pair's 894.1 ms. So it is a property of the audio path,
spent on every emission of every protocol, and it is carried here rather than
inside a protocol constant: `P3_REPLY_S = 0.890` is measured off SCS traffic and
is right.

What this file holds is the arithmetic, on a stream with no device under it.
Three claims: the audio moves by exactly the correction and by nothing without
it; the key moves with the audio, so the PTT lead the session reports is the
same number either way; and the budget between the cycle's last read and the
admission check -- the tick, the render and the drain -- is untouched.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
import subprocess
from types import MethodType
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair

FS = onair.FS
# `arm-v23-A-40-ws8eoc`'s own clocks line: block 128, ADC->DAC 1268 samples,
# DAC notice 1652, holdback 660 -- and the 40 ms settle the FT-891 is set for.
BLK, LAT, SETTLE_N = 128, 1268, round(0.04 * FS)
#: What the v24 arm flies, in samples. 20 ms, measured two ways.
CORRECTION_N = round(0.020 * FS)


class _Done:
    """`threading.Event` for a burst that is over before it is armed."""

    def clear(self) -> None:
        pass

    def wait(self, _timeout: float) -> bool:
        return True


class _Stream(onair._LiveInput):
    """`_LiveInput`'s transmit arithmetic with no card and no clock.

    Every sleep is recorded instead of taken, so the instants are the
    measurement rather than a race: `slept[0]` is where PTT would be asserted.
    """

    def __init__(self, *, tx_latency_n: int = 0) -> None:
        self.fs = FS
        self._duplex = True
        self._blk = BLK
        self._lat = LAT
        self.tx_latency_n = tx_latency_n
        self.geometry_n = 0
        self.samples = 100 * BLK
        self.pos = self.samples
        self.holdback = 660
        # A fixed epoch, so two streams are comparable to the sample.
        self._dac = (0.0, 0)
        self._last = (0.0, self.samples)
        self._tx = None
        self._tx_at = 0
        self._tx_end_time = 0.0
        self._tx_done = _Done()
        self.keyed_s = 0.0
        self.keyed_at = None
        self.slept: list[float] = []
        self.flushes: list[int] = []

    def _sleep_until(self, t: float) -> None:
        self.slept.append(t)

    def sample_now(self) -> float:
        return float(self.samples)

    def take_until(self, index: int) -> np.ndarray:
        self.pos = max(self.pos, index)
        return np.zeros(0, np.float32)

    def wait_until(self, index: int) -> None:
        pass

    def flush_to(self, index: int) -> None:
        self.flushes.append(index)
        self.pos = index


def _key(stream: _Stream, at: int) -> tuple[int, float]:
    """(output frame the burst starts on, wall clock PTT goes up) for `at`."""
    stream.slept.clear()
    stream.transmit(np.zeros(round(0.21 * FS), np.float32), at=at,
                    settle=SETTLE_N / FS)
    return stream._tx_at, stream.slept[0]


def test_the_correction_moves_the_audio_and_nothing_moves_without_it():
    at = 400 * BLK
    plain, corrected = _Stream(), _Stream(tx_latency_n=CORRECTION_N)
    assert _key(plain, at)[0] == at - LAT
    assert _key(corrected, at)[0] == at - LAT - CORRECTION_N


def test_the_burst_still_reports_the_capture_index_it_was_aimed_at():
    # `first` is what `tx_audio_start` becomes and what every peer interval is
    # measured against. The burst lands there either way -- the correction is
    # the whole loop's, so it moves the emission, not the bookkeeping.
    at = 400 * BLK
    audio = np.zeros(round(0.21 * FS), np.float32)
    for stream in (_Stream(), _Stream(tx_latency_n=CORRECTION_N)):
        first, end = stream.transmit(audio, at=at, settle=SETTLE_N / FS)
        assert first == at
        assert end == at + audio.size


def test_the_key_moves_with_the_audio_so_the_ptt_lead_is_untouched():
    # The arm reports PTT LEAD SHORT at 25 ms of the 40 it is set for. Moving
    # the audio 20 ms early without moving the key would spend the correction
    # out of what is left of that lead and cut the head off every burst.
    at = 400 * BLK
    plain, corrected = _Stream(), _Stream(tx_latency_n=CORRECTION_N)
    start_p, key_p = _key(plain, at)
    start_c, key_c = _key(corrected, at)
    lead_p = plain._dac_time(start_p) - key_p
    lead_c = corrected._dac_time(start_c) - key_c
    assert lead_p == pytest.approx(SETTLE_N / FS, abs=1e-9)
    assert lead_c == pytest.approx(lead_p, abs=1e-9)
    assert key_p - key_c == pytest.approx(CORRECTION_N / FS, abs=1e-9)


def test_the_enqueue_deadline_moves_by_exactly_the_correction():
    plain, corrected = _Stream(), _Stream(tx_latency_n=CORRECTION_N)
    assert corrected.key_notice - plain.key_notice == CORRECTION_N
    # An instant already inside the plain stream's notice, so both clamps speak.
    at = plain.samples + plain.key_notice - 300
    assert plain.clamp_late(at) == 300
    assert corrected.clamp_late(at) - plain.clamp_late(at) == CORRECTION_N
    # ...and a slot that no longer leaves the notice is refused rather than
    # slid into the peer's packet, at the new deadline.
    late = corrected.samples + corrected.key_notice - 1
    with pytest.raises(onair._MissedTxSlot):
        corrected.transmit(np.zeros(BLK, np.float32), at=late, settle=0.0)


def test_the_cycles_budget_between_the_last_read_and_the_key_is_invariant():
    # `_prekey_lead` is one number for the read deadline and for the drain, and
    # what separates it from `key_notice` is the whole budget for the tick, the
    # render and the drain. The correction moves both edges together.
    plain, corrected = _Stream(), _Stream(tx_latency_n=CORRECTION_N)
    lead_p = onair._prekey_lead(plain, SETTLE_N)
    lead_c = onair._prekey_lead(corrected, SETTLE_N)
    assert lead_c - lead_p == CORRECTION_N
    assert lead_c - corrected.key_notice == lead_p - plain.key_notice
    # And the session says so unconditionally, in the line every transmit
    # deadline in `onair` is built from.
    assert (f"+ {CORRECTION_N} samples (20.0 ms) of station transmit latency"
            in onair._clock_line(corrected, SETTLE_N))


def test_a_replay_takes_neither_term():
    # No DAC to give notice to and no clock to lose it on. The settle alone is
    # the deadline there, exactly as it was.
    replay = onair._ReplayInput.__new__(onair._ReplayInput)
    assert onair._prekey_lead(replay, SETTLE_N) == SETTLE_N


@lru_cache(maxsize=1)
def _flight_transmit():
    shown = subprocess.run([
        'git', 'show', 'pactor-hears-a-gateway:packages/hfmodem/hfmodem/shrike/onair.py'
    ], capture_output=True, text=True, cwd=Path(__file__).parent)
    if shown.returncode:
        pytest.skip('the flight branch is not in this checkout')
    source = shown.stdout
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == '_LiveInput')
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == 'transmit')
    namespace = dict(vars(onair))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<flight-tip>', 'exec'),
         namespace)
    return namespace['transmit']


@pytest.mark.parametrize('path,old_ms', [
    ('flown', 20), ('p2-mail', 20), ('p4-bench', 3), ('generic', 0)])
def test_resolved_plan_preserves_repeated_callback_placement(monkeypatch, path, old_ms):
    from hfmodem.core.rigs import resolve_tx_timing

    monkeypatch.setattr(onair.time, 'monotonic', lambda: .25)
    plan = resolve_tx_timing(rig='ft891', command_path=path, environ={})
    old = _Stream(tx_latency_n=round(old_ms * FS / 1000))
    old.transmit = MethodType(_flight_transmit(), old)
    new = _Stream(tx_latency_n=plan.latency_samples)
    new.geometry_n = plan.geometry_samples
    audio = np.ones(10080, np.float32)
    geometry = plan.geometry_samples
    old_first, old_end = old.transmit(audio, at=None, settle=SETTLE_N / FS)
    first, end = new.transmit(audio, at=None, settle=SETTLE_N / FS)
    assert (first, end) == (old_first, old_end)
    assert (new.corrected_emission_start, new.corrected_emission_end) == (
        old_first + geometry, old_end + geometry)
    assert new._tx_at == old._tx_at
    assert new.slept == old.slept
    anchor = end - audio.size
    assert anchor == old_end - audio.size
    assert new.key_notice == old.key_notice
    assert onair._prekey_lead(new, SETTLE_N) == onair._prekey_lead(old, SETTLE_N)
    assert f'logical geometry +{geometry} samples' in onair._clock_line(new, SETTLE_N)
    for turn in range(1, 5):
        logical = anchor + turn * round(1.25 * FS)
        old.slept.clear()
        new.slept.clear()
        first, end = old.transmit(audio, at=logical, settle=SETTLE_N / FS)
        new_first, new_end = new.transmit(audio, at=logical, settle=SETTLE_N / FS)
        assert new._tx_at == old._tx_at
        assert new.slept == old.slept
        assert (new_first, new_end) == (first, end)
        assert new.corrected_emission_start - new_first == geometry
        assert new._tx_end_time == old._tx_end_time
    late = old.samples + old.key_notice - 1
    for stream in (old, new):
        with pytest.raises(onair._MissedTxSlot):
            stream.transmit(audio, at=late, settle=0)
    for peer_aim in (234567, 294567, 354567):
        first, end = old.transmit(audio, at=peer_aim, settle=SETTLE_N / FS)
        new_first, new_end = new.transmit(audio, at=peer_aim, settle=SETTLE_N / FS)
        assert (new_first, new_end, new._tx_at) == (first, end, old._tx_at)


@pytest.mark.parametrize('source', ['operator', 'toml', 'generic'])
def test_zero_precedence_and_generic_flight_placement(tmp_path, source):
    from hfmodem.core.rigs import resolve_tx_timing

    station = tmp_path / 'chain.toml'
    station.write_text('schema=1\n[station]\ncontrol="local"\nregulatory="unregulated"\n'
                       'because="offline fixture"\n[audio]\ntx_latency_ms=0\n')
    plan = resolve_tx_timing(0 if source == 'operator' else None, rig='ft891',
                             station_path=station if source == 'toml' else None,
                             command_path='generic' if source == 'generic' else 'p4-bench',
                             environ={})
    stream = _Stream(tx_latency_n=plan.latency_samples)
    stream.geometry_n = plan.geometry_samples
    assert plan.geometry_samples == (960 if source == 'generic' else 0)
    for logical in [60000, 120000, 180000]:
        assert _key(stream, logical)[0] == (logical - LAT - plan.latency_samples
                                             + plan.geometry_samples)


@pytest.mark.parametrize('path,old_ms', [
    ('flown', 20), ('p2-mail', 20), ('p4-bench', 3), ('generic', 0)])
def test_radio_tx_consumers_keep_flight_coordinates(monkeypatch, tmp_path, path, old_ms):
    from hfmodem.core.rigs import resolve_tx_timing

    monkeypatch.setattr(onair.time, 'monotonic', lambda: .25)
    plan = resolve_tx_timing(rig='ft891', command_path=path, environ={})
    old = _Stream(tx_latency_n=round(old_ms * FS / 1000))
    old.transmit = MethodType(_flight_transmit(), old)
    new = _Stream(tx_latency_n=plan.latency_samples)
    new.geometry_n = plan.geometry_samples
    rig = SimpleNamespace(ptt=lambda keyed: True, key_failure=lambda: None)
    radio_old = onair.RadioTx(rig, transmit=True, outdir=tmp_path, settle=.04)
    radio_new = onair.RadioTx(rig, transmit=True, outdir=tmp_path, settle=.04)
    radio_old.live, radio_new.live = old, new
    audio = np.ones(10080, np.float32)
    anchor = None
    for turn in range(5):
        if turn:
            aim = (anchor + turn * round(1.25 * FS)
                   if turn < 4 else 321987)
            radio_old.boundary = radio_new.boundary = aim
        radio_old._tx(audio, 'timing pin')
        radio_new._tx(audio, 'timing pin')
        if anchor is None:
            anchor = radio_old.tx_end - audio.size
        assert (new._tx_at, new.slept[-2], radio_new.tx_key_up,
                radio_new.tx_audio_start, radio_new.tx_end, new.flushes[-1], new.pos,
                radio_new.keyings[-1]) == (
                old._tx_at, old.slept[-2], radio_old.tx_key_up,
                radio_old.tx_audio_start, radio_old.tx_end, old.flushes[-1], old.pos,
                radio_old.keyings[-1])
        assert radio_new.corrected_emission_start - radio_new.tx_audio_start == plan.geometry_samples
        assert radio_new._tx_events.lines() == radio_old._tx_events.lines()


@pytest.mark.parametrize('path,old_ms', [
    ('flown', 20), ('p2-mail', 20), ('p4-bench', 3), ('generic', 0)])
@pytest.mark.parametrize('source', ['operator', 'toml'])
def test_explicit_zero_moves_by_flight_delta(monkeypatch, tmp_path, path, old_ms, source):
    from hfmodem.core.rigs import resolve_tx_timing

    station = tmp_path / 'chain.toml'
    station.write_text('schema=1\n[station]\ncontrol="local"\nregulatory="unregulated"\n'
                       'because="offline fixture"\n[audio]\ntx_latency_ms=0\n')
    plan = resolve_tx_timing(0 if source == 'operator' else None, rig='ft891',
                             station_path=station, command_path=path, environ={})
    old = _Stream(tx_latency_n=round(old_ms * FS / 1000))
    old.transmit = MethodType(_flight_transmit(), old)
    new = _Stream(tx_latency_n=plan.latency_samples)
    aim = 60000
    old.transmit(np.ones(BLK, np.float32), at=aim)
    new.transmit(np.ones(BLK, np.float32), at=aim)
    assert plan.latency_source == ('operator' if source == 'operator' else 'station-toml')
    assert plan.geometry_samples == 0
    assert new._tx_at - old._tx_at == round(old_ms * FS / 1000)
