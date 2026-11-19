# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import contextlib
import io
import os
import pickle
import subprocess
import sys
import time
import math
from types import ModuleType
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hfmodem.shrike import onair, p3acquire, placement, rxfront
from hfmodem.tests.shrike import p3_deadline_support as deadline
from hfmodem.tests.shrike.p3_recovery_support import _MeasuredCosts, _RecoveryReplay
from hfmodem.tests.shrike.p3_reply_support import harness


COARSE = (0.0, *(h for h in (*p3acquire.OFFSETS_HZ, -100.0, 100.0)
                  if h != 0.0))
FINE = (5.0, -5.0)
COLD = tuple(sorted(
    (h for h in p3acquire.ENTRY_CONTROL_OFFSETS_HZ
     if h not in set(COARSE) | set(FINE)), key=lambda h: (abs(h), h)))
CASES = (
    ("changeover", "coarse", 4.0, 1.25, dict(offsets=COARSE)),
    ("changeover", "fine", 4.0, 1.25, dict(offsets=FINE)),
    ("changeover", "cold", 4.0, 1.25, dict(offsets=COLD)),
    ("changeover", "control", 1.0, .5, dict(offsets=p3acquire.CONTROL_OFFSETS_HZ)),
    ("control_signal", "entry", .46, .31, dict(offsets=p3acquire.ENTRY_CONTROL_OFFSETS_HZ)),
    ("head_aims", "aims", 1.17, .6, {}),
)


@pytest.fixture(scope="module")
def original_acquisition():
    _acquisition_archive()
    revision = "df946ce0c"
    path = "packages/hfmodem/hfmodem/shrike/p3acquire.py"
    repo = Path(__file__).resolve().parents[5]
    if not (repo / ".git").exists():
        pytest.skip(f"no git history to read {revision} from")
    source = subprocess.run(["git", "show", f"{revision}:{path}"],
                            cwd=repo, capture_output=True, text=True)
    if source.returncode:
        pytest.skip(f"acquisition source at {revision} absent")
    module = ModuleType(f"hfmodem.shrike._p3_acquire_at_{revision}")
    module.__package__ = "hfmodem.shrike"
    sys.modules[module.__name__] = module
    try:
        exec(compile(source.stdout, f"{revision}:{path}", "exec"), module.__dict__)
        yield module
    finally:
        sys.modules.pop(module.__name__, None)


def _acquisition_archive():
    archive = os.environ.get("HFMODEM_P3_ACQUISITION_ARCHIVE")
    if not archive:
        pytest.skip("HFMODEM_P3_ACQUISITION_ARCHIVE is unset")
    return Path(archive)


def _receiver():
    receiver = object.__new__(onair._SessionRx)
    receiver._p3_recovery_fits = lambda reserve: True
    return receiver


def _identity(candidate):
    if candidate is None:
        return None
    event = candidate.event
    return (event.cs, event.start, event.t, candidate.offset_hz,
            candidate.coarse_hz, repr(event.packet), event.text)


def _compare(window, offsets, receiver):
    whole = p3acquire.changeover(window, offsets=offsets)
    chunked = receiver._p3_recovery_changeover(
        window, offsets, coarse_s=.010, coarse_count=len(COARSE))
    assert _identity(chunked) == _identity(whole)
    if whole is not None:
        assert chunked.quality == pytest.approx(whole.quality, abs=1e-9)
    return _identity(chunked)


@pytest.mark.parametrize("gain", [1.0, .7, .013])
def test_chunked_changeover_preserves_bodyless_ties(gain):
    audio = np.zeros(120000)
    word = placement.control_signal(placement.BREAKIN_CS)
    for start in (120, 60120):
        audio[start:start + len(word)] = gain * word
    result = _compare(audio, (0.0, *COLD, 0.0), _receiver())
    assert result is not None


def test_chunked_changeover_matches_recorded_calls(tmp_path, original_acquisition):
    archive = _acquisition_archive()
    paths = sorted((archive / "rec").glob("*.pkl"))
    if not paths:
        pytest.skip(f"acquisition call archive absent: {archive / 'rec'}")
    receiver = _receiver()
    inventory, checked, sources = Counter(), [], set()
    for path in paths:
        with path.open("rb") as stream:
            row = pickle.load(stream)
        inventory[row["name"]] += 1
        sources.add(row["test"])
        before = getattr(original_acquisition, row["name"])(
            row["audio"], *row["args"], **row["kw"])
        after = getattr(p3acquire, row["name"])(
            row["audio"], *row["args"], **row["kw"])
        if row["name"] in ("changeover", "control_signal"):
            assert _identity(after) == _identity(before), (path.name, row["test"])
        else:
            assert after == before, (path.name, row["test"])
        if row["name"] != "changeover":
            continue
        assert not row["args"], path
        offsets = row["kw"].get("offsets", p3acquire.OFFSETS_HZ)
        try:
            result = _compare(row["audio"], offsets, receiver)
        except AssertionError as error:
            raise AssertionError((path.name, row["test"])) from error
        checked.append(dict(call=path.name, test=row["test"],
                            offsets=len(offsets), result=result))
    (tmp_path / "recorded-call-equivalence.json").write_text(json.dumps(
        dict(inventory=inventory, source_tests=sorted(sources), checked=checked),
        indent=2) + "\n")
    assert checked


@pytest.mark.wall_clock
def test_chunked_changeover_matches_november_8_windows(tmp_path, original_acquisition):
    inventory = _acquisition_archive() / "arms.txt"
    if not inventory.exists():
        pytest.skip(f"November 8 acquisition inventory absent: {inventory}")
    captures = [Path(line) for line in inventory.read_text().splitlines() if line]
    missing = [str(path) for path in captures
               if not list(path.glob("stream-l*-s*.flac"))]
    if missing:
        pytest.skip(f"November 8 streams absent: {missing}")
    receiver, rows = _receiver(), []
    for capture in captures:
        stream = sorted(capture.glob("stream-l*-s*.flac"))[0]
        audio = rxfront.load_wav(str(stream))
        for name, label, seconds, hop, kwargs in CASES:
            length = round(seconds * onair.FS)
            for start in range(0, max(1, len(audio) - length),
                               round(hop * onair.FS)):
                window = np.ascontiguousarray(audio[start:start + length])
                before = time.perf_counter()
                try:
                    old = getattr(original_acquisition, name)(window, **kwargs)
                    new = getattr(p3acquire, name)(window, **kwargs)
                    if name == "head_aims":
                        result = new
                        assert result == old
                    else:
                        result = _identity(new)
                        assert result == _identity(old)
                        if new is not None:
                            assert new.quality == pytest.approx(old.quality, abs=1e-9)
                        if name == "changeover":
                            assert result == _compare(window, kwargs["offsets"], receiver)
                except AssertionError as error:
                    raise AssertionError((capture.name, label, start)) from error
                rows.append(dict(capture=capture.name, stage=label, start=start,
                                 result=result,
                                 comparison_ms=(time.perf_counter() - before) * 1e3))
    (tmp_path / "november-8-equivalence.json").write_text(json.dumps(
        dict(captures=[path.name for path in captures], windows=rows),
        indent=2) + "\n")
    assert rows


@pytest.mark.wall_clock
def test_measure_november_8_recovery_tail(tmp_path):
    inventory = _acquisition_archive() / "arms.txt"
    if not inventory.exists():
        pytest.skip(f"November 8 acquisition inventory absent: {inventory}")
    captures = [Path(line) for line in inventory.read_text().splitlines() if line]
    missing = [str(path) for path in captures
               if not list(path.glob("stream-l*-s*.flac"))]
    if missing:
        pytest.skip(f"November 8 streams absent: {missing}")
    rows = []
    for capture in captures:
        stream = sorted(capture.glob("stream-l*-s*.flac"))[0]
        audio = rxfront.load_wav(str(stream))
        with contextlib.redirect_stdout(io.StringIO()):
            host, receiver, tx, grid, bench = harness(tmp_path)
        tx.record_audio = False
        evidence = onair._CycleEvidence(bench, tx, grid, tmp_path)
        spectrum = onair._PeerSpectrum()
        for start in range(0, max(1, len(audio) - 4 * onair.FS),
                           round(1.25 * onair.FS)):
            window = np.ascontiguousarray(audio[start:start + 4 * onair.FS])
            receiver.words_at.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                before = time.perf_counter()
                receiver._p2_packet(window)
                p2_ms = (time.perf_counter() - before) * 1e3
                before = time.perf_counter()
                bursts = evidence.record("recovery-measurement", window, start)
                onsets = [at for at, _ in bursts]
                onair._read_codeword_at_bursts(receiver, window, start, onsets)
                if not receiver.words_at:
                    onair._grid_answer(receiver, window, start, 0)
                onair._grid_reversal(grid, host)
                spectrum.hear(window[-grid.cycle_n:])
                onair._forecast_next_key(receiver, tx, grid, start)
                tail_ms = (time.perf_counter() - before) * 1e3
            rows.append(dict(capture=capture.name, start=start,
                             samples=len(window), p2_ms=p2_ms, tail_ms=tail_ms))
    summary = {
        key: dict(n=len(rows), p99=float(np.percentile(
            [row[key] for row in rows], 99)),
            maximum=max(row[key] for row in rows))
        for key in ("p2_ms", "tail_ms")}
    (tmp_path / "november-8-recovery-tail.json").write_text(json.dumps(
        dict(summary=summary, windows=rows), indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    assert rows



def _tapes(root=deadline.FIXTURES.parent):
    return sorted(path for path in root.rglob("*")
                  if path.suffix.lower() in (".wav", ".flac"))


class _TapeRecoveryReplay(_RecoveryReplay):
    def __init__(self, tag, audio, tmp_path, **kwargs):
        # A crop alone has no ARQ snapshot. Seed the post-fallback
        # ISS state and put every sample after its already-keyed P1 carrier.
        padded = np.concatenate((np.zeros(deadline.P1_DATA_N, np.float32), audio,
                                 np.zeros(2 * 60000, np.float32)))
        row = dict(samples=len(padded), own_carrier_spans=[(0, deadline.P1_DATA_N)],
                   first_slot=0, last_slot=math.ceil(len(padded) / 60000))
        with patch.object(deadline, "crop", lambda _: (row, padded)):
            super().__init__("e08", tmp_path, **kwargs)
        self.bench.limit += 60 * onair.FS
        self.tag = tag


def _compare_campaigns(baseline, recovered, tmp_path, tag, scale):
    old_slots, new_slots = set(baseline.tx.slots_used), set(recovered.tx.slots_used)
    old_crc = {tuple(row["identity"]): row for row in baseline.deliveries}
    new_crc = [row for row in recovered.deliveries
               if tuple(row["identity"]) not in old_crc
               or row["slot"] < old_crc[tuple(row["identity"])]["slot"]]
    first = min((row["at"] for row in new_crc), default=float("inf"))
    divergences = []
    for index in range(max(len(baseline.cycles), len(recovered.cycles))):
        old = baseline.cycles[index] if index < len(baseline.cycles) else None
        new = recovered.cycles[index] if index < len(recovered.cycles) else None
        if old is None or new is None or old["emissions"] != new["emissions"]:
            divergences.append(dict(cycle=index, baseline=old, recovery=new,
                cause="new CRC delivery" if any(row["cycle"] <= index for row in new_crc)
                else "charged work changed scheduling without a new CRC"))
    report = dict(tag=tag, scale=scale,
        baseline=baseline.cycles, recovery=recovered.cycles,
        baseline_slots=sorted(old_slots), recovery_slots=sorted(new_slots),
        baseline_deliveries=baseline.deliveries, recovery_deliveries=recovered.deliveries,
        new_crc_deliveries=new_crc, admissions=recovered.admissions,
        baseline_work=baseline.work, recovery_work=recovered.work,
        baseline_placements=baseline.placements, recovery_placements=recovered.placements,
        divergences=divergences)
    (tmp_path / "keyed-slot-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    assert old_slots <= new_slots, sorted(old_slots - new_slots)
    assert recovered.tx.refused <= baseline.tx.refused
    before_old = [(row["slot"], row["actual"]) for row in baseline.placements
                  if row["actual"] < first]
    before_new = [(row["slot"], row["actual"]) for row in recovered.placements
                  if row["actual"] < first]
    assert before_new == before_old
    for row in recovered.placements:
        if row["actual"] >= first:
            assert row["requested"] is not None
            assert row["actual"] == row["requested"], row
            assert row["requested"] >= row["earliest"], row
    return report


@pytest.mark.wall_clock
@pytest.mark.parametrize("tag", ["e07", "e05", "r06", "e08"])
@pytest.mark.parametrize("scale", [1.0, 2.5, 15.0])
def test_recovery_preserves_recorded_campaign_keyed_slots(tmp_path, tag, scale):
    costs = _MeasuredCosts()
    baseline = _RecoveryReplay(tag, tmp_path / "baseline", scale=scale,
                               recovery=False, measured_costs=costs).run()
    recovered = _RecoveryReplay(tag, tmp_path / "recovery", scale=scale,
                                recovery=True, measured_costs=costs).run()
    _compare_campaigns(baseline, recovered, tmp_path, tag, scale)


def test_recovery_campaign_harness_with_zero_cost(tmp_path):
    baseline = _RecoveryReplay("e07", tmp_path / "baseline", scale=0,
                               recovery=False).run(cycles=2)
    recovered = _RecoveryReplay("e07", tmp_path / "recovery", scale=0,
                                recovery=True).run(cycles=2)
    _compare_campaigns(baseline, recovered, tmp_path, "e07", 0)
    assert any(row["stage"] == "tail" for row in recovered.work)
    assert recovered.placements


def test_paired_replays_reuse_measured_cost_only_for_identical_work():
    costs = _MeasuredCosts()
    audio = np.arange(32, dtype=np.float32)
    first = costs.seconds("search", (audio,), {"offsets": (0.,)}, .010)
    assert costs.seconds("search", (audio.copy(),), {"offsets": (0.,)}, .100) == first
    assert costs.seconds("search", (audio + 1,), {"offsets": (0.,)}, .020) == .020
    assert costs.seconds("search", (audio,), {"offsets": (5.,)}, .030) == .030
    assert costs.seconds("body", (audio,), {"offsets": (0.,)}, .040) == .040


def test_replay_projection_clock_reads_charged_work(tmp_path):
    replay = _RecoveryReplay("e07", tmp_path, scale=2.5, recovery=True,
                             measured_costs=_MeasuredCosts())
    with replay._charging():
        before = onair.time.perf_counter()
        replay.bench.spend(480)
        assert onair.time.perf_counter() - before == pytest.approx(.010)


def test_tape_inventory_includes_nested_wav_and_flac(tmp_path):
    (tmp_path / "nested").mkdir()
    for name in ("one.wav", "nested/two.flac", "nested/three.wav", "ignored.json"):
        (tmp_path / name).touch()
    assert [str(path.relative_to(tmp_path)) for path in _tapes(tmp_path)] == [
        "nested/three.wav", "nested/two.flac", "one.wav"]


def test_tape_replay_keeps_short_recording_and_keys_the_following_slot(tmp_path):
    audio = np.arange(512, dtype=np.float32) / 32768
    runs = []
    for recovery in (False, True):
        run = _TapeRecoveryReplay("short.wav", audio,
            tmp_path / str(recovery), scale=0, recovery=recovery).run()
        assert np.array_equal(run.bench.audio[46080:46080 + len(audio)], audio)
        assert run.placements
        assert any(row["stage"] == "tail" for row in run.work)
        runs.append(run)
    _compare_campaigns(*runs, tmp_path, "short.wav", 0)
