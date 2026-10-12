# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The recovered pins require intact tracked audio, with no duplicated overlap."""
import errno
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.tests import evidence
from hfmodem.tests.shrike import corpora


@pytest.mark.parametrize('allow_module_level', [False, True])
def test_optional_input_absence_names_the_path(tmp_path, allow_module_level):
    path = tmp_path / 'missing.flac'
    with pytest.raises(pytest.skip.Exception) as caught:
        evidence.require_input(path, allow_module_level=allow_module_level)
    assert str(path) in str(caught.value)
    assert caught.value.allow_module_level is allow_module_level


def test_present_optional_input_keeps_decode_and_assertion_failures(tmp_path):
    path = tmp_path / 'probe.json'
    path.write_text('broken JSON')
    assert evidence.require_input(path) == path
    with pytest.raises(json.JSONDecodeError):
        json.loads(evidence.require_input(path).read_text())
    with pytest.raises(AssertionError, match='recording regression'):
        evidence.require_input(path)
        raise AssertionError('recording regression')


@pytest.mark.parametrize('suffix', ['.wav', '.flac'])
def test_optional_input_accepts_either_stored_audio_spelling(tmp_path, suffix):
    path = tmp_path / ('recording' + suffix)
    path.write_bytes(b'damaged recording')
    other = path.with_suffix('.flac' if suffix == '.wav' else '.wav')
    assert evidence.require_input(other) == path


@pytest.mark.parametrize('suffix', ['.json', '.wav', '.flac'])
@pytest.mark.parametrize('damage', ['parent-file', 'symlink-loop', 'permission'])
def test_optional_input_filesystem_errors_fail(tmp_path, monkeypatch, suffix, damage):
    path = tmp_path / ('recording' + suffix)
    if damage == 'parent-file':
        parent = tmp_path / 'parent'
        parent.write_text('not a directory')
        path = parent / path.name
        error = NotADirectoryError
    elif damage == 'symlink-loop':
        path.symlink_to(path)
        error = OSError
    else:
        path.write_bytes(b'present recording')
        stat = os.stat

        def denied(filename, *args, **kwargs):
            if os.fspath(filename) == str(path):
                raise PermissionError(errno.EACCES, 'permission denied', str(path))
            return stat(filename, *args, **kwargs)

        monkeypatch.setattr(os, 'stat', denied)
        error = PermissionError
    try:
        with pytest.raises(error) as caught:
            evidence.require_input(path)
    except pytest.skip.Exception as skipped:
        pytest.fail(f'filesystem error must fail, not skip: {skipped}')
    assert caught.value.filename == str(path)
    if damage == 'symlink-loop':
        assert caught.value.errno == errno.ELOOP


def test_optional_input_does_not_skip_absence_of_another_path(tmp_path, monkeypatch):
    path = tmp_path / 'present.json'
    path.write_text('{}')
    missing = tmp_path / 'other.json'
    stat = os.stat

    def unrelated_absence(filename, *args, **kwargs):
        if os.fspath(filename) == str(path):
            raise FileNotFoundError(errno.ENOENT, 'missing other input', str(missing))
        return stat(filename, *args, **kwargs)

    monkeypatch.setattr(os, 'stat', unrelated_absence)
    try:
        with pytest.raises(FileNotFoundError) as caught:
            evidence.require_input(path)
    except pytest.skip.Exception as skipped:
        pytest.fail(f'unrelated missing input must fail, not skip: {skipped}')
    assert caught.value.filename == str(missing)


@pytest.mark.parametrize('suffix', ['.wav', '.flac'])
@pytest.mark.parametrize('damage', ['symlink-loop', 'permission', 'unrelated-absence'])
def test_optional_input_alternate_filesystem_errors_fail(
        tmp_path, monkeypatch, suffix, damage):
    alternate = tmp_path / ('recording' + suffix)
    path = alternate.with_suffix('.flac' if suffix == '.wav' else '.wav')
    filename = alternate
    if damage == 'symlink-loop':
        alternate.symlink_to(alternate)
        error, number = OSError, errno.ELOOP
    else:
        alternate.write_bytes(b'present recording')
        if damage == 'permission':
            error, number = PermissionError, errno.EACCES
        else:
            filename = tmp_path / 'unrelated.wav'
            error, number = FileNotFoundError, errno.ENOENT
        stat = os.stat

        def faulty(filename_arg, *args, **kwargs):
            if os.fspath(filename_arg) == str(alternate):
                raise error(number, 'alternate path fault', str(filename))
            return stat(filename_arg, *args, **kwargs)

        monkeypatch.setattr(os, 'stat', faulty)
    try:
        with pytest.raises(error) as caught:
            evidence.require_input(path)
    except pytest.skip.Exception as skipped:
        pytest.fail(f'alternate filesystem error must fail, not skip: {skipped}')
    assert caught.value.filename == str(filename)
    assert caught.value.errno == number


@pytest.mark.parametrize('present', [False, True], ids=['absent-tool', 'broken-tool'])
def test_chirp_sweep_skips_only_an_absent_tool(tmp_path, monkeypatch, present):
    source = Path(__file__).with_name('test_p4_chirp_sweep_1001.py')
    copied = tmp_path / 'packages/hfmodem/hfmodem/tests/shrike' / source.name
    copied.parent.mkdir(parents=True)
    copied.write_text(source.read_text())
    tools = tmp_path / 'tools'
    tools.mkdir()
    tool = tools / 'witness_align.py'
    if present:
        tool.write_text('import missing_witness_dependency_for_regression\n')
    monkeypatch.syspath_prepend(str(tools))
    monkeypatch.delitem(sys.modules, 'witness_align', raising=False)
    if present:
        try:
            with pytest.raises(ModuleNotFoundError) as caught:
                runpy.run_path(str(copied))
        except pytest.skip.Exception as skipped:
            pytest.fail(f'a present tool with a broken import must fail: {skipped}')
        assert caught.value.name == 'missing_witness_dependency_for_regression'
    else:
        with pytest.raises(pytest.skip.Exception) as caught:
            runpy.run_path(str(copied))
        assert str(tool) in str(caught.value)
        assert caught.value.allow_module_level


@pytest.mark.parametrize('sweep', ['ladder', 'padded-transform'])
@pytest.mark.parametrize('missing', ['metadata', 'campaign-wav'])
def test_corpus_sweep_disagreement_fails_without_campaign_inputs(
        tmp_path, monkeypatch, sweep, missing):
    slot = importlib.import_module('hfmodem.tests.shrike.test_slot_deadline')
    offset = importlib.import_module('hfmodem.tests.shrike.test_offset_fine_estimate')
    campaign = tmp_path / 'slot-deadline-0910'
    campaign.mkdir()
    metadata = campaign / 'metadata.json'
    if missing == 'campaign-wav':
        metadata.write_text(json.dumps({'fixtures': [{'file': 'missing.wav'}]}))
    monkeypatch.setattr(slot, 'FIXTURES', campaign)
    monkeypatch.setattr(slot, 'METADATA', metadata)
    monkeypatch.setattr(slot, 'crops', slot.crops.__wrapped__)
    monkeypatch.setattr(offset, 'FIXTURES', tmp_path)
    wavfile.write(tmp_path / 'present.wav', slot.onair.FS,
                  np.zeros(80 * slot.onair.FS, dtype=np.int16))
    attribute = '_bands' if sweep == 'ladder' else 'compensate'
    original = getattr(slot.p3acquire, attribute)
    candidate = SimpleNamespace(
        event=SimpleNamespace(cs=1, start=0, t=0., packet=None, text='control'),
        offset_hz=0., coarse_hz=0., quality=1.)
    calls = [0, 0]

    def disagree(audio):
        baseline = getattr(slot.p3acquire, attribute) is not original
        calls[baseline] += 1
        return candidate if baseline else None

    monkeypatch.setattr(slot.p3acquire, 'control_signal', disagree)
    monkeypatch.setattr(slot.p3acquire, 'changeover', disagree)
    try:
        with pytest.raises(AssertionError, match='present.wav'):
            if sweep == 'ladder':
                slot.test_the_ladder_finds_the_same_words_over_every_fixture_in_the_tree()
            else:
                offset.test_the_padded_transform_finds_the_same_words_over_the_whole_corpus(monkeypatch)
    except pytest.skip.Exception as skipped:
        pytest.fail(f'a disagreement on present audio must fail, not skip: {skipped}')
    assert min(calls) > 500, calls


def test_n9vc_join_preserves_source_samples_and_origin():
    fs, samples = corpora.n9vc_0802()
    shorts = evidence.checked_samples(
        corpora.FIXTURES / 'n9vc-0802-robust-shorts.wav', corpora.FIXTURES / 'n9vc-0802-robust-shorts.json')
    chirps = evidence.checked_samples(
        corpora.FIXTURES / 'n9vc-0802-chirps.wav', corpora.FIXTURES / 'n9vc-0802-chirps.json')
    assert fs == 48000 and len(samples) == 844800
    np.testing.assert_array_equal(samples[:312000], shorts)
    np.testing.assert_array_equal(samples[307200:], chirps)


def test_missing_tracked_n9vc_fixture_fails(tmp_path, monkeypatch):
    for name in corpora.N9VC_NAMES:
        sidecar = (corpora.FIXTURES / name).with_suffix('.json')
        (tmp_path / sidecar.name).write_text(sidecar.read_text())
    monkeypatch.setattr(corpora, 'FIXTURES', tmp_path)
    try:
        with pytest.raises(AssertionError, match='Missing tracked evidence'):
            corpora.n9vc_0802()
    except pytest.skip.Exception as skipped:
        pytest.fail(f"missing tracked evidence must fail, not skip: {skipped}")


@pytest.mark.parametrize('loop', [False, True], ids=['absent', 'symlink-loop'])
@pytest.mark.parametrize('external', [True, False], ids=['external', 'tracked'])
@pytest.mark.parametrize('manifest_present', [False, True], ids=['no-manifest', 'manifest'])
def test_missing_evidence_is_reported_before_opening(
        tmp_path, monkeypatch, loop, external, manifest_present):
    (tmp_path / ".git").mkdir()
    (tmp_path / "publish").mkdir()
    (tmp_path / "publish" / "manifest.toml").write_text("# development source")
    monkeypatch.setattr(evidence, "TREE", tmp_path)
    path = tmp_path / 'missing.wav'
    manifest = tmp_path / ('external.json' if external else 'provenance.json')
    if manifest_present:
        manifest.write_text(json.dumps({'files': [dict(
            file=path.name, bytes=0, sha256='0'*64, sample_rate=48000,
            sample_count=0, channels=1, bits_per_sample=16)]}))
    if loop:
        path.symlink_to(path)
    error = pytest.skip.Exception if external and manifest_present else AssertionError
    try:
        with pytest.raises(error, match='(?i)missing .*evidence|Missing evidence manifest') as caught:
            evidence.checked_samples(path, manifest)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"missing manifest or tracked evidence must fail, not skip: {skipped}")
    assert str(path) in str(caught.value) and str(manifest) in str(caught.value)


@pytest.mark.parametrize('damage', ['size', 'sha256', 'manifest'])
def test_tracked_evidence_damage_fails_before_decode(tmp_path, monkeypatch, damage):
    from hfmodem.core import wav

    stream = io.BytesIO()
    wavfile.write(stream, 48000, np.arange(64, dtype=np.int16))
    raw = stream.getvalue()
    path = tmp_path / 'recording.wav'
    manifest = tmp_path / 'manifest.json'
    if damage == 'size':
        damaged = raw[:-1]
    elif damage == 'sha256':
        damaged = raw[:-1] + bytes([raw[-1] ^ 1])
    else:
        damaged = raw
    path.write_bytes(damaged)
    if damage != 'manifest':
        manifest.write_text(json.dumps({'files': [dict(
            file=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
            sample_rate=48000, sample_count=64, channels=1, bits_per_sample=16)]}))

    def must_not_decode(*args):
        pytest.fail('unverified evidence reached the decoder')

    monkeypatch.setattr(wav, 'source', must_not_decode)
    message = {'size': 'size mismatch', 'sha256': 'SHA-256 mismatch',
               'manifest': 'Missing evidence manifest'}[damage]
    try:
        with pytest.raises(AssertionError, match=message):
            evidence.checked_samples(path, manifest)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"tracked evidence damage must fail, not skip: {skipped}")
