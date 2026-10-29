# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import hashlib
import io
import json
import tarfile
import zipfile

import pytest

from hfmodem.bbs.install import InstallError, Installer, PinnedInput, Recipe, safe_extract
from hfmodem.bbs.store import OwnerLock, OwnershipError, WorldLock


def make_recipe(tmp_path, *, revision="1", schema="1", validated=True, fail=None):
    supplied = tmp_path / "source"
    supplied.write_bytes(b"known pinned source")
    item = PinnedInput("source", hashlib.sha256(supplied.read_bytes()).hexdigest(), "https://example.invalid/fixed-release", "commit1", "MIT")
    def build(output, inputs):
        if fail == "build":
            raise InstallError("missing dependency; frozen/offline")
        (output / "game").write_bytes(inputs["source"].read_bytes())
        return {"test": "fixture"}
    def smoke(output, state):
        assert not list(state.iterdir())
        (state / "fresh-state").write_text("discard")
        if fail == "smoke":
            raise InstallError("failed fresh smoke")
    return Recipe("fixture", revision, "test-fixture-not-real-game", (item,), ("game",), build, smoke,
        data_schema=schema, validated=validated), {"source": supplied}


def test_install_verify_enable_disable_remove_preserves_player_data(tmp_path):
    installer = Installer(tmp_path / "bbs")
    recipe, inputs = make_recipe(tmp_path)
    version = installer.install(recipe, inputs)
    assert not installer.inspect("fixture")["enabled"]
    assert installer.verify("fixture", version)["smoke"].startswith("passed with fresh")
    installer.enable("fixture", version)
    assert installer.inspect("fixture")["enabled"]
    state = installer.root / "games/fixture/world/player.dat"
    state.write_text("durable player")
    installer.disable("fixture")
    assert not installer.inspect("fixture")["enabled"]
    installer.remove("fixture")
    assert state.read_text() == "durable player"
    assert not (installer.installs / "fixture" / version).exists()


def test_offline_cache_resolution_never_fetches_missing_or_corrupt_inputs(tmp_path, monkeypatch):
    installer = Installer(tmp_path / 'bbs')
    recipe, supplied = make_recipe(tmp_path)
    monkeypatch.setattr(installer, 'fetch', lambda *_: pytest.fail('offline cache attempted network acquisition'))
    with pytest.raises(InstallError, match='cache is incomplete'):
        installer.cached_inputs(recipe)
    item = recipe.inputs[0]
    cache = installer.root / 'input-cache' / recipe.game / (item.name + '-' + item.sha256)
    cache.parent.mkdir(parents=True)
    cache.write_bytes(supplied['source'].read_bytes())
    assert installer.cached_inputs(recipe) == {'source': cache}
    cache.write_bytes(b'corrupt')
    with pytest.raises(InstallError, match='cache is incomplete'):
        installer.cached_inputs(recipe)


@pytest.mark.parametrize("failure", ["build", "smoke"])
def test_failed_build_or_smoke_does_not_replace_current(tmp_path, failure):
    installer = Installer(tmp_path / "bbs")
    original, inputs = make_recipe(tmp_path)
    version = installer.install(original, inputs)
    installer.enable("fixture", version)
    changed, inputs = make_recipe(tmp_path, revision="2", fail=failure)
    with pytest.raises(InstallError):
        installer.update(changed, inputs)
    assert installer.inspect("fixture")["active"] == version
    assert installer.inspect("fixture")["enabled"]


def test_corrupted_or_missing_inputs_refused_before_builder(tmp_path):
    installer = Installer(tmp_path / "bbs")
    recipe, inputs = make_recipe(tmp_path)
    with pytest.raises(InstallError, match="every declared"):
        installer.install(recipe, {})
    inputs["source"].write_bytes(b"tampered")
    with pytest.raises(InstallError, match="hash"):
        installer.install(recipe, inputs)
    assert installer.inspect("fixture")["versions"] == []


def test_optional_unvalidated_profile_cannot_enable(tmp_path):
    installer = Installer(tmp_path / "bbs")
    recipe, inputs = make_recipe(tmp_path, validated=False)
    version = installer.install(recipe, inputs)
    with pytest.raises(InstallError, match="not passed"):
        installer.enable("fixture", version)


@pytest.mark.parametrize("lock_type", ["owner", "world"])
def test_installer_refuses_frontend_or_surviving_writer(tmp_path, lock_type):
    installer = Installer(tmp_path / "bbs")
    recipe, inputs = make_recipe(tmp_path)
    lock = OwnerLock(installer.root / "owner.lock") if lock_type == "owner" else WorldLock(installer.root / "games/fixture/world/.world.lock")
    with lock:
        with pytest.raises(OwnershipError):
            installer.install(recipe, inputs)


def test_artifact_corruption_detected(tmp_path):
    installer = Installer(tmp_path / "bbs")
    recipe, inputs = make_recipe(tmp_path)
    version = installer.install(recipe, inputs)
    (installer.installs / "fixture" / version / "game").write_bytes(b"tampered")
    with pytest.raises(InstallError, match="identity"):
        installer.verify("fixture", version)


def test_schema_change_refused_without_mutating_world(tmp_path):
    installer = Installer(tmp_path / "bbs")
    recipe, inputs = make_recipe(tmp_path)
    version = installer.install(recipe, inputs)
    installer.enable("fixture", version)
    state = installer.root / "games/fixture/world/player"
    state.write_text("previous world")
    changed, inputs = make_recipe(tmp_path, revision="2", schema="2")
    with pytest.raises(InstallError, match="migration"):
        installer.update(changed, inputs)
    assert installer.inspect("fixture")["active"] == version
    assert state.read_text() == "previous world"


def test_interrupted_activation_keeps_intent_and_disables_startup(tmp_path, monkeypatch):
    import hfmodem.bbs.install as module
    installer = Installer(tmp_path / "bbs")
    recipe, inputs = make_recipe(tmp_path)
    version = installer.install(recipe, inputs)
    original = module.atomic_json
    def fail(path, value):
        if path.name == "control.json":
            raise OSError("interrupted publication")
        original(path, value)
    monkeypatch.setattr(module, "atomic_json", fail)
    with pytest.raises(OSError):
        installer.enable("fixture", version)
    assert installer.inspect("fixture")["incomplete_update"]
    with pytest.raises(InstallError, match="recovery"):
        installer.enable("fixture", version)
    intent = json.loads((installer.installs / "fixture/update-intent.json").read_text())
    assert intent["target"] == version
    assert (installer.installs / "fixture/backups" / intent["backup"]).is_dir()


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "hardlink", "device"])
def test_archive_escape_and_special_files_rejected(tmp_path, kind):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as handle:
        entry = tarfile.TarInfo({"traversal": "../escape", "absolute": "/tmp/escape"}.get(kind, "unsafe"))
        if kind == "symlink":
            entry.type, entry.linkname = tarfile.SYMTYPE, "../../escape"
        elif kind == "hardlink":
            entry.type, entry.linkname = tarfile.LNKTYPE, "../../escape"
        elif kind == "device":
            entry.type = tarfile.CHRTYPE
        handle.addfile(entry)
    with pytest.raises(InstallError):
        safe_extract(archive, tmp_path / "out")


def test_zip_expansion_and_backslash_paths_rejected(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("ok", b"x" * 100)
    with pytest.raises(InstallError, match="bounds"):
        safe_extract(archive, tmp_path / "one", max_bytes=99)
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("..\\escape", b"x")
    with pytest.raises(InstallError, match="path"):
        safe_extract(archive, tmp_path / "two")


def test_safe_regular_archive_extracted(tmp_path):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as handle:
        entry = tarfile.TarInfo("directory/file")
        entry.size = 5
        handle.addfile(entry, io.BytesIO(b"hello"))
    safe_extract(archive, tmp_path / "output")
    assert (tmp_path / "output/directory/file").read_bytes() == b"hello"


def test_undeclared_dependency_input_is_not_accepted(tmp_path):
    installer = Installer(tmp_path / "bbs")
    recipe, inputs = make_recipe(tmp_path)
    with pytest.raises(InstallError, match="undeclared"):
        installer.install(recipe, {**inputs, "download-at-build": inputs["source"]})


def test_explicit_fetch_verifies_hash_and_reuses_cache_offline(tmp_path, monkeypatch):
    import urllib.request
    recipe, inputs = make_recipe(tmp_path)
    installer = Installer(tmp_path / "bbs")
    calls = []
    def request(url, timeout):
        calls.append(url)
        return io.BytesIO(inputs["source"].read_bytes())
    monkeypatch.setattr(urllib.request, "urlopen", request)
    fetched = installer.fetch(recipe)
    assert fetched["source"].read_bytes() == inputs["source"].read_bytes()
    assert installer.fetch(recipe) == fetched
    assert len(calls) == 1


def test_corrupt_fetch_never_publishes_input(tmp_path, monkeypatch):
    import urllib.request
    recipe, _ = make_recipe(tmp_path)
    installer = Installer(tmp_path / "bbs")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(InstallError, match="hash"):
        installer.fetch(recipe)
    assert not list((installer.root / "input-cache/fixture").iterdir())


def test_compatible_update_keeps_backup_and_previous_artifacts(tmp_path):
    installer = Installer(tmp_path / "bbs")
    original, inputs = make_recipe(tmp_path)
    previous = installer.install(original, inputs)
    installer.enable("fixture", previous)
    state = installer.root / "games/fixture/world/player"
    state.write_text("saved player")
    changed, inputs = make_recipe(tmp_path, revision="2")
    current = installer.update(changed, inputs)
    assert current != previous
    assert installer.inspect("fixture")["active"] == current
    assert installer.verify("fixture", previous)["installation"] == previous
    assert state.read_text() == "saved player"
    assert any(path.read_text() == "saved player" for path in (installer.installs / "fixture/backups").rglob("player"))


def test_crash_after_activation_does_not_clear_required_recovery(tmp_path, monkeypatch):
    from pathlib import Path
    installer = Installer(tmp_path / "bbs")
    recipe, inputs = make_recipe(tmp_path)
    version = installer.install(recipe, inputs)
    original = Path.unlink
    def fail(path, *args, **kwargs):
        if path.name == "update-intent.json":
            raise OSError("crash after activation")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail)
    with pytest.raises(OSError):
        installer.enable("fixture", version)
    status = installer.inspect("fixture")
    assert status["active"] == version
    assert status["incomplete_update"]
    with pytest.raises(InstallError, match="recovery"):
        installer.enable("fixture", version)
