# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Operator-only, offline, versioned door installer.

Trusted curated builders execute as the operator; staging is not a sandbox.
Explicit fetch is separate from offline builds. No caller-facing installer,
shell, package runner, or live update service is provided.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from typing import Callable
import uuid
import zipfile

from .games.zmachine import _sync_dir, _write
from .store import OwnerLock, WorldLock


class InstallError(RuntimeError):
    pass


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(65536), b""):
            result.update(data)
    return result.hexdigest()


def _id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
        raise InstallError("invalid installation identifier")
    return value


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.parent / (".write-" + uuid.uuid4().hex)
    _write(temporary, json.dumps(value, sort_keys=True, indent=2).encode())
    os.replace(temporary, path)
    _sync_dir(path.parent)


def safe_extract(archive: Path, destination: Path, *, max_bytes: int = 67108864,
                 max_entries: int = 10000) -> None:
    """Reject links, devices, traversal and decompression expansion before writes."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    def target(name: str) -> Path:
        logical = PurePosixPath(name)
        if "\\" in name or logical.is_absolute() or ".." in logical.parts:
            raise InstallError("unsafe archive path")
        result = root.joinpath(*logical.parts)
        if not result.resolve().is_relative_to(root):
            raise InstallError("escaping archive path")
        return result
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            entries = handle.infolist()
            if len(entries) > max_entries or sum(x.file_size for x in entries) > max_bytes:
                raise InstallError("archive exceeds bounds")
            for entry in entries:
                mode = entry.external_attr >> 16
                if mode & 0o170000 not in (0, 0o040000, 0o100000):
                    raise InstallError("archive links and devices are unsupported")
                target(entry.filename)
            for entry in entries:
                path = target(entry.filename)
                if entry.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(entry) as source, path.open("xb") as output:
                        shutil.copyfileobj(source, output)
    else:
        with tarfile.open(archive) as handle:
            entries = handle.getmembers()
            if len(entries) > max_entries or sum(x.size for x in entries) > max_bytes:
                raise InstallError("archive exceeds bounds")
            for entry in entries:
                target(entry.name)
                if not (entry.isfile() or entry.isdir()):
                    raise InstallError("archive links and devices are unsupported")
            for entry in entries:
                path = target(entry.name)
                if entry.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with handle.extractfile(entry) as source, path.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    path.chmod(entry.mode & 0o777)


@dataclass(frozen=True)
class PinnedInput:
    name: str
    sha256: str
    url: str
    revision: str
    license: str

    def __post_init__(self):
        _id(self.name)
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("input requires a SHA256 pin")
        if not self.revision or not self.license:
            raise ValueError("input provenance and license are required")


@dataclass(frozen=True)
class Recipe:
    game: str
    revision: str
    profile: str
    inputs: tuple[PinnedInput, ...]
    artifacts: tuple[str, ...]
    builder: Callable[[Path, dict[str, Path]], dict]
    smoke: Callable[[Path, Path], None]
    validated: bool = True
    data_schema: str = "1"
    dependency_identity: str = "no external dependencies"

    def __post_init__(self):
        _id(self.game)
        _id(self.revision)
        if len({item.name for item in self.inputs}) != len(self.inputs):
            raise ValueError("duplicate pinned input")
        for artifact in self.artifacts:
            path = PurePosixPath(artifact)
            if path.is_absolute() or ".." in path.parts or "\\" in artifact:
                raise ValueError("unsafe artifact path")


class Installer:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.installs = self.root / "installations"

    @contextmanager
    def _mutation(self, game: str):
        _id(game)
        self.root.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            stack.enter_context(OwnerLock(self.root / "owner.lock"))
            world = self.root / "games" / game
            paths = {world / "world" / ".world.lock"}
            if world.exists():
                paths.update(world.rglob(".world.lock"))
            for path in sorted(paths, key=lambda value: str(value.resolve())):
                stack.enter_context(WorldLock(path))
            yield

    def _game(self, game: str) -> Path:
        return self.installs / _id(game)

    def inspect(self, game: str) -> dict:
        directory = self._game(game)
        if not directory.exists():
            return {"game": game, "enabled": False, "versions": []}
        control = self._control(game)
        return {"game": game, **control,
                "incomplete_update": (directory / "update-intent.json").exists(),
                "versions": sorted(p.name for p in directory.iterdir() if p.is_dir() and (p / "receipt.json").is_file())}

    def list(self) -> list[dict]:
        return [self.inspect(p.name) for p in sorted(self.installs.iterdir()) if p.is_dir()] if self.installs.exists() else []

    def _control(self, game: str) -> dict:
        path = self._game(game) / "control.json"
        return json.loads(path.read_text()) if path.exists() else {"enabled": False, "active": None}

    def fetch(self, recipe: Recipe) -> dict[str, Path]:
        """Explicit operator acquisition, separate from offline build and runtime.

        URLs and hashes come only from the curated recipe, never caller input.
        Failures preserve existing cache entries and installations.
        """
        from urllib.request import urlopen
        with self._mutation(recipe.game):
            cache = self.root / "input-cache" / recipe.game
            cache.mkdir(parents=True, exist_ok=True)
            result = {}
            for item in recipe.inputs:
                destination = cache / (item.name + "-" + item.sha256)
                if destination.is_file() and not destination.is_symlink() and digest(destination) == item.sha256:
                    result[item.name] = destination
                    continue
                if not item.url.startswith("https://"):
                    raise InstallError("curated acquisition requires HTTPS")
                temporary = cache / (".fetch-" + uuid.uuid4().hex)
                try:
                    count = 0
                    with urlopen(item.url, timeout=30) as response, temporary.open("xb") as output:
                        while True:
                            data = response.read(65536)
                            if not data:
                                break
                            count += len(data)
                            if count > 67108864:
                                raise InstallError("download exceeds input bound")
                            output.write(data)
                        output.flush()
                        os.fsync(output.fileno())
                    if digest(temporary) != item.sha256:
                        raise InstallError("download does not match curated input hash")
                    os.replace(temporary, destination)
                    _sync_dir(cache)
                except OSError as exc:
                    raise InstallError("pinned input acquisition failed") from exc
                finally:
                    temporary.unlink(missing_ok=True)
                result[item.name] = destination
            return result

    def install(self, recipe: Recipe, supplied: dict[str, Path]) -> str:
        with self._mutation(recipe.game):
            return self._install(recipe, supplied)

    def cached_inputs(self, recipe: Recipe) -> dict[str, Path]:
        """Resolve a previous explicit fetch; never acquire missing inputs."""
        result = {}
        for item in recipe.inputs:
            path = self.root / 'input-cache' / recipe.game / (item.name + '-' + item.sha256)
            if path.is_symlink() or not path.is_file() or digest(path) != item.sha256:
                raise InstallError('verified input cache is incomplete; run fetch or supply every --input')
            result[item.name] = path
        return result

    def _install(self, recipe: Recipe, supplied: dict[str, Path]) -> str:
        if set(supplied) != {item.name for item in recipe.inputs}:
            raise InstallError("supply every declared input and no undeclared inputs")
        directory = self._game(recipe.game)
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "update-intent.json").exists():
            raise InstallError("incomplete update requires offline recovery")
        identity = {"recipe_revision": recipe.revision, "profile": recipe.profile,
                    "inputs": {item.name: item.sha256 for item in recipe.inputs},
                    "dependencies": recipe.dependency_identity}
        installation = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]
        final = directory / installation
        if final.exists():
            self.verify(recipe.game, installation)
            return installation
        # Input bytes are copied and hashed again, so source replacement cannot
        # swap verified inputs between the verification and build steps.
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=directory) as temporary:
            stage = Path(temporary)
            inputs, size = {}, 0
            for item in recipe.inputs:
                source = Path(supplied[item.name])
                if source.is_symlink() or not source.is_file():
                    raise InstallError("input must be an operator-supplied regular file")
                size += source.stat().st_size
                if size > 67108864:
                    raise InstallError("input staging exceeds bounds")
                target = stage / item.name
                shutil.copyfile(source, target)
                if digest(target) != item.sha256:
                    raise InstallError("pinned input hash mismatch")
                inputs[item.name] = target
            build = stage / "output"
            build.mkdir(mode=0o700)
            tools = recipe.builder(build, inputs)
            artifacts = {}
            for name in recipe.artifacts:
                path = build / name
                if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(build):
                    raise InstallError("missing or unsafe build artifact")
                artifacts[name] = digest(path)
            # Never use an operator's existing players/world for smoke tests.
            with tempfile.TemporaryDirectory(prefix="smoke-", dir=stage) as smoke:
                recipe.smoke(build, Path(smoke))
            receipt = {**identity, "game": recipe.game, "installation": installation,
                       "data_schema": recipe.data_schema, "validated": recipe.validated,
                       "artifacts": artifacts, "toolchain": tools,
                       "smoke": "passed with fresh disposable state", "created": time.time(),
                       "sources": [item.__dict__ for item in recipe.inputs]}
            _write(build / "receipt.json", json.dumps(receipt, sort_keys=True, indent=2).encode())
            for path in build.rglob("*"):
                if path.is_file():
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
            _sync_dir(build)
            os.replace(build, final)
            _sync_dir(directory)
        return installation

    def verify(self, game: str, installation: str | None = None) -> dict:
        directory = self._game(game)
        installation = installation or self._control(game).get("active")
        if not installation:
            raise InstallError("no selected installation")
        folder = directory / _id(installation)
        receipt = json.loads((folder / "receipt.json").read_text())
        if receipt.get("installation") != installation or receipt.get("game") != game:
            raise InstallError("receipt identity mismatch")
        for name, expected in receipt["artifacts"].items():
            path = folder / name
            if not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink() or digest(path) != expected:
                raise InstallError("installed artifact identity changed")
        return receipt

    def enable(self, game: str, installation: str | None = None) -> None:
        with self._mutation(game):
            self._activate(game, installation or self._control(game).get("active"))

    def _activate(self, game: str, installation: str | None) -> None:
        directory = self._game(game)
        installation = installation or self._control(game).get("active")
        if (directory / "update-intent.json").exists():
            raise InstallError("incomplete update requires offline recovery")
        receipt = self.verify(game, installation)
        if not receipt["validated"]:
            raise InstallError("runtime profile has not passed compatibility validation")
        control = self._control(game)
        previous = control.get("active")
        if previous and previous != installation:
            previous_receipt = self.verify(game, previous)
            if previous_receipt["data_schema"] != receipt["data_schema"]:
                raise InstallError("unsupported data migration; activation refused")
        world = self.root / "games" / game
        backups = directory / "backups"
        backups.mkdir(exist_ok=True)
        backup_id = uuid.uuid4().hex
        backup = backups / backup_id
        if world.exists():
            shutil.copytree(world, backup, ignore=shutil.ignore_patterns(".world.lock"))
        else:
            backup.mkdir()
        for path in backup.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        _sync_dir(backup)
        _sync_dir(backups)
        # Intent remains on any failure through durable activation. Startup
        # refuses this game until explicit recovery restores the paired backup.
        atomic_json(directory / "update-intent.json", {
            "previous": previous, "target": installation, "backup": backup_id,
            "current_schema": self.verify(game, previous)["data_schema"] if previous else None,
            "target_schema": receipt["data_schema"]})
        atomic_json(directory / "control.json", {"active": installation, "enabled": True})
        (directory / "update-intent.json").unlink()
        _sync_dir(directory)

    def disable(self, game: str) -> None:
        with self._mutation(game):
            directory = self._game(game)
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / "control.json", {**self._control(game), "enabled": False})

    def update(self, recipe: Recipe, supplied: dict[str, Path]) -> str:
        with self._mutation(recipe.game):
            installation = self._install(recipe, supplied)
            self._activate(recipe.game, installation)
            return installation

    def remove(self, game: str, installation: str | None = None) -> None:
        with self._mutation(game):
            directory = self._game(game)
            if (directory / "update-intent.json").exists():
                raise InstallError("incomplete update requires offline recovery")
            control = self._control(game)
            selected = installation or control.get("active")
            if not selected:
                return
            _id(selected)
            if selected == control.get("active"):
                atomic_json(directory / "control.json", {"enabled": False, "active": None})
            shutil.rmtree(directory / selected)
            _sync_dir(directory)
            # Deliberately preserve root/games and the independent usage ledger.


def run_offline(argv: list[str], *, cwd: Path, timeout: int = 60) -> str:
    """Run only a curated compiler command with no package resolution.

    There is no generic external build manifest: these trusted recipes must be
    reviewed not to invoke downloaders. Proxy variables are defense in depth,
    not an OS network sandbox, and are never described as one.
    """
    environment = {"PATH": "/usr/bin:/bin", "HOME": str(cwd), "LANG": "C", "LC_ALL": "C",
        "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
        "npm_config_offline": "true", "PIP_NO_INDEX": "1", "CARGO_NET_OFFLINE": "true",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1"}
    try:
        result = subprocess.run(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise InstallError("offline build command failed") from exc
    if len(result.stdout) > 1048576:
        raise InstallError("build diagnostic bound exceeded")
    return result.stdout.decode("utf-8", "replace")


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Operator-only offline BBS game installation")
    parser.add_argument("--root", type=Path, required=True, help="canonical BBS state root (front end must be stopped)")
    parser.add_argument("operation", choices=("list", "inspect", "fetch", "install", "verify", "enable", "disable", "update", "remove"))
    parser.add_argument("game", nargs="?", default="zork")
    parser.add_argument("--installation")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=FILE")
    options = parser.parse_args(argv)
    installer = Installer(options.root)
    try:
        if options.operation == "list":
            result = installer.list()
        elif options.operation == "inspect":
            result = installer.inspect(options.game)
        elif options.operation in {"fetch", "install", "update"}:
            if options.game != "zork":
                raise InstallError("no validated recipe for this game; LORD candidate remains manual-only")
            from .recipes.zork import recipe
            supplied = {}
            for item in options.input:
                name, separator, path = item.partition("=")
                if not separator or name in supplied:
                    raise InstallError("inputs must be unique NAME=FILE pairs")
                supplied[name] = Path(path)
            if options.operation == "fetch":
                result = {name: str(path) for name, path in installer.fetch(recipe()).items()}
            else:
                if not supplied:
                    supplied = installer.cached_inputs(recipe())
                result = {"installation": getattr(installer, options.operation)(recipe(), supplied)}
        elif options.operation == "verify":
            result = installer.verify(options.game, options.installation)
        else:
            if options.operation in {"enable", "remove"}:
                getattr(installer, options.operation)(options.game, options.installation)
            else:
                installer.disable(options.game)
            result = installer.inspect(options.game)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (InstallError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"BBS installation refused: {exc}\n")


if __name__ == "__main__":
    from .install import main as entry
    raise SystemExit(entry())
