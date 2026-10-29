# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Managed Z-machine profile for the narrowly hooked Frotz recipe.

Only the patched input/file hooks establish readiness; stdout quiet never does.
Checkpoints are immutable payload+metadata generations selected by one manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from .base import GameEvent
from .door import InputProfile, ProcessAdapter

PROFILE = "dfrotz-hfmodem-v1"


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _quetzal(data: bytes) -> None:
    if len(data) < 12 or len(data) > 16777216 or data[:4] != b"FORM" or data[8:12] != b"IFZS":
        raise ValueError("invalid Quetzal checkpoint")
    if int.from_bytes(data[4:8], "big") + 8 != len(data):
        raise ValueError("incomplete Quetzal checkpoint")
    offset, chunks = 12, set()
    while offset + 8 <= len(data):
        chunks.add(data[offset:offset+4])
        size = int.from_bytes(data[offset+4:offset+8], "big")
        offset += 8 + size + (size % 2)
    if offset != len(data) or not {b"IFhd", b"Stks"} <= chunks or not ({b"CMem", b"UMem"} & chunks):
        raise ValueError("missing Quetzal state")


class Checkpoints:
    def __init__(self, directory: Path, identity: dict[str, str]):
        self.root = directory / "checkpoints"
        self.identity = identity

    def publish(self, source: Path) -> str:
        if source.is_symlink() or not source.is_file():
            raise ValueError("checkpoint is not a regular file")
        payload = source.read_bytes()
        _quetzal(payload)
        if self.root.is_symlink():
            raise ValueError("unsafe checkpoint root")
        self.root.mkdir(exist_ok=True)
        generation = uuid.uuid4().hex
        folder = self.root / generation
        folder.mkdir(mode=0o700)
        _write(folder / "payload.qzl", payload)
        metadata = {**self.identity, "sha256": hashlib.sha256(payload).hexdigest()}
        _write(folder / "metadata.json", json.dumps(metadata, sort_keys=True).encode())
        _sync_dir(folder)
        _sync_dir(self.root)
        previous = None
        try:
            previous = self._manifest()["current"]
        except FileNotFoundError:
            pass
        manifest = {"current": generation, "previous": previous}
        temporary = self.root / (".manifest-" + generation)
        _write(temporary, json.dumps(manifest, sort_keys=True).encode())
        os.replace(temporary, self.root / "manifest.json")
        _sync_dir(self.root)
        return generation

    def _manifest(self) -> dict:
        if self.root.is_symlink():
            raise ValueError("unsafe checkpoint root")
        path = self.root / "manifest.json"
        if path.is_symlink():
            raise ValueError("unsafe checkpoint manifest")
        return json.loads(path.read_text())

    def selected(self) -> tuple[str, Path]:
        generation = self._manifest()["current"]
        if not isinstance(generation, str) or len(generation) != 32 or any(c not in "0123456789abcdef" for c in generation):
            raise ValueError("unsafe checkpoint generation")
        folder = self.root / generation
        if folder.is_symlink() or (folder / "payload.qzl").is_symlink() or (folder / "metadata.json").is_symlink():
            raise ValueError("unsafe checkpoint files")
        metadata = json.loads((folder / "metadata.json").read_text())
        if any(metadata.get(key) != value for key, value in self.identity.items()):
            raise ValueError("checkpoint belongs to another runtime or story")
        payload = (folder / "payload.qzl").read_bytes()
        _quetzal(payload)
        if metadata.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError("checkpoint hash mismatch")
        return generation, folder / "payload.qzl"


class ZMachineAdapter(ProcessAdapter):
    capabilities = frozenset({"checkpoint", "restore", "structured_prompts"})

    def __init__(self, executable: Path, story: Path, *, runtime_sha256: str,
                 story_sha256: str):
        self.story = Path(story).resolve()
        self.executable = Path(executable).resolve()
        self.identity = {"profile": PROFILE, "runtime_sha256": runtime_sha256,
                         "story_sha256": story_sha256}
        super().__init__((str(self.executable), "-m", "-p", "-w", "80", "-h", "100", str(self.story)),
            profile=InputProfile(structured=True, max_input_bytes=78), env={"HF_BBS_HOOK": "1"})
        self._operation: str | None = None
        self._save_id: str | None = None

    def _prepare(self) -> None:
        for path, expected in ((self.story, self.identity["story_sha256"]),
                               (self.executable, self.identity["runtime_sha256"])):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("installed game input identity changed")
        assert self.directory is not None
        self.checkpoints = Checkpoints(self.directory, self.identity)

    def submit(self, text: str, mode: str = "default") -> bool:
        # Frotz backslash commands include debug/hotkeys and embedded macros.
        if "\\" in text or mode not in {"line", "default", "key", "pause"}:
            return False
        if self._mode == "key" and mode == "default":
            mode = "key"
        # Dumb Frotz collects even a single requested key as one pipe line.
        if mode == "key":
            if len(text) != 1:
                return False
            return super().submit(text, "line")
        return super().submit(text, mode)

    def checkpoint(self) -> bool:
        if self._mode != "line" or self._operation:
            return False
        self._operation = "save"
        if not self.submit("save"):
            self._operation = None
            return False
        return True

    def restore(self) -> bool:
        if self._mode != "line" or self._operation:
            return False
        self._operation = "restore"
        if not self.submit("restore"):
            self._operation = None
            return False
        return True

    def _control(self, line: str) -> None:
        assert self.directory is not None
        if line in {"HF_BBS line", "HF_BBS key"}:
            self._prompt(line.split()[-1])
        elif line == "HF_BBS save-file":
            self._operation = "save"
            pending = self.directory / "pending.qzl"
            pending.unlink(missing_ok=True)
            self._commands.put_nowait(b"pending.qzl\n")
        elif line == "HF_BBS restore-file":
            self._operation = "restore"
            try:
                self._save_id, source = self.checkpoints.selected()
                target = self.directory / "restore.qzl"
                target.unlink(missing_ok=True)
                shutil.copyfile(source, target)
                self._commands.put_nowait(b"restore.qzl\n")
            except (ValueError, FileNotFoundError, OSError):
                self._commands.put_nowait(b"\n")  # hook cancels empty filename
                self._emit(GameEvent("error", "No compatible saved checkpoint is available."))
        elif line == "HF_BBS saved":
            try:
                generation = self.checkpoints.publish(self.directory / "pending.qzl")
                self._emit(GameEvent("saved", "Checkpoint saved.", checkpoint_id=generation))
            except (OSError, ValueError):
                self._emit(GameEvent("error", "Checkpoint publication failed; previous save retained."))
            self._operation = None
        elif line == "HF_BBS restored":
            self._emit(GameEvent("restored", "Checkpoint restored.", checkpoint_id=self._save_id))
            self._operation = None
        elif line in {"HF_BBS save-failed", "HF_BBS restore-failed"}:
            self._emit(GameEvent("error", "The game could not complete the checkpoint operation."))
            self._operation = None
