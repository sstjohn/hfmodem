# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Actual MIT Zork I + Frotz recipe, all acquisition performed by operator.

No payloads or fetched dependencies ship in hfmodem. This recipe accepts only
these exact local inputs; Frotz's dumb target has no external build libraries.
"""
import hashlib
from pathlib import Path
import shutil
import time

from ..games.zmachine import PROFILE, ZMachineAdapter
from ..install import InstallError, PinnedInput, Recipe, digest, run_offline, safe_extract
from . import frotz_hook

FROTZ_REVISION = "fe66e2ef738354f5e2fe37c7c01b1781ef659acd"
ZORK_REVISION = "97b7b3d68c075dd9af7da499c3e9690ada3471fd"
INPUTS = (
    PinnedInput("frotz-source", "14cd7804f37afd4a63a6bb76d5ff11768ed71525d634877148037721886a3434",
        f"https://codeload.github.com/DavidGriffith/frotz/tar.gz/{FROTZ_REVISION}", FROTZ_REVISION, "GPL-2.0-or-later; COPYING retained"),
    PinnedInput("zork-story", "37084966477dff679282de42974b2077156b1bd68fad92a65d4ea94d8eb64d79",
        f"https://raw.githubusercontent.com/historicalsource/zork1/{ZORK_REVISION}/COMPILED/zork1.z3", ZORK_REVISION, "MIT; no trademark or packaging grant"),
    PinnedInput("zork-license", "98a96c38494df4963951e32b81ec4effebe4c8a812ab8d9e7d185d59e86fe2fc",
        f"https://raw.githubusercontent.com/historicalsource/zork1/{ZORK_REVISION}/LICENSE", ZORK_REVISION, "MIT notice retained"),
)


def build(output: Path, inputs: dict[str, Path]) -> dict:
    version = run_offline(["/usr/bin/cc", "--version"], cwd=output)
    if "Apple clang version 21.0.0 (clang-2100.1.1.101)" not in version.splitlines():
        raise InstallError("this validated recipe accepts Apple clang 21.0.0 only")
    source_root = output.parent / "source"
    safe_extract(inputs["frotz-source"], source_root)
    source = source_root / ("frotz-" + FROTZ_REVISION)
    frotz_hook.apply(source)
    # No package manager, submodule, library discovery, downloads or install step.
    # Exact make target only compiles bundled C and links the system C library.
    log = run_offline(["/usr/bin/make", "-j1", "dfrotz", "SOUND=none", "CC=/usr/bin/cc"], cwd=source)
    shutil.copyfile(source / "dfrotz", output / "dfrotz")
    (output / "dfrotz").chmod(0o755)
    shutil.copyfile(inputs["zork-story"], output / "zork1.z3")
    shutil.copyfile(inputs["zork-license"], output / "ZORK-LICENSE")
    shutil.copyfile(source / "COPYING", output / "FROTZ-COPYING")
    (output / "build.log").write_text(log)
    (output / "HF-FROTZ-HOOK.py").write_bytes(Path(frotz_hook.__file__).read_bytes())
    return {"compiler": version, "platform": "macOS arm64", "make": "/usr/bin/make -j1", "external_dependencies": []}


def smoke(output: Path, state: Path) -> None:
    adapter = ZMachineAdapter(output / "dfrotz", output / "zork1.z3",
        runtime_sha256=digest(output / "dfrotz"), story_sha256=digest(output / "zork1.z3"))
    events = []
    def until(kind):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            batch = adapter.poll()
            events.extend(batch)
            if any(event.kind == kind for event in batch):
                return
            if any(event.kind == "exited" for event in batch):
                raise InstallError("Zork exited during fresh-state smoke")
            time.sleep(.01)
        raise InstallError("Zork smoke timed out")
    try:
        adapter.start("smoke-player", state / "player")
        until("prompt")
        if "West of House" not in "".join(event.text for event in events):
            raise InstallError("Zork startup smoke failed")
        if not adapter.submit("open mailbox"):
            raise InstallError("Zork input smoke failed")
        until("prompt")
        if not adapter.checkpoint():
            raise InstallError("Zork checkpoint smoke failed")
        until("prompt")
        if not any(event.kind == "saved" for event in events):
            raise InstallError("Zork save did not publish")
        if not adapter.submit("north"):
            raise InstallError("Zork movement smoke failed")
        until("prompt")
        if not adapter.restore():
            raise InstallError("Zork restore smoke failed")
        until("prompt")
        if not any(event.kind == "restored" for event in events):
            raise InstallError("Zork restore did not complete")
    finally:
        adapter.close("smoke complete")
        deadline = time.monotonic() + 3
        while adapter.alive and time.monotonic() < deadline:
            time.sleep(.01)
        if adapter.alive:
            raise InstallError("smoke backend did not finish cleanup")


def recipe() -> Recipe:
    hook_identity = hashlib.sha256(Path(frotz_hook.__file__).read_bytes()).hexdigest()
    return Recipe("zork", "1", PROFILE, INPUTS,
        ("dfrotz", "zork1.z3", "ZORK-LICENSE", "FROTZ-COPYING", "HF-FROTZ-HOOK.py", "build.log"),
        build, smoke, dependency_identity="bundled C only; Apple clang 21.0.0; hook=" + hook_identity)
