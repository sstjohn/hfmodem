# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Startup validation and operator-configured factories for installed profiles."""
from dataclasses import dataclass
from pathlib import Path

from .zmachine import Checkpoints, PROFILE, ZMachineAdapter


@dataclass(frozen=True)
class ZorkFactory:
    executable: Path
    story: Path
    runtime_sha256: str
    story_sha256: str
    title: str = "Zork I"
    available: bool = True

    def __call__(self):
        return ZMachineAdapter(self.executable, self.story,
            runtime_sha256=self.runtime_sha256, story_sha256=self.story_sha256)

    def saved_game(self, directory: Path) -> bool:
        """Read-only probe, called on the core's filesystem worker.

        Missing manifest means no save; selected-but-missing, corrupt or
        incompatible checkpoint files are errors, never a silent new game.
        """
        checkpoints = Checkpoints(directory, {"profile": PROFILE,
            "runtime_sha256": self.runtime_sha256, "story_sha256": self.story_sha256})
        if checkpoints.root.is_symlink():
            raise ValueError("unsafe checkpoint root")
        try:
            (checkpoints.root / "manifest.json").lstat()
        except FileNotFoundError:
            return False
        checkpoints.selected()
        return True


def configured_games(config) -> dict:
    """Validate receipt/artifacts during startup; never auto-install or fetch.

    Missing optional games disable that game only. Arbitrary config executables
    cannot self-assert a validated profile: availability comes from the BBS's
    offline installer receipt, active selector and successful smoke test.
    """
    from ..install import Installer, InstallError
    installer = Installer(config.state_root)
    result = {}
    for name, settings in config.games.items():
        if settings.get("enabled", True) is False:
            continue
        if name != "zork":
            continue  # LORD remains gated pending actual runtime/asset proof.
        try:
            status = installer.inspect(name)
            if not status.get("enabled") or status.get("incomplete_update"):
                continue
            receipt = installer.verify(name)
            if receipt.get("profile") != PROFILE or not receipt.get("validated"):
                continue
            folder = installer.installs / name / status["active"]
            result[name] = ZorkFactory(folder / "dfrotz", folder / "zork1.z3",
                receipt["artifacts"]["dfrotz"], receipt["artifacts"]["zork1.z3"])
        except (InstallError, OSError, ValueError, KeyError):
            continue
    return result
