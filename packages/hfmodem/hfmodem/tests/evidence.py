# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Where a run's evidence is, which in a worktree is not where its code is.

Every recording this suite is measured against lives outside the package — under
``working/``, ``logs/`` and ``captures/``, or in the shared corpus beside the
checkout. Of those four only ``working/`` is in the index: ``.gitignore`` names
``logs/`` and ``captures/``, and the corpus is not in the checkout at all. A
worktree is made from the tree, not from the untracked files in it, so it gets
the code and the working record and none of the audio, and every declaration
that resolved the other three against the worktree's own root pointed at
nothing.

That is silent by construction. The tests reading those paths carry a `skipif`,
which is the right answer for a run that has no way to get the material and the
wrong one for a worktree of the very checkout holding it: an agent reading
`91 passed, 3 skipped` cannot tell that the three were the WW2MI tests covering
the stream it had been sent to investigate.

So the evidence resolves here, once, against the checkout a worktree was made
from. Code and evidence part company deliberately: `conftest.py` puts this tree's
packages first on `sys.path` precisely so a worktree tests its OWN code, while
the recordings are not a thing a worktree has its own version of.
"""
from __future__ import annotations

import os
from pathlib import Path

#: The checkout this run's code is in.
TREE = Path(__file__).resolve().parents[4]


def _origin() -> Path | None:
    """The checkout ``TREE`` was made a worktree of, or None if it is not one.

    A worktree's ``.git`` is a file naming its own git dir, and ``commondir``
    beside that names the ``.git`` the two share. A plain checkout has a
    directory there and an installed wheel has neither, and both answer for
    themselves.
    """
    dotgit = TREE / ".git"
    if not dotgit.is_file():
        return None
    gitdir = Path(dotgit.read_text().partition("gitdir:")[2].strip())
    common = gitdir / (gitdir / "commondir").read_text().strip()
    return common.resolve().parent


#: The checkout its evidence is in.
RECORD = _origin() or TREE

WORKING = RECORD / "working"
LOGS = RECORD / "logs"
CAPTURES = RECORD / "captures"

#: The shared off-air corpus, which is kept beside a checkout rather than in one.
#: One name across the tree; the per-modem names still work, because scripts and
#: notes use them.
CORPUS = Path(os.environ.get("HFMODEM_CORPUS")
              or os.environ.get("KESTREL_RF_CORPUS")
              or RECORD.parent / "rf-corpus")
