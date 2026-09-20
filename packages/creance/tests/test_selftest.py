# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The M1 gate as a pytest: same stages as `creance selftest`, smaller payload.

Skipped wholesale when neither target imports here; the selftest itself skips
whichever half is missing, so a machine with only kestrel still gates on it.

Set ``CREANCE_REQUIRE=kestrel,sabir`` on a bench that is supposed to have both
and a target that stops resolving fails the gate instead of quietly reducing it.
The list cannot be derived from ``have_kestrel()``/``have_sabir()`` — that is the
same predicate that goes False when a path moves, so it would excuse exactly the
outage it is meant to catch. It is an assertion about the machine, kept outside
the code that checks it, and unset on a clean clone so hermeticity survives.
"""

import os

import pytest

from creance import selftest


@pytest.mark.skipif(not (selftest.have_kestrel() or selftest.have_sabir()),
                    reason="neither the kestrel loopback nor the sabir server "
                           "is importable")
def test_m1_gate(capsys):
    require = os.environ.get("CREANCE_REQUIRE", "").split(",")
    run = selftest.Selftest(size="2k", require=require)
    rc = run.run()
    detail = "\n".join(f"{s.status} {s.name}: {s.detail}" for s in run.stages)
    assert rc == 0, detail
    assert any(s.status == selftest.PASS for s in run.stages), detail
