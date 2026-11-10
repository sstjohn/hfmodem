# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The working record shrike's off-air results are measured against.

`working/pactor` is committed to the source tree and denied at the publication
boundary by `publish/manifest.toml`, so an absent file here has two readings that
point opposite ways. From an installed distribution it is gone by design and
permanently, and the only useful thing a test can do is name what it would have
run. On a checkout it is in the index, so its absence is a broken working tree --
and a skip there quietly retires the coverage the README's off-air PACTOR-3
claims rest on.

A `skipif` cannot hold both readings, so it holds the one that is safe in a
stranger's hands and the other is drawn once, in
`tests/gates/test_corpus_present.py`, which reads `COMMITTED` below and FAILS on a
source tree. That gate is what makes a skip here information rather than a hole:
`test_offair.py` and `test_rx.py` skipped silently for as long as they existed,
and a reader on a checkout could not tell a run that decoded the off-air capture
from one that never opened it.

What is committed is demanded; what only this station holds is named. Failing a
fresh checkout over `captures/occ15.wav` or the rf-corpus fixtures would fail it
over something the developer cannot repair, so those stay out. The silent-gateway
recordings below are declared on the other footing the gate offers -- warned about
by name rather than demanded -- because their absence is what a worktree of this
very checkout looked like, and it read as three fewer tests and nothing else.

The WS8EOC cuts sit inside the suite rather than under `working/` and read exactly
the same way: in every clone, in no distribution, and demanded here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hfmodem.tests import evidence

ARCHIVE = evidence.WORKING / "pactor"

#: 16 s of a live off-air PACTOR-3 QSO, KE5YTA <- W4DNA on 7101.5 kHz: the
#: recording an independent decoder reads end to end, and the ground truth every
#: off-air claim shrike makes is measured against.
KE5YTA_P3 = ARCHIVE / "captures" / "offair_KE5YTA_p3.wav"

#: Reference vectors for the phase softs (`vhinF`) and de-interleaved values
#: (`deintF`) used by the raster tests.
RASTER3 = ARCHIVE / "oracle" / "vm" / "raster3"
RASTER3_DUMPS = (RASTER3 / "vhinF.bin", RASTER3 / "deintF.bin")

#: What a checkout has to hold for shrike's off-air results to be run rather than
#: reported as run. `tests/gates/test_corpus_present.py` is where this is enforced.
COMMITTED = (KE5YTA_P3, *RASTER3_DUMPS)

#: 77 s apiece of a Winlink gateway calling with our transmitter off: control-signal
#: bursts on its own 1.25 s raster, recorded by analysis/provoke_listen.py, and the
#: peer the burst-lock and break-in schedules are run against. Written by a run and
#: in no clone but this one, so nobody else can repair their absence -- which makes
#: them the shape the gate exists for, because the tests reading them skipped in
#: every worktree of this checkout and said so only in a skip count.
SILENT_GATEWAYS = (ARCHIVE / "captures" / "provoke2" / "silent.wav",
                   ARCHIVE / "captures" / "w6ids_silent" / "silent.wav")

#: 69 s of a whole PACTOR session on the capture stream's own clock -- K4MSU,
#: 3595 kHz, 2026-08-19 22:10. Self-consistent rather than complete: measured
#: against the KiwiSDR tap this arm's lag is a staircase reaching +260 ms over
#: 60 s (`tools/witness_align.py`, once corrected for an 8300 ppm framing error
#: that had read it as +760). Its own counter justified 295 ms, so the two agree
#: and the tap adds no excess. It is the only recording in which a gateway ever took the
#: channel from this station and sent a readable field behind the codeword. On
#: the same footing as the silent gateways: written by a run, under `captures/`,
#: in no clone but this one.
K4MSU_BREAKIN = evidence.CAPTURES / "onair-0819-2210" / "stream.wav"


#: WS8EOC on 30 m, 2026-09-08 and 09: the accepted entry the live receiver missed
#: and the six packets of the session after it, cut to the production listen
#: windows and indexed by the JSON beside them. Committed beside the suite rather
#: than under `working/`, and held out of the distribution all the same: they are
#: 1.0 MB of WAV, where the whole `**/*.wav` deny exception in
#: `publish/manifest.toml` is a megabyte and already spends half of it on besra.
P3_FIXTURES = Path(__file__).with_name("fixtures")
WS8EOC_P3_INDEX = (P3_FIXTURES / "ws8eoc-0908-p3.json",
                   P3_FIXTURES / "ws8eoc-0909-p3.json")


def ws8eoc_p3() -> tuple[Path, ...]:
    """The indexes and every cut they name.

    Read off the indexes rather than globbed, for the reason besra's ground truth
    is: a WAV no index names is not evidence, and one an index names that is not
    on disk is the hole. An absent index takes its own cuts with it, which is the
    distribution's case and needs no second reading.
    """
    return (*WS8EOC_P3_INDEX,
            *(P3_FIXTURES / row["file"] for index in WS8EOC_P3_INDEX
              if index.exists() for row in json.loads(index.read_text())))


def _requires(*paths: Path, what: str, where: str = "working/"):
    missing = [p for p in paths if not p.exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"{what} not present ({(missing or paths)[0]}) — committed under "
               f"{where}, which does not cross the publication boundary, so this "
               "runs on a source tree and skips from an installed distribution")


requires_offair_p3 = _requires(KE5YTA_P3, what="the off-air PACTOR-3 capture")
requires_raster_dumps = _requires(*RASTER3_DUMPS,
                                  what="the reference soft-raster vectors")
requires_ws8eoc_p3 = _requires(*ws8eoc_p3(),
                               what="the WS8EOC PACTOR-3 recordings",
                               where="tests/shrike/fixtures/")
