# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A second session in one output directory still records its own readback.

The startup readback opened `radio-readback.json` with mode `x`, so the mere
presence of an earlier session's evidence refused the next one: `NOT KEYING:
receiver mode/passband unverified: [Errno 17] File exists`. Two arms sharing an
`--outdir` is ordinary -- `test_late_entry_loop` drives two in one directory,
and the air does the same -- so that rule made a keyable radio unkeyable. What
the evidence is for is provenance, and provenance is per session: the second
record is written beside the first and neither is overwritten.
"""
import json

from hfmodem.core import rxreadiness as rx

from .test_radio_mode_readback import StartupRig


def test_a_second_session_writes_beside_the_first(tmp_path):
    rigs = StartupRig(), StartupRig()
    first, second = (rx.verify_receiver_startup(rig, expected_mode="PKTUSB", model=1036,
                                                pactor1_only=False, output_dir=tmp_path)
                     for rig in rigs)
    assert [r["status"] for r in (first, second)] == ["mode_width_verified_only"] * 2
    assert [rig.calls for rig in rigs] == [[("set", "PKTUSB", 3000), ("read",)]] * 2
    records = {json.loads(p.read_text())["utc"]
               for p in tmp_path.glob("radio-readback*.json")}
    assert records == {first["utc"], second["utc"]}
    kept = json.loads((tmp_path / "radio-readback.json").read_text())
    assert kept["utc"] == first["utc"], "the first session's evidence was overwritten"
