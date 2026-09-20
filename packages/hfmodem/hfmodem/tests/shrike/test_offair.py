# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""shrike decodes a REAL off-air PACTOR session -- not synthetic, not our own TX.

`captures/offair_KE5YTA_p3.wav` is 16 s of a live off-air PACTOR-3 QSO (KE5YTA <-
W4DNA on 7101.5 kHz), the same capture an independent decoder reads end to end
-- the first externally verified positive in the overnight corpus. This pins
shrike's receiver
against ground-truth real RF: the PACTOR-1 connect callsign, recovered through
the same decode path the monitor and live modem run, and a CRC-valid PACTOR-3
case-0 header lifted straight off the air.
"""
from pathlib import Path

from hfmodem.shrike import rx, rxfront
from hfmodem.tests.shrike import archive

FIXTURE = archive.KE5YTA_P3


def main() -> int:
    clip = rxfront.load_wav(str(FIXTURE))
    fs = rxfront.FS
    ok = True

    connects = [e for e in rxfront.decode_events(clip) if e.kind == "connect"]
    hit = any("KE5YTA" in e.text for e in connects)
    print(f"  [{'PASS' if hit else 'FAIL'}] off-air PACTOR-1 connect decoded -> KE5YTA")
    ok &= hit

    field, crc = rx.decode_case0_header(clip[12 * fs:16 * fs], fs=fs)
    print(f"  [{'PASS' if crc else 'FAIL'}] off-air PACTOR-3 case-0 header CRC-valid"
          f"  ({field.hex() if crc else '-'})")
    ok &= crc

    # A capture's levels must survive being saved. session.write_wav normalises
    # every file to 0.8 peak, so clipping -- the one fault that makes a capture
    # undecodable no matter what the far end did -- is invisible in the WAV
    # afterwards. Two readers have measured saved captures and wrongly concluded
    # there was none. The sidecar is the fix and this is what keeps it honest.
    import json
    import tempfile

    import numpy as np

    from hfmodem.shrike import onair
    d = Path(tempfile.mkdtemp())
    hot = np.clip(np.random.default_rng(0).normal(0, 0.6, rxfront.FS), -1, 1)
    onair._save_capture(d / "hot.wav", hot.astype(np.float32))
    side = json.loads((d / "hot.json").read_text())
    from_wav = float((np.abs(rxfront.load_wav(str(d / "hot.wav"))) > 0.995).mean()) * 100
    print(f"  [{'PASS' if side['railed_pct'] > 5 else 'FAIL'}] a clipped capture "
          f"is recorded as clipped ({side['railed_pct']:.1f}% railed)")
    ok &= side["railed_pct"] > 5
    print(f"  [{'PASS' if from_wav < 1 else 'FAIL'}] ...and the WAV alone could not "
          f"have told us ({from_wav:.2f}% after normalisation)")
    ok &= from_wav < 1

    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


# The skip here used to be this file's own judgement -- "it lives in the working
# record, which does not cross the publication boundary" -- and that is only half
# the truth: the recording is committed, so on a checkout it is present or the
# tree is broken. Skipping quietly made those two states look identical, on the
# one test that reads real off-air PACTOR-3 into a byte-exact assertion. The
# marker states the distribution's half; tests/gates/test_corpus_present.py fails
# over the other.
@archive.requires_offair_p3
def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
