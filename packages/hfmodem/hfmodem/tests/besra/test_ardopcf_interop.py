# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Cross-decode besra's transmit audio in the *real* ardopcf receiver.

The static fixtures prove besra agrees with ardopcf's recorded output; this proves
the live reference decoder reads besra's freshly-rendered frames. It runs the
ardopcf binary named by the ``ARDOPCF`` environment variable against a WAV besra
just rendered (the same discipline M0LTE's oracle tests use), and skips when no
binary is available so the suite stays hermetic.

    ARDOPCF=/path/to/ardopcf pytest tests/besra/test_ardopcf_interop.py

The binary needs only the file-decode path (``--decodewav``), which reads a WAV and
never opens a sound device — so a headless Linux build works. ardopcf does not
build on macOS; point ``ARDOPCF`` at a Linux build (or a wrapper that forwards to
one).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from hfmodem.besra.arq import session as S
from hfmodem.besra.phy import modulator as M

_ARDOPCF = os.environ.get("ARDOPCF")
pytestmark = pytest.mark.skipif(
    not (_ARDOPCF and Path(_ARDOPCF).exists()),
    reason="set ARDOPCF to a Linux ardopcf binary to run the live cross-decode")

_LEAD = np.zeros(1200, dtype="<i2")
_TAIL = np.zeros(2400, dtype="<i2")


def _decode(samples: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        path = tf.name
    try:
        with wave.open(path, "w") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(12000)
            w.writeframes(np.concatenate([_LEAD, samples, _TAIL]).tobytes())
        r = subprocess.run([_ARDOPCF, "--nologfile", "--decodewav", path],
                           capture_output=True, text=True, timeout=60)
        return r.stdout + r.stderr
    finally:
        os.unlink(path)


def test_ardopcf_decodes_besra_conreq():
    out = _decode(M.render_frame(0x31, caller="W9SSJ", target="K7ABC", session_id=0xFF))
    assert "Decode PASS" in out
    assert "W9SSJ" in out and "K7ABC" in out


def test_ardopcf_decodes_besra_idframe():
    out = _decode(M.render_frame(0x30, caller="W9SSJ", grid="EN63", session_id=0xFF))
    assert "Decode PASS" in out
    assert "W9SSJ" in out and "EN63" in out


def test_ardopcf_decodes_besra_data_payload_exact():
    payload = b"BESRA DECODED BY REAL ARDOPCF"
    out = _decode(M.render_frame(0x4A, payload=payload, session_id=0x5E))
    assert "Decode PASS" in out
    assert payload.decode() in out           # ardopcf prints the recovered bytes


@pytest.mark.parametrize("q", [38, 44, 60, 66, 69, 80, 94, 100])
def test_ardopcf_reads_the_quality_our_acks_report(q):
    """The number a gateway gearshifts on, read back by the real receiver.

    besra used to answer every frame with a hardcoded 100; it now reports what it
    measured, which puts codes across the whole five-bit field on the air for the
    first time. ardopcf prints the quality it recovers, so it can say whether each
    one still arrives as the frame we meant — to the two-point resolution the field
    has (an odd 69 goes out and comes back as 68).
    """
    for frame_type, kind in ((S._dataack_for(q), "DataAck"),
                             (S._datanak_for(q), "DataNak")):
        out = _decode(M.render_frame(frame_type, session_id=0x5E))
        assert f"{kind} FrameType (0x{frame_type:02X})" in out
        assert f"decode quality ({38 + 2 * (frame_type & 0x1F)}/100)" in out
        assert "RXO 5E" in out               # and addressed to the right session
