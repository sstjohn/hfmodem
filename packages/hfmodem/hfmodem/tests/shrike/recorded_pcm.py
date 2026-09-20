# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Read a recorded PCM fixture and verify its indexed sample hash."""
import hashlib
from pathlib import Path
import wave

import numpy as np

from hfmodem.shrike.rxfront import FS


def recorded_pcm(row):
    with wave.open(str(Path(__file__).with_name("fixtures") / row["file"])) as wav:
        assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) == (FS, 2, 1)
        raw = wav.readframes(wav.getnframes())
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
