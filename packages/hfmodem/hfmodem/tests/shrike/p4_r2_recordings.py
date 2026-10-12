# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fixed recorded controls: absolute capture samples, no invented TX registry.

File SHA256 hashes below refer to the actual compressed files, independently
of stream.json's historical waveform hash. Old 0926 lacks tx_event records;
its mask is explicitly unavailable and must not support a changed-TX replay.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from hfmodem.shrike import session

FS = 48000
MANIFEST = {'1001-1029': {'log': 'pactor-p4-bench-ws8eoc-40-force-20261001T152923Z-p4-mail.log',
               'hashes': {'stream.flac': '68995ac22999062fd6b519b2aa7e9335f63cdafc04ee066d92766ea74b77d17f',
                          'stream.json': 'f1cd5637b8d9164ef5ef1a978fd573fb26c52cef27d0a66a69a7bdb016d09f08',
                          'p4-probe.json': 'a4659a9604e25af8b2f3ff2ab6e690472d9e7638ac54158eb4d062f507113f5a',
                          'pactor-p4-bench-ws8eoc-40-force-20261001T152923Z-p4-mail.log': 'c0b226cf28fd067a789b857a927fba48d9f9ab885ddcb6108be181f2fc3acb72'}},
 '0926-1342': {'log': 'p4-kb5lzk-replica-1236.log',
               'hashes': {'stream.flac': 'bf4336cfb5f808d05ec96b8a1075b67911864aa1cb8f7b7ec0f1788e19d8c166',
                          'stream.json': '1b336051e15597f0f32a2c5a78de45dc9ddf319cb72424738d9e0ac4acf57eba',
                          'p4-probe.json': '7b0d785d9fee9688819b10034273bebdaf8cb45a454e0eff073c951d124b9ada',
                          'p4-kb5lzk-replica-1236.log': 'db1140b7ff1f32e5432b311e35308b86f9cf665e81cdcc3be4b13807b94fed66'}},
 '0930-2157': {'log': 'pactor-p4-bench-k4msu-80-force-20261001T025734Z-p4-mail.log',
               'hashes': {'stream.flac': '78623523e6a896a81f8c736c9b75e5f64c1571a60c6c62c7b06123e6e5f61b13',
                          'stream.json': '5b65a64feb558b7d33219db33f8bf4754c34f47a486364d66c94a8a41b36a4a8',
                          'p4-probe.json': 'ed3dedcecaff8564b89c55761ceabcaeb6e55ea7d1a40993f3fef0ab615444aa',
                          'pactor-p4-bench-k4msu-80-force-20261001T025734Z-p4-mail.log': 'e0fe2471ad510f4065b7b0af824d487c1b04edbcfac976a549586f63f58160ac'}}}
# Exact grid boundary 2729 + 60000*slot; bounded original RX tails.
KB5LZK_TAILS = (
    (1785929, 1802729, 1786811),
    (1845929, 1862729, 1846809),
    (1905929, 1922729, 1906812),
    (1965929, 1982729, 1966812),
)
# Log geometry only: capture duration is rounded, not an exact origin.
RECOVERY_1029_CS_DUE = 1793346
RECOVERY_1029_CAPTURED_S = 3.985


@dataclass(frozen=True)
class Recording:
    arm: str
    capture: Path
    log_path: Path
    metadata: dict
    probe: dict
    options: str
    log: str
    audio: np.ndarray
    own_tx: tuple

    @property
    def own_tx_mask_available(self):
        return bool(self.own_tx)

    def snippet(self, start, end, *, require_mask=False):
        """Owned audio at the original 48k origin; masked intervals are gaps."""
        if require_mask and not self.own_tx_mask_available:
            raise ValueError('recorded tx_event mask unavailable')
        if not 0 <= start < end <= len(self.audio):
            raise ValueError('snippet outside absolute stream')
        audio = self.audio[start:end].copy()
        gaps = tuple((max(start, a), min(end, b)) for a, b in self.own_tx
                     if a < end and b > start)
        for a, b in gaps:
            audio[a-start:b-start] = 0
        return start, audio, gaps


@lru_cache(maxsize=3)
def recording(arm):
    pin = MANIFEST[arm]
    roots = Path(__file__).resolve().parents
    root = next((p for p in roots if (p/'captures'/f'onair-{arm}'/'stream.flac').exists()), None)
    if root is None:
        pytest.skip(f'local recorded control {arm} absent')
    capture = root/'captures'/f'onair-{arm}'
    log_path = root/'working'/f'onair-{arm}'/pin['log']
    for name, digest in pin['hashes'].items():
        path = log_path if name == pin['log'] else capture/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, str(path)
    metadata = json.loads((capture/'stream.json').read_text())
    assert metadata['samplerate'] == FS and metadata['first_stream_sample'] == 0
    # Resolve historical stream.wav metadata against actual compressed capture.
    assert Path(metadata['wav']).name in ('stream.wav', 'stream.flac')
    audio = session.load_wav(str(capture/'stream.flac'), FS)
    assert len(audio) == metadata['samples']
    audio.flags.writeable = False
    log = log_path.read_text()
    options, = [line.split('resolved args: ', 1)[1] for line in log.splitlines()
                if 'resolved args: ' in line]
    events = [json.loads(line.split('tx_event ', 1)[1]) for line in log.splitlines()
              if line.startswith('tx_event ')]
    own_tx = tuple((e['audio_start_sample'], e['audio_end_sample']) for e in events
                   if e.get('event') == 'completed' and e.get('outcome') == 'sent')
    return Recording(arm, capture, log_path, metadata,
                     json.loads((capture/'p4-probe.json').read_text()),
                     options, log, audio, own_tx)
