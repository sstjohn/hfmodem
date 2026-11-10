# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import hashlib
import shutil
import subprocess

import numpy as np
import pytest

from hfmodem.ax25.frame import parse
from hfmodem.brubru import frame, rx, tx
from hfmodem.tests.brubru.corpus import RECORDINGS

pytestmark = pytest.mark.corpus

HASHES = ('52aad5f1764a5d595d99a5482640581856919f56bb13ff913bef2df2913cbefc', 'a96c0ea2972af5a129936418ee99a7fb3d9253324275b6b5470590377226e730', 'ad8457fb1bd9e873fe5f7bd7cd959a3caa6fd1889c18e71012b121f624d69ab7')

PAYLOADS = (
    (b'@093257h5424.61N/01824.29E&HF APRS 10.1476 / 144.850',),
    (b'WAVECOM ELEKTRONIK AG, the reference in data decoding\r',),
    (b'!5058.59N/00758.39E- Buon Natale e felice Anno nuovo\r',
     b'!5058.59N/00758.39E- Merry Christmas and a Happy New Year\r'),
)


def test_corpus_presence():
    missing = [str(path) for path in RECORDINGS if not path.is_file()]
    if len(missing) == len(RECORDINGS):
        pytest.skip('public Robust Packet recordings absent: ' + ', '.join(missing))
    assert not missing, 'incomplete Robust Packet corpus: ' + ', '.join(missing)


@pytest.mark.parametrize('tape', range(3))
def test_public_frames_and_regeneration(tape):
    path = RECORDINGS[tape]
    if not path.is_file():
        pytest.skip(f'public Robust Packet recording absent: {path}')
    assert shutil.which('ffmpeg'), 'ffmpeg is required to read the public corpus'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES[tape]
    pcm = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le',
                          '-ac', '1', '-ar', '9600', '-'], check=True, capture_output=True).stdout
    audio = np.frombuffer(pcm, '<f4').astype(float)
    got = rx.receive(audio)
    assert len(got) == len(PAYLOADS[tape])
    for i, decoded in enumerate(got):
        assert len(decoded.frames) == 1
        parsed = parse(decoded.frames[0])
        assert parsed.info == PAYLOADS[tape][i]
        calls = tuple(bytes(b >> 1 for b in a[:6]).decode().rstrip() for a in parsed.addresses)
        assert calls == (('APWW11', 'SP9MLI', 'WIDE1'), ('CQ', 'SERVER'),
                         ('APRPR', 'DH8HP', 'DB0UAL'))[tape]
        assert parsed.control == 3 and parsed.pid == 0xF0
        assert decoded.header.level == 0
        assert decoded.header.cls == (4 if i == 1 else 3)
        assert decoded.header.symbols == (227 if i == 1 else 167)
        packet = frame.assemble(list(decoded.frames), decoded.suggestion)
        assert packet.data[:-1] == decoded.packet
        if i == 1:
            assert len(decoded.differentials) == 1808
            continue
        burst, = tx.transmit(list(decoded.frames), limiter=tape == 0, suggestion=decoded.suggestion)
        regenerated, = rx.receive(burst.card_audio(), rate=48000, centre=1500)
        heard = decoded.differentials < 0
        ours = regenerated.differentials < 0
        assert heard.size == ours.size == 1328
        agree = int(np.count_nonzero(heard == ours))
        assert agree >= (1327 if tape == 0 else 1328), (path.name, agree)
