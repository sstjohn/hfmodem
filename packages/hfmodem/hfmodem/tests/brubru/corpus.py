# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from hfmodem.tests import evidence

ROOT = evidence.WORKING / 'robust-packet' / 'recordings'
NAMES = ('RPR_Frame.wav', 'Robust-packet.mp3', 'radioscanner-signal233-f.wav')
RECORDINGS = tuple(ROOT / name for name in NAMES)
