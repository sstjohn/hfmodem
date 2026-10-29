# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A channel's connect-request scanner reads the idle reader's decimated spectra."""
import numpy as np

from hfmodem.contact.passive import IdleReplies
from hfmodem.kestrel.monitor import HandshakeScanner
from hfmodem.kestrel.vara import vara_frames as VF, vara_mfsk as MK


def test_the_scanner_on_the_shared_front_names_what_it_names_alone():
    pad = np.random.default_rng(20261028).normal(0, .02, 6 * MK.FS)
    x = np.concatenate([pad, MK.synth_burst('W1AW', VF.CR500), pad, MK.synth_burst('NS0A-10', VF.CR), pad])
    idles = IdleReplies()
    shared, alone = HandshakeScanner(['W1AW', 'NS0A-10'], frontend=idles.control), HandshakeScanner(['W1AW', 'NS0A-10'])
    found, expected = [], []
    for i in range(0, len(x), MK.FS // 2):
        y = x[i:i + MK.FS // 2]
        idles.feed(y, ('W9SSJ',))
        found += shared.push(y)
        expected += alone.push(y)
    assert [(r.kind, r.gateway) for _, r in expected] == [('CR', 'W1AW'), ('CR', 'NS0A-10')]
    assert [(s, r.kind, r.gateway, r.quality) for s, r in found] == [
        (s, r.kind, r.gateway, r.quality) for s, r in expected]
    assert idles.control.cache_hits > 0
