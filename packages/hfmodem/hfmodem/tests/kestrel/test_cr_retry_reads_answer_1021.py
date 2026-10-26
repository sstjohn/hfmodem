# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A connect-response already heard whole is read before a WINLINK retry keys over it.

Pat gate, 2026-10-21, BW500 against stock W1AW: the answer started 0.11-0.14 s
after our unkey and ran 0.99 s. The 1.30 s WINLINK quiet came before the stream's
next half-second read, so the retry went out and the late link-setup landed on
stock's answer to it (pat-gate/control-51ba72a-bw500).
"""
import numpy as np

from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import VaraIO, VaraStationHandshake

FS = 48000


class IO(VaraIO):
    def __init__(self):
        self.sent = []

    def tx_metadata(self, **meta):
        self.sent.append(meta['packet_kind'])


def test_retry_reads_the_buffered_answer_first():
    io = IO()
    hs = VaraStationHandshake(['W9SSJ'], io, bw='500', high_speed=True,
                              cr_retry_form='stock')
    hs.originate('W1AW')
    answer = MK.synth_burst('W1AW', VF.for_bw(VF.connect_response('500'), '500'), 0.27)
    audio = np.zeros(int(1.30 * FS))
    lead = int(0.14 * FS)
    audio[lead:lead + len(answer)] = answer
    for i in range(0, len(audio), FS // 50):
        hs.on_rx_stream(audio[i:i + FS // 50])
    assert io.sent == ['connect-request-500']
    hs.retry_connect()
    assert io.sent == ['connect-request-500', 'link-setup']
