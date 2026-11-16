# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.onair_support import _Bench


@pytest.mark.parametrize('decode_path', ['feed', 'ahead'])
def test_ws8eoc_0924_qrt_from_receive_keeps_next_long_slot(decode_path):
    live = _Bench(seconds=120, holdback=916, lat_in=788)
    live.now = live.pos = 5008512
    host = SimpleNamespace(arq=SimpleNamespace(state=arq.State.CONNECTED),
                           protocol=spec.Protocol.PACTOR3)

    def key_from_receive():
        assert live.pos == 5140512
        _, end = live.transmit(np.ones(159498, np.float32),
                               at=5191458, settle=.04)
        live.flush_to(end)

    class Receive:
        def bridge(self, chunk):
            pass

        def feed(self, chunk):
            if decode_path == 'feed' and not live.emissions:
                key_from_receive()

        def sl3_prekey_expected(self):
            return decode_path == 'ahead'

        def prepare_p3_sl3(self, audio, origin):
            assert not live.emissions
            assert origin + audio.size == live.pos

    def ahead(audio, origin):
        if live.pos == 5140512:
            key_from_receive()

    audio = onair._listen_until_answer(
        live, 178548, host, Receive(), 60000,
        ahead=ahead if decode_path == 'ahead' else None)
    assert live.emissions == [(5191458, 5350956)]
    assert live.pos == 5350956
    assert audio.size == 0
    grid = onair._MasterGrid(571458, 60000, 8880, packet_n=38880,
                             cs_n=10080, d_max_n=6240)
    grid.protocol, grid.sending = spec.Protocol.PACTOR3, True
    grid.regear(True)
    assert onair._keyable_slot(live, grid, 80, 1920) == 80
