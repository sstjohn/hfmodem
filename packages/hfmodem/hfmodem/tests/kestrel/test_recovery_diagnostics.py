# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Logs identify the actual recovery exchange and retained DATA ownership."""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_data_replies_0919 import query_pending, through_frame
from .test_nak_repacketization import pending


@pytest.mark.parametrize('bw', ['2300', '2750'])
def test_named_intermediate_phases_preserve_the_measured_waveforms(bw):
    names = [kind.name for kind in VF.SESSION_INTERMEDIATE_ANSWER_QUERIES]
    assert len(set(names + [VF.SESSION_FINAL_ANSWER_QUERY.name])) == 3
    for kind, preadv in zip(VF.SESSION_INTERMEDIATE_ANSWER_QUERIES, (683, 745)):
        old = replace(VF.SESSION_FINAL_ANSWER_QUERY, preadv=preadv)
        np.testing.assert_array_equal(MK.synth_burst('KC9GHZ', VF.for_bw(kind, bw)),
                                      MK.synth_burst('KC9GHZ', VF.for_bw(old, bw)))
        assert VF.BURSTS[kind.name] is kind


@pytest.mark.parametrize('last_full, name, kind', [
    (False, 'stock-2300-query-answer', VF.SESSION_INTERMEDIATE_QUERY_ANSWER),
    (True, 'stock-2300-last-full-answer', VF.SESSION_RESPONDER_OVER_ANSWER),
])
def test_reply_log_names_the_actual_native_match_and_pending_context(last_full, name, kind):
    hs, io = query_pending(last_full=last_full)
    x = through_frame(hs, name, kind)
    assert hs._peer_intermediate_query_answer(x)
    hs._took_intermediate_query_answer()
    msg = next(m for m in io.msgs if ' — acknowledged ' in m)
    assert f'{kind.name} ({kind.seed_off}/{kind.preadv})' in msg
    assert 'over #1, record 3, 89 pending bytes' in msg
    assert 'intermediate queries 1/3' in msg and 'NAK retries 0/' in msg


def test_query_exhaustion_reports_unconfirmed_bytes_and_budgets():
    hs, io, _ = pending(n=94)
    for _ in range(VA._FINAL_QUERY_MAX):
        hs._intermediate_query_at -= 5
        hs._retry_data_over()
    msg = next(m for m in io.msgs if 'DATA answer unconfirmed' in m)
    assert '89 pending bytes, 5 queued bytes' in msg
    assert 'intermediate queries 3/3' in msg
    assert hs._tx_pending is not None and hs.state == VA.VaraState.DISCONNECTED


def test_refused_repacketization_reports_the_original_retained_record():
    hs, io, _ = pending(n=94)
    old = hs._tx_pending
    io.tx_went_out = lambda: False
    hs._took_responder_nak()
    msg = next(m for m in io.msgs if 'NAK retry was not transmitted' in m)
    assert 'record 3, 89 pending bytes, 5 queued bytes' in msg
    assert 'NAK retries 0/' in msg and hs._tx_pending == old
