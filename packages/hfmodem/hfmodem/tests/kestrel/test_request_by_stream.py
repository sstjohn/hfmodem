# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The request route's floor on the response route's off-air negative controls."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.evidence import require_input
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_response_by_stream import _corpus, _load_48k, _N0LCR, _PANEL
from hfmodem.tests.kestrel.test_stock_caller_connect import IO

_MEASURED_WORST = {"500": 10, "2300": 7, "2750": 6}


def _recordings():
    for path in (corpora.ONAIR_CONNECT_ATTEMPT,
                 corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav",
                 corpora.OFFAIR / "KC9GHZ_2300" / "rig_rx.wav", corpora.CLEAR_CHANNEL):
        require_input(path)
    return [*_corpus(),
            ("6800 kHz control", corpora.CARRIER_ONLY_CHANNEL, ()),
            ("7103.5 kHz control", corpora.MONITORED_QUIET, ())]


@corpora.requires_onair_connect_attempt
@corpora.requires_regress_fixtures
@corpora.requires_carrier_only_channel
@corpora.requires_monitored_quiet
@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
def test_off_air_non_requests_stay_below_the_listening_acceptance_bar(bw):
    recordings = _recordings()
    panel = (_N0LCR, *_PANEL)
    kind = VF.connect_request(bw)
    retry = VF.connect_request_retry(bw)
    assert VF.payload_bins("W9SSJ", kind) == VF.payload_bins("W9SSJ", retry)
    off = np.arange(kind.n_payload) * (MK.HOP // VA._ACK_GRID)
    best, accepted, total, seconds = 0, 0, 0, 0.0
    worst = {}
    for label, path, own in recordings:
        require_input(path)
        x = _load_48k(path)
        seconds += len(x) / MK.FS
        bins, clear, _ = VA._top3_track(x, band=MK.band_for(bw))
        n = len(bins) - off[-1]
        local = 0
        for call in (c for c in panel if c not in own):
            tones = np.asarray(VF.payload_bins(call, kind), dtype=np.int32)
            for start in range(0, max(0, n), 20000):
                idx = np.arange(start, min(start + 20000, n))[:, None] + off
                heard, ok = bins[idx, 0], clear[idx, 0] >= VA._CLEAR_DB
                comparable = ok.sum(1)
                for shift in VA._RESP_SHIFTS:
                    matches = ((heard == tones + shift) & ok).sum(1)
                    local = max(local, int(matches.max(initial=0)))
                    accepted += int(((comparable >= VA._ASK_MIN_HEARD)
                                     & (matches >= VF._ACCEPT_FRAC * comparable)).sum())
                    total += len(idx)
        worst[label] = local
        best = max(best, local)
        print(f"request BW{bw}: {label}: {local}/{kind.n_payload}", flush=True)
    print(f"request BW{bw}: {seconds:.3f} s, {len(recordings)} recordings, "
          f"{total} alignment-shifts, best {best}/{kind.n_payload}, accepts {accepted}", flush=True)
    assert seconds > 2300, f"corpus shrank to {seconds} s"
    assert total > 100_000_000, total
    assert accepted == 0, (accepted, worst)
    assert best <= _MEASURED_WORST[bw], worst
    assert best < np.ceil(VF._ACCEPT_FRAC * VA._ASK_MIN_HEARD), worst


@corpora.requires_onair_connect_attempt
@corpora.requires_regress_fixtures
@corpora.requires_carrier_only_channel
@corpora.requires_monitored_quiet
@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
def test_listening_stream_keys_nothing_on_the_off_air_negative_corpus(bw):
    panel = (_N0LCR, *_PANEL)
    for label, path, own in _recordings():
        require_input(path)
        io = IO()
        hs = VA.VaraStationHandshake([c for c in panel if c not in own], io, bw=bw)
        hs.listen(True)
        x = _load_48k(path)
        for start in range(0, len(x), 1024):
            hs.on_rx_stream(x[start:start + 1024])
        assert not io.keyed, (bw, label, io.lines)
        assert hs.state is VA.VaraState.LISTENING
        print(f"request BW{bw}: {label}: no transmission", flush=True)
