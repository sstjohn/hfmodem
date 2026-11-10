# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from creance import propagation as prop
from creance.propagation import CLOSED, OPEN, UNKNOWN


def _rows(*pairs):
    return [{"band": b, "n": n, "best": s, "km": 500}
            for b, n, s in pairs]


def test_bands_with_spots_are_open_and_bands_without_are_closed():
    def fetch(sql, timeout):
        return _rows((7, 225, 5), (14, 97, -4))
    p = prop.path((43.0, -88.0), (38.5, -95.0), fetch=fetch)
    assert p.bands["40m"].verdict == OPEN
    assert p.bands["40m"].spots == 225 and p.bands["40m"].best_snr == 5
    assert p.bands["20m"].verdict == OPEN
    assert p.bands["80m"].verdict == CLOSED
    assert p.bands["10m"].verdict == CLOSED


def test_closed_is_the_only_verdict_that_stops_a_run():
    """The evidence is asymmetric: no spots is a trustworthy negative, spots
    are a weak positive, and no data must never stop us transmitting."""
    def fetch(sql, timeout):
        return _rows((7, 12, -14))
    p = prop.path((43.0, -88.0), (38.5, -95.0), fetch=fetch)
    assert p.bands["40m"].worth_trying is True
    assert p.bands["30m"].worth_trying is False        # CLOSED
    assert prop.unknown().bands == {}
    assert prop.BandReport("40m", UNKNOWN).worth_trying is True


def test_a_network_failure_is_unknown_not_an_exception():
    """A remote site has opportunistic IP at best. A propagation lookup must
    never delay or fail a session."""
    def fetch(sql, timeout):
        raise OSError("no route to host")
    p = prop.path((43.0, -88.0), (38.5, -95.0), fetch=fetch)
    assert p.bands == {}
    assert "no route to host" in p.error
    assert p.best() is None


def test_best_picks_the_band_with_the_most_evidence():
    def fetch(sql, timeout):
        return _rows((7, 225, 5), (10, 69, 6), (14, 97, -4))
    p = prop.path((43.0, -88.0), (38.5, -95.0), fetch=fetch)
    assert p.best().band == "40m"          # most spots, not best SNR


def test_own_beacon_filters_on_our_callsign():
    seen = {}

    def fetch(sql, timeout):
        seen["sql"] = sql
        return _rows((10, 4, -12))
    p = prop.own_beacon("N0CRE", (38.5, -95.0), fetch=fetch)
    assert "tx_sign = 'N0CRE'" in seen["sql"]
    assert p.bands["30m"].source == "own"


def test_a_callsign_is_refused_rather_than_escaped():
    """The call is interpolated into SQL, so anything implausible is rejected
    outright instead of quoted and hoped for."""
    for bad in ("'; DROP TABLE wspr.rx --", "N0CRE OR 1=1", ""):
        with pytest.raises(ValueError):
            prop.own_beacon(bad, (0.0, 0.0), fetch=lambda *a: [])


def test_summary_serialises_for_the_session_record():
    def fetch(sql, timeout):
        return _rows((7, 5, -1))
    d = prop.path((43.0, -88.0), (38.5, -95.0), fetch=fetch).as_dict()
    assert d["bands"]["40m"]["verdict"] == OPEN
    assert d["bands"]["40m"]["source"] == "path"
    import json
    json.dumps(d)          # must survive the transcript
