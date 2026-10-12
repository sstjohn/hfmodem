# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import time

from creance.monitor.activity import (ACTIVE_CONFIRMED, ACTIVE_TENTATIVE,
                                      CONFIRMED, OURS, QUIET, STREAM_CLOCK,
                                      TENTATIVE, TRACKER, UNKNOWN, WALL_CLOCK,
                                      ActivityView, Detection, format_detection,
                                      render_summary)


def d(t, modem, proto, kind, grade, station="", role="", wall=0.0):
    return Detection(t=t, modem=modem, protocol=proto, kind=kind, grade=grade,
                     station=station, role=role, wall=wall)


def test_from_json_defaults_unknown_grade_to_tentative():
    got = Detection.from_json({"t": 1.0, "modem": "x", "grade": "bogus"})
    assert got.grade == TENTATIVE
    assert got.protocol == UNKNOWN


def test_confirmed_line_and_tentative_line_are_marked_differently():
    conf = format_detection(d(1.0, "kestrel", "VARA", "CR", CONFIRMED, "W1AW",
                              "gateway"))
    tent = format_detection(d(1.0, "shrike", "PACTOR-3", "DETECT", TENTATIVE))
    assert "?" not in conf and "W1AW" in conf
    assert "?" in tent


def test_a_stream_position_is_marked_as_one():
    """It printed as bare seconds and was read as an offset into the recording of
    the session. On live audio it provably is not one — a pass on 2026-08-14
    printed 309.57 for an event the recording puts at 380.18 — so the unit is on
    the line and cannot be mistaken for a time of day."""
    line = format_detection(d(309.57, "kestrel", "VARA", "CR", CONFIRMED),
                            clock=STREAM_CLOCK)
    assert line.startswith("[   309.57s]")
    assert ":" not in line.split("]")[0]


def test_a_live_position_is_the_wall_clock_the_audio_arrived_at():
    """The figure a lost sample cannot move. A live capture drops audio, so any
    count of samples drifts against the world; the time of day does not."""
    when = 1786000000.7
    line = format_detection(d(1.0, "kestrel", "VARA", "CR", CONFIRMED,
                              wall=when), clock=WALL_CLOCK)
    stamp = line.split("]")[0].lstrip("[")
    assert stamp == time.strftime("%H:%M:%S", time.localtime(when)) + ".7"
    assert stamp.count(":") == 2 and "s" not in stamp


def test_confirmed_station_yields_active_confirmed_and_names_it():
    v = ActivityView(window_s=20)
    v.add(d(5.0, "kestrel", "VARA", "CR", CONFIRMED, "W1AW", "gateway"))
    s = v.summary(6.0)
    assert s["protocols"]["VARA"]["state"] == ACTIVE_CONFIRMED
    assert s["overall"] == ACTIVE_CONFIRMED
    assert s["protocols"]["VARA"]["stations"][0]["call"] == "W1AW"
    assert "W1AW" in render_summary(s)


def test_tentative_never_promotes_to_confirmed():
    v = ActivityView(window_s=20)
    v.add(d(5.0, "shrike", "PACTOR-3", "DETECT", TENTATIVE))
    s = v.summary(6.0)
    assert s["protocols"]["PACTOR-3"]["state"] == ACTIVE_TENTATIVE
    assert s["overall"] == ACTIVE_TENTATIVE
    line = render_summary(s)
    assert "tentative" in line and "confirmed" not in line


def test_confirmed_outranks_tentative_regardless_of_arrival_order():
    v = ActivityView(window_s=20)
    v.add(d(5.0, "shrike", "PACTOR-3", "HEADER", CONFIRMED))
    v.add(d(5.5, "shrike", "PACTOR-3", "DETECT", TENTATIVE))   # weaker, later
    assert v.summary(6.0)["protocols"]["PACTOR-3"]["state"] == ACTIVE_CONFIRMED


def test_stale_detections_fall_out_of_the_window():
    v = ActivityView(window_s=10)
    v.add(d(5.0, "kestrel", "VARA", "CR", CONFIRMED, "W1AW", "gateway"))
    assert v.summary(30.0)["protocols"] == {}
    assert v.summary(30.0)["overall"] == QUIET


def test_unidentified_energy_counts_but_names_no_protocol():
    v = ActivityView(window_s=20)
    v.add(d(5.0, "kestrel", UNKNOWN, "unknown", TENTATIVE))
    s = v.summary(6.0)
    assert s["protocols"] == {}
    assert s["unidentified_energy"] is True
    assert s["overall"] == ACTIVE_TENTATIVE
    assert "unidentified energy" in render_summary(s)


def test_two_modems_merge_on_one_timeline():
    v = ActivityView(window_s=20)
    v.add(d(3.0, "kestrel", "VARA", "connect-response", CONFIRMED, "K1ABC",
            "gateway"))
    v.add(d(4.0, "shrike", "PACTOR-3", "HEADER", CONFIRMED))
    s = v.summary(5.0)
    assert set(s["protocols"]) == {"VARA", "PACTOR-3"}
    assert s["overall"] == ACTIVE_CONFIRMED


def test_quiet_channel_advises_nothing_heard():
    s = ActivityView().summary(100.0)
    assert s["overall"] == QUIET
    assert "nothing heard" in render_summary(s)


def test_a_measured_place_survives_the_runner_contract():
    full = Detection(t=3.2, modem="kestrel", protocol="VARA", kind="CR",
                     grade=CONFIRMED, station="K0SI", role="called",
                     centre_hz=1488.3, rf_hz=7101488.3, bw_hz=2310.0,
                     offset_hz=-11.7, level_dbfs=-41.5, cadence_s=1.25, track=3,
                     text="WS8EOC")
    assert Detection.from_json(json.loads(json.dumps(full.to_json()))) == full


def test_an_unmeasured_place_is_none_and_left_off_the_wire():
    bare = Detection.from_json({"t": 1.0, "modem": "shrike", "grade": "confirmed",
                                "offset_hz": None})
    assert bare.offset_hz is None and bare.centre_hz is None and bare.track is None
    assert set(bare.to_json()) == {"t", "modem", "protocol", "kind", "grade",
                                   "station", "role", "detail", "wall"}


def test_a_role_is_one_the_record_names():
    for role in ("called", "caller", "answering", "sender", "gateway"):
        assert Detection.from_json({"t": 0, "modem": "x", "role": role}).role == role
    assert Detection.from_json({"t": 0, "modem": "x", "role": "boss"}).role == ""


def test_a_track_is_energy_and_names_no_protocol_or_station():
    view = ActivityView(window_s=20)
    view.add(Detection(t=1.0, modem=TRACKER, protocol="FSK-shaped", kind="track",
                       grade=TENTATIVE, station="W1AW", centre_hz=1615.0, track=3))
    s = view.summary(2.0)
    assert s["protocols"] == {} and s["unidentified_energy"]
    assert s["overall"] == ACTIVE_TENTATIVE


def test_our_own_keying_is_not_the_channel():
    view = ActivityView(window_s=20)
    view.add(Detection(t=1.0, modem=TRACKER, protocol=UNKNOWN, kind=OURS,
                       grade=CONFIRMED))
    s = view.summary(2.0)
    assert s["overall"] == QUIET and not s["unidentified_energy"]
