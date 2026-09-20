# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import time

from creance.monitor.activity import (ACTIVE_CONFIRMED, ACTIVE_TENTATIVE,
                                      CONFIRMED, QUIET, STREAM_CLOCK, TENTATIVE,
                                      UNKNOWN, WALL_CLOCK, ActivityView,
                                      Detection, format_detection,
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
