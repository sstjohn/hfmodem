# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Every reply/notification form the kestrel and harrier servers emit."""

import pytest

from hfhost import wire
from hfhost.wire import classify


def test_replies():
    assert classify("OK").kind == wire.REPLY_OK
    assert classify("WRONG").kind == wire.REPLY_WRONG
    assert classify(" OK ").kind == wire.REPLY_OK   # framing strips around CR


@pytest.mark.parametrize("line,name,fields", [
    ("CONNECTED N0CAL K7ABC 2300", "CONNECTED",
     {"src": "N0CAL", "dst": "K7ABC", "bw": "2300"}),
    ("CONNECTED N0CAL K7ABC", "CONNECTED",
     {"src": "N0CAL", "dst": "K7ABC", "bw": None}),
    ("DISCONNECTED", "DISCONNECTED", {}),
    ("PTT ON", "PTT", {"on": True}),
    ("PTT OFF", "PTT", {"on": False}),
    ("BUFFER 0", "BUFFER", {"n": 0}),
    ("BUFFER 12345", "BUFFER", {"n": 12345}),
    ("BUSY ON", "BUSY", {"on": True}),
    ("BUSY OFF", "BUSY", {"on": False}),
    ("PENDING", "PENDING", {}),
    ("CANCELPENDING", "CANCELPENDING", {}),
    ("IAMALIVE", "IAMALIVE", {}),
    ("LINK REGISTERED", "LINK", {"registered": True}),
    ("LINK UNREGISTERED", "LINK", {"registered": False}),
    ("REGISTERED K7ABC", "REGISTERED", {"call": "K7ABC"}),
    ("VERSION 4.9.0.KestrelOpen", "VERSION", {"version": "4.9.0.KestrelOpen"}),
    ("VERSION 4.9.0.harrier", "VERSION", {"version": "4.9.0.harrier"}),
    ("BITRATE (7) 1329 bps TX", "BITRATE",
     {"level": 7, "bps": 1329, "direction": "TX"}),
    ("BITRATE (0) 232 bps RX", "BITRATE",
     {"level": 0, "bps": 232, "direction": "RX"}),
    ("SN 12", "SN", {"value": 12.0}),
    ("SN -3.5", "SN", {"value": -3.5}),
    ("CQFRAME K7ABC 2300", "CQFRAME", {"src": "K7ABC", "bw": "2300"}),
])
def test_notifications(line, name, fields):
    got = classify(line)
    assert got.kind == wire.NOTIFICATION
    assert got.name == name
    assert got.fields == fields
    assert got.raw == line


@pytest.mark.parametrize("line", [
    "",                          # blank
    "FROBNICATE",                # unknown verb
    "connected N0CAL K7ABC",     # wrong case is a finding, not a match
    "BUFFER",                    # missing argument
    "BUFFER x",                  # non-numeric argument
    "BUFFER -1",                 # negative depth
    "CONNECTED N0CAL",           # too few fields
    "PTT MAYBE",                 # bad ON/OFF
    "PENDING NOW",               # unexpected argument
    "IAMALIVE 2",
    "LINK",                      # missing REGISTERED/UNREGISTERED
    "VERSION",                   # reply always carries a version string
    "BITRATE 1329 bps TX",       # missing (N)
    "BITRATE (7) fast bps TX",
    "SN loud",
    "CQFRAME K7ABC",
])
def test_unknown(line):
    got = classify(line)
    assert got.kind == wire.UNKNOWN
    assert got.fields == {}


def test_vocabulary_and_deltas():
    assert wire.CR == b"\r"
    assert wire.BANDWIDTHS == ("500", "2300", "2750")
    # The pre-connect-buffering delta is resolved (kestrel's LoopbackModem now
    # buffers pre-CONNECTED writes and flushes on CONNECT, per spec §7.4), so
    # only the IAMALIVE cadence delta remains.
    assert len(wire.SPEC_DELTAS) == 1
    assert any("IAMALIVE" in d for d in wire.SPEC_DELTAS)
