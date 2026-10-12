# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The amateur ALE network's conventions (``ale.hflink``) and its position
report (``ale.gpr``), through simulated channels."""
from __future__ import annotations

from datetime import datetime, timezone

from hfmodem.bateleur.ale import gpr, hflink, words as W
from hfmodem.tests.bateleur.ale import ether as E

CHANNELS = ["14109", "18106", "21096", "28146"]


def _pair(**kw):
    eth = E.Ether(snr_db=15.0, seed=1)
    sam = eth.add(hflink.config("W9SSJ", CHANNELS, listen_s=0.0, **kw))
    joe = eth.add(hflink.config("KQ6XA", CHANNELS, listen_s=0.0, **kw))
    eth.run(1.0)
    return eth, sam, joe


def test_the_published_gpr_example_reads():
    r = gpr.parse("GPR*KQ6XA*37N654321*122W987654*000003M*20050821Z135235*EVERYTHING FINE")
    assert r.object == "KQ6XA" and r.comment == "EVERYTHING FINE" and not r.manual
    assert abs(r.lat - 37.654321) < 1e-9 and abs(r.lon + 122.987654) < 1e-9
    assert r.alt_m == 3.0 and r.when == datetime(2005, 8, 21, 13, 52, 35, tzinfo=timezone.utc)


def test_a_gpr_is_written_as_it_is_read_and_fits_an_amd():
    r = gpr.Report("W9SSJ/M", 43.0731, -89.4012, 270, datetime(2026, 9, 29, 12, 0, 5,
                                                                    tzinfo=timezone.utc),
                   "mobile, all well")
    t = gpr.format(r)
    assert t == "GPR*W9SSJ#M*43N073100*089W401200*270M*20260929Z120005*MOBILE# ALL WELL"
    assert len(t) <= 90 and W.amd_words(t)
    back = gpr.parse(t)
    assert back.object == "W9SSJ/M" and abs(back.lat - 43.0731) < 1e-6
    assert back.lon < 0 and back.when == r.when


def test_a_hand_keyed_or_unfixed_position_is_marked():
    t = gpr.format(gpr.Report("KQ6XA", None, None))
    assert t.startswith("GPR*KQ6XA*##*###*#M*") and gpr.parse(t).manual
    assert gpr.parse(t).lat is None


def test_the_preset_takes_the_networks_conventions():
    eth, sam, joe = _pair()
    c = sam.ctl.cfg
    assert set(c.nets) == {"HFN", "HFL", "QRZ", "GPR", "RPT"}
    assert all(v == ("W9SSJ", 1) for v in c.nets.values())
    assert c.activity_s == 360 and c.tune_s == 3 and c.sound_terminator == W.TWAS
    assert abs(sam.ctl._scan_words() * 0.392 - 15.0) < 0.4        # a 15 s call
    assert c.sound_s == 0                                          # not a base station


def test_a_base_station_sounds_hourly_on_its_pilot_channels_only():
    eth, sam, joe = _pair(base=False)
    base = eth.add(hflink.config("K4PWO", CHANNELS, base=True, pilot_channels=[0]))
    due = base.ctl._sound_due
    assert due[0] <= 3600 and all(d == float("inf") for d in due[1:])


def test_a_net_call_to_hfn_links_with_a_member():
    """The amateur CQ: a net call, answered in slot 1 by the station's own
    callsign."""
    eth, sam, joe = _pair()
    sam.ctl.call("HFN", eth.now, channel=0, members=hflink.net_members("HFN", "W9SSJ"))
    eth.run(60, until=lambda: sam.ctl.state == joe.ctl.state == "linked")
    assert sam.ctl.linked == {"KQ6XA"}


def test_a_gpr_broadcast_reaches_the_net_and_links_no_one():
    eth, sam, joe = _pair()
    text = gpr.format(gpr.Report("W9SSJ", 43.07, -89.40, 270))
    sam.ctl.broadcast("GPR", text, eth.now, channels=[0, 1])
    eth.run(90, until=lambda: any(e.kind == "SENT" for e in sam.ctl.events))
    eth.run(3)
    assert any(e.kind == "SENT" for e in sam.ctl.events)
    assert ("W9SSJ", text) in joe.ctl.amd_log
    assert gpr.parse(joe.ctl.amd_log[0][1]).object == "W9SSJ"
    assert not sam.ctl.linked and not joe.ctl.linked
    assert sam.ctl.state == "available"
