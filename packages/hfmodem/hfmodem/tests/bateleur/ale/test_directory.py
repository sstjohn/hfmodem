# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The ALE directory: the monitoring list's layout read (from a sample in
its layout, written here), other sources merged, and answers given on the
console and the host port."""
from __future__ import annotations

from hfmodem.bateleur.ale import directory as D

SAMPLE = """
                  Automatic Link Establishment

==========================================================================
Introduction:
==========================================================================

Prose about ALE.

==========================================================================
1: Agencies & Frequencies (kHz)
==========================================================================

Prose about the lists, which runs on for a while
and has no frequencies in it.

---



Example Coast Watch
CW = Coast Watch HQ

A note after a blank line.
\t4995.0
\t7527.0\tCWHQ snd
\t11161.O\ttypo in the source



Example Relief Net
\t7527.0
\t14109\tpilot



==========================================================================
2: Call Signs ("Addresses")
==========================================================================

x = letter
# = number digit 0-9
- = varies

CWHQ\t\tCoast Watch\tHeadquarters
CW##\t\tCoast Watch\tBoats
R-\t\tRelief Net
@?@\t\tSystem\t\tGlobal Allcall

==========================================================================
3: Appendixes/ Notes
==========================================================================

Some notes.

==========================================================================
4: Acronym List
==========================================================================

CW\tCoast Watch
HQ\tHeadquarters
"""


def test_the_lists_four_parts_are_read():
    d = D.parse(SAMPLE, "sample")
    assert [a.name for a in d.agencies] == ["Example Coast Watch", "Example Relief Net"]
    cw = d.agencies[0]
    assert cw.notes == ["CW = Coast Watch HQ", "A note after a blank line."]
    assert cw.freqs == [(4995.0, ""), (7527.0, "CWHQ snd"), (11161.0, "typo in the source")]
    assert d.agencies[1].freqs[1] == (14109.0, "pilot")
    assert len(d.addresses) == 4 and d.acronyms == {"CW": "Coast Watch", "HQ": "Headquarters"}
    assert d.notes == "Some notes."


def test_who_takes_exact_addresses_first_then_patterns():
    d = D.parse(SAMPLE, "sample")
    assert [e.pattern for e in d.who("cwhq")] == ["CWHQ"]
    assert [e.pattern for e in d.who("CW12")] == ["CW##"]
    assert d.who("CWAB") == []
    assert [e.pattern for e in d.who("R42X")] == ["R-"]
    assert str(d.who("@?@")[0]) == "@?@: System — Global Allcall"


def test_who_uses_a_frequency():
    d = D.parse(SAMPLE, "sample")
    assert [(a.name, n) for a, _, n in d.at(7527.0)] == \
        [("Example Coast Watch", "CWHQ snd"), ("Example Relief Net", "")]
    assert d.at(7530.0) == [] and len(d.at(7527.3)) == 2


def test_sources_merge_and_say_where_each_entry_is_from(tmp_path):
    lst = tmp_path / "list.txt"
    lst.write_text(SAMPLE)
    extra = tmp_path / "extra.csv"
    extra.write_text("kind,key,agency,description\n"
                     "address,CWHQ,Coast Watch,also here\n"
                     "freq,7527.0,Harbour Net,shared\n"
                     "acronym,HN,,Harbour Net\n")
    d = D.load([str(lst), str(extra)])
    assert [e.source for e in d.who("CWHQ")] == [str(lst), str(extra)]
    assert {a.name for a, _, _ in d.at(7527.0)} == \
        {"Example Coast Watch", "Example Relief Net", "Harbour Net"}
    assert d.expand("HN") == "Harbour Net"


def test_a_list_it_cannot_read_is_refused(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("nothing like a directory\n")
    try:
        D.load(str(p))
    except ValueError as exc:
        assert "not a directory format" in str(exc)
    else:
        raise AssertionError("read a list it should not have")


def test_the_console(tmp_path, capsys):
    from hfmodem.cli import main
    lst = tmp_path / "list.txt"
    lst.write_text(SAMPLE)
    assert main(["ale-list", "--source", str(lst), "who", "CW07"]) == 0
    assert "CW##: Coast Watch — Boats" in capsys.readouterr().out
    assert main(["ale-list", "--source", str(lst), "freq", "14109000"]) == 0
    assert "Example Relief Net  (pilot)" in capsys.readouterr().out
    assert main(["ale-list", "--source", str(lst), "channels", "Coast Watch"]) == 0
    assert "channels = [4995000, 7527000, 11161000]" in capsys.readouterr().out


def test_the_host_port_names_a_station_the_first_time_it_is_heard():
    from hfmodem.bateleur.ale.controller import Config, Event
    from hfmodem.bateleur.ale.host import AleHost
    from hfmodem.bateleur.ale.modem import AleModem
    m = AleModem(Config(["W9SSJ"], ["C0"], scanning=False), send=lambda a, r: 0,
                 card_rate=48000)
    host = AleHost(m, port=0)
    assert host.command("WHO CWHQ").startswith("ERR no directory")
    m.directory = D.parse(SAMPLE, "sample")
    assert host.command("WHO CWHQ") == "OK WHO CWHQ CWHQ=Coast Watch: Headquarters"
    assert host.command("WHO XYZ") == "OK WHO XYZ not listed"
    host.event(Event("HEARD", peer="CW12", channel=0))
    host.event(Event("HEARD", peer="CW12", channel=0))
    assert list(host.log) == ["HEARD CW12 CHANNEL C0", "WHO CW12 CW##=Coast Watch: Boats",
                        "HEARD CW12 CHANNEL C0"]
