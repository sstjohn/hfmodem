# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CLI surface: argument parsing, exit codes, and the two commands that read
live modems (`modems`, `conform`) against FakeModems."""

import json

import pytest

from creance import cli
from creance import config as config_mod
from creance.metrics import SessionMetrics
from creance.report import write_session
from hfhost.supervisor import Supervisor

from hfhost.testing.fakemodem import FakeModem


# -- sizes -------------------------------------------------------------------

def test_size_converts_at_the_parser_and_keeps_the_written_form():
    size = cli.Size("10k")
    assert size == 10240 and size.text == "10k" and str(size) == "10k"


@pytest.mark.parametrize("text", ["", "big", "10g", "k10", "-1k"])
def test_size_rejects_at_the_parser(text):
    with pytest.raises(ValueError):
        cli.Size(text)


def test_run_rejects_an_unknown_payload():
    with pytest.raises(SystemExit):
        parse("run", "-c", "s.toml", "--modem", "k", "--dst", "K7XYZ",
              "--scenario", "unidir", "--payload", "prbs99")


# -- parser ------------------------------------------------------------------

def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_subcommand_required(capsys):
    with pytest.raises(SystemExit):
        parse()


def test_run_arguments():
    args = parse("run", "-c", "s.toml", "--modem", "kestrel", "--dst", "K7XYZ",
                 "--scenario", "unidir", "--size", "10k", "--repeat", "3",
                 "--interval", "60", "--label", "mytag")
    assert (args.modem, args.dst, args.scenario) == ("kestrel", "K7XYZ", "unidir")
    assert (args.size, args.size.text, args.repeat, args.interval, args.label) \
        == (10240, "10k", 3, 60.0, "mytag")
    assert args.payload == "prbs9"


def test_run_size_and_duration_are_exclusive():
    with pytest.raises(SystemExit):
        parse("run", "-c", "s.toml", "--modem", "k", "--dst", "K7XYZ",
              "--scenario", "unidir", "--size", "10k", "--duration", "60")


def test_run_rejects_unknown_scenario():
    with pytest.raises(SystemExit):
        parse("run", "-c", "s.toml", "--modem", "k", "--dst", "K7XYZ",
              "--scenario", "nonesuch")


def test_campaign_plan_and_all_are_exclusive():
    with pytest.raises(SystemExit):
        parse("campaign", "-c", "s.toml", "--plan", "p.toml", "--all")


def test_conform_modem_and_pair_are_exclusive():
    with pytest.raises(SystemExit):
        parse("conform", "-c", "s.toml", "--modem", "a", "--pair", "a", "b")


def test_verbose_survives_the_subcommand():
    assert cli._verbose(parse("-v", "report", "results")) is True
    assert cli._verbose(parse("report", "results", "-v")) is True
    assert cli._verbose(parse("report", "results")) is False


# -- graceful failures -------------------------------------------------------

def test_missing_config_is_not_a_traceback(capsys):
    assert cli.main(["modems", "-c", "/nonesuch/site.toml"]) == 2
    assert "creance:" in capsys.readouterr().err


def test_bad_config_is_reported(tmp_path, capsys):
    bad = tmp_path / "site.toml"
    bad.write_text("[site]\nname = 'x'\n")          # no mycall, no modems
    assert cli.main(["run", "-c", str(bad), "--modem", "k", "--dst", "K7XYZ",
                     "--scenario", "connect"]) == 2
    assert "creance:" in capsys.readouterr().err


def test_unknown_modem_is_reported(tmp_path, capsys):
    cfg = write_config(tmp_path, [("kestrel", 9, 10)])
    assert cli.main(["run", "-c", str(cfg), "--modem", "nope", "--dst", "K7XYZ",
                     "--scenario", "connect"]) == 2
    assert "unknown modem" in capsys.readouterr().err


def test_conform_rejects_dst_equal_to_mycall(tmp_path, capsys):
    cfg = write_config(tmp_path, [("kestrel", 9, 10)])
    assert cli.main(["conform", "-c", str(cfg), "--modem", "kestrel",
                     "--dst", "N0CRE"]) == 2
    assert "must differ" in capsys.readouterr().err


# -- helpers -----------------------------------------------------------------

def write_config(tmp_path, modems, *, results=None) -> str:
    lines = ["[site]", 'name = "test"', 'mycall = "N0CRE"',
             f'results_dir = "{results or tmp_path / "results"}"']
    for name, cmd_port, data_port in modems:
        lines += ["", "[[modem]]", f'name = "{name}"',
                  f"cmd_port = {cmd_port}", f"data_port = {data_port}",
                  'bandwidth = "2300"']
    path = tmp_path / "site.toml"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def write_result(root, sid, **fields):
    m = SessionMetrics(sid=sid, site="test", **fields)
    d = root / "test" / (m.wall_start[:10] or "2026-07-01") / sid
    d.mkdir(parents=True)
    (d / "transcript.jsonl").write_text(json.dumps(
        {"seq": 1, "t": 0.0, "wall": m.wall_start, "sid": sid, "modem": m.modem,
         "chan": "sys", "dir": "", "kind": "note", "text": "session_end"}) + "\n")
    write_session(d, m)
    return m


# -- report ------------------------------------------------------------------

def test_report_empty_tree(tmp_path, capsys):
    assert cli.main(["report", str(tmp_path)]) == 1
    assert "no sessions" in capsys.readouterr().err


def test_report_aggregate_and_single(tmp_path, capsys):
    root = tmp_path / "results"
    write_result(root, "20260701T120000-kestrel-aaaa", modem="kestrel",
                 scenario="unidir", outcome="ok", wall_start="2026-07-01T12:00:00")
    write_result(root, "20260702T120000-kestrel-bbbb", modem="kestrel",
                 scenario="echo", outcome="failed:connect_timeout",
                 wall_start="2026-07-02T12:00:00")

    assert cli.main(["report", str(root)]) == 0
    out = capsys.readouterr().out
    assert "scenario" in out and "unidir" in out and "echo" in out

    assert cli.main(["report", str(root), "--since", "2026-07-02"]) == 0
    out = capsys.readouterr().out
    assert "echo" in out and "unidir" not in out

    assert cli.main(["report", str(root), "--sid",
                     "20260701T120000-kestrel-aaaa"]) == 0
    out = capsys.readouterr().out
    assert "session 20260701T120000-kestrel-aaaa" in out and "integrity" in out

    assert cli.main(["report", str(root), "--sid", "nope"]) == 1


def test_report_filters_narrow_a_multi_modem_tree(tmp_path, capsys):
    root = tmp_path / "results"
    write_result(root, "20260701T120000-kestrel-aaaa", modem="kestrel",
                 scenario="unidir", outcome="ok", label="before",
                 wall_start="2026-07-01T12:00:00")
    write_result(root, "20260701T130000-sabir-bbbb", modem="sabir",
                 scenario="echo", outcome="ok", label="after",
                 wall_start="2026-07-01T13:00:00")

    assert cli.main(["report", str(root), "--modem", "kestrel"]) == 0
    out = capsys.readouterr().out
    assert "kestrel" in out and "sabir" not in out

    assert cli.main(["report", str(root), "--scenario", "echo"]) == 0
    assert "unidir" not in capsys.readouterr().out

    assert cli.main(["report", str(root), "--label", "after"]) == 0
    assert "sabir" in capsys.readouterr().out

    assert cli.main(["report", str(root), "--label", "nonesuch"]) == 1
    assert "matching the filters" in capsys.readouterr().err


# -- campaign ----------------------------------------------------------------

def test_await_ready_returns_only_the_modems_that_came_up(tmp_path):
    live = FakeModem().start()
    dead = FakeModem(data_listener=False).start()
    try:
        cfg = config_mod.load(write_config(
            tmp_path, [("live", live.cmd_port, live.data_port),
                       ("dead", dead.cmd_port, dead.data_port)]))
        sup = Supervisor(cfg, str(tmp_path / "logs"))
        ready = cli._await_ready(cfg, sup, ["live", "dead"], "127.0.0.1",
                                 timeout=0.5)
        assert ready == ["live"]
    finally:
        live.stop()
        dead.stop()


def test_campaign_runs_the_surviving_subset(tmp_path, capsys, monkeypatch):
    live = FakeModem().start()
    dead = FakeModem(data_listener=False).start()
    ran = []

    def fake_campaign(cfg, runs, **kw):
        ran.extend(runs)
        return [], "no sessions\n"

    monkeypatch.setattr(cli.campaign_mod, "run_campaign", fake_campaign)
    monkeypatch.setattr(cli, "READY_TIMEOUT_S", 0.5)
    try:
        cfg = write_config(tmp_path, [("live", live.cmd_port, live.data_port),
                                      ("dead", dead.cmd_port, dead.data_port)])
        cli.main(["campaign", "-c", cfg, "--all", "--dst", "K7XYZ"])
    finally:
        live.stop()
        dead.stop()
    assert ran and {r.modem for r in ran} == {"live"}
    assert "dropping every run on dead" in capsys.readouterr().err


def _fake_metrics(*outcomes):
    return [SessionMetrics(sid=f"s{i}", modem="m", scenario="unidir",
                           outcome=o) for i, o in enumerate(outcomes)]


@pytest.mark.parametrize("outcomes,fail_under,want", [
    (("ok", "ok"), None, 0),
    (("ok", "failed:tx_stall"), None, 1),          # one bad session fails the gate
    (("ok", "failed:tx_stall"), 50.0, 0),
    (("ok", "failed:tx_stall"), 60.0, 1),
    (("echo_peer",), None, 0),                     # graded by GOOD_OUTCOMES
])
def test_campaign_exit_code_policy(tmp_path, monkeypatch, outcomes, fail_under,
                                   want):
    modem = FakeModem().start()
    try:
        cfg = write_config(tmp_path, [("live", modem.cmd_port, modem.data_port)])
        monkeypatch.setattr(cli.campaign_mod, "run_campaign",
                            lambda *a, **k: (_fake_metrics(*outcomes), "table\n"))
        argv = ["campaign", "-c", cfg, "--all", "--dst", "K7XYZ"]
        if fail_under is not None:
            argv += ["--fail-under", str(fail_under)]
        assert cli.main(argv) == want
    finally:
        modem.stop()


def test_campaign_interval_overrides_the_derived_plan(tmp_path, monkeypatch):
    modem = FakeModem().start()
    seen = []
    try:
        cfg = write_config(tmp_path, [("live", modem.cmd_port, modem.data_port)])
        monkeypatch.setattr(cli.campaign_mod, "run_campaign",
                            lambda c, runs, **k: (seen.extend(runs), ([], "x"))[1])
        cli.main(["campaign", "-c", cfg, "--all", "--dst", "K7XYZ",
                  "--interval", "45"])
    finally:
        modem.stop()
    assert seen and all(r.interval_s == 45.0 for r in seen)


def test_rev_override_relabels_every_modem(tmp_path):
    cfg = config_mod.load(write_config(tmp_path, [("a", 9, 10), ("b", 11, 12)]))
    assert [m.rev for m in cli._with_rev(cfg, "sabir-g3").modems] \
        == ["sabir-g3", "sabir-g3"]
    assert cli._with_rev(cfg, None) is cfg


# -- modems ------------------------------------------------------------------

def test_modems_table_against_fakes(tmp_path, capsys):
    live = FakeModem(version="9.9-fake").start()
    dead = FakeModem(data_listener=False).start()
    try:
        cfg = write_config(tmp_path, [("live", live.cmd_port, live.data_port),
                                      ("dead", dead.cmd_port, dead.data_port)],
                           results=tmp_path / "results")
        write_result(tmp_path / "results", "20260701T120000-live-aaaa",
                     modem="live", scenario="unidir", outcome="ok",
                     wall_start="2026-07-01T12:00:00")
        rc = cli.main(["modems", "-c", cfg, "--timeout", "1"])
    finally:
        live.stop()
        dead.stop()
    out = capsys.readouterr().out
    rows = {line.split()[0]: line for line in out.splitlines()[2:]}
    assert "external" in rows["live"] and "ok" in rows["live"]
    assert "9.9-fake" in rows["live"] and "ago" in rows["live"]
    assert "unreachable" in rows["dead"] or "fail" in rows["dead"]
    assert rc == 1                          # one modem could not be reached


def test_modems_all_reachable(tmp_path, capsys):
    modem = FakeModem().start()
    try:
        cfg = write_config(tmp_path, [("live", modem.cmd_port, modem.data_port)])
        assert cli.main(["modems", "-c", cfg, "--timeout", "1"]) == 0
    finally:
        modem.stop()
    assert "never" in capsys.readouterr().out       # no sessions on record


# -- conform -----------------------------------------------------------------

def test_conform_grades_a_fake_modem(tmp_path, capsys):
    """The fake answers OK/WRONG but never emits BUFFER or CONNECTED, so the
    session probes must FAIL — and the command must exit 1 saying so."""
    modem = FakeModem().start()
    try:
        cfg = write_config(tmp_path, [("fake", modem.cmd_port, modem.data_port)])
        rc = cli.main(["conform", "-c", cfg, "--modem", "fake", "--timeout", "1"])
    finally:
        modem.stop()
    out, err = capsys.readouterr()
    assert "reply_discipline" in out and "probes:" in out
    assert rc == 1 and "FAIL on" in err


def test_conform_golden_deviation_exits_nonzero(tmp_path, capsys):
    modem = FakeModem().start()
    try:
        cfg = write_config(tmp_path, [("fake", modem.cmd_port, modem.data_port)])
        rc = cli.main(["conform", "-c", cfg, "--modem", "fake", "--timeout", "1",
                       "--golden", "kestrel-loopback"])
    finally:
        modem.stop()
    assert rc == 1
    assert "deviations from golden table" in capsys.readouterr().out


def test_conform_unreachable_modem(tmp_path, capsys):
    modem = FakeModem()          # bound but never started: nothing accepts
    modem.stop()
    cfg = write_config(tmp_path, [("gone", modem.cmd_port, modem.data_port)])
    assert cli.main(["conform", "-c", cfg, "--modem", "gone", "--timeout", "1"]) == 2
    assert "not reachable" in capsys.readouterr().err
