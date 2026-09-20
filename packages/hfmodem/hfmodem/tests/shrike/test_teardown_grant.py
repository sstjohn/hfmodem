# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Two defects an aligned third-party witness found in the WS8EOC arm of
2026-09-11.

A KiwiSDR 160 mi from the gateway, registered to our own cycle grid to a median
residual of 0.2 ms, says the peer answered 42 of the arm's 45 cycles and
commanded PACTOR-3 in the last seven of them. Our receiver read eight cycles and
five grants, and the session keyed no entry packet and reported no reason.

  * every grant we did read arrived after `max retries -> QRT`, and
    `ptc.PtcHost._take_grant` dropped each one with no line at all; and
  * `--retries 20` was passed on every arm of the evening and reached
    `max_connect_retries`, while the counter that spends a LIVE link into that
    QRT stayed at its default.

The two meet in the counterfactual at the bottom: on the budget the operator
believed was flown, the same grant lands in a link that is still up and the
entry packet goes out.
"""
from __future__ import annotations

import pytest

from hfmodem.shrike import onair, pactor1, rxfront
from hfmodem.shrike.arq import ArqConfig, State
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_armdefaults import _parsed
from hfmodem.tests.shrike.test_grid import _Rig
from hfmodem.tests.shrike.test_identify import Clock
from hfmodem.tests.shrike.test_p3_offer import calling_station

GRANT = rxfront.Event(0.1, "unassigned", "0x59A", protocol=Protocol.PACTOR1,
                      spare=pactor1.CS_59A)
"""The word itself, as `rxfront` hands it up -- the arm read five of these at
zero bit errors and the witness read seven."""


def _spent_link(budget: int | None = None):
    """A calling ISS in PACTOR-1 that has just spent its retry budget on a peer
    it could not hear, which is where the arm stood when the grant arrived."""
    host, keyed = calling_station()
    if budget is not None:
        host.arq.cfg.max_retries = budget
    host.arq.on_host_data(b"TEST DE W9SSJ -- " + b"X" * 200)
    for _ in range(ArqConfig().max_retries + 2):
        host.tick()
    return host, keyed


def test_a_grant_read_in_the_teardown_is_refused_and_said_out_loud():
    host, _ = _spent_link()
    assert host.arq.state == State.DISCONNECTING
    said = len(host.log_lines)
    host.on_rx_event(GRANT)

    # The goodbye is already on the air and PACTOR-1 has nothing that un-says
    # one, so the upgrade is declined -- but the session is told, and told which
    # state declined it.
    assert host.protocol is Protocol.PACTOR1
    assert not host.arq.entry_pending
    line = next(ln for ln in host.log_lines[said:] if "0x59A grant" in ln)
    assert "REFUSED" in line and State.DISCONNECTING in line
    assert "goodbye" in line and "budget" in line


def test_the_grant_is_still_taken_with_the_link_up():
    """The refusal is about the teardown and nothing else: the same word, the
    same station, one cycle earlier."""
    host, keyed = _spent_link(budget=20)
    assert host.arq.state == State.CONNECTED
    host.on_rx_event(GRANT)
    assert host.protocol is Protocol.PACTOR3 and host.arq.entry_pending
    assert keyed.packets[-1][0] is Protocol.PACTOR3


def _wired(tmp_path, monkeypatch, *argv: str):
    """The config a session actually runs on, read off the host `run` builds."""
    built = []
    monkeypatch.setattr(onair.ota, "Rig", lambda *a, **kw: _Rig())
    monkeypatch.setattr(onair, "find_device", lambda *a, **kw: 0)
    monkeypatch.setattr(onair, "_LiveInput", lambda *a, **kw: Clock())
    monkeypatch.setattr(onair, "_save_capture_async", lambda *a, **kw: None)
    host_cls = onair.PtcHost
    monkeypatch.setattr(onair, "PtcHost",
                        lambda **kw: built.append(host_cls(**kw)) or built[-1])
    onair.run(_parsed("--transmit", "--serial", "/dev/null",
                      "--dial", "7100000", "--outdir", str(tmp_path),
                      "--pactor1-only", "--hold", "1", "--max-cycles", "1",
                      *argv))
    return built[0].arq.cfg


def test_link_retries_reaches_the_counter_that_ends_a_session(
        tmp_path, monkeypatch, capsys):
    cfg = _wired(tmp_path, monkeypatch, "--retries", "20", "--link-retries", "20")
    assert cfg.max_connect_retries == 20
    assert cfg.max_retries == 20
    # Neither number is a default any more, so both are said where the arm's own
    # transcript keeps them.
    banner = next(ln for ln in capsys.readouterr().out.splitlines()
                  if "RETRY BUDGETS" in ln)
    assert "connect 20" in banner and "live link 20" in banner


def test_retries_alone_is_the_connect_budget_and_says_so(
        tmp_path, monkeypatch, capsys):
    cfg = _wired(tmp_path, monkeypatch, "--retries", "20")
    assert cfg.max_connect_retries == 20
    assert cfg.max_retries == ArqConfig().max_retries
    assert f"live link {cfg.max_retries}" in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
