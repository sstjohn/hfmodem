# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The bandwidth an ARDOP call goes out at is the one the launcher was given.

`onair.sh ardop NS0A 3586500 200` announced BW200, handed besra `--bandwidth
200`, and the session came up at `ARQBW now 2000MAX`: the host client that
drives the call carried its own default and pushed it over the server's ceiling
after start(), which is what governs the next ConReq. BW200 is the only setting
that has ever moved mail from this station, so a call at 2000 is not a
cosmetic loss.

Off-air entirely: no rig, no card, no socket.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hfhost.ardop import ARQ_BW_TOKENS, _arqbw

_REPO = Path(__file__).resolve().parents[5]
_TOOLS = _REPO / "tools"
sys.path.insert(0, str(_TOOLS))

_off_tree = pytest.mark.skipif(
    not (_TOOLS / "ardop_call.py").exists(),
    reason=f"{_TOOLS} is not in this tree -- ardop_call ships from the repo "
           "rather than the package")


class _Reached(Exception):
    """Far enough: the config is built, and nothing has opened a socket."""


@pytest.fixture
def called(monkeypatch):
    """What ardop_call would hand the host link, without opening one."""
    import ardop_call

    seen: list = []

    def fake_open_link(cfg, host, transcript):
        seen.append(cfg)
        raise _Reached

    monkeypatch.setattr(ardop_call, "open_link", fake_open_link)

    def call(*argv):
        with pytest.raises(_Reached):
            ardop_call.main(["NS0A", "0", *argv])
        return seen[-1]

    return call


@_off_tree
@pytest.mark.parametrize("hz", [200, 500, 1000, 2000])
def test_the_call_asks_for_the_bandwidth_it_was_given(called, hz):
    cfg = called("--bandwidth", str(hz))
    # The configured enum is VARA's and ARDOP's is 200/500/1000/2000, so what
    # reaches the modem is the token derived from it -- MAX, so a session still
    # negotiates down to whatever the path will carry.
    assert _arqbw(cfg) == f"{hz}MAX"
    assert _arqbw(cfg) in ARQ_BW_TOKENS


@_off_tree
def test_a_call_that_names_no_bandwidth_takes_besras_own(called):
    """500 is `run_server --bandwidth`'s default, and the two must not differ:
    a client that widens the server's ceiling is a call at a bandwidth nobody
    asked for."""
    assert _arqbw(called()) == "500MAX"


@_off_tree
def test_a_bandwidth_ardop_has_no_frame_for_is_refused():
    """2300 is VARA's, and the host client rounds it DOWN to 2000 without a
    word. Asked for at the command line it is a typo, on a verb that keys."""
    import ardop_call

    with pytest.raises(SystemExit):
        ardop_call.parser().parse_args(["NS0A", "--bandwidth", "2300"])
