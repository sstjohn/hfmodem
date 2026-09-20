# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What this station calls itself to a Winlink gateway, and who gets to say.

A production CMS refuses a client type it does not know — KX8U's answered ours
with `*** Unknown client types are not allowed on production servers`, ahead of
the `;PQ:` challenge, so no mail could ever move — and the only way past that is
a name the servers accept. What a station announces is a claim its operator
makes about their own station, so it is theirs to set: `[station] client_sid`,
`--mail-sid` for one run, and the shipped default stays our own honest name.

The capability letters are the other half of the SID and are nobody's setting.
They say what this code can do, so they come from the code.

Off-air entirely: no rig, no card, no link.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest
from hfhost.config import ConfigError

from hfmodem.core import config
from hfmodem.winlink import CLIENT_SID, B2FSession, sid_line

_REPO = Path(__file__).resolve().parents[5]
_TOOLS = _REPO / "tools"
sys.path.insert(0, str(_TOOLS))

_off_tree = pytest.mark.skipif(
    not (_TOOLS / "kestrel_connect.py").exists(),
    reason=f"{_TOOLS} is not in this tree -- kestrel_connect ships from the "
           "repo rather than the package")

#: Nothing else in the tree announces this, so a test passing on it cannot be
#: passing on a leftover default.
_SET = "Sparrowhawk-3.2"
_OVERRIDE = "Sparrowhawk-9.9"


@pytest.fixture
def station(tmp_path, monkeypatch):
    """The example station file, announcing something of its own."""
    def _write(sid: str = _SET) -> Path:
        raw = (_REPO / "examples" / "station.toml").read_text(encoding="utf-8")
        out = tmp_path / "station.toml"
        out.write_text(re.sub(r"(?m)^client_sid.*$", f'client_sid = "{sid}"', raw),
                       encoding="utf-8")
        monkeypatch.setenv(config.STATION_ENV, str(out))
        return out
    return _write


# -- the resolver ---------------------------------------------------------

def test_without_a_station_file_we_announce_our_own_name(monkeypatch):
    monkeypatch.delenv(config.STATION_ENV, raising=False)
    assert config.client_sid() == CLIENT_SID == "HFM-0.1"


def test_the_shipped_example_announces_the_shipped_default():
    """The example is copied verbatim to start a station. A different client's
    identifier in it would be everyone's declaration but nobody's choice."""
    raw = (_REPO / "examples" / "station.toml").read_text(encoding="utf-8")
    assert f'client_sid = "{CLIENT_SID}"' in raw


def test_the_station_file_sets_what_we_announce(station):
    station()
    assert config.client_sid() == _SET


def test_a_run_may_still_overrule_the_station(station):
    station()
    assert config.client_sid(_OVERRIDE) == _OVERRIDE


def test_a_station_file_that_will_not_load_names_itself(tmp_path, monkeypatch):
    monkeypatch.setenv(config.STATION_ENV, str(tmp_path / "nothing.toml"))
    with pytest.raises(ConfigError, match=config.STATION_ENV):
        config.client_sid()


@pytest.mark.parametrize("bad", [
    "[HFM-0.1-B2FHM$]",          # the whole SID, brackets and letters and all
    "HFM-0.1-B2FHM$",            # the capability field, which is not theirs
    "HFM–0.1",                   # an en dash out of a word processor
    "",
])
def test_a_client_sid_the_wire_could_not_carry_is_refused(bad, station):
    station(bad)
    with pytest.raises(ConfigError, match="client_sid"):
        config.client_sid()


def test_a_carriage_return_in_the_name_is_refused_at_the_flag():
    """Not a malformed SID: a second protocol line of the setter's choosing,
    injected into the handshake."""
    with pytest.raises(ConfigError, match="client_sid"):
        config.client_sid("HFM-0.1\r;FW: N0CALL")


# -- what reaches the wire ------------------------------------------------

def _handshake(**kw) -> list[str]:
    """The lines a calling session transmits in answer to a greeting."""
    s = B2FSession("W9SSJ", role="calling", target="KX8U", **kw)
    out = s.feed(b"[WL2K-5.0-B2FWIHJM$]\rCMS via KX8U >\r")
    return out.decode("latin-1").split("\r")


def test_the_default_sid_on_the_wire_is_unchanged():
    assert "[HFM-0.1-B2FHM$]" in _handshake()


def test_the_configured_name_is_what_goes_out():
    assert f"[{_SET}-B2FHM$]" in _handshake(client_sid=_SET)


@pytest.mark.parametrize("name", [CLIENT_SID, _SET, "RMS Express-1.5.36.0"])
def test_the_capability_letters_are_the_codes_promise_and_not_a_setting(name):
    """B2 forwarding, F compression, H hierarchical addresses, M message IDs.
    Every one is implemented in `winlink/session.py`, so no operator setting
    reaches them — announcing a capability we do not have would break a session
    further in and far more obscurely than being refused at the greeting."""
    assert sid_line(name).endswith("-B2FHM$]")
    assert f"[{name}-B2FHM$]" in _handshake(client_sid=name)


@_off_tree
def test_the_configured_name_reaches_the_vara_mail_path(station):
    import kestrel_connect

    station()
    args = types.SimpleNamespace(
        mail_send=None, mail_fetch=True, mycall="W9SSJ", gateway="KX8U",
        mail_to="", mail_subject="", mail_password="", mail_sid="")
    _, session = kestrel_connect.attach_mail(
        args, types.SimpleNamespace(send=lambda blob: None),
        types.SimpleNamespace())
    assert session.sid == f"[{_SET}-B2FHM$]"


#: Every program that calls a gateway must take its client type from the
#: resolver. One that constructs its own announces `HFM-0.1` whatever the
#: operator set, which is a setting that can be changed and does nothing — the
#: exact shape `[audio] tx_drive` was in before it was wired up.
_CALLERS = re.compile(r"B2FSession\((?:[^()]|\([^()]*\))*\)", re.S)


def test_every_path_that_calls_a_gateway_takes_its_name_from_the_resolver():
    sources = [p for p in (_REPO / "packages").rglob("*.py")
               if not {"tests", "winlink", "build"} & set(p.parts)]
    # tools/ ships from the repo rather than the package, so this is empty in a
    # distribution. The packaged paths still carry the gate; there is nothing to
    # skip for, because there is nothing there to have got wrong.
    sources += sorted(_TOOLS.glob("*.py"))
    missing = [f"{p}: {call.split(chr(10))[0]}"
               for p in sources
               for call in _CALLERS.findall(p.read_text(encoding="utf-8"))
               if 'role="calling"' in call
               and "client_sid=config.client_sid(" not in "".join(call.split())]
    assert not missing, (
        "these build an outbound session without client_sid=config.client_sid("
        "…), so they announce the shipped name whatever the operator set:\n"
        + "\n".join(missing))
