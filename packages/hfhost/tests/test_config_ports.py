# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from hfhost.config import ConfigError, parse_modems

import pytest


def _m(name, cmd, data, dialect="vara"):
    return {"name": name, "cmd_port": cmd, "data_port": data, "dialect": dialect}


def test_single_socket_modems_do_not_collide_on_the_unused_port():
    """data_port 0 means 'no second socket'. Two structured-dialect modems both
    leave it at 0 and are not thereby sharing a port."""
    modems = parse_modems([_m("a", 8340, 0, "hostapi"),
                           _m("b", 8350, 0, "hostapi")], ())
    assert [m.name for m in modems] == ["a", "b"]


def test_real_port_collisions_are_still_caught():
    with pytest.raises(ConfigError, match="used by both"):
        parse_modems([_m("a", 8300, 8301), _m("b", 8301, 8302)], ())
