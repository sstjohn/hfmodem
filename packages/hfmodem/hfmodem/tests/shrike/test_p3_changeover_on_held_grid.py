# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A changeover with no placement against the peer keys on the grid we hold.

An aged packet/control pair, or a comb the clamp says sits in front of the
peer's reader, used to refuse the changeover ("CHANGEOVER NOT PLACED -- NOT
KEYING"), and a refused slot is a missed one. The reason is warned and the
held comb stands in.
"""
import pytest

from hfmodem.shrike import arq
from hfmodem.tests.shrike.test_p3_breakin_timing import (
    PAYLOAD, STATUS, changeover_lead, duplex, observed_reply, transitioned)

SLOT = 44


def _grid(problem):
    g = transitioned()
    if problem == 'clamp':
        observed_reply(g)
        g.anchor -= 1000
    else:
        g.note_p3_packet(2568704, 38880, 60000)
        assert g._p3_reply_timing() is None
    return g


@pytest.mark.parametrize('problem,reason', [
    ('aged_pair', 'no fresh corroborated'),
    ('clamp', 'boundary sits'),
])
def test_an_unplaceable_changeover_keys_on_the_held_grid(problem, reason, tmp_path, capsys):
    g = _grid(problem)
    tx = duplex(g, tmp_path)
    tx._flip = lambda: False
    tx.breakin_due = True
    tx.aim(g, SLOT)
    held = tx._breakin_reference(g, SLOT)
    assert tx._place_breakin() == ' on the held grid' and tx.placed
    assert tx.boundary == held and reason in tx.unplaceable

    tx.aim(g, SLOT)
    tx.breakin_due = True
    carrier = held - changeover_lead()
    tx.live.now = tx.live.pos = carrier - 2000 - tx.live.key_notice
    assert tx.send_packet(3, PAYLOAD, STATUS, breakin=True) != arq.REFUSED
    said = capsys.readouterr().out
    assert 'CHANGEOVER ON THE HELD GRID' in said and reason in said
    assert 'NOT KEYING' not in said
    assert len(tx.live.emissions) == 1 and not tx.refused
    assert tx.slot == SLOT and tx.boundary == carrier
