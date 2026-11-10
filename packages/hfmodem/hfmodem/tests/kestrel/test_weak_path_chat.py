# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A weak but audible path holds a chat, W9SSJ-1 to W9SSJ, 7105 kHz, 2026-10-28.

The live link read about 19 on the tone measure one way and about 9 the other.
It came up, the answerer's first over drew an answer it could not read, its
queries went unread, and it closed. On simulated air at the same two readings
both stations must carry several overs each way, through turn changes, with
the weak direction held at the low levels it reads.
"""
import collections
import re
from types import SimpleNamespace

import pytest

from hfmodem.contact import varac as v
from hfmodem.contact.bbs import HostCatalog
from hfmodem.kestrel.vara import vara_arq as A
from hfmodem.tests.contact.test_kestrel_pair import Channel, Peer

CALLER, CALLED = 'W9SSJ-1', 'W9SSJ'


class OneWay(Channel):
    snrs = (-3., -17.)          # what station 0 (callee) and 1 (caller) receive at

    @property
    def snr(self):
        return self.snrs[self.active]

    @snr.setter
    def snr(self, value):
        pass


@pytest.mark.parametrize('snrs', [(-3., -17.), (-17., -3.)])
def test_a_weak_path_holds_a_chat_of_several_overs(tmp_path, monkeypatch, snrs):
    channel = OneWay(None, 65536)
    channel.snrs = snrs
    channel.trace = collections.deque(maxlen=200000)
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=channel.clock))
    monkeypatch.setitem(v.CALLING_HZ, 20, 14_074_500)
    callee = Peer(channel, CALLED, 500, tmp_path / 'a', HostCatalog())
    caller = Peer(channel, CALLER, 500, tmp_path / 'b', HostCatalog())
    caller.command(f'/call {CALLED} cf')
    channel.until(lambda: all(p.client.connected and p.session.steps is None
                              for p in (callee, caller)), 'connect')
    lines = [(caller, callee, 'de W9SSJ-1 portable, weak here'),
             (callee, caller, 'de W9SSJ copy, you are weak but solid here in Madison, '
                              'running 25 watts to a dipole, how copy?'),
             (callee, caller, 'name Saul, qth Madison'),
             (caller, callee, 'ok 73 and thanks, a longer line from the portable end '
                              'to make it run more than one over'),
             (callee, caller, 'tu 73')]
    for sender, receiver, text in lines:
        sender.command(text)
        channel.until(lambda: any(text in line for line in receiver.said),
                      f'chat delivery from {sender.call}', seconds=300)
        assert all(p.client.connected for p in (callee, caller)), '\n'.join(channel.trace)
    weak = snrs.index(min(snrs))
    read = collections.Counter(
        (int(m[1]), int(m[2])) for m in re.finditer(
            r'^(\d): rx BW500 burst: \d frame\(s\) at level (\d+)', '\n'.join(channel.trace), re.M))
    assert sum(n for (who, _), n in read.items() if who == weak) >= 5, read
    assert all(level <= 2 for (who, level) in read if who == weak), read
    assert not any('closing' in line for line in channel.trace)
