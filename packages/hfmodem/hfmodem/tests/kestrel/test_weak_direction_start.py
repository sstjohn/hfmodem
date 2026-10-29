# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The weaker direction's reading starts the link, carried in the link-setup's ask.

W9SSJ-1 called W9SSJ on 7105 kHz, 2026-10-28 16:57Z: the callee heard the
requests at 17-21 on the tone measure and offered level 4, the caller heard the
answers at 7.1-8.6 and logged "asking its greeting at level 1", and the link-setup
it keyed still asked 6. The ask is the one on-wire field a caller has for its own
reading, and stock high speed capable callers grade it: 06 on the clean bench
tapes, 05 from KD2WNY and 01 from KQ4QWH on the air.
"""
import collections
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F
from .test_idle_poll import clock as clock, _IO as _IO
from .test_responder_initial_greeting import confirm

CALLER, CALLED = 'W9SSJ-1', 'W9SSJ'


@pytest.mark.parametrize('ask,wire', [(4, 6), (3, 3), (2, 2), (1, 1)])
def test_a_high_speed_link_setup_asks_what_its_caller_heard(ask, wire):
    frame = F.link_setup_frame(CALLER, bw='500', ask=ask, high_speed=True)
    assert F.link_setup_marks(frame) == (True, wire)


def answerer(*, marked, chat, ask):
    io = _IO()
    hs = A.VaraStationHandshake([CALLED], io, bw='500')
    hs.role, hs.called, hs.caller = 'responder', CALLED, CALLER
    hs.state = A.VaraState.CONNECTING
    hs.chat = hs._chat_profile = chat
    hs.high_speed = True
    assert hs._take_link_setup(F.link_setup_frame(CALLER, bw='500', ask=ask, high_speed=marked))
    return hs, io


@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('ask', [1, 2, 3])
def test_a_free_answerer_keys_its_first_over_at_the_ask(chat, ask):
    hs, _ = answerer(marked=False, chat=chat, ask=ask)
    assert hs._rec500 == A._phy.level_key('500', ask)


def test_a_free_answerer_asked_four_stays_at_its_base():
    hs, _ = answerer(marked=False, chat=False, ask=4)
    assert hs._rec500 == A._phy.base_level('500')


@pytest.mark.parametrize('ask', [1, 2])
def test_under_chat_the_confirm_cannot_raise_the_start_past_the_ask(ask):
    hs, _ = answerer(marked=True, chat=True, ask=ask)
    confirm(hs)
    assert hs._rec500 == A._phy.level_key('500', ask)


def test_without_chat_the_confirm_still_starts_the_first_over():
    hs, _ = answerer(marked=True, chat=False, ask=1)
    confirm(hs)
    assert hs._rec500 == A._phy.level_key('500', 7)


def weak_caller(ask, *, low_confirm=False, chat=False):
    io = _IO()
    hs = A.VaraStationHandshake([CALLER], io, bw='500', low_confirm=low_confirm)
    hs.role, hs.called, hs.caller = 'initiator', CALLED, CALLER
    hs.state, hs.step = A.VaraState.CONNECTED, A._I_CONNECTED
    hs.high_speed = hs._link_high_speed = True
    hs.chat = hs._chat_profile = chat
    hs._setup_ask, hs._heard_db = ask, 8.9
    hs._tx_session_confirm()
    return hs, io


def keyed_confirm(io, start):
    from hfmodem.kestrel.vara import vara_mfsk as M
    kind = next(k for k, lv in A._CONFIRM_START.items() if lv == start)
    return np.array_equal(io.sent[-1], M.synth_burst(CALLED, F.for_bw(kind, '500')))


@pytest.mark.parametrize('ask', [1, 2, 3])
def test_by_default_a_weak_callers_confirm_stays_canonical_and_expects_it(ask):
    """Confirm frames 0-2 have never been seen from or answered by a stock
    station; the default keys frame 3, and reads the greeting where it says."""
    hs, io = weak_caller(ask)
    assert keyed_confirm(io, 4)
    assert hs._expect_level == 4 == hs._confirm_start


@pytest.mark.parametrize('ask', [1, 2, 3])
def test_under_chat_a_weak_caller_expects_the_greeting_at_its_ask(ask):
    hs, io = weak_caller(ask, chat=True)
    assert keyed_confirm(io, 4)
    assert hs._expect_level == ask


@pytest.mark.parametrize('ask', [1, 2, 3])
def test_the_low_confirm_option_names_the_ask(ask):
    hs, io = weak_caller(ask, low_confirm=True)
    assert keyed_confirm(io, ask)
    assert hs._expect_level == ask == hs._confirm_start


def test_weak_direction_greeting_is_read_at_level_one(tmp_path, monkeypatch):
    """Simulated air, one way at -3 dB and the other at -17 dB: the caller reads
    the callee's connect-response at about 9 on the tone measure, the callee
    reads the request at about 19, and the callee's greeting must come at the
    level the caller can read."""
    from hfmodem.contact import varac as v
    from hfmodem.contact.bbs import HostCatalog
    from hfmodem.tests.contact.test_kestrel_pair import Channel, Peer

    class OneWay(Channel):
        snrs = (-3., -17.)          # what station 0 (callee) and 1 (caller) receive at

        @property
        def snr(self):
            return self.snrs[self.active]

        @snr.setter
        def snr(self, value):
            pass

    channel = OneWay(None, 65536)
    channel.trace = collections.deque(maxlen=100000)
    monkeypatch.setattr(A, 'time', SimpleNamespace(monotonic=channel.clock))
    monkeypatch.setitem(v.CALLING_HZ, 20, 14_074_500)
    callee = Peer(channel, CALLED, 500, tmp_path / 'a', HostCatalog())
    caller = Peer(channel, CALLER, 500, tmp_path / 'b', HostCatalog())
    caller.command(f'/call {CALLED} cf')
    channel.until(lambda: callee.modem.confirmed_total > 0, 'callee greeting delivered')
    trace = list(channel.trace)
    assert any('asking level 1' in line for line in trace if line.startswith('1: tx link-setup'))
    assert any(line.startswith('1: rx BW500 burst: 1 frame(s) at level 1') for line in trace)
