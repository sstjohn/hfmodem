# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace
import threading
import time

import pytest

from hfmodem.bbs.config import BBSConfig
from hfmodem.bbs.core import BBSCore
from hfmodem.bbs.store import Store
from hfmodem.tests.bbs.test_core import Game, drain, settle
from hfmodem.tests.bbs import test_varac_entrance as radio

board = radio.board


def paused_delivery(d, core):
    """Wait for one real wire page without the usual test helper's auto-MORE."""
    for _ in range(400):
        d.tick()
        if not d.host.client.data_writer.pending:
            d.hear('BUFFER 0')
        if core.session.paused and not d.session.bbs.inflight:
            return
        time.sleep(.001)
    pytest.fail('Radio did not reach the caller-controlled pager boundary')


def test_default_radio_entry_is_one_short_plain_frame(board):
    d, core, *_ = board
    start = len(radio.sent(d))
    radio.enter(board)
    frames = radio.sent(d)[start:]
    assert frames == ['Radio Board\nMain: D doors; ? help']
    assert len(frames[0].encode()) < 64
    assert core.session.prefs['expert'] and core.session.presentation.mode == 'plain'
    d.receive('D')
    radio.drive(d, core)
    assert radio.sent(d)[-1] == 'Doors: 1 adventure\nDoor: ? help; Q back'
    assert len(radio.sent(d)) == start + 2


def test_verbose_content_and_classic_menus_require_explicit_request(tmp_path):
    config = BBSConfig(state_root=tmp_path, title='Tiny BBS', news='Long news. ' * 100,
                       sysop='Operator')
    with Store(tmp_path) as store:
        core = BBSCore(config, store, games={'zork': Game})
        try:
            sid = core.open('K1ABC')
            assert drain(core, sid) == 'Tiny BBS\nMain: D doors; ? help'
            core.input(sid, 'help', '?')
            assert '[D] Doors' in drain(core, sid)
            assert core.session.prefs['expert']  # one-shot help is not a persistent opt-in
            core.input(sid, 'news', 'NEWS')
            assert 'Long news.' in drain(core, sid)
            core.input(sid, 'classic', 'MENUS ON')
            assert '[D] Doors' in drain(core, sid)
            assert not store.preferences(core.session.player.id)['expert']
            assert core.session.presentation.mode == 'plain'  # menus do not enable ANSI
            core.input(sid, 'terse', 'MENUS OFF')
            assert drain(core, sid) == 'Preferences saved.\nTiny BBS\nMain: D doors; ? help'
            core.close(sid)
            settle(core)
            sid = core.open('K1ABC-2')
            assert drain(core, sid) == 'Tiny BBS\nMain: D doors; ? help'
        finally:
            core.shutdown()
            settle(core)
            core.shutdown()


def test_radio_paging_explains_continuation_and_help_without_auto_more(board):
    d, core, *_ = board
    core.config = replace(core.config, news='News line.\n' * 100)
    radio.enter(board)
    start = len(radio.sent(d))
    d.receive('NEWS')
    paused_delivery(d, core)
    frames = radio.sent(d)[start:]
    assert len(frames) == 1
    assert frames[0].endswith('[BBS MORE | R | Q]')
    assert len(frames[0].encode()) <= 600
    pending = len(core.session.pending)
    d.receive('?')
    paused_delivery(d, core)
    assert radio.sent(d)[-1] == 'BBS MORE; R repeat; Q skip'
    assert len(core.session.pending) == pending
    d.receive('R')
    paused_delivery(d, core)
    assert radio.sent(d)[-1] == frames[0]
    assert len(core.session.pending) == pending
    d.receive('BBS MORE')
    paused_delivery(d, core)
    assert len(core.session.pending) == pending - 1
    assert len(radio.sent(d)) == start + 4


def test_color_setup_uses_real_attachment_negotiation(board):
    d, core, *_ = board
    radio.enter(board)
    d.receive('P')
    radio.drive(d, core)
    d.receive('C')
    radio.drive(d, core)
    assert radio.sent(d)[-1] == 'Color: BBS TERM ANSI16 or BBS TERM PLAIN.'
    assert core.session.presentation.mode == 'plain'
    assert core.store.preferences(core.session.player.id)['color'] == 'plain'
    for mode in ('ANSI16', 'PLAIN'):
        d.receive('BBS TERM ' + mode)
        radio.drive(d, core)
        core._display('\x1b[31mRed\x1b[0m')
        radio.drive(d, core)
        assert ('\x1b[' in radio.sent(d)[-1]) == (mode == 'ANSI16')


def test_inflight_preference_write_refuses_stale_toggle_then_allows_retry(board, monkeypatch):
    d, core, *_ = board
    radio.enter(board)
    d.receive('P')
    radio.drive(d, core)
    entered, release = threading.Event(), threading.Event()
    original = core.store.set_preferences
    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)
    monkeypatch.setattr(core.store, 'set_preferences', gated)
    try:
        assert core.input(core.session.id, 'classic', 'MENUS ON')
        assert entered.wait(1)
        assert not core.input(core.session.id, 'early-toggle', 'E')
    finally:
        release.set()
    radio.drive(d, core)
    assert not core.session.prefs['expert']
    assert core.input(core.session.id, 'retry-toggle', 'E')
    radio.drive(d, core)
    assert core.session.prefs['expert']
