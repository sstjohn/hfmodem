# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
import time
import threading

import pytest

from hfmodem.bbs.config import BBSConfig, Limits
from hfmodem.bbs.core import BBSCore
from hfmodem.bbs.frontends.varac import VarACEntrance, escape_markup
from hfmodem.bbs.store import Store
from hfmodem.bbs.policy import ConnectionLease
from hfmodem.contact import varac as v
from hfmodem.contact.bbs import HostCatalog
from hfmodem.contact.bbs_download import decode_file
from hfmodem.contact.bbs_server import BBSServer
from hfmodem.contact.filetransfer import parse_offer
from hfmodem.tests.bbs.test_core import Game, settle
from hfmodem.tests.contact.test_varac import Desk


@pytest.fixture
def board(tmp_path):
    d = Desk(mailbox=tmp_path / 'mail')
    clock = lambda: d.host.now
    utc = lambda: datetime(2026, 10, 26, tzinfo=timezone.utc) + timedelta(seconds=clock())
    config = BBSConfig(state_root=tmp_path / 'state', title='Radio Board')
    store = Store(config.state_root)
    made = []

    def factory():
        game = Game()
        made.append(game)
        return game

    exports = tmp_path / 'exports'
    exports.mkdir()
    (exports / 'hello.txt').write_text('Bulletin hello\n')
    catalog = HostCatalog()
    catalog.enable(exports)
    core = BBSCore(config, store, games={'adventure': factory}, catalog=catalog, clock=clock, utcnow=utc)
    d.session.bbs_host = BBSServer(d.session, catalog)
    d.session.bbs = VarACEntrance(d.session, core)
    d.flow.dial = 14_105_750
    d.hear('CONNECTED W9SSJ KQ4QWH 500')
    yield d, core, made, catalog
    d.session.bbs.ended()
    d.host.client.close()
    core.shutdown()
    settle(core)
    core.shutdown()
    if d.session.bbs_host._executor:
        d.session.bbs_host._executor.shutdown(wait=True, cancel_futures=True)
    catalog.close()
    store.close()


def drive(d, core):
    """Supply generation-fresh BUFFER 0 after actual serialized writes."""
    for _ in range(400):
        d.tick()
        if not d.host.client.data_writer.pending:
            d.hear('BUFFER 0')
        s = core.session
        if s and s.paused and not d.session.bbs.inflight:
            d.receive('BBS MORE')
        if (not core._jobs and not d.session.bbs.inflight and not d.host.client.data_writer.pending
                and (not s or not s.pending and not s.outstanding)
                and s and s.state in ('lobby', 'playing')):
            return
        time.sleep(.001)
    pytest.fail('BBS application did not settle')


def enter(board):
    d, core, *_ = board
    d.receive('BBS')
    assert d.session.bbs.active
    drive(d, core)
    assert core.session.state == 'lobby'


def play(board):
    d, core, *_ = board
    enter(board)
    d.receive('PLAY adventure')
    drive(d, core)
    assert core.session.state == 'playing'


def sent(d):
    return [raw.split(' ', 1)[1] for raw in d.texts]


def test_explicit_entry_after_readiness_normal_chat_stays_chat(board):
    d, core, *_ = board
    d.receive('hello')
    assert core.session is None
    d.flow.dial = v.CALLING_HZ[20]
    d.receive('BBS')
    assert not d.session.bbs.active
    d.flow.dial = 14_105_750
    drive_writer(d)
    enter(board)
    assert any('Radio Board' in text for text in sent(d))


def drive_writer(d):
    for _ in range(100):
        d.tick()
        if not d.host.client.data_writer.pending:
            d.hear('BUFFER 0')
            return
        time.sleep(.001)
    pytest.fail('Writer did not settle')


def test_fragmented_coalesced_messages_once_and_legitimate_repeats(board):
    d, core, games, _ = board
    play(board)
    frame = v.frame_message('north')
    for byte in frame:
        d.host.data.incoming.append(bytes([byte]))
        d.tick()
    drive(d, core)
    d.receive('north')
    drive(d, core)
    assert games[0].submissions == [('north', 'default'), ('north', 'default')]
    d.receive('north<SM>mixed protocol')
    assert games[0].submissions == [('north', 'default'), ('north', 'default'),
                                    ('north<SM>mixed protocol', 'default')]
    before = len(games[0].submissions)
    d.receive('<BLR>north')
    assert len(games[0].submissions) == before


def test_game_markup_never_becomes_protocol_and_six_hundred_byte_bound(board):
    d, core, games, _ = board
    play(board)
    from hfmodem.bbs.games.base import GameEvent
    games[0].events = [GameEvent('text', '<DISC><QSYF><SM><BLR>&IT&' * 50), GameEvent('prompt', '> ', 'line')]
    baseline = len(sent(d))
    drive(d, core)
    output = sent(d)[baseline:]
    assert output and all(len(text.encode()) <= 600 for text in output)
    assert all('<' not in text and '>' not in text and '&IT&' not in text for text in output)
    assert len(escape_markup('<&IT&>')) == len('<&IT&>')


def test_stale_zero_buffer_cannot_release_outstanding_page(board):
    d, core, *_ = board
    d.host.data.respond = lambda payload: None
    d.hear('BUFFER 0')
    d.receive('BBS')
    for _ in range(100):
        d.tick()
        if d.session.bbs.inflight and d.session.bbs.inflight['after'] is not None:
            break
        time.sleep(.001)
    event = core.session.outstanding
    assert event is not None
    for _ in range(5):
        d.tick()
    assert core.session.outstanding.output_id == event.output_id
    d.hear('BUFFER 0')
    assert core.session.outstanding is None


def test_game_refuses_native_files_vmail_email_and_operator_chat(board):
    d, core, games, _ = board
    play(board)
    d.receive('<BLR>')
    drive_writer(d)
    assert sent(d)[-1] == '<BLJ>'
    d.receive('<BG:hello.txt>')
    drive_writer(d)
    assert sent(d)[-1] == '<BGJ>'
    count = len(sent(d))
    d.receive('<SM><SBJ:secret><MSG:mail>')
    d.receive('<E>')
    d.session.command('operator chat')
    d.session.command('/email send a@b c@d subject')
    drive_writer(d)
    assert len(sent(d)) == count + 1 and sent(d)[-1] == '<SMF>'
    assert games[0].submissions == []
    assert not d.session.mailbox.entries()
    assert not d.session.email.waiting


def test_lobby_get_uses_server_offer_and_receipt_releases_lease(board):
    d, core, *_ = board
    enter(board)
    d.receive('FILES')
    drive(d, core)
    d.receive('GET 1')
    for _ in range(100):
        d.tick()
        if d.session.bbs_host.phase == 'offer':
            break
        time.sleep(.001)
    drive_writer(d)
    offer = parse_offer(sent(d)[-1])
    assert core.session.state == 'transfer'
    assert not any(text.startswith('<BG:') for text in sent(d))
    d.receive('<SFRD>')
    drive_writer(d)
    assert decode_file(offer, sent(d)[-1]) == b'Bulletin hello\n'
    d.receive('<SFOK>')
    drive(d, core)
    assert core.session.state == 'lobby' and not d.session.bbs_host.active


def test_native_bg_from_settled_lobby_uses_same_transfer_lease(board):
    d, core, *_ = board
    enter(board)
    d.receive('<BG:hello.txt>')
    for _ in range(100):
        d.tick()
        if d.session.bbs_host.phase == 'offer':
            break
        time.sleep(.001)
    assert core.session.state == 'transfer'
    d.receive('<SFFA>')
    drive(d, core)
    assert core.session.state == 'lobby' and not d.session._closing


def test_hard_deadline_precedes_sfrd_and_prepared_file_output(board):
    d, core, *_ = board
    enter(board)
    d.receive('FILES')
    drive(d, core)
    d.receive('GET 1')
    for _ in range(100):
        d.tick()
        if d.session.bbs_host.phase == 'offer':
            break
        time.sleep(.001)
    drive_writer(d)
    before = len(sent(d))
    d.host.now = d.session.bbs.connection.deadline
    d.receive('<SFRD>')
    assert len(sent(d)) == before
    assert not d.host.client.connected and core.session.state == 'closed'
    assert not any('<DISC>' in text for text in sent(d)[before:])


def test_qsy_stop_disconnect_fence_game_and_never_renew_lease(board):
    d, core, games, _ = board
    play(board)
    deadline = d.session.bbs.connection.deadline
    d.session.bbs.detach('QSY')
    assert games[0].closed
    settle(core)
    assert d.session.bbs.connection.deadline == deadline
    d.hear('DISCONNECTED')
    assert d.session.bbs.connection is None


def test_poisoned_writer_uses_host_control_no_farewell_or_disc(board):
    d, core, *_ = board
    d.host.data.sendall = lambda payload: (_ for _ in ()).throw(OSError('partial write'))
    d.receive('BBS')
    for _ in range(100):
        d.tick()
        if d.host.client._stream_poisoned:
            break
        time.sleep(.001)
    assert d.host.client._stream_poisoned and not d.host.client.connected
    assert core.session.state == 'closed'
    assert 'DISCONNECT' in d.host.sent and 'ABORT' in d.host.sent
    assert not any('<DISC>' in text for text in sent(d))


def test_native_client_plain_optin_filter_and_blank_enter(tmp_path):
    d = Desk(mailbox=tmp_path)
    rendered = []
    d.session.bbs.say = rendered.append
    d.flow.dial = 14_105_750
    d.hear('CONNECTED W9SSJ KQ4QWH 500')
    try:
        d.session.command('/bbs enter')
        drive_writer(d)
        d.receive('\x1b[31mred\x1b]52;c;clipboard\x07')
        assert rendered == ['red']
        d.session.command('/bbs term ansi16 80 24')
        drive_writer(d)
        d.receive('BBS TERM OK ANSI16 80 24')
        d.receive('\x1b[91mcolor\x1b[2J')
        assert rendered[-1].endswith('\x1b[0m') and '\x1b[2J' not in rendered[-1]
        d.session.command('')
        drive_writer(d)
        assert sent(d)[-1] == 'BBS MORE' and not d.session._closing
        assert not any('KQ4QWH:' in line for line in rendered)
        d.receive('BBS EXIT OK')
        assert not d.session.bbs.remote_view
    finally:
        d.host.client.close()


def test_operator_reload_applies_revision_on_owner_tick(board):
    d, core, *_ = board
    enter(board)
    future = Future()
    d.session.bbs.reload = lambda: future
    d.session.command('/bbs reload')
    assert d.session.bbs.reload_future is future
    old = core.quota.revision
    from dataclasses import replace
    future.set_result(replace(core.config, limits=Limits(900, 3600, 0)))
    d.tick()
    assert core.quota.revision == old + 1
    assert any('reloaded' in text for text in d.legacy_said)


def test_operator_stop_interrupts_blocked_writer_before_clear_and_abort(board):
    d, core, *_ = board
    started, cancelled = threading.Event(), threading.Event()

    def blocked(payload):
        started.set()
        if not cancelled.wait(2):
            raise AssertionError('Writer was not interrupted by stop')
        raise OSError('socket interrupted')

    d.host.data.sendall = blocked
    d.host.data.shutdown = lambda how: cancelled.set()
    d.receive('BBS')
    for _ in range(100):
        d.tick()
        if started.is_set():
            break
        time.sleep(.001)
    assert started.is_set()
    d.session.command('/stop')
    assert cancelled.is_set() and d.host.client._stream_poisoned
    assert core.session.state == 'closed'
    assert 'CLEANTXBUFFER' in d.host.sent and 'DISCONNECT' in d.host.sent
    with pytest.raises(v.HostError):
        d.host.client.send_text('must not follow a poisoned frame')


def test_completed_native_preparation_cannot_send_at_hard_expiry(board):
    d, core, _, catalog = board
    enter(board)
    host = d.session.bbs_host
    host.application_lease = d.session.bbs.native_file_lease()
    assert host.application_lease
    host._bind(catalog.generation)
    host.name, host.phase, host.deadline = '<BLR>', 'preparing', d.host.now + 60
    future = Future()
    host._future = future
    future.set_result(('directory', catalog.list_files(d.session.peer)))
    baseline = len(sent(d))
    d.host.now = d.session.bbs.connection.deadline
    d.tick()
    assert len(sent(d)) == baseline
    assert core.session.state == 'closed'


def test_unknown_late_identity_selects_policy_from_original_link_start(board):
    d, core, *_ = board
    from dataclasses import replace
    core.config = replace(core.config, callsigns={'K1ABC': Limits(900, 7200, 3600)})
    d.hear('DISCONNECTED')
    d.flow.dial = 14_105_750
    d.hear('CONNECTED W9SSJ W9SSJ 500')
    lease = d.session.bbs.connection
    assert not lease.selected and d.session.peer is None
    started = lease.started
    d.host.now += 50
    d.session.peer = 'K1ABC-2'
    d.tick()
    assert lease.selected and lease.started == started
    assert lease.deadline == started + 7200
    assert d.session.bbs.identity == 'K1ABC'
    d.session.peer = 'K2XYZ'
    d.receive('BBS')
    drive_writer(d)
    assert core.session is None
    assert 'identity changed' in sent(d)[-1]


def test_late_shorter_identity_is_selected_before_native_application_ingress(board):
    d, core, *_ = board
    from dataclasses import replace
    core.config = replace(core.config, callsigns={'K1ABC': Limits(10, 20, 3600)})
    d.session.bbs.connection = ConnectionLease.start(core.config.limits, 0, selected=False)
    d.session.bbs.identity = None
    d.session.peer = 'K1ABC-2'
    d.host.now = 50
    before = len(sent(d))
    # Ingress is dispatched during Client.poll, before entrance.tick. The short
    # original-start override must already win over the unselected default.
    d.receive('<BG:hello.txt>')
    assert not d.host.client.connected
    assert d.session.bbs_host._future is None
    assert len(sent(d)) == before


def test_waiting_mail_has_priority_over_bbs_entry_then_releases_it_after_smr(board):
    from hfmodem.tests.contact.test_varac_cf_qsy import parked
    d, core, *_ = board
    row = parked(d, 'KQ4QWH')
    deadline = d.session.bbs.connection.deadline
    d.receive('BBS')
    assert core.session is None and not d.session.bbs.active
    drive_writer(d)
    drive_writer(d)
    assert any('settled link' in text for text in sent(d))
    assert d.session.parking.store.get(row['id'])['state'] == 'in_flight'
    assert sum('<SM>' in text for text in sent(d)) == 1
    d.receive('<SMR>')
    drive_writer(d)
    enter(board)
    assert d.session.parking.store.get(row['id'])['state'] == 'delivered'
    assert d.session.bbs.connection.deadline == deadline


@pytest.mark.parametrize('mode', ['lobby', 'game'])
@pytest.mark.parametrize('parking', [False, True])
def test_bbs_owned_mail_gets_explicit_refusal_without_game_input(board, mode, parking):
    d, core, games, _ = board
    (play if mode == 'game' else enter)(board)
    d.session.parking.store.configure(parking=parking)
    d.receive('<SM><TO:K1ABC><SBJ:deposit><MSG:do not feed the game>')
    drive_writer(d)
    assert sent(d)[-1] == ('<SMF>' if parking else '<SMFP>')
    assert not d.session.parking.store.records and not d.mailbox.entries()
    assert not games or not games[0].submissions


def test_return_to_chat_resumes_queued_mail_without_renewing_bbs_allowance(board):
    from hfmodem.tests.contact.test_varac_cf_qsy import parked
    d, core, *_ = board
    enter(board)
    deadline = d.session.bbs.connection.deadline
    row = parked(d, 'KQ4QWH')
    for _ in range(5):
        d.tick()
    assert not any('<SM>' in text for text in sent(d))
    d.receive('BBS EXIT')
    for _ in range(100):
        d.tick()
        if not d.session.bbs.active:
            break
        time.sleep(.001)
    assert not d.session.bbs.active
    drive_writer(d)
    drive_writer(d)
    assert d.session.parking.store.get(row['id'])['state'] == 'in_flight'
    assert sum('<SM>' in text for text in sent(d)) == 1
    assert d.session.bbs.connection.deadline == deadline


def test_bbs_allowance_closes_unacknowledged_mail_and_retains_the_remainder(board):
    from hfmodem.tests.contact.test_varac_cf_qsy import parked
    d, core, *_ = board
    first = parked(d, 'KQ4QWH')
    second = parked(d, 'KQ4QWH', 'later')
    deadline = d.session.bbs.connection.deadline
    drive_writer(d)
    drive_writer(d)
    assert d.session.parking.store.get(first['id'])['state'] == 'in_flight'
    d.host.now = deadline
    d.tick()
    d.drive()
    assert not d.host.client.connected
    assert d.session.parking.store.get(first['id'])['state'] == 'uncertain'
    assert d.session.parking.store.get(second['id'])['state'] == 'queued'
