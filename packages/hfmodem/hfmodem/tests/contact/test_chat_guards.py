# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace

import pytest

from hfmodem.contact import chat_state as s
from hfmodem.contact.chat_controller import Controller
from hfmodem.contact.chat_session import SessionAdapter
from hfmodem.tests.contact.test_chat_controller import adapter, completed, execute, request
from hfmodem.tests.contact.test_chat_state import sample
from hfmodem.tests.contact.test_varac import Desk


ACTIONS = tuple(sample(cls) for cls, _ in s.ACTIONS.values())
MUTATIONS = tuple(a for a in ACTIONS if s.effect(a) is not s.Effect.R)


@pytest.mark.parametrize('action', ACTIONS, ids=lambda a: a.kind)
@pytest.mark.parametrize('connected', (False, True))
def test_every_action_preflight_is_pure_on_real_session(action, connected):
    desk, port, _ = adapter(connected=connected)
    before = (desk.sent, desk.texts, list(desk.tunes), list(desk.armed),
              desk.session.compose, desk.session.email.draft)
    assert port.preflight(action).kind == action.kind
    assert (desk.sent, desk.texts, desk.tunes, desk.armed,
            desk.session.compose, desk.session.email.draft) == before


@pytest.mark.parametrize('action', MUTATIONS, ids=lambda a: a.kind)
def test_every_mutation_rechecks_generation_and_authority_on_real_owner(tmp_path, action):
    owned = {}
    def factory():
        desk = Desk(mailbox=tmp_path)
        port = SessionAdapter(desk.session, ownership=s.Ownership(
            'station', 'owner', 'lease', 4, None, 'ready', True))
        owned['desk'] = desk
        return port
    controller = Controller(factory, controller_id='test')
    controller.ready.result(2)
    desk = owned['desk']
    before = (desk.sent, desk.texts, list(desk.tunes), list(desk.armed))
    try:
        req = request(action, generation=-1, control_id='lease', control_generation=4)
        assert completed(controller, req).state.reason_code == 'link-stale'
        req = replace(req, id=req.id + '-lease', link_generation=0, control_generation=3)
        assert completed(controller, req).state.reason_code == 'authority-stale'
        assert (desk.sent, desk.texts, desk.tunes, desk.armed) == before
    finally:
        controller.shutdown().result(2)
        assert controller.join(2)


@pytest.mark.parametrize('action', (
    s.ChatCall('INVALID!', None, False), s.ChatAnswer(999), s.ChatCQ(999, False),
    s.ChatQSY(999), s.ChatSend('<BLR>'), s.ChatTag('SM', None), s.BBSInput('context', 0, '<BG:file>'),
    s.BeaconConfigure(0), s.BeaconHop((20, 20), False, 5), s.BandSet(999),
    s.AwaySet('unsafe\nline'), s.EmailCompose('bad', 'bad', 'subject'), s.BBSReload(),
    s.BBSTerm('ansi16', 0, 0), s.DraftAppend('missing', 0, '.'), s.DraftDrop('missing', 0),
    s.EmailSend('missing', 0), s.VMailSend('missing', 0), s.BroadcastSend('unselected'),
))
def test_guard_refusals_do_not_write_or_change_drafts(action):
    desk, port, _ = adapter(connected=True)
    before = (desk.sent, desk.texts, list(desk.tunes), list(desk.armed))
    assert port.preflight(action).availability == 'blocked'
    _, result = execute(port, action)
    assert result.state == 'refused'
    assert (desk.sent, desk.texts, desk.tunes, desk.armed) == before
    assert port.project().application.draft is None


def test_real_draft_reads_settings_and_failed_append_preserve_reviewed_revision():
    desk, port, _ = adapter(connected=True)
    _, composed = execute(port, s.VMailCompose('subject'))
    draft = composed.result.draft
    _, appended = execute(port, s.DraftAppend(draft.id, 0, 'body'))
    draft = appended.result.draft
    for action in (s.HeardList(), s.FilesList(), s.VMailList(), s.VerboseSet(True), s.AwayBack()):
        _, result = execute(port, action)
        assert result.state == 'done'
        assert port.project().application.draft == draft
    _, failed = execute(port, s.DraftAppend(draft.id, 1, 'x' * 65537))
    assert failed.state == 'refused' and port.project().application.draft == draft
    _, foreign = execute(port, s.VMailSend(draft.id, 1), actor='second-view')
    assert foreign.reason_code == 'draft-owned' and not desk.texts
    desk.hear('DISCONNECTED')
    _, stale = execute(port, s.VMailSend(draft.id, 1))
    assert stale.state == 'refused' and not desk.texts


def test_real_broadcast_acknowledgements_allow_retune_and_keep_host_queue():
    pytest.importorskip('textual', reason='install hfmodem[tui] to run the Textual chat tests')
    from hfmodem.ui.textual_chat import parse_action
    desk = Desk(kiss=True)
    port = SessionAdapter(desk.session)
    def act(line):
        action = parse_action(line)
        assert s.decode(s.encode(action)) == action
        return execute(port, action)[1]
    assert act('/all cf').state == 'done'
    assert act('/all hello').state == 'running'
    for _ in range(100):
        port.poll(.1)
        if not desk.session.broadcast.active:
            break
    assert port.project().broadcast.possibly_outstanding == 1
    written = len(desk.host.kiss.sent)
    assert act('/all slot 1').state == 'done'
    assert desk.session.broadcast._retune_command == '/all slot 1'
    assert act('/all no').state == 'done'
    assert desk.session.broadcast.possibly_outstanding == 1
    assert desk.flow.dial == desk.session.calling_hz
    assert act('/all slot 1').state == 'done'
    assert act('/all yes').state == 'done'
    assert desk.flow.dial != desk.session.calling_hz
    assert port.project().broadcast.possibly_outstanding == 0
    desk.session.broadcast.possibly_outstanding = 2
    assert act('/all clear').state == 'done'
    assert desk.session.broadcast.possibly_outstanding == 0
    assert len(desk.host.kiss.sent) == written
