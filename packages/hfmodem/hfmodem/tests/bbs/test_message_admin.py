# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from hfmodem import cli
from hfmodem.bbs.config import load
from hfmodem.bbs.message_admin import main
from hfmodem.bbs.messages import MessageStore, OperatorMessages
from hfmodem.bbs.store import Store


def seed(tmp_path):
    config_path = tmp_path / 'bbs.toml'
    config_path.write_text('[bbs]\nstate_root="state"\n[bbs.message_areas]\ngeneral="General"\n')
    config = load(config_path)
    with Store(config.state_root) as store:
        player = store.player('K1ABC')
        messages = MessageStore(store, config.message_areas)
        public = messages.post_public(player.id, 'general', 'Hello board', 'Public body', 'public-request')
        private = messages.post_sysop(player.id, 'Question', 'Please contact me here.', 'sysop-request')
    return config_path, config, player, public, private


def test_operator_inbox_read_reply_and_caller_receipt(tmp_path, capsys):
    path, config, player, _, private = seed(tmp_path)
    assert cli.main(['bbs', 'messages', '--config', str(path), 'list']) == 0
    assert f'* {private.id} K1ABC: Question' in capsys.readouterr().out
    assert main(['--config', str(path), 'read', str(private.id)]) == 0
    assert 'Please contact me here.' in capsys.readouterr().out
    assert main(['--config', str(path), 'list']) == 0
    assert f'* {private.id}' not in capsys.readouterr().out
    body = tmp_path / 'reply.txt'
    body.write_text('Reply from the operator.')
    command = ['--config', str(path), 'reply', str(private.id), '--body-file', str(body),
               '--request-key', 'stable-operator-reply']
    assert main(command) == 0
    assert 'Request key: stable-operator-reply' in capsys.readouterr().out
    assert main(command) == 0  # exact retry cannot create a second reply
    with Store(config.state_root) as store:
        messages = MessageStore(store, config.message_areas)
        items = messages.list_sysop(player.id).messages
        assert len(items) == 2
        reply = messages.get(player.id, items[-1].id)
        assert reply.body == 'Reply from the operator.' and not reply.read
        assert OperatorMessages(store).get(private.id).read


def test_operator_moderation_redacts_callers_and_preserves_threads(tmp_path, capsys):
    path, config, player, public, _ = seed(tmp_path)
    assert main(['--config', str(path), 'areas']) == 0
    assert 'general: General' in capsys.readouterr().out
    assert main(['--config', str(path), 'list', '--area', 'general']) == 0
    assert 'Hello board' in capsys.readouterr().out
    assert main(['--config', str(path), 'hide', str(public.id)]) == 0
    with Store(config.state_root) as store:
        hidden = MessageStore(store).get(player.id, public.id)
        assert hidden.hidden and not hidden.body and not hidden.author
        assert OperatorMessages(store).get(public.id).body == 'Public body'
    assert main(['--config', str(path), 'unhide', str(public.id)]) == 0
    with Store(config.state_root) as store:
        assert MessageStore(store).get(player.id, public.id).body == 'Public body'


def test_operator_cli_cannot_open_a_running_service_or_send_oversize_reply(tmp_path, capsys):
    path, config, _, _, private = seed(tmp_path)
    with Store(config.state_root):
        assert main(['--config', str(path), 'list']) == 2
        assert 'already owned' in capsys.readouterr().err
    body = tmp_path / 'too-long.txt'
    body.write_bytes(b'x' * 2049)
    assert main(['--config', str(path), 'reply', str(private.id), '--body-file', str(body)]) == 2
    assert '2048' in capsys.readouterr().err
    with Store(config.state_root) as store:
        assert len(OperatorMessages(store).list_sysop().messages) == 1
