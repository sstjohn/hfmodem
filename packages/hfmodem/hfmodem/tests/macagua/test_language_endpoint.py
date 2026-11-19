# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Endpoint QSO requests and both waveform lifecycles, without models or hardware."""
from dataclasses import asdict, replace
import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from hfhost.config import ConfigError
from hfmodem.core.config import Voice
from hfmodem.contact import language
from hfmodem.contact.runtime import Session, model_hashes
from hfmodem.contact.txmark import Tag
from hfmodem.tests.macagua.test_voice import cfg


def remote(**kwargs):
    return replace(Voice(language_backend='openai', language_model='test-model',
                         language_api_key_env=''), **kwargs)


@pytest.fixture
def endpoint(monkeypatch):
    state = SimpleNamespace(calls=[], content={'reply': 'Sounds interesting.', 'claims': {}},
                            finish='stop', error=None, refusal=None)

    class HTTP:
        def open(self, req, timeout):
            state.calls.append((req, timeout))
            if state.error:
                raise state.error
            return io.BytesIO(json.dumps({'choices': [{'finish_reason': state.finish,
                'message': {'content': json.dumps(state.content), 'refusal': state.refusal}}]}).encode())

    monkeypatch.setattr(language, 'build_opener', lambda *args: HTTP())
    monkeypatch.setattr(language.subprocess, 'Popen', lambda *a, **k: pytest.fail('started local host'))
    return state


def test_endpoint_request_auth_history_and_wire_schema(endpoint, monkeypatch, tmp_path):
    monkeypatch.setenv('QSO_TEST_KEY', 'secret-for-test')
    settings = remote(language_base_url='https://example.test/custom/v1/',
                      language_api_key_env='QSO_TEST_KEY', language_timeout_s=12,
                      language_max_tokens=1234)
    host = language.Language.from_settings(settings, tmp_path)
    endpoint.content['action'] = 'continue'
    history = [dict(role='station' if i % 2 else 'partner', text=str(i)) for i in range(10)]
    history.append(dict(role='partner', text='Interesting hobby.'))
    result = host.converse('Interesting hobby.', {}, history)
    req, timeout = endpoint.calls[0]
    payload = json.loads(req.data)
    assert result['reply'] == 'Sounds interesting.'
    assert req.full_url == 'https://example.test/custom/v1/chat/completions'
    assert req.get_method() == 'POST' and timeout == 12
    assert req.get_header('Authorization') == 'Bearer secret-for-test'
    assert payload['model'] == 'test-model'
    assert payload['response_format'] == {'type': 'json_object'}
    assert payload['max_completion_tokens'] == 1234
    assert 'temperature' not in payload and 'max_tokens' not in payload
    assert 'chat_template_kwargs' not in payload
    assert len(payload['messages']) == 10
    assert payload['messages'][1] == {'role': 'user', 'content': '2'}
    assert payload['messages'][-1] == {'role': 'user', 'content': 'Interesting hobby.'}
    assert 'schema' in payload['messages'][0]['content']
    assert 'secret-for-test' not in json.dumps(asdict(settings))
    assert not list(tmp_path.iterdir())
    host.close()
    host.close()


def test_unauthenticated_endpoint_and_legacy_token_parameter(endpoint, monkeypatch, tmp_path):
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-be-used')
    host = language.Language.from_settings(remote(language_token_parameter='max_tokens'), tmp_path)
    assert host.draft('Nice day.', {}) == 'Sounds interesting.'
    req, _ = endpoint.calls[0]
    assert not req.has_header('Authorization')
    assert json.loads(req.data)['max_tokens'] == 2048
    assert 'max_completion_tokens' not in json.loads(req.data)


def test_missing_key_fails_before_any_request(endpoint, monkeypatch, tmp_path):
    monkeypatch.delenv('QSO_TEST_KEY', raising=False)
    with pytest.raises(ValueError, match='Set QSO_TEST_KEY'):
        language.Language.from_settings(remote(language_api_key_env='QSO_TEST_KEY'), tmp_path)
    assert not endpoint.calls


@pytest.mark.parametrize('content', [None, [], {}, {'reply': 3, 'claims': {}},
    {'reply': 'Hi.', 'claims': []}, {'reply': 'Hi.', 'claims': {'rig': 'invented'}},
    {'reply': 'Hi.', 'claims': {}, 'unexpected': 'field'}])
def test_invalid_remote_candidates_are_rejected(endpoint, tmp_path, content):
    endpoint.content = content
    host = language.Language.from_settings(remote(), tmp_path)
    with pytest.raises(RuntimeError, match='required fields'):
        host.draft('Nice day.', {})


def test_remote_extraction_keeps_callsign_and_report_constraints(endpoint, tmp_path):
    host = language.Language.from_settings(remote(), tmp_path)
    endpoint.content = dict(to='W9SSJ', de='K4MCK', report='57', name='', qth='', free='', end='over')
    assert host.extract('K4MCK, five seven', 'W9SSJ', ['K4MCK'])['de'] == 'K4MCK'
    endpoint.content['de'] = 'K1FAKE'
    with pytest.raises(RuntimeError, match='required fields'):
        host.extract('K4MCK', 'W9SSJ', ['K4MCK'])
    endpoint.content.update(de='K4MCK', report='599')
    with pytest.raises(RuntimeError, match='required fields'):
        host.extract('K4MCK', 'W9SSJ', ['K4MCK'])


@pytest.mark.parametrize('finish,refusal,message', [('length', None, 'truncated'),
    ('content_filter', None, 'declined'), ('stop', 'No.', 'declined')])
def test_truncation_and_refusal_are_not_drafts(endpoint, tmp_path, finish, refusal, message):
    endpoint.finish, endpoint.refusal = finish, refusal
    with pytest.raises(RuntimeError, match=message):
        language.Language.from_settings(remote(), tmp_path).draft('Nice day.', {})


@pytest.mark.parametrize('response', [None, {}, {'choices': []}, {'choices': [None]},
    {'choices': [{'message': None}]}, {'choices': [{'message': {'content': None}}]},
    {'choices': [{'message': {'content': 'not JSON'}}]}])
def test_malformed_chat_response_is_an_actionable_error(endpoint, tmp_path, monkeypatch, response):
    host = language.Language.from_settings(remote(), tmp_path)
    monkeypatch.setattr(host, 'request', lambda *a: response)
    with pytest.raises(RuntimeError, match='invalid JSON reply'):
        host.draft('Nice day.', {})


@pytest.mark.parametrize('failure,message', [
    (HTTPError('https://example.test', 401, 'secret-for-test', {}, io.BytesIO(b'secret-for-test')), 'HTTP 401'),
    (HTTPError('https://example.test', 429, 'quota', {}, io.BytesIO(b'secret-for-test')), 'HTTP 429'),
    (URLError('secret-for-test'), 'unavailable'), (TimeoutError('secret-for-test'), 'timed out'),
    (ValueError('secret-for-test'), 'invalid JSON')])
def test_endpoint_failures_are_bounded_and_do_not_echo_secrets(endpoint, tmp_path, failure, message):
    endpoint.error = failure
    with pytest.raises(RuntimeError, match=message) as caught:
        language.Language.from_settings(remote(), tmp_path).draft('Nice day.', {})
    assert 'secret-for-test' not in str(caught.value)
    assert len(endpoint.calls) == 1  # No automatic billable retry.


def test_redirect_cannot_forward_key():
    req = language.Request('https://example.test/v1/chat/completions',
                           headers={'Authorization': 'Bearer secret-for-test'})
    assert language._NoRedirect().redirect_request(req, None, 302, 'redirect', {},
                                                    'https://other.test') is None


@pytest.mark.parametrize('field,value', [('language_backend', 'typo'),
    ('language_timeout_s', 0), ('language_timeout_s', float('nan')),
    ('language_max_tokens', True), ('language_max_tokens', -1),
    ('language_token_parameter', 'tokens'), ('language_model', ''),
    ('language_base_url', 'file:///tmp/model'), ('language_base_url', 'https://user:secret@example.test/v1'),
    ('language_base_url', 'https://example.test/v1?key=secret'),
    ('language_base_url', 'https://example.test:bad/v1'),
    ('language_base_url', 'https://example.test:70000/v1'),
    ('language_base_url', 'https://[invalid/v1'),
    ('language_base_url', 'https://example.test /v1'),
    ('language_base_url', 'https://example.test/v1#fragment')])
def test_bad_endpoint_settings_fail_early(field, value):
    with pytest.raises(ConfigError):
        remote(**{field: value})


@pytest.mark.parametrize('waveform', ['voice', 'cw'])
def test_remote_session_starts_without_local_process_or_tokenizer(endpoint, tmp_path, monkeypatch, waveform):
    settings = cfg()
    settings.voice = remote()
    monkeypatch.setattr(language.subprocess, 'run', lambda *a, **k: pytest.fail('queried local RSS'))
    session = Session(settings, tmp_path, test=True, waveform=waveform)
    closed = []
    monkeypatch.setattr(session.waveform, 'open_modem', lambda *a: SimpleNamespace(
        ready={'rss_gb': .25}, close=lambda: closed.append(True)))
    try:
        session.start_models()
        assert isinstance(session.language, language.OpenAILanguage)
        assert session.mark_codec is None
        assert session.language.draft('Nice day.', {}, mode=waveform) == 'Sounds interesting.'
        events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
        assert events[-1]['kind'] == 'models_loaded' and events[-1]['measured_rss_gb'] == .25
        with pytest.raises(ValueError, match='require language_backend'):
            session.set_tx_mark(Tag('hfmodem'))
        assert session.tx_mark.mode == 'off'
    finally:
        session.close()
    assert closed == [True]


@pytest.mark.parametrize('options', [dict(watermark='hfmodem-v1'), dict(tx_mark=Tag('hfmodem'))])
def test_remote_marks_refuse_before_models_open(tmp_path, monkeypatch, options):
    settings = cfg()
    settings.voice = remote()
    session = Session(settings, tmp_path, test=True, **options)
    monkeypatch.setattr(session.waveform, 'open_modem', lambda *a: pytest.fail('opened speech model'))
    try:
        with pytest.raises(ValueError, match='require language_backend'):
            session.start_models()
    finally:
        session.close()


def test_model_ids_are_not_hashed_as_local_files(tmp_path):
    model = tmp_path / 'test-model'
    model.write_bytes(b'local model')
    assert model_hashes(remote(language_model=str(model)), 'cw') == {}
    assert str(model) in model_hashes(Voice(language_model=str(model)), 'cw')


def test_local_default_still_requires_gguf(tmp_path):
    assert Voice().language_backend == 'local'
    with pytest.raises(ValueError, match='local GGUF'):
        language.Language.from_settings(Voice(language_model=str(tmp_path / 'absent.gguf')), tmp_path)


def test_channel_rejects_remote_before_creating_a_session(tmp_path, monkeypatch):
    from hfmodem.contact import channel
    settings = cfg()
    settings.voice = remote()
    monkeypatch.setattr(channel, 'Session', lambda *a, **k: pytest.fail('created channel session'))
    with pytest.raises(ValueError, match='requires language_backend'):
        channel.simulate(settings, tmp_path, b'hello', None)
