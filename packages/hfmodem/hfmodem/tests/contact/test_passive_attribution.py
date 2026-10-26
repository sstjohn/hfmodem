# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Passive payload mentions must not manufacture QSO participants. No DSP."""
from types import SimpleNamespace

import pytest

from hfmodem.contact import slots as S

# Exact fragments supplied by the operator, not a claim of complete RF decoding.
LHR_FRAGMENTS = (
    '5 <LHR>', '386 [2]', 'VO1CBL> 15:52:27 / BEACON / 20m / -',
    '15:49:19 / BEACON / 20m / -08dB', 'KC0TAF> 15:',
    '1 / BEACON / 20m / -03dB', 'W0PM> 15:46:39 / B',
    'N / 20m / +01dB', 'W0TRM> 15:41:02 / BEACON /',
    '20m / -10dB', 'W9SSJ> 15:40:42 / BEACON / 20m',
    '/ -14dB', 'K9JAJ> 15:40:33 / BEACON / 20m / +0',
)


@pytest.fixture
def passive(monkeypatch):
    monitor = object.__new__(S.SlotMonitor)
    monitor.samples = S.FS
    monitor.mycall = 'W9SSJ'
    monitor.calls = []
    monitor._link = lambda ch, ev: False
    heard = []
    monitor.heard = heard.append
    monkeypatch.setattr(S.B, 'parse_payload', lambda body: None)
    monkeypatch.setattr(S.VF, 'is_link_setup', lambda body: False)
    ch = SimpleNamespace(slot=1, excess=[], audio_hz=750, home=False)
    ev = S._Evidence(level=4)
    monitor._evidence = {1: ev}
    frame = SimpleNamespace(frame_bytes=b'\x00' * 6, is_connect=False, level=4)
    def feed(text):
        raw = text.encode() if isinstance(text, str) else text
        monkeypatch.setattr(S.phy, 'vara_payload', lambda *args, **kwargs: raw)
        return [item[2] for item in monitor._frame(ch, ev, frame)]
    return monitor, ch, ev, heard, feed


def test_exact_fragmented_last_heard_report_is_not_a_multiparty_link(passive):
    monitor, ch, ev, heard, feed = passive
    ev.caller, ev.called = 'N0XYZ', 'K0EPI'
    shown = [said for text in LHR_FRAGMENTS for said in feed(text) if said.kind == 'text']
    assert [said.text for said in shown] == list(LHR_FRAGMENTS)
    assert all(not said.station for said in shown)
    assert not ev.calls and not heard
    assert monitor._judge(ch).calls == ('N0XYZ', 'K0EPI')
    assert monitor._judge(ch).who() == 'N0XYZ ↔ K0EPI'


def test_same_report_without_endpoint_evidence_stays_unknown(passive):
    monitor, ch, ev, heard, feed = passive
    for text in LHR_FRAGMENTS:
        feed(text)
    assert monitor._judge(ch).calls == ()
    assert not ev.caller and not ev.called and not ev.keyed and not heard


def test_absolute_qsy_is_reported_after_prior_bare_readiness_without_sender_claim(passive):
    monitor, ch, ev, heard, feed = passive
    feed('3 <Q>')
    said = feed('13 <QF:14105750>')
    assert any(s.kind == 'stage' and s.text == 'QSY <QF:14105750>' for s in said)
    assert ev.stage == 'QSY' and not ev.calls and not heard


@pytest.mark.parametrize('text', [
    'I heard VO1CBL and KC0TAF earlier',
    'quoted: de N0XYZ <R+03>',
    '> de N0XYZ <R+03>',
    '<SM><FR:K0EPI><MSG:de N0XYZ <R+03>>',
    'Mail body\nde N0XYZ <R+03>',
    'de N0XYZ <R+03> is what another station wrote',
    b'\x00de N0XYZ <R+03>',
])
def test_mentions_quotes_mail_and_binary_do_not_identify_sender(passive, text):
    monitor, ch, ev, heard, feed = passive
    said = feed(text)
    assert not any(item.station for item in said)
    assert not ev.calls and not heard and not ev.called and not ev.caller
    if isinstance(text, bytes):
        assert said[-1].kind == 'payload' and ev.text == ''


@pytest.mark.parametrize('text', ['15 de AI7EQ <R-10>', 'de AI7EQ <R-10>',
                                  'de AI7EQ <R-10> <S:VAB>'])
def test_standalone_signed_report_retains_declared_sender(passive, text):
    monitor, ch, ev, heard, feed = passive
    ev.caller = 'KD5LEE'
    said = feed(text)
    assert said[-1].station == 'AI7EQ'
    assert ev.called == 'AI7EQ' and ev.calls == ['AI7EQ']
    assert heard == ['AI7EQ']
    assert monitor._judge(ch).calls == ('KD5LEE', 'AI7EQ')


def test_declaration_cannot_add_third_endpoint_to_known_link(passive):
    monitor, ch, ev, heard, feed = passive
    ev.caller, ev.called = 'N0XYZ', 'K0EPI'
    said = feed('de AI7EQ <R-10>')
    assert not said[-1].station and not ev.calls and not heard
    assert monitor._judge(ch).calls == ('N0XYZ', 'K0EPI')


def test_setup_identity_is_preserved_without_payload_scanning(passive, monkeypatch):
    monitor, ch, ev, heard, feed = passive
    monkeypatch.setattr(S.VF, 'is_link_setup', lambda body: True)
    monkeypatch.setattr(S.VF, 'caller_from_link_setup', lambda body: 'N0XYZ')
    said = feed('ignored setup bytes')
    assert ev.caller == 'N0XYZ' and heard == ['N0XYZ']
    assert said[-1].kind == 'stage' and 'N0XYZ' in said[-1].text
