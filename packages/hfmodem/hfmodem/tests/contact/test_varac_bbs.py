# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Public ON2AD manual V16.1.8 p65/p126; fake host only, no transmissions."""
import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk

# First two rows from the p65 screenshot, with its trailing line before >.
LIST = '<BL:current_wx_oh8xat.txt|2026-02-11|8758\nhf_r1_bandplan.pdf|2026-02-05|63366\n>'


def linked():
    desk = Desk()
    desk.flow.dial = 14_106_500
    desk.hear('CONNECTED W9SSJ OH8XAT 500')
    return desk


def test_list_round_trip_fragmented_reply_keeps_lossless_evidence():
    desk = linked()
    desk.session.command('/bbs list')
    assert desk.texts == ['5 <BLR>']
    wire = v.frame_message(LIST)
    for byte in wire[:-1]:
        desk.host.data.incoming.append(bytes([byte]))
    desk.tick()
    assert desk.session.steps is not None
    desk.host.data.incoming.append(wire[-1:])
    desk.tick()
    assert desk.session.steps is None
    assert any('current_wx_oh8xat.txt  2026-02-11  8758 bytes' in line for line in desk.said)
    assert desk.kinds('bbs_directory')[-1]['text'] == LIST
    assert desk.host.client.connected
    assert desk.texts == ['5 <BLR>']


def test_single_row_public_example_and_crlf_screenshot_shape():
    assert v.bbs_files('<BL:hello.txt|2025-01-01|5>') == [v.BBSFile('hello.txt', '2025-01-01', 5)]
    assert len(v.bbs_files(LIST.replace('\n', '\r\n'))) == 2


@pytest.mark.parametrize('text', ['<BL:>', '<BL:bad>', '<BL:x|2026-01-01|-1>',
                                 '<BL:x|today|1>', '<BL:x|2026-01-01|1|extra>',
                                 '<BL:x\x1b|2026-01-01|1>', '<BL:' + 'a' * 65536 + '>'],
                         ids=['empty', 'short-row', 'negative-size', 'date', 'extra-field', 'control', 'oversize'])
def test_unknown_or_excessive_directory_preserves_raw_but_does_not_fabricate_entries(text):
    desk = linked()
    desk.session.command('/bbs list')
    desk.receive(text)
    assert desk.session.steps is None
    assert desk.kinds('bbs_directory')[-1]['text'] == text
    assert any('raw reply retained' in line for line in desk.said)
    assert desk.texts == ['5 <BLR>']


def test_rejection_and_no_implicit_retry():
    desk = linked()
    desk.session.command('/bbs list')
    desk.receive('<BLJ>')
    assert 'OH8XAT rejected the BBS directory request' in desk.said
    assert desk.texts == ['5 <BLR>']


@pytest.mark.parametrize('state', ['unlinked', 'cf', 'exchange'])
def test_list_guard_sends_nothing(state):
    desk = Desk() if state == 'unlinked' else linked()
    if state == 'cf':
        desk.flow.dial = desk.session.calling_hz
    elif state == 'exchange':
        desk.session.steps = iter([.1])
    before = desk.texts[:]
    desk.session.command('/bbs list')
    assert desk.texts == before
    assert desk.said[-1].startswith('error:')


def test_stop_cancels_wait_but_does_not_claim_request_unsent():
    desk = linked()
    desk.session.command('/bbs list')
    desk.session.command('/stop')
    assert desk.session.steps is None and not desk.session._bbs_waiting
    assert 'may still be delivered' in desk.said[-1]
    desk.receive(LIST)
    assert not desk.kinds('bbs_directory')
    assert desk.host.client.connected


def test_timeout_and_old_reply_cannot_satisfy_a_new_request():
    desk = linked()
    desk.receive(LIST)
    desk.session.command('/bbs list')
    desk.drive()
    assert desk.session.steps is None and not desk.session._bbs_waiting
    assert any('timed out' in line for line in desk.said)
    assert desk.host.client.connected
    assert not desk.kinds('bbs_directory')


def test_bye_and_quit_remain_available_during_bbs_wait():
    for close in ('/bye', '/quit'):
        desk = linked()
        desk.session.command('/bbs list')
        if close == '/quit':
            assert desk.session.command(close) is False
            desk.session.quit()
        else:
            desk.do(close)
        assert not desk.session._bbs_waiting
        assert not desk.host.client.connected
        assert sum('<BLR>' in text for text in desk.texts) == 1


def test_get_requires_a_current_directory():
    desk = linked()
    desk.session.command('/bbs get 1')
    assert desk.texts == []
    assert 'Run /bbs list' in desk.said[-1]
    with pytest.raises(ValueError):
        v.emit_tag('BG', 'hello.txt')


def test_explicit_tag_uses_the_same_calling_frequency_guard():
    desk = linked()
    desk.flow.dial = desk.session.calling_hz
    desk.session.command('/tag BLR')
    assert desk.texts == []
    assert 'QSY off' in desk.said[-1]


def test_directory_row_limit_and_tag_argument_validation():
    with pytest.raises(ValueError, match='oversized'):
        v.bbs_files('<BL:' + '\n'.join(['x.txt|2026-01-01|1'] * 501) + '>')
    desk = linked()
    desk.session.command('/tag BLR unexpected')
    assert desk.texts == []
    assert 'takes no value' in desk.said[-1]


@pytest.mark.parametrize('frequency', v.CALLING_HZ.values())
def test_list_rejects_every_configured_calling_frequency(frequency):
    desk = linked()
    desk.flow.dial = frequency
    desk.session.command('/bbs list')
    assert desk.texts == []
    assert 'QSY off' in desk.said[-1]


@pytest.mark.parametrize('ending', ['disconnect', 'eof'])
def test_incomplete_list_is_retained_but_never_rendered(ending):
    desk = linked()
    desk.session.command('/bbs list')
    partial = v.frame_message(LIST)[:-5]
    desk.host.data.incoming.append(partial)
    desk.tick()
    if ending == 'disconnect':
        desk.hear('DISCONNECTED')
    else:
        desk.host.data.incoming.append(b'')
        with pytest.raises(v.HostError, match='closed'):
            desk.tick()
        # The terminal's finally owns quit on socket failure.
        try:
            desk.session.quit()
        except v.HostError:
            pass
    assert not desk.session._bbs_waiting
    assert not desk.kinds('bbs_directory')
    assert desk.kinds('bbs_incomplete')[-1]['text'] == partial.decode()
    assert not any('files (sizes in bytes)' in line for line in desk.said)
