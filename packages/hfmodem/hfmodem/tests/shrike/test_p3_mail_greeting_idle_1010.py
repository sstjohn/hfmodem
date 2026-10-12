"""A Trimode banner that ends without a SID expires the mail stage; one that
carries the SID reaches its turn. Frames are the new P3 payloads of the arms,
one per 1.25 s cycle, the idle tail continued in the gateway's own rotation."""
import re
from itertools import chain, cycle, islice

import pytest

from hfmodem.shrike import arq, spec
from hfmodem.shrike.p3progress import MailProgress
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import IO, acknowledge_all, modem
from hfmodem.winlink import B2FSession, MailClient
from hfmodem.winlink.client import summarize
from hfmodem.winlink.session import _THEIR_TURN

CYCLE = 1.25

# pactor-current-ws8eoc-40-force-20261010T224247Z
WS8EOC_1010 = [
    (0x00, b'RMS'), (0x21, b' Trim'), (0x22, b'ode 1'), (0x23, b'.4.4.'),
    (0x20, b'0\r\nW9'), (0x21, b'SSJ h'), (0x22, b'as 11'),
    (0x23, b'2 daily minutes remaini'), (0x00, b'ng with WS8EOC (EN72QQ)'),
    (0x01, b'\r{SFI = 121 On 2026-10-'), (0x02, b'10 21:00 UTC}\r\n'),
] + [(s, b'') for s in (
    0x03, 0x40, 0x01, 0x42, 0x03, 0x00, 0x01, 0x42, 0x03, 0x00, 0x41, 0x02,
    0x03, 0x40, 0x01, 0x02, 0x43, 0x00, 0x01, 0x02, 0x43,
    *[0x00, 0x41, 0x02, 0x43] * 7,
    0x40, 0x01, 0x42, 0x03)]

# pactor-current-ve3kpg-40-force-20261010T224619Z
VE3KPG_1010 = [
    (0x00, b'RMS'), (0x21, b' Trim'), (0x22, b'ode 1'), (0x23, b'.4.4.'),
    (0x20, b'0\r\nW9SSJ has 116 daily '), (0x21, b'minutes remaining with '),
    (0x22, b'VE3KPG (FN04VE)\r{SFI = '), (0x23, b'121 On 2026-10-10 21:00'),
    (0x20, b' UTC}\r\n****************'), (0x21, b'*' * 23),
    (0x22, b'**************\rKawartha'), (0x23, b' Packet Group\rHiawatha,'),
    (0x00, b' Ontario\rSysop is Harme'), (0x01, b'n,VA3OU/VE3EMA\rMTD Conn'),
    (0x02, b'ections: VARA=69 PACTOR'), (0x03, b'=47\r*******************'),
    (0x00, b'*' * 23), (0x01, b'************\r\n'),
] + [(s, b'') for s in (0x02, 0x43, 0x00, 0x41, 0x02, 0x43, 0x00)]

# pactor-current-ws8eoc-40-force-20261004T225456Z
WS8EOC_1004 = [
    (0x00, b'RMS'), (0x21, b' Trim'), (0x22, b'ode 1'), (0x23, b'.4.4.'),
    (0x20, b'0\r\nW9'), (0x21, b'SSJ h'), (0x22, b'as 120 daily minutes re'),
    (0x03, b'maining with WS8EOC (EN'), (0x00, b'72QQ)\r{SFI = 93 On 2026'),
    (0x01, b'-10-04 20:00 UTC}\r\n[WL2'), (0x02, b'K-5.0-B2FWIHJM$]\r;PQ: 7'),
    (0x03, b'0693571\rCMS via WS8EOC '), (0x00, b'>\r'), (0x41, b''),
]

IDLE = [(s, b'') for s in (0x00, 0x41, 0x02, 0x43)]


class LinkIO(IO):
    progress = None
    def disconnected(self):
        self.progress.link_down()


def closing():
    """Ptc's half: a link that ends tells the mail timer at once. The hold
    budget has already asked for the close, as in the 23:46 WS8EOC arm."""
    io = LinkIO()
    a = arq.PactorArq(io, arq.ArqConfig(max_retries=200))
    a._enter_connected()
    a.role = arq.IRS
    a.on_host_disconnect()
    return a, io


def failures(session):
    return [line for line in session.log_lines if line.startswith('session failed')]


def fly(frames, seconds=200., link=None, qualified=True):
    a, io = link or modem()
    session = B2FSession('W9SSJ', target='WS8EOC')
    client = MailClient(session, a.on_host_data)
    now = [0.]
    timer = client.progress = io.progress = MailProgress(client, a, now=lambda: now[0])
    client.link_up()
    last, stages = None, []
    for status, payload in islice(chain(frames, cycle(IDLE)), round(seconds / CYCLE)):
        now[0] += CYCLE
        if payload:
            client.on_link_data(payload)
            last = now[0]
        elif status & spec.STATUS_CHANGEOVER:
            client.on_empty_turn()
        stages.append(session.stage)
        if timer.poll(qualified=qualified):
            break
    return a, session, timer, now[0], last, stages


@pytest.mark.parametrize('frames', [WS8EOC_1010, VE3KPG_1010], ids=['WS8EOC', 'VE3KPG'])
def test_banner_without_sid_expires_and_closes(frames):
    a, session, timer, at, last, _ = fly(frames)
    assert not session.remote_sid
    assert timer.expired and timer.close_requested and a._qrt_pending
    assert last + 120. <= at < last + 120. + CYCLE
    assert 'banner ended without a SID' in session.failure


def test_banner_stall_keeps_its_reason_when_the_close_ends_the_link():
    a, session, timer, at, last, _ = fly(WS8EOC_1010, link=closing(), qualified=False)
    assert timer.expired and a.state == arq.State.DISCONNECTED
    assert last + 120. <= at < last + 120. + CYCLE
    assert failures(session) == ['session failed: gateway banner ended without a SID (CMS side?)']
    said = re.findall(r'^mail: peer said: (.*)$', summarize(session), re.M)
    assert said[-2:] == ['W9SSJ has 112 daily minutes remaining with WS8EOC (EN72QQ)',
                         '{SFI = 121 On 2026-10-10 21:00 UTC}']


def test_their_turn_expiry_names_its_phase_when_the_close_ends_the_link():
    a, io = closing()
    session = B2FSession('W9SSJ', target='WS8EOC')
    client = MailClient(session, a.on_host_data)
    now = [0.]
    timer = client.progress = io.progress = MailProgress(client, a, now=lambda: now[0])
    session._state = _THEIR_TURN
    client._push(b'FF\r')
    acknowledge_all(a)
    assert not timer.poll(qualified=False)
    now[0] = 120.
    assert timer.poll(qualified=False) and a.state == arq.State.DISCONNECTED
    assert failures(session) == [
        'session failed: application progress timeout while their turn; transfer unconfirmed']


def test_banner_with_sid_reaches_their_turn():
    a, session, timer, _, _, stages = fly(WS8EOC_1004)
    assert session.remote_sid == '[WL2K-5.0-B2FWIHJM$]'
    assert stages[len(WS8EOC_1004)-2] == 'their turn'
    assert not timer.expired and not a._qrt_pending and not session.failure


# pactor-current-k4msu-40-force-20261010T225605Z: the CMS line never terminated.
K4MSU_1010 = [
    (0x00, b'RMS'), (0x21, b' Trim'), (0x22, b'ode 1'), (0x23, b'.4.4.'),
    (0x20, b'0 Wel'), (0x21, b'come '), (0x22, b'to K4'), (0x23, b'MSU W'),
    (0x20, b'inlin'), (0x21, b'k Gateway at Murray Sta'),
    (0x02, b'te University\r\nW9SSJ ha'), (0x03, b's 116 daily minutes rem'),
    (0x00, b'aining with K4MSU (EM56'), (0x01, b'UO)\r'), (0x02, b''), (0x43, b''),
    (0x00, b'*** Trimode is unable t'), (0x01, b'o connect to a Winlink '),
    (0x02, b'CMS.'), (0x03, b''), (0x00, b''),
]
CMS_DOWN = '*** Trimode is unable to connect to a Winlink CMS.'


def test_unterminated_cms_line_is_in_the_summary_when_the_link_ends():
    _, session, timer, _, _, _ = fly(K4MSU_1010, seconds=len(K4MSU_1010) * CYCLE)
    timer.link_down()
    said = re.findall(r'^mail: peer said: (.*)$', summarize(session), re.M)
    assert said[-1] == CMS_DOWN
    assert 'link ended' in session.failure


def test_unterminated_cms_line_fails_the_stage_at_the_deadline():
    a, session, timer, at, last, _ = fly(K4MSU_1010)
    assert timer.expired and timer.close_requested and a._qrt_pending
    assert last + 120. <= at < last + 120. + CYCLE
    assert session.remote_refusal == CMS_DOWN
    assert CMS_DOWN in session.failure
    assert session.remote_text.count(CMS_DOWN) == 1


def test_terminated_cms_line_fails_the_stage_at_once():
    frames = [(s, p + b'\r' if p == b'CMS.' else p) for s, p in K4MSU_1010]
    _, session, timer, _, _, stages = fly(frames)
    assert stages.index('failed') == frames.index((0x02, b'CMS.\r'))
    assert session.remote_refusal == CMS_DOWN
    assert not timer.expired


@pytest.mark.parametrize('line, refused', [
    ('*** W9SSJ Connected to CMS', False),
    ('*** W9SSJ not connected to CMS', True),
    ('*** Unable to get connected to CMS', True),
    (CMS_DOWN, True),
])
def test_only_the_connection_notice_escapes_the_star_rule(line, refused):
    s = B2FSession('W9SSJ')
    s.feed((line + '\r').encode())
    assert (s.remote_refusal == line) is refused


def test_a_connection_notice_cut_off_at_the_deadline_is_no_refusal():
    s = B2FSession('W9SSJ')
    s.feed(b'ws.com\r*** W9SSJ Connec')
    s.greeting_stalled()
    assert not s.remote_refusal
    assert 'without a SID' in s.failure
