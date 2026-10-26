# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Automatic speakers-only pilot. Never admits a Rig, even if attached later."""
import time
from hfmodem.contact.calls import checked
from hfmodem.contact.simulation import asks_question, decision
from .runtime import utc


class AutoTest:
    def __init__(self, terminal):
        self.terminal = terminal
        self.session = terminal.session
        if not self.session.test or self.session.rig is not None or self.session.monitor or self.session.replay:
            raise ValueError('Automatic QSO playback requires computer test mode with no rig.')
        self.options = self.session.simulation
        self.paused = False
        self.replies = 0
        self.ready_at = None
        self.listen_at = None
        self.next_cq_at = None
        if self.options.partner:
            self.session.contact.confirm(checked(self.options.partner))

    def heard(self, over, text):
        session, terminal = self.session, self.terminal
        action = decision(session.contact, over, text, self.replies, self.options)
        session.event('simulation_decision', action=action, reply_count=self.replies, partner=session.contact.partner)
        if action == 'ignore':
            return
        session.contact.receive(over)
        session.cancel_draft()
        terminal.draft_text = ''
        session.history.append(dict(role='partner', text=text))
        self.listen_at, self.next_cq_at, self.ready_at = None, None, None
        if action == 'wait':
            self.next_cq_at = time.monotonic() + self.options.next_cq_s
        elif action == 'fill':
            terminal.draft('fill', self.session.codec.fill(session.contact.own_call))
            terminal.notice = 'Uncertain copy; asking for the call again automatically.'
        elif action == 'repeat_message':
            terminal.draft('fill', self.session.codec.fill(session.contact.own_call, 'last message'))
        elif action == 'ack_call':
            terminal.draft('free', self.session.codec.identify('I have your call. Please go ahead with your question.', session.contact.own_call))
        elif action == 'close':
            terminal.draft('close', self.session.codec.close(session.contact.own_call, session.contact.partner))
        else:
            if action == 'accept':
                session.contact.confirm(checked(over.de))
                session.contact_started = utc()
                session.contact.report_sent = self.options.report
            if action == 'accept' or (self.replies == 0 and not asks_question(text) and not over.free):
                terminal.draft('exchange', self.session.codec.greeting(session.contact.own_call, session.contact.partner,
                    session.facts, self.options.report))
            else:
                session.contact.state = 'FREE'
                session.make_free(text, conversation=True,
                                  may_close=self.replies >= 2 and not asks_question(text))
                terminal.draft_text = 'Drafting a reply using the recent conversation…'

    def ready(self, now=None):
        self.ready_at = (time.monotonic() if now is None else now) + .4

    def finished(self, playing, now=None):
        if playing is None or playing['preview']:
            return
        now = time.monotonic() if now is None else now
        if playing['cancelled'] or playing.get('audio_status'):
            self.paused = True
            return
        kind = playing['draft']['kind']
        self.ready_at = None
        if kind == 'close':
            self.next_cq_at, self.listen_at = now + self.options.next_cq_s, None
            self.terminal.notice = 'QSO finished; calling CQ again shortly. Space pauses.'
        else:
            self.listen_at = now
            if kind in ('free', 'exchange'):
                self.replies += 1

    def cq(self):
        session = self.session
        session.contact.reset()
        session.history.clear()
        session.edits.clear()
        self.replies = 0
        self.ready_at = self.listen_at = self.next_cq_at = None
        self.terminal.draft('cq', self.session.codec.cq(session.contact.own_call))

    def tick(self, now=None):
        session = self.session
        if session.rig is not None or not session.test:
            self.paused = True
            raise RuntimeError('Automatic speaker pilot cannot operate with a rig attached.')
        if self.paused or not self.terminal.receive_ok or self.terminal.editor or session.playing or session.receiving:
            return
        now = time.monotonic() if now is None else now
        if session.draft is not None and self.ready_at is not None and now >= self.ready_at:
            session.play()
            self.ready_at = None
        elif session.pending is None and session.draft is None:
            if self.next_cq_at is not None and now >= self.next_cq_at:
                self.cq()
            elif self.listen_at is not None:
                wait = self.options.reply_timeout_s if session.contact.partner else self.options.cq_interval_s
                if now - self.listen_at >= wait:
                    self.listen_at = None
                    if session.contact.partner:
                        session.event('simulation_decision', action='close', reason='reply timeout')
                        self.terminal.draft('close', self.session.codec.close(session.contact.own_call, session.contact.partner))
                    else:
                        self.cq()
