# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VarAC application ownership and one-page socket/drain handoff.

This is attached to the existing contact.varac.Session. It never consumes a
second socket, tunes a radio, or asserts that drain means human receipt.
"""
import json
import re

from ..model import BusyError, Presentation
from ..policy import ConnectionLease
from ..presentation import StyleParser, pages


def escape_markup(text):
    """Length-preserving printable substitutions before VarAC framing."""
    return text.replace('<', '[').replace('>', ']').replace('&IT&', '&I T')


class VarACEntrance:
    def __init__(self, session, core=None, *, say=None, reload=None):
        self.session, self.core = session, core
        self.say = say or session.say
        self.reload = reload
        self.reload_future = None
        self.sid = None
        self.connection = None
        self.identity = None
        self.generation = None
        self.dial = None
        self.inflight = None
        self.serial = 0
        self.remote_view = False
        self.remote_ansi = False
        self.remote_dimensions = (80, 24)
        self.requested_term = None
        self.parser = StyleParser()
        self._poison_closed = False
        if core:
            core.transfer_supported = session.bbs_host.catalog is not None
            session.client.enable_writer()

    @property
    def active(self):
        return self.sid is not None

    @property
    def owned(self):
        return self.active or self.remote_view

    @property
    def game(self):
        if not self.active:
            return False
        s = self.core.session
        return s is not None and s.id == self.sid and s.state not in ('lobby', 'transfer', 'closed')

    def linked(self):
        c, s = self.session.client, self.session
        self.generation, self.dial = c.link_generation, s.flow.dial
        self._poison_closed = False
        if self.core:
            try:
                limits = self.core.config.effective_limits(s.peer)
                self.identity = self.core.config.resolve(s.peer)
                selected = True
            except (TypeError, ValueError):
                limits, selected = self.core.config.limits, False
                self.identity = None
            self.connection = ConnectionLease.start(limits, c.clock(), selected=selected)

    def detach(self, reason='attachment ended'):
        if self.active:
            self.core.close(self.sid, reason)
        self.sid, self.inflight = None, None
        self.remote_view, self.remote_ansi = False, False
        self.requested_term = None
        self.parser = StyleParser()

    def ended(self, reason='link ended'):
        self.detach(reason)
        self.connection = None
        self.identity = None

    def hard_stop(self, reason='operator stop'):
        """Fence a blocked local writer before modem clear/abort commands."""
        c = self.session.client
        if c.data_writer and c.data_writer.pending:
            c._stream_poisoned = True
            c.data_writer.close()
        self.detach(reason)

    def _select_identity(self):
        if self.core and self.connection and not self.connection.selected:
            try:
                limits = self.core.config.effective_limits(self.session.peer)
                self.connection.select(limits, self.session.client.clock())
                self.identity = self.core.config.resolve(self.session.peer)
            except (TypeError, ValueError):
                pass

    def settled(self):
        from ...contact.varac import State, CALLING_HZ
        s, c = self.session, self.session.client
        return bool(c.connected and not c.closed and not s._closing and not s._quitting
                    and s.flow.state is State.CONNECTED and s.flow.dial not in CALLING_HZ.values()
                    and not (s.steps is not None or s._opening or s.flow.qsy_phase or s.flow.qsy_receiving
                             or s.files.active or s.bbs_host.active or s.email.waiting or s.mail_ack.busy or s.parking.active
                             or s.compose is not None or s.email.draft is not None
                             or c.pending or c.ptt or s._data_status()[0]
                             or c.data_writer and c.data_writer.pending))

    @staticmethod
    def _control(text):
        from ...contact.varac import decode_application
        application = decode_application(text)
        if '<SM>' in text and application.envelope is None:
            return False
        return application.envelope is not None or any(part.tag for part in application.parts) or bool(
            re.search(r'<(?:SF|BG|BL|EA|EJ|EM|SM|E:|E>)[^>]*>', text))

    @staticmethod
    def _link_control(text):
        return bool(re.fullmatch(r'<(?:Q:-?[0-9]+|QF:[0-9]+)>|<QSYF>[0-9]{12}</QSYF>|<DISC>', text))

    def message(self, raw):
        """Called once, from complete messages, before service side effects."""
        s, c = self.session, self.session.client
        self._select_identity()
        if self.connection and self.connection.expired(c.clock()):
            self._close_contact('Connection allowance expired.')
            return True
        text = raw[:-2] if raw.endswith('\r\n') else raw[:-1] if raw.endswith('\n') else raw
        if self.remote_view:
            if self._link_control(text):
                self.detach('peer link control')
                return False
            if self._control(text):
                if text.startswith('<SF:'):
                    c.send_text('<SFFA>')
                    s.say('Binary offers in the interactive view are refused; use the native file service separately.')
                else:
                    s.emit('bbs_control_refused', text=raw, owner='remote-bbs')
                return True
            match = re.fullmatch(r'BBS TERM OK (PLAIN|ANSI16) ([0-9]+) ([0-9]+)', text)
            if match:
                selected = (match[1].lower(), int(match[2]), int(match[3]))
                if self.requested_term == selected:
                    self.remote_ansi = selected[0] == 'ansi16'
                    self.remote_dimensions = selected[1:]
                    self.requested_term = None
            if text == 'BBS EXIT OK':
                self.detach('return to chat')
                s.say('Returned to ordinary chat.')
                return True
            document = self.parser.feed(raw)
            mode = 'ansi16' if self.remote_ansi else 'plain'
            width, height = self.remote_dimensions
            presentation = Presentation(mode, width, height, 32768, self.remote_ansi)
            for page in pages(document, presentation):
                self.say(page)
            return True
        if self.active:
            if self._link_control(text):
                self.detach('peer link control')
                return False
            if self._control(text):
                # Native file requests receive their documented refusal or
                # acquire the settled lobby lease in BBSServer.message.
                if text == '<BLR>' or text.startswith('<BG:') or text in ('<SFRD>', '<SFOK>', '<SFFA>', '<SFAB>'):
                    return False
                s.emit('bbs_control_refused', text=raw, owner='game' if self.game else 'lobby')
                return True
            self.serial += 1
            input_id = f'{c.link_generation}:{self.serial}'
            accepted = self.core.input(self.sid, input_id, text)
            s.emit('bbs_input', session_id=self.sid, input_id=input_id, accepted=accepted,
                   player_id=self.core.session.player.id if self.core.session.player else None)
            return True
        if text.strip().upper() in ('BBS', 'BBS HELP') and self.core:
            if not self.settled():
                c.send_text('BBS entry requires a settled link off the calling frequency and no active exchange.')
                return True
            if self.connection is None or not self.connection.selected:
                # Never manufacture a fresh allowance from BBS entry.
                c.send_text('BBS connection policy is unavailable; reconnect before entry.')
                return True
            if self.core.config.resolve(s.peer) != self.identity:
                c.send_text('Connection identity changed; reconnect before BBS entry.')
                return True
            try:
                self.sid = self.core.open(s.peer, Presentation(page_bytes=600),
                                          connection=self.connection, mode='varac')
                self.generation, self.dial = c.link_generation, s.flow.dial
                s.emit('bbs_enter', session_id=self.sid, peer=s.peer)
            except (BusyError, ValueError):
                c.send_text('BBS is busy or this connection allowance has expired.')
            return True
        return False

    def native_file_lease(self):
        """None means refusal; False means no interactive attachment owns it."""
        if not self.active:
            return False
        token = self.core.claim_transfer(self.sid)
        if token is None:
            return None
        return self.sid, token, self.session.client.link_generation, self.session.flow.dial

    def transfer_done(self, lease, disposition):
        if not lease:
            return
        sid, output_id, generation, dial = lease
        if sid == self.sid and generation == self.session.client.link_generation and dial == self.session.flow.dial:
            self.core.output_done(sid, output_id, disposition)

    def writer_completion(self, raw):
        record = json.loads(raw)
        if record['disposition'] == 'unknown':
            self._close_contact('Data write extent is unknown; no further framed output is safe.')
            return
        if self.inflight and self.inflight['write_id'] == record['write_id'] and record['generation'] == self.generation:
            if record['disposition'] == 'complete':
                self.inflight['after'] = record['after']
            else:
                self._close_contact('BBS output was discarded before completion.')

    def _close_contact(self, reason):
        s = self.session
        self.hard_stop(reason)
        if s._closing or s._quitting:
            return
        if s.steps is not None:
            s.steps.close()
            s.steps = None
        s._force_close = s._closing = True
        s.repeat = False
        s.say(reason)
        # Host control only: never append farewell or DISC on a poisoned stream.
        s._run(s.flow.close_steps(cancelled=lambda: True))

    def tick(self):
        s, c = self.session, self.session.client
        if self.core:
            self.core.tick()
        self._select_identity()
        if self.reload_future is not None and self.reload_future.done():
            future, self.reload_future = self.reload_future, None
            try:
                config = future.result()
                revision = self.core.reload_policy(config)
                s.say(f'BBS daily policy reloaded (revision {revision}); connection allowances keep their snapshots.')
            except (OSError, ValueError, RuntimeError) as error:
                s.say(f'BBS policy reload refused: {error}')
        if c._stream_poisoned and not self._poison_closed:
            self._poison_closed = True
            self._close_contact('Data stream poisoned; closing through host control without application frames.')
            return
        if not c.connected or c.closed:
            self.ended()
            return
        if self.connection and self.connection.expired(c.clock()) and not s._closing:
            self._close_contact('Connection allowance expired.')
            return
        if self.owned and (self.generation != c.link_generation or self.dial != s.flow.dial):
            self.detach('link generation or frequency changed')
            return
        if not self.active:
            return
        state = self.core.session
        if state and state.id == self.sid and state.state == 'closed':
            reason = state.pending[0].text if state.pending else 'BBS closed'
            self.inflight = None
            if reason == 'exit':
                self.detach('exit')
                c.send_text('BBS EXIT OK')
            elif reason == 'goodbye':
                self.detach('goodbye')
                s._run(s._farewell())
            else:
                self._close_contact(reason)
            return
        if self.inflight:
            after = self.inflight['after']
            if after is None:
                return
            if c._discard_sequence > after:
                self._close_contact('BBS output queue was discarded; delivery unconfirmed.')
                return
            if (c.buffer_sequence > after and c.buffer == 0 and not c.ptt and not c.pending
                    and not (c.data_writer and c.data_writer.pending)):
                event, self.inflight = self.inflight['event'], None
                self.core.output_done(self.sid, event.output_id, 'complete')
            return
        if c.data_writer and c.data_writer.pending or s._data_status()[0]:
            return
        events = self.core.poll(self.sid)
        if not events:
            return
        event = events[0]
        if event.kind == 'transfer':
            lease = self.sid, event.output_id, c.link_generation, s.flow.dial
            try:
                s.bbs_host.offer_snapshot(event.payload, lease=lease)
            except (OSError, ValueError):
                self.transfer_done(lease, 'failed')
            return
        if event.kind in ('exit', 'close'):
            return  # closed state handled at the next owner tick
        write_id = c.send_text(escape_markup(event.text))
        self.inflight = dict(event=event, write_id=write_id, after=None)

    def command(self, line):
        """True if the explicit local command/BBS client view consumed it."""
        s, c = self.session, self.session.client
        words = line.split()
        if words in (['/status'], ['/help'], ['/help', 'bbs']):
            return False
        if words[:2] == ['/bbs', 'reload']:
            if words != ['/bbs', 'reload'] or self.reload is None or self.core is None:
                s.say('BBS reload is available only for a configured local service.')
            elif self.reload_future is not None:
                s.say('BBS policy reload is already pending.')
            else:
                self.reload_future = self.reload()
                s.say('BBS policy reload requested.')
            return True
        if self.active:
            if words[:1] in (['/stop'], ['/bye'], ['/qsy'], ['/quit']):
                if words[:1] == ['/qsy'] and (self.inflight or c.data_writer and c.data_writer.pending):
                    s.say('Wait for BBS output to settle before QSY, or /stop ends this contact.')
                    return True
                self.detach('operator control')
                return False
            s.say('Remote BBS owns this conversation. End it with /bye or /stop before operator chat or another service.')
            return True
        if words == ['/bbs', 'enter']:
            if not self.settled():
                s.say('Interactive BBS entry requires a settled off-calling-frequency link.')
                return True
            c.enable_writer()
            self.remote_view, self.remote_ansi = True, False
            self.generation, self.dial = c.link_generation, s.flow.dial
            self.parser = StyleParser()
            c.send_text('BBS')
            return True
        if words[:2] == ['/bbs', 'term']:
            if not self.remote_view:
                s.say('Use /bbs enter before choosing interactive terminal presentation.')
                return True
            try:
                mode = words[2].lower()
                width, height = map(int, words[3:]) if len(words) == 5 else (80, 24)
                if len(words) not in (3, 5):
                    raise ValueError()
                Presentation(mode, width, height, ansi_capable=mode == 'ansi16')
            except (IndexError, ValueError):
                s.say('Use /bbs term plain or /bbs term ansi16 80 24.')
                return True
            self.requested_term = mode, width, height
            if mode == 'plain':
                self.remote_ansi = False  # local disable is always immediate
            c.send_text(f'BBS TERM {mode.upper()} {width} {height}')
            return True
        if self.remote_view:
            if words[:1] in (['/stop'], ['/bye'], ['/qsy'], ['/quit']):
                if words[:1] == ['/qsy'] and c.data_writer and c.data_writer.pending:
                    s.say('Wait for BBS output to settle before QSY, or /stop ends this contact.')
                    return True
                self.detach('operator control')
                return False
            if line.startswith('/'):
                s.say('Interactive BBS view: send a menu/game line, /bbs term, /bye, or /stop.')
            else:
                c.send_text(line if line else 'BBS MORE')
            return True
        return False
