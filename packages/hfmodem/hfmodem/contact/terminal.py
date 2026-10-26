# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Supervised radio terminal and optional automatic computer-only QSO pilot."""
import curses
import queue
import textwrap
import time
import unicodedata
from dataclasses import replace
from hfmodem.contact.calls import checked
from hfmodem.core.rig import RigError
from .txmark import Tag


def edit_layout(value, width):
    """Wrapped text plus the screen position of every insertion point."""
    width = max(2, width)
    rows, positions = [''], []
    row = column = 0
    for char in value:
        size = 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if char != '\n' and column + size > width:
            rows.append('')
            row, column = row + 1, 0
        positions.append((row, column))
        if char == '\n':
            rows.append('')
            row, column = row + 1, 0
        else:
            rows[row] += char
            column += size
            if column == width:
                rows.append('')
                row, column = row + 1, 0
    positions.append((row, column))
    return rows, positions


class Terminal:
    def __init__(self, session):
        self.session = session
        self.notice = 'Enter sends the ready draft. v confirms/edits the caller; s chooses your report.'
        self.editor = None
        self.edit_width = 78
        self.edit_height = 10
        self.last_escape = 0.
        self.heard = ''
        self.partner_text = ''
        self.reviews = []
        self.draft_text = ''
        self.state_at_draft = ''
        self.receive_ok = True
        self.received_mark = None
        self.mark_evidence = None
        self.mark_redraft = None
        self.auto = None
        if session.simulation is not None:
            from .autotest import AutoTest
            self.auto = AutoTest(self)

    def draft(self, kind, text):
        self.draft_text = text
        self.state_at_draft = self.session.contact.state
        self.session.make_draft(kind, text)
        self.notice = 'Preparing the draft…'

    def exchange(self):
        session = self.session
        if not session.contact.partner:
            self.notice = 'Press v to confirm/edit the caller first.'
        elif self.auto:
            self.draft('exchange', self.session.codec.greeting(session.contact.own_call, session.contact.partner,
                                                 session.facts, session.contact.report_sent))
        elif not session.contact.report_sent:
            self.edit('report', '', 'Your signal report to the partner (e.g. 57)')
        else:
            self.draft('exchange', self.session.codec.exchange(session.contact.own_call, session.contact.partner,
                session.contact.report_sent, session.facts, session.contact.heard, session.cfg.voice.exchange))

    def edit(self, kind, value, label):
        self.session.cancel_draft()
        self.editor = dict(kind=kind, value=value, label=label, cursor=len(value), scroll=0)

    def edited(self):
        session = self.session
        kind, value = self.editor['kind'], self.editor['value'].strip()
        self.editor = None
        session.edits.append(dict(kind=kind, value=value))
        session.event('operator_edit', field=kind, value=value)
        if kind == 'mark_code':
            self.change_mark(Tag('code', value) if value else Tag())
        elif kind == 'call':
            call = checked(value)
            if call != session.contact.partner:
                from .runtime import utc
                session.contact_started = utc()
            session.contact.confirm(call)
            self.exchange()
        elif kind == 'report':
            lengths = (2, 3) if session.waveform.name == 'cw' else (2,)
            if len(value) not in lengths or value[0] not in '12345' or any(c not in '123456789' for c in value[1:]):
                raise ValueError('A voice report is two digits: readability 1–5, strength 1–9.')
            session.contact.report_sent = value
            self.exchange()
        elif kind == 'script':
            # The renderer identifies once, after the operator's edit is final.
            self.draft(self.edit_draft_kind, value)
        elif kind.startswith('heard_'):
            field = kind.removeprefix('heard_')
            from hfmodem.contact.frame import Over
            session.contact.heard = replace(session.contact.heard or Over(), **{field: value})
            self.notice = 'Received field updated. v confirms the call; s sets your report; t opens conversation.'

    def heard_event(self, event):
        session = self.session
        session.latest = event
        text = event.get('content_text', event['recognition']['text'])
        self.received_mark = event.get('transmission_mark')
        self.mark_evidence = event.get('mark_evidence')
        if event['kind'] == 'self':
            self.notice = ('Self copy: ' + text) if event.get('loopback_calls_ok') else 'Self copy: callsign missing; replay/check the audio.'
            return
        self.heard = text or '(speech segment produced no trusted words)'
        if text:
            self.partner_text = text
        self.reviews = []
        if not text or not event['over']:
            return
        from hfmodem.contact.frame import Over
        over = Over(**event['over'])
        self.reviews = list(over.review)
        if self.auto and not self.editor:
            self.auto.heard(over, text)
            return
        if (session.waveform.name == 'voice' and session.contact.partner
                and text.strip().lower().strip('.!?') == 'over'
                and not any((over.de, over.report, over.name, over.qth))):
            self.notice = 'Over heard; the current draft is unchanged.'
            return
        session.contact.receive(over)
        if session.monitor:
            self.notice = 'Receive only.'
            return
        session.cancel_draft()
        self.draft_text = ''
        if self.editor:
            self.notice = 'New over heard; finish your edit to draft a reply.'
            return
        if self.reviews:
            self.notice = 'Review the heard over; v confirms/edits the call, s sets your report, f asks for a fill.'
            return
        if session.contact.state == 'CONFIRM' or (session.contact.state == 'LISTEN' and not session.contact.partner):
            self.notice = f'First copy: {over.de or "uncertain"}. Press v to confirm or edit.'
        elif session.contact.state == 'FREE':
            session.make_free(over.free)
            self.draft_text = 'Drafting a brief reply…'
            self.notice = 'The local model is drafting from station facts and partner words.'
        elif session.contact.state == 'CLOSE':
            self.draft('close', self.session.codec.close(session.contact.own_call, session.contact.partner))
        else:
            self.exchange()

    def remember_mark_draft(self):
        draft = self.session.draft
        self.mark_redraft = (draft['kind'], draft.get('source_text', draft['text'])) if draft else None

    def change_mark(self, tag):
        if self.auto:
            self.auto.paused = True
        self.session.set_tx_mark(tag)
        self.draft_text = ''
        if self.mark_redraft:
            kind, source = self.mark_redraft
            self.mark_redraft = None
            self.draft(kind, source)
        self.notice = f'TX mark: {tag.label}. Draft regenerated; Space resumes automatic mode.' if self.auto else f'TX mark: {tag.label}.'

    def key(self, key):
        session = self.session
        if key == '\x1b':
            now = time.monotonic()
            session.finish_playback(cancelled=True)
            session.cancel_draft()
            self.editor = None
            self.draft_text = ''
            if self.auto:
                self.auto.paused = True
                self.auto.ready_at = self.auto.listen_at = self.auto.next_cq_at = None
            self.notice = 'Cancelled and unkeyed. Esc again within 2s quits; c drafts CQ.'
            if now - self.last_escape < 2:
                return False
            self.last_escape = now
            return True
        if self.editor:
            editor = self.editor
            value, cursor = editor['value'], min(editor['cursor'], len(editor['value']))
            if key in ('\n', '\r', curses.KEY_ENTER):
                self.edited()
            elif key in (curses.KEY_LEFT, curses.KEY_RIGHT):
                editor['cursor'] = max(0, min(len(value), cursor + (-1 if key == curses.KEY_LEFT else 1)))
                editor.pop('column', None)
            elif key in (curses.KEY_HOME, '\x01'):
                editor['cursor'] = 0
                editor.pop('column', None)
            elif key in (curses.KEY_END, '\x05'):
                editor['cursor'] = len(value)
                editor.pop('column', None)
            elif key in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_PPAGE, curses.KEY_NPAGE):
                _, positions = edit_layout(value, self.edit_width)
                row, column = positions[cursor]
                editor.setdefault('column', column)
                amount = self.edit_height if key in (curses.KEY_PPAGE, curses.KEY_NPAGE) else 1
                row += -amount if key in (curses.KEY_UP, curses.KEY_PPAGE) else amount
                row = max(0, min(positions[-1][0], row))
                choices = [i for i, point in enumerate(positions) if point[0] == row]
                editor['cursor'] = min(choices, key=lambda i: abs(positions[i][1] - editor['column']))
            elif key in (curses.KEY_BACKSPACE, '\x7f', '\b'):
                editor['value'] = value[:max(0, cursor - 1)] + value[cursor:]
                editor['cursor'] = max(0, cursor - 1)
                editor.pop('column', None)
            elif key == curses.KEY_DC:
                editor['value'] = value[:cursor] + value[cursor + 1:]
                editor.pop('column', None)
            elif key == '\x15':
                editor['value'], editor['cursor'] = value[cursor:], 0
                editor.pop('column', None)
            elif key == '\x0b':
                editor['value'] = value[:cursor]
                editor.pop('column', None)
            elif isinstance(key, str) and key.isprintable() and len(value) < 1000:
                editor['value'] = value[:cursor] + key + value[cursor:]
                editor['cursor'] = cursor + len(key)
                editor.pop('column', None)
            return True
        if key == 'q':
            session.finish_playback(cancelled=True)
            return False
        if self.auto and key == ' ':
            self.auto.paused = not self.auto.paused
            if self.auto.paused:
                self.auto.finished(session.finish_playback(cancelled=True))
                self.notice = 'Automatic computer QSO paused. Space resumes; q quits.'
            else:
                self.auto.ready()
                self.auto.listen_at = time.monotonic()
                self.notice = 'Automatic computer QSO resumed.'
            return True
        if session.playing:
            return True
        if key == 'p' and session.latest:
            session.replay_heard()
            return True
        if session.monitor:
            return True
        if key in ('m', 'd', 'z'):
            if not session.test:
                self.notice = 'Mark controls are available in computer test mode.'
                return True
            if self.auto:
                self.auto.paused = True
            self.remember_mark_draft()
            if key == 'd':
                self.edit('mark_code', session.tx_mark.code, 'Four-digit implicit code (blank switches marking off)')
            else:
                self.change_mark(Tag() if key == 'z' or session.tx_mark.mode == 'hfmodem' else Tag('hfmodem'))
            return True
        if key in ('\n', '\r', curses.KEY_ENTER):
            if not self.receive_ok:
                self.notice = 'Receive failed; restart before sending.'
            else:
                session.play()
        elif key == 'a':
            session.play(preview=True)
        elif key == 'c':
            if self.auto:
                self.auto.cq()
                return True
            session.contact.reset()
            session.history.clear()
            session.edits.clear()
            self.heard = self.partner_text = ''
            from .runtime import utc
            session.contact_started = utc()
            self.draft('cq', self.session.codec.cq(session.contact.own_call))
        elif key == 'v':
            if self.auto:
                self.auto.paused = True
            call = session.contact.heard.de if session.contact.heard else session.contact.partner
            self.edit('call', call, 'Confirmed partner callsign')
        elif key == 's':
            if self.auto:
                self.auto.paused = True
            self.edit('report', session.contact.report_sent, 'Your signal report to the partner')
        elif key in ('h', 'n', 'w'):
            field = {'h': 'report', 'n': 'name', 'w': 'qth'}[key]
            heard = session.contact.heard
            self.edit('heard_' + field, getattr(heard, field, ''), 'Confirmed received ' + field)
        elif key == 'e':
            if self.auto:
                self.auto.paused = True
            text = session.draft['text'] if session.draft else self.draft_text
            self.edit_draft_kind = session.draft['kind'] if session.draft else 'free'
            self.edit('script', text, 'Edit the next over (existing callsign and ending are preserved)')
        elif key == 'f':
            self.draft('fill', self.session.codec.fill(session.contact.own_call))
        elif key in ('t', 'r'):
            if not session.contact.partner:
                self.notice = 'Confirm the partner with v first.'
            elif key == 'r' and session.contact.state != 'FREE':
                self.notice = 'Redraft applies to FREE only; t opens conversation.'
            else:
                session.contact.state = 'FREE'
                heard = self.partner_text or 'The operator opens conversation.'
                session.make_free(heard)
                self.draft_text = 'Drafting a brief reply…'
        return True

    def draw(self, screen):
        session = self.session
        screen.erase()
        height, width = screen.getmaxyx()
        if self.editor:
            self.draw_editor(screen, height, width)
            return
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        line = 0
        def show(text, attr=0):
            nonlocal line
            for part in textwrap.wrap(str(text), max(1, width - 2), replace_whitespace=False) or ['']:
                if line < height - 1:
                    try:
                        screen.addnstr(line, 0, part, max(1, width - 1), attr)
                    except curses.error:
                        pass
                    line += 1
        mode = 'COMPUTER AUDIO — NO RIG' if session.test else 'RECEIVE ONLY' if session.monitor else 'RADIO'
        show(f'{session.waveform.name.upper()} | {mode} | {session.contact.own_call} | {session.contact.state}', curses.A_BOLD)
        show('TX mark: ' + session.tx_mark.label + (' (shared key)' if session.watermark_key else ''))
        if self.received_mark:
            received = Tag(**self.received_mark)
            show('RX mark: ' + received.label + ' (experimental evidence; not authentication)')
        elif self.mark_evidence:
            show(f"RX mark: accumulating ({self.mark_evidence['evidence']} distinct words; "
                 f"{self.mark_evidence['overs']} overs)")
        if self.auto:
            show('AUTOMATIC SIMULATED QSO' + (' — PAUSED' if self.auto.paused else '')
                 + ' | Space pauses/resumes | say 73 to finish')
            if self.auto.options.report:
                show('Simulated report to send: ' + self.auto.options.report)
        if session.waveform.name == 'voice' and session.cfg.voice.synthesis_backend == 'piper':
            show('Banjo / Timothy Ferguson; Data Craftsman CC BY 4.0; base Jenny (Dioco)')
        if session.dial_hz:
            show(f'Dial {session.dial_hz / 1e6:.6f} MHz | {session.cfg.selected_rig.upper()}')
        show('HEARD', curses.A_BOLD)
        show(self.heard[-1400:])
        if session.contact.heard:
            over = session.contact.heard
            show(f'Call: {over.de or "?"} | report received: {over.report or "?"} | name: {over.name or "?"} | QTH: {over.qth or "?"}')
        for review in self.reviews:
            show('REVIEW: ' + review, curses.A_BOLD)
        show('DRAFT', curses.A_BOLD)
        show(session.draft['text'] if session.draft else self.draft_text)
        if session.draft:
            show(f'{session.draft["duration_s"]:.1f}s | '
                 + ('automatic speaker reply | Space pauses' if self.auto and not self.auto.paused else 'Enter sends | a previews'))
        if session.playing:
            show('PLAYING — Esc cancels immediately', curses.A_BOLD)
        elif session.pending:
            show('Working…')
        show('CONTACTS', curses.A_BOLD)
        for record in session.contacts[-3:]:
            show(f'{record["call"]} sent {record["report_sent"]} received {record["report_received"]}'
                 + (' (computer test)' if record['test'] else ''))
        show(self.notice)
        show('Enter send | c CQ | v call | s report | e edit | t talk | r redraft | f fill | a preview | p replay | Esc stop | q quit')
        if session.test and not session.monitor:
            show('m hfmodem/off | d four-digit code | z mark off')
        show('Record: ' + str(session.directory))
        screen.refresh()

    def draw_editor(self, screen, height, width):
        editor = self.editor
        self.edit_width = max(2, width - 2)
        start = min(5, max(1, height - 3))
        self.edit_height = max(1, height - start - 2)
        rows, positions = edit_layout(editor['value'], self.edit_width)
        cursor = min(editor['cursor'], len(editor['value']))
        row, column = positions[cursor]
        scroll = min(editor['scroll'], max(0, len(rows) - self.edit_height))
        if row < scroll:
            scroll = row
        elif row >= scroll + self.edit_height:
            scroll = row - self.edit_height + 1
        editor['scroll'] = scroll
        def put(y, text, attr=0):
            if 0 <= y < height - 1:
                try:
                    screen.addnstr(y, 0, text, max(1, width - 1), attr)
                except curses.error:
                    pass
        put(0, f'EDIT | {self.session.contact.own_call} | {self.session.contact.state}', curses.A_BOLD)
        put(1, editor['label'])
        put(2, 'Arrows move | Home/End | Backspace/Delete | Ctrl-U/K trim | Enter accepts | Esc cancels')
        if start > 3:
            put(3, 'HEARD: ' + self.heard[-max(1, width - 9):])
        for i, text in enumerate(rows[scroll:scroll + self.edit_height]):
            put(start + i, text, curses.A_REVERSE)
        put(height - 2, f'Character {cursor + 1}/{len(editor["value"]) + 1} | Enter accepts; another Enter sends')
        try:
            curses.curs_set(1)
            screen.move(start + row - scroll, column)
        except curses.error:
            pass
        screen.refresh()

    def run(self, screen):
        screen.timeout(40)
        screen.keypad(True)
        if not self.session.monitor:
            self.draft('cq', self.session.codec.cq(self.session.contact.own_call))
        while not self.session.stop.is_set():
            try:
                self.step()
                self.draw(screen)
                try:
                    key = screen.get_wch()
                except curses.error:
                    continue
                if not self.key(key):
                    return
            except (ValueError, RuntimeError, OSError, RigError) as exc:
                if self.auto:
                    self.auto.paused = True
                self.notice = str(exc)

    def step(self):
        """One nonblocking terminal iteration, also usable by a headless rehearsal."""
        self.session.check_receive()
        if self.session.receive_failure:
            self.receive_ok = False
            self.session.cancel_draft()
            if self.auto:
                self.auto.paused = True
            if self.session.playing and not self.session.playing['preview']:
                self.session.finish_playback(cancelled=True)
                self.notice = 'Receive failed; playback cancelled and unkeyed. Restart the session.'
        if self.session.output_done.is_set() and self.session.playing:
            playing = self.session.finish_playback()
            self.notice = ('Playback failed or cancelled; draft retained for review.' if playing['cancelled'] else
                           'Listening for the replying station. c prepares another CQ.')
            if not playing['cancelled'] and playing['draft']['kind'] == 'cq' and not self.auto:
                self.notice = 'Listening. Enter repeats this CQ; c prepares a CQ; v confirms the caller.'
            if self.auto:
                self.auto.finished(playing)
        result = self.session.poll_draft()
        if result:
            self.draft_text = result['text']
            if result.get('review_error'):
                self.notice = result['review_error'] + '. e edits this text; r redrafts; f asks for a fill.'
                if self.auto:
                    self.auto.paused = True
                    self.auto.ready_at = None
            else:
                self.notice = 'Reply ready; playing through the computer speakers.' if self.auto else 'Draft ready. Review it, then Enter sends.'
                if self.auto:
                    self.auto.ready()
        # A new heard over invalidates a pending reply before any automatic playback.
        if not self.session.playing:
            try:
                event = self.session.events.get_nowait()
            except queue.Empty:
                event = None
            if event:
                if event['kind'] in ('heard', 'self'):
                    self.heard_event(event)
                else:
                    self.notice = event.get('text', '')
                    if event['kind'] == 'receive_error':
                        self.receive_ok = False
                        self.session.cancel_draft()
                        if self.auto:
                            self.auto.paused = True
        if self.auto and self.session.events.empty():
            self.auto.tick()


def run(session):
    curses.wrapper(Terminal(session).run)
