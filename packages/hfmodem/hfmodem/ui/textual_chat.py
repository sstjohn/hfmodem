# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import os
import signal
import sys
import termios
import time
from uuid import uuid4

from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.drivers.linux_driver import LinuxDriver
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, RichLog, Static, TabbedContent, TabPane, TextArea

from ..contact import chat_state as s
from ..contact.chat_controller import Controller, TERMINAL
from ..core.rig import FATAL_SIGNALS

HISTORY_BYTES = 2 * 1024 * 1024
RECOVERY_RATE = 4 * 1024 * 1024
DEPENDENCIES = frozenset((s.ProjectionSection.LIVE, s.ProjectionSection.APPLICATION,
                          s.ProjectionSection.CAPABILITIES))


class _DeferredInterrupt:
    def __init__(self, handlers):
        self.handlers = handlers
        self.error = None
        self.callbacks = []

    def __call__(self, signum, frame):
        try:
            handler = self.handlers[signum]
            if callable(handler):
                handler(signum, frame)
        except (KeyboardInterrupt, SystemExit) as error:
            # Panic must run now; unwinding a lock acquisition can strand its owner.
            self.error = error
            for callback in tuple(self.callbacks):
                callback(error)


@contextmanager
def defer_interrupts(callback=None):
    previous = {sig: signal.getsignal(sig) for sig in FATAL_SIGNALS}
    handler = previous[signal.SIGINT]
    shared = isinstance(handler, _DeferredInterrupt) and all(
        value is handler for value in previous.values())
    if not shared:
        handler = _DeferredInterrupt(previous)
    if callback is not None:
        handler.callbacks.append(callback)
    if not shared:
        for sig in previous:
            signal.signal(sig, handler)
    try:
        if callback is not None and handler.error is not None:
            callback(handler.error)
        yield
    finally:
        if callback is not None:
            handler.callbacks.remove(callback)
        if not shared:
            for sig, saved in previous.items():
                signal.signal(sig, saved)
            if handler.error is not None:
                raise handler.error


def finish_controller(controller):
    with defer_interrupts():
        completion = controller.shutdown()
        while not completion.done():
            time.sleep(.01)
        while not controller.join(0):
            time.sleep(.01)
        completion.result()


def merge_stamp(left, right):
    return tuple(max(a, b) for a, b in zip(left, right))


def behind(stamp, floor):
    return any(a < b for a, b in zip(stamp, floor))


COLORS = ('#000000', '#800000', '#008000', '#808000', '#000080', '#800080', '#008080', '#c0c0c0',
          '#808080', '#ff0000', '#00ff00', '#ffff00', '#0000ff', '#ff00ff', '#00ffff', '#ffffff')


def plain(text):
    return ''.join(c for c in text if c in '\n\t' or ord(c) >= 32 and not 127 <= ord(c) < 160)


def color(code, background=False):
    base, bright = (40, 100) if background else (30, 90)
    if code is not None:
        if base <= code <= base + 7:
            return COLORS[code - base]
        if bright <= code <= bright + 7:
            return COLORS[code - bright + 8]
    return None


def styled(page):
    result = Text()
    for span in page.spans:
        style = None
        if page.mode == 'ansi16':
            foreground = color(span.foreground)
            background = color(span.background, background=True)
            style = Style(color=foreground, bgcolor=background, bold=span.bold)
        result.append(plain(span.text), style)
    return result


def parse_action(line):
    words = line.split()
    if not words:
        raise ValueError('Choose an action')
    command, args = words[0], words[1:]
    fixed = {
        '/status': s.ChatProjection(), '/help': s.ChatHelp(None), '/heard': s.HeardList(),
        '/slots': s.SlotsList(), '/stop': s.ChatCancelWait(), '/quit': s.ChatExit(),
        '/clear': s.BufferClear(), '/back': s.AwayBack(), '/vmail': s.VMailList(),
    }
    if not args and command in fixed:
        return fixed[command]
    if command == '/help':
        return s.ChatHelp(' '.join(args))
    if command == '/call' and 1 <= len(args) <= 2:
        return s.ChatCall(args[0], int(args[1]) if len(args) == 2 and args[1] != 'cf' else None,
                          not (len(args) == 2 and args[1] == 'cf'))
    if command == '/answer' and len(args) == 1:
        return s.ChatAnswer(int(args[0]))
    if command == '/cq':
        repeat = bool(args and args[-1] == 'repeat')
        slots = args[:-1] if repeat else args
        if len(slots) <= 1:
            return s.ChatCQ(int(slots[0]) if slots else None, repeat)
    if command == '/qsy' and len(args) == 1:
        return s.ChatQSY(int(args[0]))
    if command == '/tag' and args:
        name, separator, value = line[5:].partition(' ')
        return s.ChatTag(name, value if separator else None)
    if command == '/bye':
        return s.ChatClose(line[5:] or '73')
    if command == '/follow' and len(args) == 1:
        return s.FollowSet(None if args[0] == 'off' else int(args[0]) if args[0].isdigit() else args[0])
    if command == '/band' and len(args) == 1:
        return s.BandReady(args[0] == 'ready') if args[0] in ('ready', 'unready') else s.BandSet(int(args[0]))
    if command == '/auto' and args in (['on'], ['off']):
        return s.AutoSet(args[0] == 'on')
    if command == '/verbose' and args in (['on'], ['off']):
        return s.VerboseSet(args[0] == 'on')
    if command == '/away' and args:
        return s.AwaySet(line[5:].strip())
    if command == '/pskreporter' and args in (['status'], ['off']):
        return s.PSKReporterOff() if args[0] == 'off' else s.PSKReporterStatus()
    if command == '/inquire' and args == ['heard']:
        return s.InquireHeard()
    if command == '/beacon':
        if args == ['hop', 'status']:
            return s.BeaconHopStatus()
        if args and args[0] == 'hop':
            values = args[1:]
            auto = bool(values and values[0] == 'auto')
            if auto:
                values = values[1:]
            if len(values) >= 2:
                return s.BeaconHop(tuple(int(v) for v in values[:-1]), auto, int(values[-1]))
        if args in (['now'], ['force']):
            return s.BeaconOnce(args[0] == 'force')
        if len(args) == 1:
            return s.BeaconConfigure(None if args[0] == 'off' else int(args[0]))
    if command == '/all':
        if args == ['cf']:
            return s.BroadcastSelect(None)
        if not args or args == ['help']:
            return s.ChatHelp('all')
        if args == ['status']:
            return s.BroadcastStatus()
        if args == ['clear']:
            return s.BroadcastClear()
        if args in (['yes'], ['no']):
            return s.BroadcastConfirm(args == ['yes'])
        if args == ['off']:
            return s.BroadcastOff()
        if len(args) == 2 and args[0] == 'slot':
            return s.BroadcastSelect(None if args[1] == 'cf' else int(args[1]))
        if args:
            return s.BroadcastSend(line[4:].strip())
    if command == '/bbs':
        if len(args) == 1 and args[0] in ('list', 'enter', 'reload', 'capture'):
            return {'list': s.BBSList, 'enter': s.BBSEnter,
                    'reload': s.BBSReload, 'capture': s.BBSCapture}[args[0]]()
        if len(args) >= 2 and args[0] in ('get', 'request', 'offer'):
            return (s.BBSGet if args[0] == 'get' else s.BBSRequest)(' '.join(args[1:]))
        if args[:1] == ['term'] and len(args) in (2, 4):
            return s.BBSTerm(args[1], int(args[2]) if len(args) == 4 else 80,
                             int(args[3]) if len(args) == 4 else 24)
    if command == '/vmail':
        if args == ['pending']:
            return s.VMailPending()
        if len(args) == 1 and args[0].isdigit():
            return s.VMailRead(int(args[0]))
        if len(args) >= 2 and args[0] == 'send':
            return s.VMailCompose(line.split(None, 2)[2].strip())
        if len(args) == 2 and args[0] in ('collect', 'dismiss'):
            return (s.VMailCollect if args[0] == 'collect' else s.VMailDismiss)(args[1])
    if command == '/email':
        if args == ['status']:
            return s.EmailStatus()
        if args[:1] == ['send'] and len(args) >= 4:
            return s.EmailCompose(args[1], args[2], ' '.join(args[3:]))
    raise ValueError('Unknown action or missing parameters; /help lists commands')


class Composer(TextArea):
    class LimitReached(Message):
        pass

    class Submitted(Message):
        pass

    def __init__(self, **kwargs):
        super().__init__(tab_behavior='focus', **kwargs)
        self.buffering = False

    def _can_insert(self, value):
        removed = self.get_text_range(*self.selection)
        size = len(self.text.encode('utf-8')) - len(removed.encode('utf-8')) + len(value.encode('utf-8'))
        if size > 64 * 1024:
            self.post_message(self.LimitReached())
            return False
        return True

    async def _on_key(self, event):
        if event.key == 'enter' and not self.buffering:
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted())
            return
        insert = '\n' if event.key == 'enter' else event.character if event.is_printable else ''
        if insert and not self._can_insert(insert):
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)

    async def _on_paste(self, event):
        event.prevent_default()
        value = event.text.replace('\r\n', '\n').replace('\r', '\n')
        if not self._can_insert(value):
            event.stop()
            return
        if '\n' in value:
            self.buffering = True
        await super()._on_paste(events.Paste(value))
        event.stop()


class Information(ModalScreen):
    BINDINGS = [('escape', 'dismiss', 'Back')]

    def __init__(self, title, text):
        super().__init__()
        self.heading, self.content = title, text

    def compose(self):
        with Vertical(classes='dialog'):
            yield Label(self.heading)
            yield RichLog(id='information', markup=False, wrap=True)
            yield Button('Back', id='dismiss')

    def on_mount(self):
        self.query_one(RichLog).write(Text(plain(self.content)))

    @on(Button.Pressed, '#dismiss')
    def close_dialog(self):
        self.dismiss()


class MailScreen(ModalScreen):
    BINDINGS = [('escape', 'dismiss', 'Back')]

    def compose(self):
        with Vertical(classes='dialog'):
            yield Label('Mail')
            yield RichLog(id='mail-list', markup=False, wrap=True)
            yield Input(placeholder='Subject', id='subject')
            yield Input(placeholder='Recipient email address', id='recipient')
            yield Input(placeholder='Reply email address', id='reply-to')
            with Horizontal():
                yield Button('Compose VMail', id='vmail-compose')
                yield Button('Compose email', id='email-compose')
                yield Button('Back', id='dismiss')

    def on_mount(self):
        app = self.app
        application = app.view.application
        text = '\n'.join(f'{m.number}: {m.sender} → {m.recipient}: {m.subject}'
                         for m in application.mailbox)
        text += '\n' + '\n'.join(f'Parked mail: {n.source} → {n.destination}' for n in application.notices)
        self.query_one('#mail-list', RichLog).write(Text(plain(text.strip() or 'No mail')))
        self.query_one('#subject', Input).focus()

    @on(Button.Pressed)
    def button(self, event):
        self.run_worker(self._button(event), group='mail', exclusive=True)

    async def _button(self, event):
        if event.button.id == 'dismiss':
            self.dismiss()
            return
        subject = self.query_one('#subject', Input).value
        if event.button.id == 'vmail-compose':
            action = s.VMailCompose(subject)
        else:
            action = s.EmailCompose(self.query_one('#recipient', Input).value,
                                    self.query_one('#reply-to', Input).value, subject)
        if await self.app.perform(action):
            self.dismiss()
            self.app.action_focus_composer()


class ActionPalette(ModalScreen):
    BINDINGS = [('escape', 'dismiss', 'Back')]

    def compose(self):
        with Vertical(classes='dialog'):
            yield Label('Actions')
            yield Static('/call CALL [SLOT] · /cq [SLOT] · /bbs enter · /vmail send SUBJECT\n'
                         '/auto on|off · /beacon now|force · /band N · /help', markup=False)
            yield Input(placeholder='Type an action', id='action')
            yield Static('', id='preview', markup=False)
            with Horizontal():
                yield Button('Apply', id='apply')
                yield Button('Back', id='dismiss')

    def on_mount(self):
        self.query_one(Input).focus()
        self._version = 0
        self._checking = False
        self._dirty = False
        self._applying = False
        self._last_check = float('-inf')
        self.set_interval(.2, self.preview)

    @on(Input.Changed)
    def changed(self):
        self._version += 1
        self._dirty = True

    def preview(self):
        if self._checking or not self._dirty or self.app.clock() - self._last_check < .2:
            return
        self._dirty = False
        self._checking = True
        self._last_check = self.app.clock()
        self.run_worker(self._preview(), group='preview')

    async def _preview(self):
        version = self._version
        try:
            action = parse_action(self.query_one(Input).value)
            if s.effect(action) is s.Effect.R:
                text = 'Read only'
            else:
                result = await asyncio.shield(asyncio.wrap_future(self.app.controller.check(action)))
                if version != self._version or result.controller_id != self.app.view.controller_id:
                    return
                if result.revision < self.app.view.revision or result.action != action:
                    self._dirty = True
                    return
                cap = result.capability
                text = cap.reason or cap.availability
            if version == self._version:
                self.query_one('#preview', Static).update(plain(text))
        except (ValueError, RuntimeError) as error:
            self.query_one('#preview', Static).update(plain(str(error)))
        finally:
            self._checking = False

    @on(Input.Submitted)
    @on(Button.Pressed, '#apply')
    def apply_action(self):
        if not self._applying:
            self._applying = True
            self.run_worker(self._apply_action(), group='action')

    async def _apply_action(self):
        try:
            action = parse_action(self.query_one(Input).value)
        except ValueError as error:
            self.query_one('#preview', Static).update(plain(str(error)))
            self._applying = False
            return
        try:
            outcome = await self.app.perform(action)
            if outcome:
                self.dismiss(outcome)
        finally:
            self._applying = False

    @on(Button.Pressed, '#dismiss')
    def close_dialog(self):
        self.dismiss()


class TTYDriver(LinuxDriver):
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self._file = app.chat_terminal
        self.fileno = app.chat_terminal.fileno()
        self.input_tty = True
        self._in_alternate_screen = False
        self._kitty_keyboard_owned = False
        self._terminal_restored = False

    @classmethod
    def _patch_lflag(cls, attrs):
        # The kernel must deliver Ctrl-C even before the input thread exists.
        return super()._patch_lflag(attrs) | termios.ISIG | termios.NOFLSH

    def _get_terminal_size(self):
        try:
            return tuple(os.get_terminal_size(self.fileno))
        except OSError:
            return super()._get_terminal_size()

    def write(self, data):
        # Kitty's report-all-keys encodes Ctrl-C instead of sending VINTR.
        if data.startswith('\x1b[>') and data.endswith('u'):
            return
        # A suppressed push gives Textual no keyboard stack entry to pop.
        if data == '\x1b[<u':
            if not self._kitty_keyboard_owned or not self._in_alternate_screen:
                return
            self._kitty_keyboard_owned = False
        super().write(data)
        if data == '\x1b[?1049h':
            self._in_alternate_screen = True
        elif data == '\x1b[?1049l':
            self._in_alternate_screen = False

    def start_application_mode(self):
        self._saved_attrs = termios.tcgetattr(self.fileno)
        self._terminal_restored = False
        try:
            attrs = termios.tcgetattr(self.fileno)
            attrs[3] |= termios.ISIG | termios.NOFLSH
            attrs[6][termios.VINTR] = b'\x03'
            disabled = os.fpathconf(self.fileno, 'PC_VDISABLE')
            attrs[6][termios.VQUIT] = disabled
            attrs[6][termios.VSUSP] = disabled
            termios.tcsetattr(self.fileno, termios.TCSANOW, attrs)
            super().start_application_mode()
        except BaseException as error:
            # Textual's startup precedes its stop-application-mode finalizer.
            with defer_interrupts():
                try:
                    self._app.prepare_terminal_restore(error)
                finally:
                    self._restore_terminal()
            raise

    def stop_application_mode(self):
        if self._terminal_restored:
            return
        # Textual restores the driver before dispatching Unmount on failures.
        with defer_interrupts():
            try:
                self._app.prepare_terminal_restore(sys.exception())
            finally:
                self._restore_terminal()

    def run_input_thread(self):
        # Readiness can go stale before Textual's read; VTIME bounds that read.
        attrs = termios.tcgetattr(self.fileno)
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 1
        termios.tcsetattr(self.fileno, termios.TCSANOW, attrs)
        super().run_input_thread()

    def disable_input(self):
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        self.exit_event.set()
        thread = self._key_thread
        if thread is not None and thread.ident is not None:
            while thread.is_alive():
                thread.join(0)
                time.sleep(.01)

    def close(self):
        writer = self._writer_thread
        if writer is not None and writer.ident is not None:
            writer._queue.put(None)
            while writer.is_alive():
                writer.join(0)
                time.sleep(.01)
        self._writer_thread = None

    def _restore_terminal(self):
        try:
            if self._writer_thread is not None and self._writer_thread.ident is not None:
                super().stop_application_mode()
        finally:
            self.disable_input()
            self.close()
            # Drain the writer first so no queued frame can undo this reset.
            keyboard = '\x1b[<u' if self._kitty_keyboard_owned and self._in_alternate_screen else ''
            screen = '\x1b[?1049l' if self._in_alternate_screen else ''
            try:
                self._file.write('\x1b[?2004l\x1b[?7h\x1b[?2048l'
                                 '\x1b[?1000l\x1b[?1003l\x1b[?1015l\x1b[?1006l\x1b[?1016l'
                                 + keyboard + screen + '\x1b[?25h\x1b[?1004l')
                self._file.flush()
                termios.tcdrain(self.fileno)
            finally:
                termios.tcsetattr(self.fileno, termios.TCSANOW, self._saved_attrs)
            self._kitty_keyboard_owned = False
            self._in_alternate_screen = False
            self._terminal_restored = True


class ChatFooter(Footer):
    def compose(self):
        yield from super().compose()
        self.styles.grid_size_columns = 4


class ChatApp(App):
    TITLE = 'Kestrel chat — offline prototype'
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding('tab', 'focus_next', 'Next', priority=True),
        Binding('shift+tab', 'focus_previous', 'Previous', priority=True),
        Binding('f2', 'commands', 'Actions'), Binding('f3', 'diagnostics', 'Diagnostics'),
        Binding('f4', 'mail', 'Mail'), Binding('ctrl+l', 'focus_composer', 'Composer', priority=True),
        Binding('ctrl+s', 'submit', 'Send', priority=True),
        Binding('ctrl+x', 'cancel_wait', 'Cancel', priority=True),
        Binding('ctrl+b', 'close_contact', 'Close', priority=True),
        Binding('ctrl+q', 'ordered_exit', 'Exit', priority=True),
        Binding('ctrl+c', 'interrupt', 'Panic', priority=True),
    ]
    CSS = '''
    Screen { layout: vertical; }
    #status { height: auto; max-height: 5; padding: 0 1; background: $panel; }
    #content { height: 1fr; }
    #main-tabs { width: 2fr; }
    #side { width: 1fr; min-width: 24; }
    RichLog { height: 1fr; }
    #mode, #notice, #history-notice, #resize-notice { height: auto; max-height: 3; }
    #composer { height: 4; border: solid $primary; }
    #bbs-input { dock: bottom; }
    .dialog { width: 90%; max-width: 90; height: 22; max-height: 100%; padding: 0 1; border: solid $primary; background: $surface; }
    ModalScreen { align: center middle; }
    .dialog Horizontal { height: auto; }
    .dialog Button { margin-right: 1; }
    .dialog Static { height: auto; max-height: 5; }
    #mail-list { height: 1fr; min-height: 1; max-height: 4; }
    ChatFooter { layout: grid; grid-size: 4; grid-rows: 1; height: 3; }
    ChatFooter FooterKey { width: 1fr; margin: 0; }
    #resize-notice { display: none; }
    .compact #content, .compact #mode, .compact #composer { display: none; }
    .compact #resize-notice { display: block; }
    .narrow #side { display: none; }
    '''

    class Updates(Message):
        def __init__(self, events):
            super().__init__()
            self.events = events

    def __init__(self, controller, *, actor=None, clock=None, title=None,
                 terminal=None, stdout=None, stderr=None, exit_on_failure=False, interrupt=None):
        self.chat_terminal = terminal
        self.interrupt = interrupt
        self._terminal_failure = None
        super().__init__(driver_class=TTYDriver if terminal is not None else None)
        if stdout is not None:
            self._capture_stdout = stdout
        if stderr is not None:
            self._capture_stderr = stderr
        if title is not None:
            self.title = title
        self.controller = controller
        self.exit_on_failure = exit_on_failure
        self.actor = actor or str(uuid4())
        self.clock = clock or controller.clock
        self.view = s.project(replace(s.initial_snapshot(), controller_id=controller.controller_id))
        self.records = {}
        self._record_sizes = {}
        self._record_bytes = 0
        self.history_evicted_through = None
        self._local_bbs_loss_context = None
        self._recovery_pending = False
        self._recovering = False
        self._next_recovery_at = float('-inf')
        self._bulk_credit = float(RECOVERY_RATE)
        self._bulk_at = self.clock()
        self._bulk_reading = False
        self._recovery_gap_through = 0
        self._recovery_buffer = []
        self._recovery_buffer_bytes = 0
        self.display_cursor = 0
        self._dirty = {}
        self._applied = {section: (0, 0) for section in s.ProjectionSection}
        self._last_projection = float('-inf')
        self._upload_epoch = 0
        self._submitting = False
        self._exiting_chat = False
        self._orphan = False
        self._editor_generation = self.view.live.link.generation
        self._bbs_fence = None
        self._bbs_submitting = False
        self.unread = 0
        self.history_reads = 0
        self.projection_reads = 0
        self._subscription = None
        self._rendered_records = None
        self._rendered_history = None

    @property
    def failure(self):
        # Textual records handler/driver failures instead of raising from run().
        return self._exception or self._terminal_failure

    def prepare_terminal_restore(self, error=None):
        if error is not None:
            self._terminal_failure = error
        finish_controller(self.controller)

    def _interrupted(self, error):
        self._terminal_failure = error
        self.exit()

    def run(self, **kwargs):
        if self.interrupt is None:
            return super().run(**kwargs)
        with defer_interrupts(self._interrupted):
            return super().run(**kwargs)

    def compose(self) -> ComposeResult:
        yield Static('', id='status', markup=False)
        yield Static('', id='history-notice', markup=False)
        yield Static('Terminal too small. Resize to at least 40 × 12; Ctrl+Q exits.', id='resize-notice', markup=False)
        with Horizontal(id='content'):
            with TabbedContent(id='main-tabs'):
                with TabPane('Contact', id='contact-tab'):
                    yield RichLog(id='contact', markup=False, wrap=True, max_lines=2000)
                with TabPane('BBS', id='bbs-tab'):
                    yield RichLog(id='bbs', markup=False, wrap=True, max_lines=2000)
                    yield Input(placeholder='BBS menu input', id='bbs-input')
                with TabPane('Heard / Slots', id='receive-tab'):
                    yield RichLog(id='receive', markup=False, wrap=True, max_lines=2000)
            with Vertical(id='side'):
                yield Input(placeholder='Filter heard / slots', id='filter')
                yield RichLog(id='heard-slots', markup=False, wrap=True, max_lines=2000)
                yield RichLog(id='passband', markup=False, wrap=True, max_lines=2000)
        yield Static('', id='mode', markup=False)
        yield Composer(id='composer', placeholder='Enter sends one line; multiline paste uses Ctrl+S')
        yield Static('', id='notice', markup=False)
        yield ChatFooter(compact=True)

    def _widget(self, selector, expected=None):
        if not hasattr(self, '_widgets'):
            return self.screen_stack[0].query_one(selector, expected)
        if isinstance(selector, str):
            return self._widgets[selector.removeprefix('#')]
        return next(widget for widget in self._widgets.values() if isinstance(widget, selector))

    def _visible(self):
        return hasattr(self, "_status_widget") and self._status_widget.is_mounted

    async def on_mount(self):
        self._widgets = {widget.id: widget for widget in self.screen_stack[0].walk_children() if widget.id}
        self._status_widget = self._widget("#status", Static)
        await asyncio.shield(asyncio.wrap_future(self.controller.ready))
        self._subscription = self.controller.subscribe(self.display_cursor)
        self.recover(initial=True)
        self.set_interval(.02, self.collect_updates)
        self._widget(Composer).focus()
        self.on_resize(events.Resize(self.size, self.size))

    def collect_updates(self):
        if not self._visible():
            return
        # Lifecycle completion cannot depend on a retained display event.
        if self.exit_on_failure and self.controller.join(0):
            self.exit()
            return
        if self._subscription is not None:
            batch = self._subscription.drain()
            if batch:
                self.post_message(self.Updates(batch))
        self._maybe_recover()
        if not self._recovery_pending and self._dirty and self.clock() - self._last_projection >= .2:
            self.refresh_projection()
        self.render_status()

    def on_chat_app_updates(self, message):
        if not self._visible():
            return
        for event in message.events:
            if event.controller_id != self.view.controller_id:
                continue
            if isinstance(event.payload, s.Gap):
                if event.payload.through_seq <= self.display_cursor:
                    continue
                if not self._recovery_pending:
                    self._upload_epoch += 1
                self._recovery_pending = True
                self._recovery_gap_through = max(self._recovery_gap_through,
                                                 event.payload.through_seq)
                self._recovery_buffer.clear()
                self._recovery_buffer_bytes = 0
                self._bbs_fence = None
                self._widget('#bbs-input', Input).disabled = True
                self.notice('Event gap; recovery pending. Input disabled.')
                continue
            if event.seq <= self.display_cursor:
                continue
            if self._recovery_pending:
                size = len(s.encode(event))
                if len(self._recovery_buffer) >= 128 or self._recovery_buffer_bytes + size > 256 * 1024:
                    self._recovery_buffer.clear()
                    self._recovery_buffer_bytes = 0
                    self._recovery_gap_through = max(self._recovery_gap_through, event.seq)
                else:
                    self._recovery_buffer.append(event)
                    self._recovery_buffer_bytes += size
                continue
            payload = event.payload
            if isinstance(payload, s.Lifecycle) and self.exit_on_failure:
                self.exit()
                continue
            if isinstance(payload, s.DisplayChanged):
                old = self.records.get(payload.record.id)
                if old is None or old.updated_seq < payload.record.updated_seq:
                    self._retain(payload.record)
            elif isinstance(payload, s.StateChanged):
                for section in payload.changed_sections:
                    self._invalidate(section, (payload.revision, event.seq))
            elif isinstance(payload, s.RequestChanged) and s.ACTIONS[payload.request.kind][1] is not s.Effect.R:
                section = s.ProjectionSection.REQUESTS
                self._invalidate(section, (self.view.revision, event.seq))
            self.display_cursor = event.seq
        self._maybe_recover()
        self.render_records()
        self.render_state()
        if not self._recovery_pending and self._dirty and self.clock() - self._last_projection >= .2:
            self.refresh_projection()

    def _invalidate(self, section, stamp):
        if behind(self._applied[section], stamp):
            self._dirty[section] = merge_stamp(self._dirty.get(section, (0, 0)), stamp)

    def _begin_bulk_read(self):
        if self._bulk_reading:
            return False
        now = self.clock()
        self._bulk_credit = min(RECOVERY_RATE,
                                self._bulk_credit + max(0., now - self._bulk_at) * RECOVERY_RATE)
        self._bulk_at = now
        if self._bulk_credit < RECOVERY_RATE - 1e-6:
            self._next_recovery_at = now + (RECOVERY_RATE - self._bulk_credit) / RECOVERY_RATE
            return False
        self._bulk_reading = True
        return True

    def _charge_bulk_read(self, value):
        size = len(s.encode(value))
        self._bulk_credit -= size
        self._bulk_at = self.clock()
        self._next_recovery_at = self._bulk_at + size / RECOVERY_RATE

    def _retain(self, record):
        self._record_bytes -= self._record_sizes.get(record.id, 0)
        self.records[record.id] = record
        self._record_sizes[record.id] = len(s.encode(record))
        self._record_bytes += self._record_sizes[record.id]
        while len(self.records) > 2000 or self._record_bytes > HISTORY_BYTES:
            removed = min(self.records.values(), key=lambda item: item.created_seq)
            del self.records[removed.id]
            self._record_bytes -= self._record_sizes.pop(removed.id)
            self.history_evicted_through = max(self.history_evicted_through or 0, removed.created_seq)
            if removed.bbs_context is not None and removed.bbs_context == self.view.live.bbs_context:
                self._local_bbs_loss_context = removed.bbs_context

    def recover(self, *, initial=False):
        self._recovery_pending = True
        if initial:
            self._next_recovery_at = float('-inf')
        self._maybe_recover()
        self.render_state()

    def _maybe_recover(self):
        now = self.clock()
        if (not self._recovery_pending or self._recovering or now < self._next_recovery_at or
                not self._begin_bulk_read()):
            return
        self._recovering = True
        try:
            snapshot = self.controller.snapshot()
            self.history_reads += 1
            self._charge_bulk_read(snapshot)
            if (snapshot.controller_id != self.controller.controller_id or
                    snapshot.through_seq < self._recovery_gap_through):
                return
            self.records.clear()
            self._record_sizes.clear()
            self._record_bytes = 0
            self._local_bbs_loss_context = None
            self.display_cursor = snapshot.through_seq
            self.view = s.project(snapshot)
            self._applied = {section: (snapshot.revision, snapshot.through_seq)
                             for section in s.ProjectionSection}
            self._dirty.clear()
            self._bbs_fence = None
            evicted = snapshot.presentation.evicted_through
            if evicted is not None:
                self.history_evicted_through = max(self.history_evicted_through or 0, evicted)
            for record in snapshot.presentation.records:
                self._retain(record)
            buffered = tuple(event for event in self._recovery_buffer
                             if event.seq > snapshot.through_seq)
            self._recovery_buffer.clear()
            self._recovery_buffer_bytes = 0
            self._recovery_pending = False
            self._recovery_gap_through = 0
            self.render_records()
            self.render_state()
            if buffered:
                self.post_message(self.Updates(buffered))
        finally:
            self._bulk_reading = False
            self._recovering = False

    def refresh_projection(self):
        if (self._recovery_pending or self.clock() - self._last_projection < .2 or
                not self._begin_bulk_read()):
            return False
        try:
            projection = self.controller.projection()
            self._charge_bulk_read(projection)
            self._last_projection = self.clock()
            self.projection_reads += 1
            return self.apply_projection(projection)
        finally:
            self._bulk_reading = False

    def apply_projection(self, projection):
        if self._recovery_pending or projection.controller_id != self.view.controller_id:
            return False
        stamp = projection.revision, projection.through_seq
        changes = {}
        for section in s.ProjectionSection:
            if behind(stamp, self._applied[section]) or behind(stamp, self._dirty.get(section, (0, 0))):
                continue
            changes[section.value] = getattr(projection, section.value)
            self._applied[section] = stamp
            self._dirty.pop(section, None)
        if not changes:
            return False
        self.view = replace(self.view, **changes, revision=max(self.view.revision, projection.revision),
                            through_seq=max(self.view.through_seq, projection.through_seq),
                            sampled_at=projection.sampled_at, utc=projection.utc,
                            omitted_sections=projection.omitted_sections)
        self.render_state()
        return True

    def render_status(self):
        if not self._visible():
            return
        live = self.view.live
        age = max(0, self.clock() - live.sampled_at)
        channel = '?' if live.identity.channel_hz is None else f'{live.identity.channel_hz / 1000:.3f} kHz'
        text = (f'{live.identity.mycall} · {live.identity.band_m} m · {channel} · '
                f'{live.link.peer or "no peer"} · {live.link.state} · {age:.1f}s old\n'
                f'{live.operation.phase.value}: {live.operation.description}\n'
                f'{live.tx.description}\n{live.operation.next_action}')
        b = live.broadcast
        if b.selected_hz is not None or b.possibly_outstanding:
            text += (f'\nBroadcast: {b.phase or b.result or "idle"}; '
                     f'{b.possibly_outstanding} possibly outstanding; reception unconfirmed')
        self._widget('#status', Static).update(plain(text))

    def render_state(self):
        if not self._visible():
            return
        generation = self.view.live.link.generation
        composer = self._widget(Composer)
        if generation != self._editor_generation:
            if composer.text:
                self._orphan = True
            self._upload_epoch += 1
            self._editor_generation = generation
        draft = self.view.application.draft
        mode = f'{draft.kind} draft {draft.revision}' if draft else 'chat'
        if self._orphan:
            mode = 'Orphan draft: review and start a new composition'
        if self._dependent_stale():
            mode += ' (refresh pending; Send disabled)'
        self._widget('#mode', Static).update(plain(
            f'{mode} · {self.view.live.link.peer or "no peer"} · generation {generation} · '
            f'{"Ctrl+S submits reviewed body" if composer.buffering else "Enter submits one line"}'
            f' · Cancel wait: {self.view.live.operation.next_action}'))
        p = self.view.live
        available = any(r.bbs_context == p.bbs_context for r in self.records.values())
        stale = self._dependent_stale()
        self._widget('#bbs-input', Input).disabled = (
            stale or not self.view.application.bbs.remote_view or not p.bbs_complete or
            p.bbs_reconnect_required or not available or
            p.bbs_context is not None and self._local_bbs_loss_context == p.bbs_context)
        self.render_status()
        self.render_receive()

    def render_records(self):
        if not self._visible():
            return
        records = tuple(sorted(self.records.values(), key=lambda r: r.created_seq))
        if records == self._rendered_records and self.history_evicted_through == self._rendered_history:
            return
        self._rendered_records = records
        self._rendered_history = self.history_evicted_through
        self._widget('#history-notice', Static).display = self.history_evicted_through is not None
        self._widget('#history-notice', Static).update(
            '' if self.history_evicted_through is None else
            f'History unavailable through event {self.history_evicted_through}')
        contact = self._widget('#contact', RichLog)
        autoscroll, position = contact.is_vertical_scroll_end, contact.scroll_y
        old_count = len(contact.lines)
        contact.clear()
        if self.history_evicted_through is not None:
            contact.write(Text(f'History unavailable through event {self.history_evicted_through}'),
                          scroll_end=autoscroll)
        bbs = self._widget('#bbs', RichLog)
        bbs_position, bbs_bottom = bbs.scroll_y, bbs.is_vertical_scroll_end
        bbs.clear()
        passband = self._widget('#passband', RichLog)
        passband.clear()
        for record in records:
            payload = record.payload
            if isinstance(payload, s.BBSPage):
                bbs.write(styled(payload), scroll_end=bbs_bottom)
            elif isinstance(payload, s.SlotRecord):
                passband.write(Text(plain(f'{payload.slot}: {payload.text}')))
            else:
                text = payload.text
                if isinstance(payload, s.TextRecord):
                    text = f'{payload.direction} {payload.peer or ""}: {text}'
                if record.tx is not None:
                    text += f' [{record.tx.status}; not proof of human receipt]'
                contact.write(Text(plain(text)), scroll_end=autoscroll)
        if not autoscroll:
            contact.scroll_to(y=position, animate=False)
            self.unread += max(0, len(contact.lines) - old_count)
            self.notice(f'{self.unread} unread records; scroll to the end to resume')
        else:
            self.unread = 0
        if not bbs_bottom:
            bbs.scroll_to(y=bbs_position, animate=False)

    @on(Input.Changed, '#filter')
    def filter_changed(self):
        self.render_receive()

    def render_receive(self):
        query = self._widget('#filter', Input).value.casefold()
        receive = self.view.receive
        rows = [f'{h.n} {h.call} {h.frequency_hz / 1000:.3f} kHz {h.sn if h.sn is not None else "?"} dB'
                for h in receive.heard]
        rows += [f'Slot {r.slot}: {r.state} {" ".join(r.calls) or "anonymous"} {r.audio_hz} Hz'
                 for r in receive.slots]
        rows += [f'Coverage {c.slot}: {c.last_note}; loss '
                 f'{c.share_60s if c.share_60s is not None else "unknown"}' for c in receive.coverage]
        for ident in ('receive', 'heard-slots'):
            widget = self._widget('#' + ident, RichLog)
            position = widget.scroll_y
            widget.clear()
            for row in rows:
                if query in row.casefold():
                    widget.write(Text(plain(row)))
            widget.scroll_to(y=position, animate=False)

    @on(Input.Changed, '#bbs-input')
    def bbs_edited(self, event):
        if event.value and self._bbs_fence is None:
            p = self.view.live
            self._bbs_fence = (p.bbs_context, p.bbs_revision, p.link.generation)
        elif not event.value:
            self._bbs_fence = None

    @on(Input.Submitted, '#bbs-input')
    def bbs_submit(self, event):
        p = self.view.live
        context, revision, generation = self._bbs_fence or (p.bbs_context, p.bbs_revision, p.link.generation)
        if context is None or event.input.disabled or self._bbs_submitting:
            return
        self._bbs_submitting = True
        action = s.BBSInput(context, revision, event.value)
        self.run_worker(self._bbs_submit(action, generation, event.input), group='bbs-input')

    async def _bbs_submit(self, action, generation, field):
        try:
            if await self.perform(action, generation=generation) and field.value == action.text:
                field.value = ''
                self._bbs_fence = None
        finally:
            self._bbs_submitting = False

    def notice(self, text):
        if not self._visible():
            return
        self._widget('#notice', Static).update(plain(text))

    def _dependent_stale(self):
        return self._recovery_pending or bool(DEPENDENCIES & self._dirty.keys())

    async def _await_current(self, epoch):
        while self._dependent_stale() and self.is_running and not self._exiting_chat:
            if self._upload_epoch != epoch:
                return False
            await asyncio.sleep(.01)
        return self._upload_epoch == epoch and not self._exiting_chat

    def _request_covered(self, state):
        requests_stamp = self._applied[s.ProjectionSection.REQUESTS]
        return state in self.view.requests and all(
            not behind(self._applied[section], requests_stamp) for section in DEPENDENCIES)

    async def perform(self, action, *, generation=None):
        if (s.effect(action) is not s.Effect.R and
                action.kind not in ('chat.cancel_wait', 'chat.close', 'chat.exit') and
                self._dependent_stale()):
            self.notice('State refresh pending; review before submitting')
            return None
        live = self.view.live
        request = s.Request(str(uuid4()), self.view.controller_id,
                            live.link.generation if generation is None else generation,
                            self.actor, live.ownership.control_id, live.ownership.control_generation, action)
        receipt = self.controller.submit(request)
        if not receipt.queued:
            self.notice(receipt.reason or 'Queue refused')
            return None
        self.notice('Request queued')
        epoch = self._upload_epoch
        while self.is_running and not self._exiting_chat:
            outcome = self.controller.outcome(request.id)
            if isinstance(outcome, s.RequestResult) and outcome.state.state in TERMINAL:
                if outcome.state.state == 'done':
                    if s.effect(action) is not s.Effect.R and not self._request_covered(outcome.state):
                        for section in s.ProjectionSection:
                            self._dirty.setdefault(section, (0, 0))
                        if self.clock() - self._last_projection >= .2:
                            self.refresh_projection()
                    if isinstance(action, (s.VMailCompose, s.EmailCompose)):
                        self._orphan = False
                        self._editor_generation = self.view.live.link.generation
                    self.notice('Operation complete' if s.effect(action) is s.Effect.R else 'Action accepted; delivery evidence is separate')
                    if isinstance(action, s.ChatExit):
                        self.action_ordered_exit()
                    return outcome
                self.notice(outcome.state.detail or outcome.state.reason_code or outcome.state.state)
                return None
            if epoch != self._upload_epoch:
                self.notice('View changed; upload stopped. Query the pending outcome before retrying.')
                return None
            await asyncio.sleep(.01)
        return None

    @on(Composer.LimitReached)
    def composer_limit(self):
        self.notice('Draft exceeds 64 KiB; paste was not retained')

    @on(TextArea.Changed, '#composer')
    def composer_changed(self, event):
        if not event.text_area.text:
            self._orphan = False
            self._editor_generation = self.view.live.link.generation
        if self._visible():
            self.render_state()

    @on(Composer.Submitted)
    def line_submitted(self):
        self.action_submit()

    def action_submit(self):
        if not self._submitting:
            self.run_worker(self._submit(), group="composer")

    async def _submit(self):
        if self._submitting or self._exiting_chat:
            return
        if self._dependent_stale():
            self.notice('State refresh pending; editor retained')
            return
        composer = self._widget(Composer)
        text = composer.text
        if len(text.encode('utf-8')) > 64 * 1024:
            self.notice('Draft exceeds 64 KiB')
            return
        if self._orphan:
            self.notice('Link changed. Start a new composition explicitly before sending retained text.')
            return
        self._submitting = True
        epoch = self._upload_epoch
        try:
            if text.startswith('/') and not composer.buffering:
                outcome = await self.perform(parse_action(text))
            else:
                draft = self.view.application.draft
                if draft is not None and draft.owner != self.actor:
                    self.notice('The backend draft belongs to another view')
                    return
                if draft is not None:
                    if composer.buffering:
                        for line in text.split('\n'):
                            if not await self._await_current(epoch):
                                return
                            result = await self.perform(s.DraftAppend(draft.id, draft.revision, line))
                            if result is None or epoch != self._upload_epoch:
                                return
                            if not isinstance(result.result, s.DraftAccepted) or result.result.draft is None:
                                self.notice('Draft acknowledgement missing; review before sending')
                                return
                            draft = result.result.draft
                        if not await self._await_current(epoch):
                            return
                        action = s.VMailSend if draft.kind == 'vmail' else s.EmailSend
                        outcome = await self.perform(action(draft.id, draft.revision))
                    elif text.strip() == '.':
                        action = s.VMailSend if draft.kind == 'vmail' else s.EmailSend
                        outcome = await self.perform(action(draft.id, draft.revision))
                    else:
                        outcome = await self.perform(s.DraftAppend(draft.id, draft.revision, text))
                else:
                    if text.strip():
                        outcome = await self.perform(s.ChatSend(text))
                    elif self.view.live.operation.active:
                        outcome = await self.perform(s.ChatCancelWait())
                    else:
                        outcome = None
            if outcome:
                self.show_read(outcome)
            if outcome and epoch == self._upload_epoch and composer.text == text:
                composer.load_text('')
                composer.buffering = False
        except ValueError as error:
            self.notice(str(error))
        finally:
            self._submitting = False
            self.render_state()

    def action_focus_composer(self):
        self._widget(Composer).focus()

    def show_read(self, outcome):
        if outcome is None:
            return
        value = outcome.result
        if isinstance(value, s.HelpText):
            self.push_screen(Information('Help', value.text))
        elif isinstance(value, s.MailBody):
            self.push_screen(Information(value.header.subject, value.body))

    def action_commands(self):
        self.push_screen(ActionPalette(), self.show_read)

    def action_diagnostics(self):
        lines = [f'{r.kind}: {r.state} {r.detail}' for r in self.view.requests]
        lines.append(f'Controller {self.view.controller_id}; revision {self.view.revision}')
        lines.append(f'Ownership: {self.view.live.ownership.mode} / {self.view.live.ownership.health}')
        self.push_screen(Information('Diagnostics', '\n'.join(lines)))

    def action_mail(self):
        self.push_screen(MailScreen())

    def action_cancel_wait(self):
        self.run_worker(self.perform(s.ChatCancelWait()), group="actions")

    def action_close_contact(self):
        self.run_worker(self.perform(s.ChatClose()), group="actions")

    def action_ordered_exit(self):
        if not self._exiting_chat:
            self._exiting_chat = True
            self.run_worker(self._ordered_exit(), group="shutdown")

    def action_interrupt(self):
        if self.interrupt is not None:
            self.interrupt()
        else:
            self.action_ordered_exit()

    async def _ordered_exit(self):
        self._upload_epoch += 1
        self.notice('Closing; waiting for owner teardown')
        try:
            await self._finish_owner()
        except BaseException as error:
            self.notice(f'Close failed: {error}')
            if self.exit_on_failure:
                self.exit()
                return
            self._exiting_chat = False
            return
        self.exit()

    async def _finish_owner(self):
        completion = self.controller.shutdown()
        while not completion.done():
            await asyncio.sleep(.01)
        while not self.controller.join(0):
            await asyncio.sleep(.01)
        completion.result()

    async def on_unmount(self):
        if self._subscription is not None:
            self._subscription.close()
        await self._finish_owner()

    def on_resize(self, event):
        self.screen_stack[0].set_class(event.size.width < 100, 'narrow')
        self.screen_stack[0].set_class(event.size.width < 40 or event.size.height < 12, 'compact')


def main(argv=None):
    from ..contact.chat_fake import FakeSession, SCENARIOS
    parser = argparse.ArgumentParser(description='Kestrel chat offline prototype; no radio attachment')
    parser.add_argument('--scenario', choices=tuple(SCENARIOS), default='idle-monitoring')
    parser.add_argument('--list', action='store_true', help='list the fake scenarios and exit')
    args = parser.parse_args(argv)
    if args.list:
        print('\n'.join(SCENARIOS))
        return
    began = time.monotonic()
    clock = lambda: time.monotonic() - began
    controller = Controller(FakeSession(clock=clock, scenario=args.scenario), clock=clock)
    ChatApp(controller).run()


if __name__ == '__main__':
    main()
