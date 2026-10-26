# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""BBS downloads and explicit file-offer/opaque evidence capture."""
import json

from .filetransfer import RawTransferCapture, CaptureStopped, parse_offer


class BBSClient:
    def __init__(self, session):
        self.session = session
        self.entries = ()
        self.list_generation = None
        self.offer = None
        self.offer_path = None
        self.generation = None
        self.dial = None
        self.requested = None
        self.recorder = None
        self.waiting = False
        self.opaque = False
        self.retired_opaque = False
        self.failure = None
        self.offer_deadline = None
        self.downloading = False
        self.receiving = False
        self.received_bytes = 0
        self._progress_at = self._reported_at = None
        self._reported_bytes = None

    @property
    def active(self):
        return self.waiting or self.opaque or self.receiving

    def listed(self, entries):
        self.entries = tuple(entries)
        self.list_generation = self.session.client.link_generation
        self.offer = None

    def guard(self):
        from .varac import HostError, State, CALLING_HZ
        s, c = self.session, self.session.client
        if (c.closed or not c.connected or s._closing or s.flow.state is not State.CONNECTED):
            raise HostError('BBS file evidence requires an established link')
        if s.flow.dial in CALLING_HZ.values():
            raise HostError('BBS file evidence requires QSY off the calling frequency')
        if (s.bbs_host.active or s.steps is not None or s._opening or s.email.waiting or s._vmail_sent
                or c.pending or c.ptt or s._data_status()[0]):
            raise HostError('BBS file evidence requires other exchanges and transmission to settle')

    def _new_capture(self, seconds):
        s = self.session
        return RawTransferCapture(s.mailbox.root / 'file-captures', peer=s.peer,
                                  filename=self.requested, link_generation=self.generation,
                                  clock=s.client.clock, time_limit=seconds)

    def command(self, mode, selection=None):
        try:
            if mode in ('offer', 'request', 'get'):
                self.request_offer(selection)
                self.downloading = mode == 'get'
            else:
                self.capture()
        except OSError as error:
            self.session._error(error)
            self.stop(f'file evidence I/O failed: {error}')

    def request_offer(self, selection):
        from .varac import HostError
        self.guard()
        s, c = self.session, self.session.client
        if self.active:
            raise HostError('A BBS file evidence operation is already active')
        if self.list_generation != c.link_generation or not self.entries:
            raise HostError('Run /bbs list on this link before selecting a file')
        if selection.isdecimal():
            n = int(selection)
            if not 1 <= n <= len(self.entries):
                raise ValueError('BBS file number is outside the current directory')
            name = self.entries[n - 1].name
        else:
            if selection not in {entry.name for entry in self.entries}:
                raise ValueError('Filename is not in this link\'s BBS directory')
            name = selection
        # Use the offer parser's conservative flat-name policy before emitting BG.
        parse_offer(f'<SF:{name}|0|{"0" * 32}>')
        self.requested, self.generation, self.dial = name, c.link_generation, s.flow.dial
        self.offer = None
        self.failure = None
        self.recorder = self._new_capture(60)
        self.waiting = True
        self.offer_deadline = c.clock() + 60
        try:
            c.send_text(f'<BG:{name}>')
        except BaseException:
            self.finish('request failed')
            raise
        s.say(f'File offer requested for {name}; no readiness sent. /stop ends this contact.')

    def message(self, text):
        c = self.session.client
        if (self.waiting or self.receiving) and (
                not c.connected or c.link_generation != self.generation
                or self.session._closing):
            self.end('link ended or changed')
            return True
        if self.receiving:
            if text in ('<SFAB>', '<SFFA>', '<BGJ>'):
                self.failure = 'Peer refused or cancelled the file transfer'
                return True
            if not text.startswith('<SFB>'):
                return False
            from .bbs_download import decode_file, save_file
            try:
                data = decode_file(self.offer, text)
                path = save_file(self.session.mailbox.root / 'downloads', self.requested, data)
                self.receiving = self.downloading = False
                self.offer = None
                self.finish('file validated and saved')
                self.session.say(f'BBS file saved: {path} ({len(data)} bytes; checksum verified)')
                self.session.emit('bbs_download', peer=self.session.peer, path=str(path), size=len(data))
                self.session.client.send_text('<SFOK>')
            except (ValueError, OSError) as error:
                self.failure = f'File download failed: {error}'
            return True
        if not self.waiting:
            return
        if text == '<BGJ>':
            self.downloading = False
            self.finish('offer rejected')
            self.session.say('Peer rejected the file request; no readiness sent')
        elif text.startswith('<SF:'):
            try:
                offer = parse_offer(text)
                if offer.wire_name not in (self.requested, self.requested + '.VarAC_gzip'):
                    raise ValueError('Offer filename does not match the selected file')
            except ValueError as error:
                self.failure = str(error)
                return
            self.offer = offer
            self.offer_path = self.recorder.directory
            self.finish('offer recorded')
            if self.downloading:
                from .bbs_download import validate_offer, DownloadCapture, DOWNLOAD_SECONDS
                try:
                    validate_offer(offer)
                    self.recorder = DownloadCapture(
                        self.session.mailbox.root / 'file-captures', peer=self.session.peer,
                        filename=self.requested, link_generation=self.generation,
                        clock=c.clock, time_limit=DOWNLOAD_SECONDS)
                    self.receiving = True
                    self.received_bytes = 0
                    self._progress_at = c.clock()
                    self._reported_at = self._reported_bytes = None
                    self.session.client.send_text('<SFRD>')
                    self.session.say('Receiving BBS file; validation and saving are automatic.')
                    self.report_progress()
                except (ValueError, OSError) as error:
                    self.failure = str(error)
                return
            self.session.say('Offer recorded. /bbs capture explicitly sends readiness and records '
                             'opaque bytes; this is not a completed download.')

    def capture(self):
        from .varac import HostError
        self.guard()
        c = self.session.client
        if self.active or self.offer is None or self.generation != c.link_generation:
            raise HostError('Capture requires a validated offer from this current link')
        if self.dial != self.session.flow.dial:
            raise HostError('Frequency changed since the file offer; request it again')
        self.recorder = self._new_capture(120)
        try:
            (self.recorder.directory / 'offer.json').write_text(json.dumps({
                'raw_offer': self.offer.raw, 'offer_capture': str(self.offer_path),
                'link_generation': self.generation, 'peer': self.session.peer,
                'dial_hz': self.dial}, ensure_ascii=True) + '\n')
        except OSError:
            self.finish('offer provenance write failed')
            raise
        self.opaque = True  # Before any SFRD response can enter the sole data pump.
        self.offer = None  # One use, never reused after cancellation/rejection.
        try:
            c.send_text('<SFRD>')
        except BaseException:
            self.failure = 'readiness send failed'
            raise
        self.session.say('Opaque file evidence capture active (64 KiB / 120 seconds). '
                         '/stop ends capture and this contact; no file success is claimed.')

    def raw(self, direction, data):
        c = self.session.client
        if self.retired_opaque:
            if not (c.connected or c.pending) or self.generation == c.link_generation:
                return True
            self.retired_opaque = False
        if self.generation != c.link_generation:
            was_opaque = self.opaque
            self.finish('link changed')
            self.opaque = False
            self.offer = None
            self.retired_opaque = was_opaque and not (c.connected or c.pending)
            return self.retired_opaque
        if self.recorder is not None:
            try:
                self.recorder.append(direction, data)
            except (CaptureStopped, OSError) as error:
                self.failure = str(error)
                self.opaque = True  # Never feed capped/failed evidence to the text parser.
        return self.opaque

    def finish(self, reason):
        recorder, self.recorder = self.recorder, None
        self.waiting = False
        if recorder is not None:
            try:
                result = recorder.finish(reason)
                self.session.say(f'File evidence retained: {result.directory} ({result.reason}); '
                                 f'{reason if reason == "file validated and saved" else "download not validated"}')
            except OSError as error:
                self.session.say(f'File evidence write failed: {error}; partial bytes in {recorder.directory}')

    def end(self, reason='link ended'):
        self.finish(reason)
        self.receiving = self.downloading = False
        self.offer = None
        self.retired_opaque = self.retired_opaque or self.opaque
        self.opaque = False
        self.failure = None

    def close_steps(self, reason):
        from .varac import State, HostError
        s = self.session
        self.finish(reason)
        self.offer = None
        s._closing = True
        s.flow._state(State.FAILED)
        # Remain opaque while bounded close consumes late data; never resync it.
        yield from s.flow.close_steps()
        if s.line_state() is True:
            raise HostError('File capture ended but keying line remains HIGH')
        self.end(reason)

    def stop(self, reason='operator stopped'):
        s = self.session
        if s.steps is not None:
            s.steps.close()
            s.steps = None
        self.opaque = True
        s._run(self.close_steps(reason))

    def tick(self):
        c = self.session.client
        if self.generation != c.link_generation or not c.connected or c.closed:
            self.end('link ended or changed')
            return
        if self.recorder is not None:
            try:
                self.recorder.check_deadline()
            except (CaptureStopped, OSError) as error:
                self.failure = str(error)
        if self.failure and not self.session._closing:
            reason, self.failure = self.failure, None
            self.stop(reason)
        elif self.receiving and not self.session._closing:
            self.report_progress()

    def progress_text(self):
        """Count complete base64 groups in the application parser's file body.

        Modem retries, TX bytes, message headers and other messages are not file
        progress. These bytes are received, not yet checksum-verified or saved.
        """
        if not self.receiving or self.offer is None:
            return None
        c = self.session.client
        total = self.offer.declared_integer
        header, sep, content = c._messages.pending.partition(b' ')
        expected = 5 + 4 * ((total + 2) // 3)
        if (sep and header.isdigit() and int(header) == expected
                and content.startswith(b'<SFB>')):
            received = min(total, max(0, len(content) - 5) // 4 * 3)
            if received > self.received_bytes:
                self.received_bytes = received
                self._progress_at = c.clock()
        quiet = max(0, c.clock() - self._progress_at)
        state = (f'no new data for {quiet:.0f}s' if quiet >= 15 else
                 'receiving' if self.received_bytes else 'waiting for file data')
        return (f'File: {self.requested} — {100 * self.received_bytes // total}% '
                f'({self.received_bytes:,} / {total:,} compressed bytes); {state}')

    def report_progress(self):
        text = self.progress_text()
        if text is None:
            return
        now = self.session.client.clock()
        if self._reported_at is not None:
            interval = 5 if self.received_bytes != self._reported_bytes else 15
            if now - self._reported_at < interval:
                return
        self._reported_at, self._reported_bytes = now, self.received_bytes
        self.session.say(text)
        self.session.emit('bbs_progress', peer=self.session.peer, filename=self.requested,
                          received_bytes=self.received_bytes,
                          total_bytes=self.offer.declared_integer,
                          quiet_seconds=max(0, now - self._progress_at))
