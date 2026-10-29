# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)


class MailAckOwner:
    def __init__(self, session):
        self.session = session
        self.generation = None
        self.state = 'idle'
        self.kind = None
        self.result = None
        self.callback = None
        self.last_sequence = -1

    def linked(self):
        generation = self.session.client.link_generation
        if self.generation != generation:
            self.interrupt('link changed')
            self.generation = generation
            self.state, self.kind, self.result = 'idle', None, None
            self.callback = None

    @property
    def busy(self):
        return self.generation == self.session.client.link_generation and self.state != 'idle'

    def reserve(self, kind, attempt, callback):
        from .varac import HostError
        self.linked()
        if self.busy:
            raise HostError('Another VMail acknowledgement owns this link; resolve it or reconnect')
        c = self.session.client
        if c._messages.pending:
            raise HostError('VMail waits for the incomplete incoming application frame')
        self.kind, self.attempt, self.callback = kind, attempt, callback
        self.peer, self.after = self.session.peer, c.message_sequence
        self.state, self.result = 'active', None
        self.started = self.progress_at = c.clock()
        self.progress = None
        self.drained_at = None

    def release_unsent(self):
        self.state, self.kind, self.result = 'idle', None, None
        self.callback = None

    def interrupt(self, reason):
        if self.state != 'active':
            return
        self.state = 'unresolved'
        if self.callback:
            try:
                self.callback(None, reason)
            except (OSError, ValueError) as error:
                self.session.say(f'VMail uncertainty could not be committed; link remains reserved: {error}')

    def message(self, application, sequence, generation):
        if (not self.busy or generation != self.generation or sequence <= self.after
                or sequence <= self.last_sequence or self.peer != self.session.peer
                or not self.session.client.connected):
            return False
        tags = [p.tag for p in application.parts
                if p.tag in ('SMR', 'SMF', 'SMFP', 'ER', 'EJ')]
        if not tags:
            return False
        if self.kind == 'email' and set(tags) & {'EJ', 'SMF', 'SMFP'}:
            tags = [tag for tag in tags if tag in ('EJ', 'SMF', 'SMFP')]
        self.last_sequence = sequence
        for tag in dict.fromkeys(tags):
            if self.kind != 'email' and tag in ('ER', 'EJ'):
                continue
            try:
                self.callback(tag, None)
            except (OSError, ValueError) as error:
                self.state = 'unresolved'
                self.session.say(f'VMail outcome could not be committed; link remains reserved: {error}')
                return True
            self.result = tag
            self.state = 'email-reserved' if self.kind == 'email' else 'idle'
            if self.kind != 'email':
                break
        return True

    def tick(self):
        s, c = self.session, self.session.client
        if self.state != 'active':
            return
        if (not c.connected or c.closed or c.link_generation != self.generation
                or c._stream_poisoned or s._closing or s._quitting):
            self.interrupt('link or writer ended before acknowledgement')
            return
        if self.kind == 'email':
            return
        now = c.clock()
        progress = (c.buffer, tuple(c._writes))
        if progress != self.progress:
            self.progress, self.progress_at = progress, now
        drained = not s._data_status()[0] and not (c.data_writer and c.data_writer.pending)
        if drained and self.drained_at is None:
            self.drained_at = now
        if (self.drained_at is not None and now - self.drained_at >= 90
                or not drained and now - self.progress_at >= 180):
            self.interrupt('application acknowledgement timed out')
