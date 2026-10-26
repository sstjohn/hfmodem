# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Outbound RF email client. Public ON2AD V16.1.8 manual p113 wire example."""
from dataclasses import dataclass
import re

ADDRESS = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,63}")


def address(value):
    if len(value) > 254 or not ADDRESS.fullmatch(value):
        raise ValueError('use one plain ASCII email address; no display name or recipient list')
    return value


@dataclass(frozen=True)
class Email:
    recipient: str
    reply_to: str
    subject: str
    body: str = ''

    def wire(self):
        address(self.recipient)
        address(self.reply_to)
        if not self.subject or any(c in self.subject for c in '<>\r\n'):
            raise ValueError('email needs a one-line subject without < or >')
        if any(c in self.body for c in '<>') or any(ord(c) < 32 and c not in '\r\n\t' for c in self.body):
            raise ValueError('email body cannot contain protocol brackets or control characters')
        if any(ord(c) < 32 for c in self.subject):
            raise ValueError('email subject cannot contain control characters')
        text = f'<SM><TO:{self.recipient}><SBJ:{self.subject}><MSG:{self.body}><EG><RE:{self.reply_to}>'
        if len(text.encode('utf-8')) > 500:
            raise ValueError('email envelope exceeds the 500-byte client limit')
        return text


class EmailClient:
    """One explicit send, tied to the link on which composition began; no auto retry."""
    def __init__(self, session):
        self.session = session
        self.draft = None
        self.waiting = False
        self.phase = None
        self.sent_generation = None

    def _guard(self, generation=None):
        from .varac import CALLING_HZ, HostError, State, TextParser
        s, c = self.session, self.session.client
        if c.closed or not c.connected or s._closing or s.flow.state is not State.CONNECTED:
            raise HostError('email needs an established link')
        if generation is not None and c.link_generation != generation:
            raise HostError('email link changed; no retry sent')
        if s.flow.dial in CALLING_HZ.values():
            raise HostError('email requires QSY off the calling frequency first')
        ready = any(part.raw in ('<E>', '<EA>') or
                    (re.fullmatch(r'<S:[VEAB]+>', part.raw) and 'E' in part.raw[3:-1])
                    for seq, text in c.messages if seq > c.link_message_start and not text.startswith('<SM>')
                    for part in TextParser().feed(text, final=True) if part.tag)
        if not ready:
            raise HostError('this link has not advertised email gateway service (<E> or <S:...E...>)')

    def command(self, words):
        s = self.session
        if words[:1] == ['send'] and len(words) >= 4:
            self._guard()
            if self.sent_generation == s.client.link_generation:
                raise ValueError('one email per link; reconnect before another send to avoid ambiguous late confirmations')
            if s.steps is not None:
                raise ValueError('wait for the current exchange before composing email')
            email = Email(words[1], words[2], ' '.join(words[3:]))
            email.wire()
            self.phase = None
            self.draft = (s.client.link_generation, email, [])
            s.say(f'Email via {s.peer} to {email.recipient}; replies to {email.reply_to}. '
                  'Type body; . sends, /stop drops draft. No message sent yet.')
        elif words == ['status']:
            s.say('email: ' + (self.phase or ('composing' if self.draft else 'idle')))
        elif words[:1] in (['get'], ['receive']):
            s.say('Email reply retrieval is not implemented; use an external reply email address')
        else:
            s.say('usage: /email send TO_EMAIL REPLY_EMAIL SUBJECT; /email status; external replies only')

    def line(self, line):
        generation, email, lines = self.draft
        if line.strip() != '.':
            candidate = Email(email.recipient, email.reply_to, email.subject, '\r\n'.join(lines + [line]))
            candidate.wire()  # Bound the draft before retaining another line.
            lines.append(line)
            return
        self._guard(generation)
        if self.session.steps is not None:
            raise ValueError('another exchange is active; email draft retained')
        email = Email(email.recipient, email.reply_to, email.subject, '\r\n'.join(lines))
        email.wire()
        self.draft = None
        self.session._run(self.send_steps(email, generation))

    def send_steps(self, email, generation, timeout=120):
        from .varac import HostError, TextParser
        s, c = self.session, self.session.client
        self._guard(generation)
        after, end = c.message_sequence, c.clock() + timeout
        self.waiting = True
        self.sent_generation = generation
        self.phase = 'queued; gateway outcome unknown'
        accepted = delivered = False
        try:
            c.send_vmail(email)
            s.emit('email', event='queued', peer=s.peer, recipient=email.recipient,
                   reply_to=email.reply_to, subject=email.subject)
            s.say('Email queued; awaiting modem delivery and gateway confirmation')
            sent_after = c._sent_after
            while True:
                self._guard(generation)
                tags = {part.raw for seq, text in c.messages if seq > after and not text.startswith('<SM>')
                        for part in TextParser().feed(text, final=True) if part.tag}
                if tags & {'<EJ>', '<SMF>', '<SMFP>'}:
                    raise HostError('gateway rejected the email/VMail; no automatic retry')
                if '<SMR>' in tags and not accepted:
                    accepted = True
                    self.phase = 'gateway accepted VMail; awaiting email-send confirmation'
                    s.say('Gateway accepted the VMail; email send is not yet confirmed')
                    s.emit('email', event='vmail_accepted', peer=s.peer)
                if not delivered and 'BUFFER 0' in c.since(sent_after) and c.buffer == 0:
                    delivered = True
                    s.say('Modem acknowledged email payload; this does not confirm email delivery')
                    s.emit('email', event='modem_acknowledged', peer=s.peer)
                if '<ER>' in tags:
                    self.phase = 'gateway reported email sent; recipient inbox delivery not verified'
                    s.say(self.phase)
                    s.emit('email', event='gateway_reported_sent', peer=s.peer, recipient=email.recipient)
                    return
                if c.clock() >= end:
                    raise TimeoutError('email confirmation timed out; outcome unknown, do not blindly resend')
                yield .1
        except GeneratorExit:
            self.phase = 'wait cancelled; queued email may still be sent'
            s.say(self.phase)
            s.emit('email', event='wait_cancelled', peer=s.peer)
            raise
        except (HostError, TimeoutError):
            self.phase = 'failed or unconfirmed; inspect session evidence before retrying'
            s.emit('email', event='unconfirmed', peer=s.peer)
            raise
        finally:
            self.waiting = False
