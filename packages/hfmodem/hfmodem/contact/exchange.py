# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Contact decisions are pure; advancing requires a completed approved over."""
from dataclasses import dataclass, replace
from .frame import Over


@dataclass
class Contact:
    own_call: str
    state: str = 'CQ'
    partner: str = ''
    heard: Over | None = None
    report_sent: str = ''
    exchanged: bool = False
    edit_revision: int = 0

    def receive(self, over):
        if self.heard and self.partner and (not over.de or over.de == self.partner):
            over = replace(over, **{key: getattr(over, key) or getattr(self.heard, key)
                                    for key in ('report', 'name', 'qth')})
        self.heard = over
        if not self.partner:
            self.state = 'CONFIRM' if over.de else 'LISTEN'
        elif over.de and over.de != self.partner:
            self.state = 'CONFIRM'
        elif over.end == 'clear':
            self.state = 'CLOSE'
        elif over.free:
            self.state = 'FREE'
        else:
            self.state = 'CLOSE' if self.exchanged else 'EXCHANGE'
        return self.state

    def confirm(self, call):
        if self.partner and self.partner != call:
            self.report_sent, self.exchanged = '', False
        self.partner = call
        self.state = 'EXCHANGE'
        self.edit_revision += 1

    def sent(self, kind):
        if kind == 'cq':
            self.state = 'LISTEN'
        elif kind == 'exchange':
            self.exchanged = True
            self.state = 'LISTEN'
        elif kind == 'close':
            self.state = 'LOGGED'
        else:
            self.state = 'LISTEN'

    def reset(self):
        self.partner, self.heard, self.report_sent = '', None, ''
        self.exchanged, self.state = False, 'CQ'
        self.edit_revision += 1
