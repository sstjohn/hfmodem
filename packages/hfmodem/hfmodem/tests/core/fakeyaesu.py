# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import errno
import os
import pty
import select
import threading
import time
import tty

# Wire shapes from rigs.py's recorded readbacks; IDs from Yaesu's CAT reference
# manuals (FT-891 p.11, FTDX10 p.14), not a new bench capture.
READBACKS = {
    'ft891': {'ID;': 'ID0650;', 'VS;': 'VS0;', 'FA;': 'FA007100000;',
              'MD0;': 'MD0C;', 'MD1;': 'MD1C;', 'SH0;': 'SH0117;',
              'NA0;': 'NA00;', 'RG0;': 'RG0030;', 'EX0811;': 'EX0811045;'},
    'ftdx10': {'ID;': 'ID0761;', 'VS;': 'VS0;', 'FA;': 'FA007100000;',
               'MD0;': 'MD0C;', 'MD1;': 'MD1C;', 'SH0;': 'SH0021;',
               'RF0;': 'RF06;', 'NA0;': 'NA00;', 'RG0;': 'RG0255;',
               'EX010413;': 'EX010413050;'},
}


class PtySerial:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.rts = self.dtr = True
        self.fd = None

    def open(self):
        self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        tty.setraw(self.fd)

    def fileno(self):
        return self.fd

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class FakeYaesu:
    def __init__(self, model='ft891', *, replies=None, chunk=4096):
        self.master, self.slave = pty.openpty()
        tty.setraw(self.slave)
        self.port = os.ttyname(self.slave)
        self.replies = dict(READBACKS[model])
        self.replies.update({'TX;': 'TX0;', 'RT;': 'RT0;', 'XT;': 'XT0;',
                             'FT;': 'FT0;', 'IS0;': 'IS0+0000;', 'PC;': 'PC005;',
                             'GT0;': 'GT00;', 'PA0;': 'PA00;', 'RA0;': 'RA00;',
                             'NR0;': 'NR00;', 'BC0;': 'BC00;', 'NB0;': 'NB00;',
                             'CO00;': 'CO000;'})
        self.replies.update(replies or {})
        self.chunk = chunk
        self.log = []
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._scheduled = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait_for(self, command, count=1, timeout=1):
        deadline = time.monotonic() + timeout
        with self._condition:
            while self.log.count(command) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f'fake did not receive {command}: {self.log}')
                self._condition.wait(remaining)

    def send(self, reply, delay=0):
        with self._condition:
            self._scheduled.append((time.monotonic() + delay, reply.encode()))

    def _answer(self, command):
        if command in self.replies:
            reply = self.replies[command]
            if callable(reply):
                reply = reply(command)
            if isinstance(reply, tuple):
                delay, reply = reply
            else:
                delay = 0
            if reply:
                self.send(reply, delay)
            return
        if command in ('AI0;', 'TX0;'):
            return
        matches = [query for query in self.replies
                   if command.startswith(query[:-1]) and command != query]
        if matches:
            query = max(matches, key=len)
            self.replies[query] = command
        else:
            self.send('?;')

    def _run(self):
        buf = b''
        while not self._stop.is_set():
            with self._condition:
                due = [r for at, r in self._scheduled if at <= time.monotonic()]
                self._scheduled = [(at, r) for at, r in self._scheduled if at > time.monotonic()]
            for raw in due:
                for i in range(0, len(raw), self.chunk):
                    try:
                        os.write(self.master, raw[i:i + self.chunk])
                    except OSError:
                        return
            try:
                readable, _, _ = select.select([self.master], [], [], 0.002)
                if not readable:
                    continue
                buf += os.read(self.master, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    continue
                return
            while b';' in buf:
                command, buf = buf.split(b';', 1)
                command = (command + b';').decode('ascii')
                with self._condition:
                    self.log.append(command)
                    self._condition.notify_all()
                self._answer(command)

    def close(self):
        self._stop.set()
        self._thread.join(1)
        os.close(self.slave)
        os.close(self.master)
