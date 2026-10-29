# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Local operator tools for BBS messages; never a caller privilege check."""
import argparse
from pathlib import Path
import sqlite3
import sys
import uuid

from .config import load
from .presentation import plain
from .store import Store


def _line(value):
    return plain(str(value)).replace('\n', ' ')


def _body(path):
    with Path(path).open('rb') as source:
        data = source.read(2049)
    if len(data) > 2048:
        raise ValueError('reply body exceeds 2048 UTF-8 bytes')
    return data.decode('utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='hfmodem bbs messages', description=__doc__)
    parser.add_argument('--config', required=True, help='BBS configuration; stop its service before administration')
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('areas', help='list configured public areas')
    listing = sub.add_parser('list', help='list at most ten headers; * means unread by the operator')
    listing.add_argument('--area', default='sysop', help='sysop inbox or public area slug')
    listing.add_argument('--after', type=int, default=0, help='continue after this stable message ID')
    for action in ('read', 'hide', 'unhide'):
        command = sub.add_parser(action)
        command.add_argument('id', type=int)
    reply = sub.add_parser('reply', help='store a reply in a caller\'s sysop thread')
    reply.add_argument('id', type=int)
    reply.add_argument('--body-file', required=True)
    reply.add_argument('--subject')
    reply.add_argument('--request-key', help='reuse the printed key to retry an uncertain submission')
    args = parser.parse_args(argv)
    try:
        config = load(args.config)
        if args.operation == 'areas':
            for key, title in config.message_areas.items():
                print(f'{key}: {_line(title)}')
            return 0
        from .messages import OperatorMessages
        with Store(config.state_root) as store:
            messages = OperatorMessages(store, config.message_areas)
            if args.operation == 'list':
                result = (messages.list_sysop(after=args.after) if args.area == 'sysop'
                          else messages.list_public(args.area, after=args.after))
                for message in result.messages:
                    parent = f' (re {message.parent_id})' if message.parent_id is not None else ''
                    print(f'{" " if message.read else "*"} {message.id} '
                          f'{_line(message.author)}: {_line(message.subject)}{parent}')
                if not result.messages:
                    print('No messages.')
                if result.next_cursor is not None:
                    print(f'Continue: list --area {_line(args.area)} --after {result.next_cursor}')
            elif args.operation == 'read':
                message = messages.get(args.id)
                hidden = ' [hidden]' if message.hidden else ''
                print(f'{message.id} {_line(message.author)}: {_line(message.subject)}{hidden}')
                print(plain(message.body))
                sys.stdout.flush()
                messages.mark_read(message.id)
            elif args.operation == 'reply':
                body = _body(args.body_file)
                key = args.request_key or uuid.uuid4().hex
                # The operator can retry this same request after uncertain I/O.
                print(f'Request key: {_line(key)}', flush=True)
                message = messages.reply_sysop(args.id, body, key, subject=args.subject)
                print(f'Reply stored: {message.id}')
            else:
                getattr(messages, args.operation)(args.id)
                print(f'Message {args.id}: {args.operation}.')
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        print(f'BBS messages: {_line(error)}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
