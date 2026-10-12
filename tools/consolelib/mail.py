"""THE MESSAGE AN ARM CARRIES OUT, and the one claim a mailbox can check.

VARA transmit arms default to a delivery test; `--send delivery` selects the
same message explicitly. Every new arm gets its own MID and UTC date, addressed
to the operator's internet mailbox, with the gateway in its subject and body.
Reconnects and DATA retries within one arm reuse the same message. Historical
matrix fixtures remain available as evidence, but are no longer the default.

It is written into the arm's own directory, so the message and the transcript
that moved it are one record, and the body carries session metadata that the
record's own fields regenerate byte for byte: an arriving message is checked
against the arm that sent it rather than believed.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import shlex

from hfmodem.winlink.message import compose, new_mid

#: What `--send` is given to ask for a delivery test rather than a named file.
DELIVERY = 'delivery'

FILENAME = 'delivery.b2f'


def subject(mode_name, gateway, now, sender):
    """Mode, gateway and the minute it was built, short enough to read in a list."""
    return (f'{sender} {mode_name.upper()} test via {gateway} '
            f'{now.strftime("%Y-%m-%d %H%Mz")}')


def body(mode_name, gateway, now, mid, metadata=None):
    """Session details frozen in the record, so writing needs no live lookup."""
    content = (f'{mode_name.upper()} delivery test via {gateway}, '
            f'{now.strftime("%Y-%m-%d %H%Mz")}.\n'
            f'mid {mid}\n')
    if metadata:
        content += (f"Channel frequency: {metadata['centre_hz']} Hz\n"
                    f"Dial frequency: {metadata['dial_hz']} Hz USB\n")
        argv = metadata.get('console_argv')
        content += 'Console command: ' + (shlex.join(['tools/console', *argv])
                                        if argv else 'unavailable') + '\n'
        git = metadata['git']
        content += f"Git revision: {git['head']} (dirty: {git['dirty']})\n"
        content += ('Runtime source SHA-256: '
                    + metadata['source'].get('working_tree_sha256', 'unknown') + '\n')
    return content


def render(mode_name, gateway, now, mid, profile=None, metadata=None, *, sender=None, recipient=None):
    """The message itself, rebuilt entirely from its recorded fields."""
    if sender is None or recipient is None:
        from .operator import delivery_identity
        sender, recipient = delivery_identity()
    title = subject(mode_name, gateway, now, sender)
    content = body(mode_name, gateway, now, mid, metadata)
    if profile:
        level = profile.get('requested_level')
        label = f"BW{profile['bw_hz']}" + (f' L{level}' if level is not None else '')
        title += f' {label}'
        content += f'{label} requested.\n'
    content += '\nhttps://hfmodem.ssj.lol\n'
    # Keep a closing-level test's final block short despite changing the MID,
    # gateway and date. The B2F compression means rendered size is insufficient.
    for padding in range(200):
        message = compose(sender, recipient, title,
                          content + ' ' * (32 * padding),
                          mid=mid, date=now)
        if not profile or profile.get('requested_level') != 3:
            return message
        from hfmodem.winlink.session import B2FSession, _Outbound
        wire = bytearray()
        B2FSession(sender)._send_body(_Outbound(message), 0, wire)
        block = 43 if profile['bw_hz'] == 500 else 89
        if 0 < len(wire) % block <= (34 if block == 43 else 47):
            return message
    raise ValueError('No closing-level message geometry found.')


def delivery_message(mode_name, peer, directory, now, mid=None, profile=None, metadata=None, *,
                     station=None, sender=None):
    """This arm's own delivery test: the whole message, as a record of itself.

    Composed here and written by `write` when the arm flies. A campaign previews
    by planning every candidate on the list, and none of those is an arm.
    """
    from .operator import delivery_identity
    configured_sender, recipient = delivery_identity(station=station)
    sender = configured_sender if sender is None else sender
    message = render(mode_name, peer['call'], now, mid or new_mid(), profile, metadata,
                     sender=sender, recipient=recipient)
    data = message.render()
    return dict(mid=message.mid, sender=sender, recipient=recipient, to=message.get('To'), subject=message.subject,
                gateway=peer['call'], mode=mode_name, profile=profile, metadata=metadata,
                path=str(Path(directory) / FILENAME),
                composed_utc=now.isoformat(), rendered_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def write(delivery):
    """The message on disk, rebuilt from the record that names it.

    The record is what a mailbox checks an arriving message against, so the
    rebuild is proved here rather than promised: a body the record cannot
    reproduce is not evidence about anything.
    """
    data = render(delivery['mode'], delivery['gateway'],
                  datetime.fromisoformat(delivery['composed_utc']),
                  delivery['mid'], delivery.get('profile'), delivery.get('metadata'),
                  sender=delivery.get('sender'), recipient=delivery.get('recipient')).render()
    if hashlib.sha256(data).hexdigest() != delivery['sha256']:
        raise ValueError('The delivery message does not rebuild from its own record.')
    Path(delivery['path']).write_bytes(data)


def outbound(mode_name, peer, named, directory, now, profile=None, metadata=None, station=None,
             sender=None):
    """What this arm sends, and the record of it: (path, delivery) and either may be None.

    `--send delivery` is composed here; anything else is a file the operator
    named, and it has to be on disk before the rig is claimed.
    """
    if not named:
        return None, None
    if named == DELIVERY:
        delivery = delivery_message(mode_name, peer, directory, now, profile=profile,
                                    metadata=metadata, station=station, sender=sender)
        return Path(delivery['path']), delivery
    path = Path(named).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'No message to send at {path}.')
    return path, None
