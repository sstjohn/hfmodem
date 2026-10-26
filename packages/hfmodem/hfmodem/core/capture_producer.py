# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Pure capture delegation preflight for optional independent recorder children."""
from pathlib import Path
from . import capture_identity as identity


def arguments(parser):
    for name in ('id', 'root', 'role'):
        parser.add_argument('--capture-' + name, default=None, help=None)


def prepare(path, *, capture_id=None, capture_root=None, capture_role=None, roles=()):
    """Claim output before any device query or external client import.

    Legacy invocations without delegation retain their existing writer behavior.
    Delegation is an all-or-none explicit tuple; environment variables confer no
    writer authority. The output must belong to the exact registered role root.
    """
    values = (capture_id, capture_root, capture_role)
    if not any(value is not None for value in values):
        return None
    if not all(values):
        raise identity.IdentityError('Capture delegation requires identity, root and role')
    if capture_role not in roles:
        raise identity.IdentityError('Independent recorder role mismatch')
    capture = identity.load(capture_root)
    if capture.record['kind'] != 'arm':
        raise identity.IdentityError('Recorder requires an individual arm')
    root = Path(capture_root).absolute()
    output = Path(path).absolute()
    try:
        name = output.relative_to(root)
    except ValueError as exc:
        raise identity.IdentityError('Recorder output is outside its delegated root') from exc
    if any(candidate.exists() or candidate.is_symlink() for candidate in
           (output, output.with_suffix('.flac'), output.with_suffix('.json'))):
        raise identity.IdentityError('Recorder artifact path is already occupied')
    # Validate the whole artifact path before the persistent role claim.
    identity.relative(name)
    writer = capture.claim(capture_role, capture_id=capture_id, root=root)
    clock = capture_role + ':' + capture.capture_id
    return identity.Recording.reserve(writer, name, clock_id=clock)


def restore_signals(function):
    """Return caller handlers after an offline/CLI producer exits or refuses."""
    import functools
    import signal
    import threading

    @functools.wraps(function)
    def restored(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            return function(*args, **kwargs)
        prior = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
        try:
            return function(*args, **kwargs)
        finally:
            for sig, handler in prior.items():
                signal.signal(sig, handler)
    return restored


class Lifecycle:
    """Producer-close journal; identity remains incomplete until explicit finish."""
    def __init__(self, ownership, *, component='producer'):
        self.ownership, self.component = ownership, component
        self.finished = False
        self.graceful_signal = False
        self.written = 0

    def __enter__(self):
        if self.ownership is not None:
            self.ownership.audio.writer.update_status(state='ongoing',pending=1,component=self.component)
        return self

    def complete(self, *, written=1):
        self.finished, self.written = True, written

    def __exit__(self, kind, error, trace):
        if self.ownership is None:
            return False
        state='finalized' if self.finished and (kind is None or self.graceful_signal) else 'partial'
        try:
            self.ownership.audio.writer.update_status(state=state,written=self.written,partial=0 if state=='finalized' else 1,component=self.component)
        except Exception as trouble:
            if kind is None:
                raise
            # Keep the original producer failure; unchanged ongoing metadata
            # remains incomplete rather than claiming a successful close.
            self.ownership.audio.writer._status['journal_error']=type(trouble).__name__
        return False


def owned_record(function):
    """Wrap a recording entry, retaining graceful signal-close evidence."""
    import functools
    import json
    import signal
    @functools.wraps(function)
    def owned(*args,ownership=None,**kwargs):
        with Lifecycle(ownership) as lifecycle:
            try:
                code=function(*args,**kwargs,**({'ownership':ownership} if ownership is not None else {}))
            except SystemExit as exc:
                if ownership is not None and exc.code in (128+signal.SIGINT,128+signal.SIGTERM,128+signal.SIGHUP):
                    path=ownership.sidecar.artifact.path
                    if path.is_file() and json.loads(path.read_text()).get('capture_state')=='finalized':
                        # The existing recorder's finally already flushed and
                        # finalized its owned file. Signal semantics stay intact.
                        lifecycle.complete(written=0)
                        lifecycle.graceful_signal=True
                raise
            if code==0:
                lifecycle.complete(written=0)
            return code
    return owned
