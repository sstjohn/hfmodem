# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Exclusive external VARA attachment lifetime on the shared station owner."""
import time


def serve_pair(runtime, command, data, session_factory, *, publish=None, retirement_timeout=5):
    """Reserve before modem creation; release only after both owners retire.

    A stalled teardown retains the reservation until a later reservation's
    nonblocking retirement check confirms both owners have stopped.
    """
    owner = object()
    runtime.attachments.reserve(owner)
    try:
        modem = runtime.modem_factory()
    except BaseException:
        runtime.attachments.release(owner)
        raise
    session = None
    retired = False
    try:
        session = session_factory(command, data, modem)
        if publish is not None:
            publish(session)
        session.run()
    finally:
        try:
            deadline = time.monotonic() + retirement_timeout
            retirement_error = None
            if session is not None:
                try:
                    session.close()
                    session.ensure_retired(deadline=deadline)
                except BaseException as error:
                    retirement_error = error
            modem.stop()
            while not modem._retired_attachment:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RuntimeError('VARA audio owner did not acknowledge attachment retirement')
                time.sleep(min(.01, left))
            if retirement_error is not None:
                raise retirement_error
            retired = True
        finally:
            if publish is not None:
                publish(None)
            if retired:
                runtime.attachments.release(owner)
            else:
                def check_retired():
                    if not modem._retired_attachment:
                        return False
                    try:
                        if session is not None:
                            session.ensure_retired(deadline=time.monotonic())
                    except Exception:
                        return False
                    return True
                runtime.attachments.await_retirement(owner, check_retired)
