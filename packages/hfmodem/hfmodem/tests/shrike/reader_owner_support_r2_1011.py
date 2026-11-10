# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""R1 full input ledger; delegate all genuine input and decoder calls unchanged."""
import hashlib
import os
from pathlib import Path
from hfmodem.tests.shrike import second_delta_support_1011 as previous

INPUTS = {}
ORIGINALS = []
CAPS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def digest(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def remember(path):
    from hfmodem.core.wav import stored
    p = Path(path)
    if p.suffix in ('.wav', '.flac'):
        p = stored(p)
    p = p.resolve()
    if p.is_file():
        for item in (p, p.with_suffix('.json')):
            if item.is_file() and str(item) not in INPUTS:
                value = digest(item)
                INPUTS[str(item)] = value
                previous.record('actual_input_before', path=str(item), sha256=value)


def pytest_sessionstart(session):
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 19
    assert all(os.environ.get(k) == '1' for k in CAPS)
    previous.pytest_sessionstart(session)
    from hfmodem.shrike import onair
    from hfmodem.tests import evidence
    for obj, name in ((onair.session, 'load_wav'), (evidence, 'require_input'), (evidence, 'checked_record')):
        original = getattr(obj, name)
        def audited(path, *args, _original=original, **kw):
            remember(path)
            if args and isinstance(args[0], Path):
                remember(args[0])
            return _original(path, *args, **kw)
        ORIGINALS.append((obj, name, original))
        setattr(obj, name, audited)


def pytest_sessionfinish(session, exitstatus):
    for path, before in INPUTS.items():
        after = digest(Path(path))
        previous.record('actual_input_after', path=path, sha256=after)
        assert after == before, path
    for obj, name, original in ORIGINALS:
        setattr(obj, name, original)
    previous.pytest_sessionfinish(session, exitstatus)
