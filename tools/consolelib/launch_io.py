# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The launcher's output contract, including Python children and saved output."""

import os
import sys

from hfmodem.core.pipeio import Reader as PipeReader


ENCODING = 'utf-8'
ERRORS = 'replace'


def initialize():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding=ENCODING, errors=ERRORS)
    # Set these even when the caller selected a different Python encoding.
    os.environ.update(PYTHONUTF8='1', PYTHONIOENCODING=f'{ENCODING}:{ERRORS}')


class Reader(PipeReader):
    def __init__(self, stream=None, **options):
        super().__init__(stream, encoding=ENCODING, errors=ERRORS, **options)
