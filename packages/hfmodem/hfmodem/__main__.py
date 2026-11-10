# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`python -m hfmodem`, for when the console script is not on PATH.

At the radio the venv is usually addressed by path rather than activated, and
`.venv/bin/hfmodem` and `.venv/bin/python -m hfmodem` should not disagree about
whether the station will start.
"""
from hfmodem.cli import main

raise SystemExit(main())
