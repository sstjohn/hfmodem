# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Arms hermetic_guard in every Python a hermetic test starts, whatever else it
dropped; the interpreter's own sitecustomize, which this one shadows, still runs
after it."""
import importlib.util
import os
import sys

import hermetic_guard

hermetic_guard.arm()

here = os.path.dirname(os.path.abspath(__file__))
for entry in sys.path:
    if os.path.abspath(entry or ".") == here:
        continue
    spec = importlib.util.spec_from_file_location("_sitecustomize", os.path.join(entry or ".", "sitecustomize.py"))
    if spec and os.path.isfile(spec.origin):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        break
