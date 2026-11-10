# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the rig is hearing: the tracker, the CW reader, and what they report."""

import re

#: The rate every monitor stream runs at.
FS = 48000

#: Also in tools/rehear/kestrel.py, which a package cannot import.
CALLSIGN = re.compile(r"[A-Z0-9]{0,2}\d[A-Z]{1,3}(-\d+)?")
