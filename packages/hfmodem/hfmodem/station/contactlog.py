# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Compatibility name for the shared contact publisher."""
import sys
from hfmodem.core import contactlog


if __name__ == '__main__':
    raise SystemExit(contactlog.main())
else:
    sys.modules[__name__] = contactlog
