# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Re-export of hfhost.link.

The Link seam lives in hfhost because it is about host dialects, not about
benchmarking, and the orchestration layer needs the same one. This shim keeps
`creance.link` working for everything already written against it.
"""

from hfhost.link import *                                    # noqa: F401,F403
from hfhost.link import (ATTACHED, BUFFER, BUSY, CANCELPENDING,  # noqa: F401
                         CAPS, CONNECTED, DETACHED, DISCONNECTED, ERROR,
                         OTHER, PENDING, PTT, STATS, HostApiLink, Link,
                         LinkEvent, VaraLink, open_link)
