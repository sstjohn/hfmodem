# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""creance monitor: one live audio source, fanned to every modem's receive-only
decoder, merged into a single confidence-graded "what is on this frequency now".

Live audio is the design center — a rig input or a KiwiSDR streamed in real time
and fanned to the decoders as it arrives; WAV is only a replay source for tests.
Each modem's decode runs in its own interpreter as a subprocess (numpy stays out
of creance), and the merge preserves each detection's confidence grade so a
tentative signal is never dressed up as a confirmed station.
"""

from .activity import (ACTIVE_CONFIRMED, ACTIVE_TENTATIVE, CONFIRMED, QUIET,
                       STREAM_CLOCK, TENTATIVE, WALL_CLOCK, ActivityView,
                       Detection, format_detection, render_summary)
from .aggregate import run
from .driver import (ModemMonitor, MonitorSpec, default_specs,
                     device_source)

__all__ = ["run", "ActivityView", "Detection", "MonitorSpec", "ModemMonitor",
           "default_specs", "device_source", "format_detection",
           "render_summary", "CONFIRMED", "TENTATIVE", "ACTIVE_CONFIRMED",
           "ACTIVE_TENTATIVE", "QUIET", "STREAM_CLOCK", "WALL_CLOCK"]
