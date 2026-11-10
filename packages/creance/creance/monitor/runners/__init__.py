# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The monitor's spawned processes: the only code here that needs the modem stack.

Each per-modem runner is executed by the modem's *own* interpreter
(creance/monitor/driver launches it), reads live s16le audio on stdin, and writes
normalised detection JSON on stdout. Its ``normalize`` function is pure —
importable and testable in any interpreter, no numpy — while ``main`` does the
audio loop against the real modem decode. When a modem's monitor interface
shifts, this one file is the edit.

``capture_runner`` runs the other way round: it opens the sound card through
PortAudio and writes the audio the rest of them read. It is here rather than in
creance because the library and the device table are the modems', and because
opening a card from inside creance is what cost every live pass an eighth of its
audio.
"""
