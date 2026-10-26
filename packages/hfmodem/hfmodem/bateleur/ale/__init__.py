# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""2G ALE word coding, FSK reception, calls, links and monitoring.

``controller`` runs scanning and exchanges; ``modem`` and ``host`` attach it
to the station. ``lp`` handles AL-1/AL-2, ``directory`` labels monitored
traffic, and ``hflink``/``gpr`` supply network conventions and position text.
The interoperability limits are in docs/implementation/bateleur/ALE.md.
"""
