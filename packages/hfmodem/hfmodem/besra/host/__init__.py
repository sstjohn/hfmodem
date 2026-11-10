# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""besra's host interface — ARDOP's control+data TCP dialect.

The server (`server.py`) speaks ARDOP's documented host interface to a client
(Pat, Winlink Express); everything below it — link establishment, over-the-air
transfer, flow control — is delegated to a `ModemCore` (`modem_core.py`). The
constants and the port-derivation live in `protocol.py`.
"""
