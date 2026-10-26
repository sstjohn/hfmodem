<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Application and modem hosts

Host support in 0.3.0:

| Mode | Host interfaces | Scope |
|---|---|---|
| PACTOR | PTC serial, managed serial | Native PACTOR contacts, terminal and host mode, scanner holds, CW identification and recovery of unconfirmed data |
| VARA | Command/data TCP, managed TCP, optional KISS | Calling and answering, TEXT/FILES/OFF compression, chat, CQ, beacons and UI broadcasts |
| ARDOP | Command/data TCP, managed TCP | Calling and answering; legacy Trimode CONNECT syntax |
| Robust Packet | KISS TCP, Tracker serial, AGWPE TCP, managed Tracker | Raw frames through KISS; connected AX.25 through Tracker and AGWPE |

The station supplies the modem and radio ownership. Standalone development
hosts may use loopback. Check the startup report before attaching an application.
See [configuration](CONFIG.md) and [station operation](README-station.md).

VARA defaults to TEXT; legacy `COMPRESSION ON` selects TEXT. Compression changes
require an idle contact. PTC supports Normal calls and MYLevel 1 or 3; unsupported
Robust/longpath and AES selections are refused. ARDOP2 is not provided.

Windows adapters expose the VARA and ARDOP TCP launch contracts and PTC/Tracker
serial streams. They require a supervised application and an authenticated
managed station host. They open no radio or audio hardware themselves.

Pat mail checks cover PTC, ARDOP and AGWPE over virtual audio. Native Windows
checks cover application startup, attachment and binary component exchanges.
These checks do not establish complete Winlink Express mail compatibility,
stock SCS interoperability or physical-radio timing.

Dated reports under `docs/compatibility/spec/` retain individual results and
limits; older reports may describe a feature before it was completed.
