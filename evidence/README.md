<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Recorded evidence

This index covers recordings in this directory and the site examples. Shorter
recordings also ship as test fixtures.
The external recording corpus and private station logs are not part of the release.

| Record | Result | Included material |
|---|---|---|
| [VarAC reception, 24 October](varac-receive-1024/README.md) | Passive BW500 receive at L5, L6 and L7; stations unidentified | 125 s FLAC, provenance, CRC-valid frame hashes |
| [KB8AY, 18 September](vara-mail-0918/README.md) | VARA mail accepted; destination receipt reported by the operator | Rig speaker WAV and provenance |
| [VARA send: NS0A](../site/examples/vara-send/index.html) | L9 acknowledged transmit; destination receipt confirmed | Monitor audio, packet replay, session record |
| [VARA receive: N5TW](../site/examples/vara-receive/index.html) | L4 receive; two messages received, one sent | Receiver audio, packet replay, session record |
| [PACTOR-3 send: VE1YZ](../site/examples/pactor3-send/index.html) | SL6 acknowledged transmit; one message sent, two received | Monitor, receiver and witness audio; packet replay |
| [PACTOR-3 receive: WS8EOC](../site/examples/pactor3-receive/index.html) | SL3 receive; one message received | Receiver and witness audio; packet replay |
| [ARDOP send: WW2MI](../site/examples/ardop-send/index.html) | Destination receipt confirmed; subsequent receive failed | Remote witness audio, packet replay, session record |
| [ARDOP receive: W6IDS](../site/examples/ardop-receive/index.html) | Three messages received; normal close | Remote witness audio, packet replay, session record |
| [K5DAT-13 VARA closing frame](../site/recordings/vara-ofdm-k5dat-0930.json) | BW500 L7 decoded from local transmit-monitor audio | 4.4 s FLAC and provenance; not proof of peer reception |

Test fixtures under `packages/hfmodem/hfmodem/tests/` include shorter receive
crops and synthetic inputs. Their adjacent provenance identifies the source.
They are not all completed contacts.

The older high-speed Winlink VARA receive recording and original ARDOP rig
recordings were lost. The VarAC recording above is a separate retained session.
See [implementation status](../docs/STATUS.md) for results supported only by
station logs or operator reports.
