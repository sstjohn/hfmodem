<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Changelog

## 0.3.2 — 2026-10-26

- Added an opt-in Xara validation source kit with explicit peer selection and
  recorded source/build provenance; normal builds and tests do not fetch Xara.
- Refreshed the site’s PACTOR and VARA evidence, added a VarAC BBS replay, and
  improved replay spectrum views and navigation.
- Updated release-test fixtures for current VARA handshakes, portable terminal
  editing, and deterministic station timing.

## 0.3.1 — 2026-10-26

- Fixed near-capacity payload accounting when closing a Kestrel contact.
- Recognized solicited long grants while closing a VARA contact.
- Fixed role-correct continuation for generated responder DATA packets.
- Corrected wideband confirmation and empty-turn sequencing.

## 0.3.0 — 2026-10-25

Changes since 0.2.1:

- Added a recorded-evidence index and VarAC BW500 receive audio at L5, L6 and L7.

- Added Kestrel chat with VarAC peers at 500, 2300 and 2750 Hz, including
  CQ, beacons, slot changes, away replies and VMail.
- Added passive QSO transcripts, slot and frequency following, all-stations
  broadcasts and optional PSKReporter reception reports.
- Added email-gateway submission, BBS directory requests, verified BBS downloads
  and serving an explicitly selected directory. Download validation is covered
  by recorded replay and software round trips; live RF interoperability of the
  automatic receiver and sender remains unverified. Raw capture remains separate.
- Improved VARA weak-signal reception, signal reports, connection recovery,
  turn changes and closing.
- Added experimental supervised voice and CW contacts.
- Added connected Robust Packet hosts for Tracker and AGWPE clients.
- Expanded PTC, VARA and ARDOP host compatibility, including VARA TEXT
  compression and legacy `COMPRESSION ON`.
- Added supervised Windows TCP and serial adapters for station hosts.
- Improved PACTOR session timing and delivery accounting.
- Added a public PACTOR-4 protocol guide and exact C2 PRNG description.
- Improved partial experimental PACTOR-4 reception and turn handling.
  On-air PACTOR-4 mail transfer remains unproven.
- Added station-service ownership controls and native Yaesu CAT support.
  Transmit operation requires a separate PTT line; `--ptt-type` was removed.
- Added console attach, own and release commands, station session records,
  logbook export and explicit inbox delivery checks.
- Improved interruption, shutdown and host reconnection handling.
- Added VARA transmit calibration and transmit-level monitoring; preserved
  FTDX10 filter settings across frequency changes.
- Renamed VARA's `--speed-limited` option to `--winlink-cartel-mode`.

## 0.2.1 — 2026-10-12

- Fixed release-test portability across FFT implementations.
- Fixed responder shutdown during session-worker startup.
- Fixed release checks for shared CI runners while retaining separate receiver
  performance qualification with the existing timing requirements.

Python 3.13 or later is required. Modem behavior is unchanged from 0.2.0.

## 0.2.0 — 2026-10-11

Changes since 0.1.0:

- Improved PACTOR reception, rate transitions and session handling, with
  PACTOR-3 Winlink mail demonstrated in both directions through WS8EOC.
- Expanded VARA high-speed operation and improved rate selection, retransmission
  and session recovery across the supported bandwidths.
- Added experimental Robust Packet support with AX.25 framing and a station
  KISS interface, validated with recordings and software loopback.
- Added experimental MIL-STD and STANAG waveforms, the STANAG 5066 link layer,
  2G ALE and HF selcall, with station interfaces and monitoring tools.
- Added lanner for Baudot RTTY, SITOR mode B and NAVTEX encoding and decoding.
- Expanded PACTOR-4 decoding and offline validation.
- Added Yaesu FTDX10 rig and audio profiles while retaining FT-891 support.
  Improved station controls, audio handling, monitoring and diagnostics.
- Released the station console and operator tools, with distinct receive-only
  monitors, ALE/HFLINK scanning and sounding, native protocol text commands,
  portable station/data configuration, and a logbook with audio replay.
- Added an installed, guided Winlink console with current gateway listings,
  native text-message sending and an offline received-mail reader.
- Added successful PACTOR-3, VARA and ARDOP replay examples to the project site.
  Improved decoded-byte displays, audio alignment and sender-coloured waterfalls.
- Replaced expanded PACTOR-4 permutation tables with compact reconstruction data.
- Updated documentation and protocol guides, expanded regression coverage,
  and made numerous reliability improvements and bug fixes.

The project remains an experimental prerelease for developers only.

## 0.1.0 — 2026-09-20

Initial public release of the PACTOR, VARA, ARDOP and experimental sabir
implementations, shared station tools, host clients and conformance bench.
