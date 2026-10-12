# Changelog

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
<!-- site:omit:start -->
- Expanded PACTOR-4 decoding and offline validation.
<!-- site:omit:end -->
- Added Yaesu FTDX10 rig and audio profiles while retaining FT-891 support.
  Improved station controls, audio handling, monitoring and diagnostics.
- Released the station console and operator tools, with distinct receive-only
  monitors, ALE/HFLINK scanning and sounding, native protocol text commands,
  portable station/data configuration, and a logbook with audio replay.
- Added an installed, guided Winlink console with current gateway listings,
  native text-message sending and an offline received-mail reader.
- Added successful PACTOR-3, VARA and ARDOP replay examples to the project site.
  Improved decoded-byte displays, audio alignment and sender-coloured waterfalls.
<!-- site:omit:start -->
- Replaced expanded PACTOR-4 permutation tables with compact reconstruction data.
<!-- site:omit:end -->
- Updated documentation and protocol guides, expanded regression coverage,
  and made numerous reliability improvements and bug fixes.

The project remains an experimental prerelease for developers only.

## 0.1.0 — 2026-09-20

Initial public release of the PACTOR, VARA, ARDOP and experimental sabir
implementations, shared station tools, host clients and conformance bench.
