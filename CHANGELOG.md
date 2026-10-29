<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Changelog

## 0.4.0 — 2026-10-29

Includes the changes tagged as 0.3.1–0.3.4, which were not released on PyPI.

- VarAC VMail for a station that is not on the air is now parked for up to
  30 days. The station is told on its next beacon, and the mail is forwarded
  when it connects.
- VarAC `/heard`, `/slots`, `/vmail`, `/auto`, `/away` and other local commands
  work while a link is closing or busy and keep a draft. An away station sends
  `<AWQ>` so callers can QSY, and `/bbs list` waits for a pending QSY.
- Auto-answer now stays on after `/bye` or `/stop` until `/auto off`; `/status`
  and the monitoring line say when it is off. VarAC farewell and shutdown
  waits are bounded, and `/quit` cancels pending operations.
- `vara chat --dial` holds a link on an agreed dial instead of the VarAC calling
  frequency, and `--idcall` keys the closing CW ID inside the bounded close.
- Fixed VarAC links between two SSIDs of one station: level-6 DATA is no longer
  mistaken for a poll, peers are identified by full call, and a seventh call
  character survives link setup.
- Advanced beacons whose locator has a lowercase subsquare are now heard.
- The last packet of another station's two-packet VarAC broadcast, heard
  alone during a linked session, is no longer delivered as session DATA.
- A call's final connect request now gets its full reply window before the
  attempt gives up and keys the closing CW ID.
- A VARA caller no longer completes its connect on another station's burst
  heard after the connected-ack's answer window.
- An answering VARA station re-sends its connected-ack when the caller repeats
  its link setup, and closes a link it cannot hear the caller on within 30 s
  instead of holding it open at one end. Any part of an over from the caller
  counts as hearing it.
- Under CHAT, an answering VARA station sends text queued when the link comes
  up, such as VarAC's readiness or away message: on a Kestrel caller's
  confirm, and to a stock caller on the handover after its opening over or
  keepalive. With text queued it answers the caller's keepalive by asking for
  the turn, and asks again after a lost handover within one 12 s cadence
  instead of about 20 s.
- A high speed capable VARA caller that hears the called station weakly now
  asks for its greeting at a level it can read instead of always level 6, and
  the answering station starts at that level. A confirm naming the level is an
  off-default option, since no stock station has been seen using one.
- A VARA caller answers the answering station's query about an over it has
  acknowledged instead of letting the link close.
- A VARA station queried about an over it continued answers with the
  32-symbol frame at the same step, as stock does, instead of repeating the
  eight-symbol burst a weak path had already missed.
- An answering VARA station on a high speed capable link reads the caller's
  32-symbol answer to its over, including through a weak path, instead of
  querying it until the link closed.
- An answering VARA station reads the turnaround once more before querying
  its own over, and no longer takes a repeated caller handover heard behind
  its over as a new turn.
- A connected VARA station no longer repeats a turn request within a second
  of its first; the idle cadence counts from every transmission.
- At BW500, a weak over below the top level is no longer read as a pair with a
  lost second frame. An over at level 4 or under on a weak path asks the
  sender one level down, as stock does, and a weak closing over names one
  level lower for the sender's next delivery.
- A VARA chat station now queries an unanswered over seven times rather than
  four, outlasting the peer's own recovery cadence.
- Corrected wideband VARA confirmation and empty-turn sequencing, including
  through the whole-burst receive interface.
- Fixed near-capacity payload accounting when closing a Kestrel contact,
  recognition of solicited long VARA close grants, and generated responder DATA
  continuation streams.
- A VARA session whose audio stream wedges now exits with its own status after
  unkeying and closing its recording, instead of crashing on exit.
- VARA sessions warn when the receive audio device runs more than 2% off its
  sample rate, as a radio's USB codec can after re-enumerating; an overrun on
  such a device is reported as slow rather than as a CoreAudio deadlock.
- The passive follower needs about a third of the CPU it did. It runs its
  numerical libraries single-threaded at its own lowered priority, searches
  for idle replies once a second, and reads idle replies and connect requests
  off one decimated spectrum per channel. A follower that still falls behind
  stops reading its quietest channel and says so, instead of dropping audio
  on every channel.
- The passive follower keeps its readers across a short capture dropout, and
  keeps audio-gap notes in the event log rather than printing them as
  traffic, with one summary line every few minutes under heavy loss.
- The passive follower hears broadcasts, CQs and beacons from another SSID of
  our own call on neighbouring slots, and no longer names the caller of an
  earlier link as the sender of a later connect request.
- Added an optional BBS with a guide and replay. Doors recover across sleep and
  restarts, and daily quotas keep their charges. It is not yet qualified on
  the air.
- A PACTOR station answering a call now grants the PACTOR-3 upgrade the
  caller's announcement asks for, as a stock called station does, instead of
  keeping the link at PACTOR-1.
- The PACTOR-2 receiver combines the peer's repeats of a field it could not
  read (memory ARQ), so a faded link recovers where single decodes fail.
- A PACTOR caller whose first packet draws `0x6A9` three times running now
  resends it as a plain PACTOR-1 packet instead of running out of retries, so
  a PACTOR-2-only station keeps the link in PACTOR-1.
- The station PTC host accepts `MYLevel 2` and runs as a PACTOR-2-only modem,
  as do `onair --pactor2-only` and `console pactor fly --pactor2-only`: no
  PACTOR-3 or PACTOR-4 announcement, grant or follow.
- `onair --p2-6a9-grant` (off by default, unproven) reads `0x6A9` after the
  announcement as a PACTOR-2 grant and enters PACTOR-2 for that contact.
- Fixed PTC host-output byte accounting so output limits are enforced; pending
  PTC connects no longer spin, and running packet commands return their
  eventual results instead of false timeouts.
- Bounded the PACTOR-4 receiver's emission history.
- `[rig] power_clamp_w` caps every RF power the station sets, and lowers a
  radio found above it when a flight or chat starts. `--power` sets one
  PACTOR, VARA or ARDOP flight's power, and VarAC `/power` shows or sets it.
  `[rig] tune_power_w` sets the `--tune` carrier's power, and the FTDX10 no
  longer keys a tune carrier, which its tuner does not use.
- On Linux, a station that held its claim while the clock was adjusted could
  be refused its own release; claim holders are now identified by boot and
  process start ticks.
- A launcher no longer misses its modem's admission acknowledgement when a
  read times out mid-line.
- Fixed a race that could admit a silent native audio capture when the WAV
  header was written after the first size check.
- TX meter summaries report a key-down shorter than half a second as power
  unmeasured instead of the low reading the rig gives mid-attack.
- Faster CLI and VARA startup through lazy table generation and cached receive
  calculations.
- A creance run gives a modem a few seconds to accept its probe before
  reporting it unreachable.
- Added a checkout-only shackbook push/pull utility that reports transfer
  errors and removes retired source files on deployment.
- Added an opt-in Xara validation kit for public source releases, with a pinned
  peer or explicit tag, hash or HEAD, checks on generated audio, and recorded
  source and build provenance. Normal builds and tests do not fetch or run
  Xara.
- Fixed installed voice-console resources, excluded tests from wheels, and
  consolidated the operator-file inventory used for packaging.
- Public source drops now include the required regression recordings; fixed
  CI fixture readiness, asynchronous waits, terminal portability and timing.
- Refreshed the site's PACTOR and VARA evidence, replay controls, spectrum
  views and navigation, and corrected the homepage exchange diagram.
- Lanner's RTTY, SITOR-B and NAVTEX are now labelled alpha.

### Changed behaviour

For users upgrading from 0.2.1:

- FT-891 and FTDX10 station files without a `[rig] cat` setting now use native
  Yaesu CAT for station-owned radio control. Set `cat = "rigctld"` to retain
  rigctld control.
- ARDOP host commands and replies now wait for the data socket to connect,
  so an incomplete attachment cannot control the modem.
- The ARDOP host now refuses a second client and keeps the incumbent connection
  instead of replacing it, preventing contact takeover.
- Dropping the ARDOP data socket now closes the command socket without sending
  `DISCONNECTED` or `NEWSTATE DISC`, retiring the entire controlling attachment.
- The VARA host now refuses a second client instead of queuing it, and closes
  a command socket after 5 s if no data socket connects, preventing delayed
  takeover and stale incomplete attachments.
- Wideband VARA responders queue writes made at `CONNECTED` until the caller
  confirms; an empty initial turn polls before sending later writes. Wide
  `mfsk_only` responders can still request a turn for responder-first writes
  without full link setup.
- VARA now accepts called-station `DISCONNECT` and legacy `COMPRESSION ON`.
  `--speed-limited` was replaced by `--winlink-cartel-mode`, without an alias.
- PTC terminal mode enters converse during a contact after `JHOST0`; prefix
  terminal commands with ESC while the link is active.
- The logbook now uses the station service at `127.0.0.1:8737`. The standalone
  `tools/logbook_view/serve.py` server on port 8738 was removed; use
  `console logbook --export` for an offline copy.
- `[station] port` now reserves port 8737 by default. Choose distinct station
  ports when running multiple stations or when another service uses that port.
- Transmit tools now require a separate PTT line, including `shrike.onair`,
  ARDOP host radio mode and `tools/vara_rig_bridge.py`. `--ptt-type` was removed;
  configure `--line-ptt` and `--ptt-device`.
- The FTDX10 receive passband now applies a +500 Hz IF shift. The
  `shrike.beacon --rig` default changed from `x6100` to `ftdx10`.
- The hfhost client now sends `VERSION` before replaying state on attachment.
- ARDOP host recordings now default to `logs/onair` under `HFMODEM_DATA_DIR`
  or the XDG `hfmodem` data directory (`~/.local/share/hfmodem` by default).
  `onair.sh` retains its previous location.
- Removed internal import paths: use `core.hostmode` instead of
  `shrike.hostmode`, and `contact.runtime`, `contact.terminal` and
  `contact.autotest` instead of their `macagua` counterparts.

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
