<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# The ALE host dialect

One TCP port (8330 by default, `[protocols.ale] port`), ASCII lines ending in
CR LF. Each command gets one answer, `OK ...` or `ERR <reason>`. Between
answers the controller writes what happens as unsolicited lines, so a client
reads the port as a stream and matches answers to its commands in order. One
application at a time: a new connection replaces the old one.

MIL-STD-188-141B specifies the controller, not how an application drives one.
This dialect is small, and it is not any radio's remote control.

Connections use the [normal host opening](HOST-API.md#host-connections). A pending
command that exceeds its five-second wait is cancelled before it can run;
a command already running returns its actual result. DTM, DBM and protected
exchanges are self-tested only; see [ALE.md](ALE.md) for unresolved wire
readings and the recording evidence needed.

## Commands

| Command | Does |
|---|---|
| `VERSION` | `OK bateleur-ale 1` |
| `CALL <address> [message]` | Link with a station: each channel in turn, best measured first, or within the link if there is one. An optional message goes with the call as an AMD. |
| `CALL <a>,<b>,... [message]` | A star group call (A.5.5.4.3): each station answers in the slot its place in the list gives it, and the link is made with those heard. At most 12 address words, 5 different first words. |
| `CALL <wildcard> [message]` | A wildcard call (A.5.5.4.6), `?` for any one character (`?O?` reaches JOE and BOB). Answered in random slots, as an AnyCall is. |
| `NET <net> <member>,<member>... [message]` | A star net call (A.5.5.4.2): one net address, the members listed in their slot order, which the caller needs to time the slots. |
| `ANYCALL [X\|XY]` | An AnyCall (A.5.5.4.5): every station hearing it answers in a random one of 16 slots, which takes about 45 s. `X` or `XY`: only addresses ending so. |
| `ALLCALL [/X] [message]` | An AllCall (A.5.5.4.4): every station hearing it stops, none answers, and all are linked. `/X`: only addresses ending in X. |
| `AMD <address> <message>` | Send an AMD message: inside the link with that station, or carried by a call. Up to 90 characters of the expanded 64 set, which covers space to `_` in ASCII; lowercase is sent as uppercase. |
| `BROADCAST <address> <message>` | An AMD to an address or net, ended TWAS on the present channel: no one answers and no link is made. `SENT` when it is off. |
| `GPR <lat> <lon> [<altitude m> [comment]]` | This station's position now, as an ALE-GPR (`ale.gpr`), broadcast to the `GPR` net. Degrees, north and east positive. |
| `WHO <address>` | What the station's directories (`[protocols.ale] directory`) say of an address: `OK WHO <address> <pattern>=<agency>: <description> \| ...`, or `not listed`. |
| `DTM <address> [NOARQ] <text>` | A DTM message (A.5.7.3), 7-bit ASCII, over the link with that station (called first if there is none). Sent in data blocks each acknowledged before the next, and sent at most seven times until it is; `NOARQ` sends the blocks once each, one frame after another. |
| `DTMBIN <address> [NOARQ] <hex>` | The same in binary form: the bytes given in hex, sent as bits. |
| `DBM`, `DBMBIN` | As `DTM` and `DTMBIN`, in DBM's deeply interleaved blocks (A.5.7.4): some three times as fast. |
| `TIME <address>` | Ask a station the time (linking protection, B.5.5): protected if this station's time is good to the minute (under the coarse seed if not to the protection interval), in the clear with an authenticator if not. |
| `TIMECAST [address]` | Broadcast this station's time, protected, TWAS (B.5.5.5); to all (`@?@`) by default. |
| `QUERY <address> LQA\|VERSION\|CAPABILITIES ...` | Call the station asking for those reports in its response (A.5.4.2, A.5.6.6). A call also links. |
| `SOUND` | Sound on the present channel, if available and the channel is clear. |
| `HANGUP` | End every link, sending TWAS to each station. |
| `SCAN ON\|OFF` | Scan the channel list while available, or stay put. With scanning off, new calls and scheduled sounds use the held channel only. |
| `CHANNEL <index or name>` | Stop scanning and tune to one channel while available; a linked or busy controller refuses (hang up a link or wait for the current call to finish). A failed retune returns `ERR`. |
| `CHANNELS` | `OK CHANNELS 0=7102.0 1=10145.5 ...` |
| `STATUS` | `OK STATUS <state> CHANNEL <i> <name> LINKED <stations or -> SCAN ON\|OFF`, and `LP AL-<n> TIME Q<quality>` when protected |
| `HEARD` | The stations heard most recently, each with its channel, BER and SINAD. |

`NET HFN` (also HFL) uses the preset response-slot
list when members are omitted. Explicit members remain available for
other slot plans.

## Events

| Line | Means |
|---|---|
| `SOUNDED CHANNEL <name>` | A sound completed. |
| `TUNE_FAILED CHANNEL <name> <reason>` | A channel tune failed; its previous dial is not treated as the requested channel. |
| `TX_REFUSED CHANNEL <name> <reason>` | A complete frame exceeded the key-up limit. |
| `STATE <state>` | `AVAILABLE`, `CALLING`, `AWAIT-RESPONSE`, `AWAIT-ACK`, `LINKED`, `SOUNDING`, and for one-to-many calls `COLLECT` (gathering slotted responses) and `AWAIT-SLOT` (waiting for our slot); `CHANNEL BUSY` when a channel is skipped. |
| `CALLED <station> CHANNEL <name> [(<kind>)]` | A station called us, and we responded; the kind is `net`, `group`, `anycall` or `wildcard` for a slotted response. |
| `RESPONDED <station> CHANNEL <name>` | A slotted response to our one-to-many call was read. |
| `SENT <address> CHANNEL <name> allcall[: <text>]` | Our AllCall went out. |
| `LINKED <station> CHANNEL <name> [BER n SINAD n] [(allcall)]` | A three-way handshake completed, or an AllCall linked us. |
| `CALL FAILED <station> <why>` | Every channel was tried, or the handshake broke off. |
| `REJECTED <station> CHANNEL <name>` | The station answered with TWAS. |
| `LINK ENDED <station> <why>` | By `HANGUP`, by the far end's TWAS, or by the activity timer. |
| `AMD <station> <text>` | A message arrived. A version report arrives this way. |
| `WHO <station> <pattern>=<agency>: <description> \| ...` | After the first `HEARD`, `CALLED` or `LINKED` of a station the directories list: who it is. Once a session per station. |
| `DTM <station> <text>` / `DTMBIN <station> <hex>` | A DTM message arrived whole, every block's CRC good. `DBM` and `DBMBIN` likewise, and `DBM SENT`, `DBM FAILED`, `DBM DAMAGED` as for DTM. |
| `DTM SENT <station> <length>` | Every block of our DTM message was acknowledged (or, without ARQ, sent). |
| `DTM FAILED <station> <why>` | A block exhausted seven attempts or its five-minute exchange limit, a transmission was refused, or the link ended. |
| `DTM DAMAGED <station> <what>` | A block arrived failing its CRC; with ARQ it is asked for again and the copies combined. |
| `TIME <station> adjusted <seconds> s, quality <q>` | A time service response checked and taken. `TIME REQUESTED`, `TIME SERVED`, `TIME FAILED` as they say. |
| `LQA <station> CHANNEL <name> BER n SINAD n` | The station's measurement of our call: BER as table A-XIII's code (average non-unanimous votes), SINAD in dB. |
| `CAPABILITIES <station> scan_rate=... channels=... ...` | The station's capabilities report, field by field (figure A-45). |
| `HEARD <station> CHANNEL <name> [BER n SINAD n] (<what>)` | A sound, or a call to someone else. |
