# ARDOP Host Interface — Functional Record

This document is the functional record of the ARDOP host interface: the ASCII/TCP
dialect a client (Pat, Winlink Express, ARIM, hamChat) speaks to drive an ARDOP modem.
It was compiled **entirely by reading open reference implementations** — no reverse
engineering, no traffic capture. It states functional facts only (command names,
argument value-types, reply/echo grammar, notification grammar, port and framing
behaviour) so that the `besra` modem can consume it; every non-obvious claim carries a
`file:line` citation into one of the sources below, marked `[code]`, `[doc]`, or
`[spec-pdf]` for provenance.

## Sources

| Source | What it is | Licence |
|---|---|---|
| `reference/wl2k-go/transport/ardop/` | Pat's Go client transport (the driving host). `command.go`'s `parseCtrlMsg` is the most compact statement of the reply grammar. | MIT |
| `reference/ardopcf/src/common/HostInterface.c`, `TCPHostInterface.c`, `ARDOPC.c`, `ARQ.c`, `SoundInput.c` | The reference server (modem side): the command dispatcher and the notification emitters. | MIT |
| `reference/ardopcf/docs/Host_Interface_Commands.md` | ardopcf's own descriptive command reference (explicitly *descriptive of current behaviour, not prescriptive*). | MIT |
| `reference/ardopcf/docs/refs/*_20171130.pdf`, `*_20171109.pdf` | The 2017 ARDOP native-command / host-interface specs (describe the older ARDOP_Win TNC 1.0.2). | public spec |
| `reference/M0LTE.Ardop/src/M0LTE.Ardop/Host/` | A clean managed reimplementation (`ArdopHostTnc.cs`, `ArdopHostServer.cs`) that ports ardopcf's command surface byte-for-byte, with citations and quirk notes. | AGPL-3.0 |

Citations are keyed as: `command.go:114` = wl2k-go; `HostInterface.c:241` = ardopcf
server; `ArdopHostTnc.cs:280` = M0LTE; `[spec-pdf]` = the 2017 PDFs;
`[doc]` = `Host_Interface_Commands.md`.

---

## 1. Transport model

ARDOP exposes **two TCP sockets**:

- **Command socket** — CR-terminated ASCII commands host→modem, and CR-terminated
  ASCII replies + asynchronous notifications modem→host.
- **Data socket** — length-prefixed binary payload, both directions.

**Ports.** The command port defaults to **8515**; the data port is **always the
command port + 1** (default **8516**). ardopcf states this directly in its usage text:
"port is TCP Command Port Number. Data port number is automatically 1 higher"
(`ARDOPCommon.c:118-121` `[code]`; default reaffirmed `TCPHostInterface.c` header and
`ArdopHostServer.cs:49,56`). ardopcf binds `0.0.0.0` by default (any interface)
`[doc]`; the managed reimplementation binds loopback by default (`ArdopHostServer.cs:54`).

Pat derives the data address by **incrementing the last character of the command
address string** — literally byte `addr[len-1]+1` — so `localhost:8515` →
`localhost:8516`. The source comments this `// Oh no he didn't!`
(`tnc.go:62` `[code]`). This is a string hack, not integer arithmetic: it works for
`…8515` but would misbehave on an address ending in `9`. In practice the default is
`localhost:8515` (`ardop.go:17`).

**Socket lifecycle.** One host at a time per socket; a new connection replaces the
previous one (`ArdopHostServer.cs:19` remark, porting ardopcf). If either the command
or data socket drops mid-session, the modem runs a host-link failsafe: request an
orderly ARQ disconnect and revert to receive (`LostHost`, `ArdopHostServer.cs:335`,
`ArdopHostTnc.cs:264`). Listeners set `SO_REUSEADDR` so a restarted modem can rebind
its well-known ports immediately (`OpenSocket4`, `ArdopHostServer.cs:57`).

**Encoding.** Command lines are CR (`0x0D`) terminated. Commands are
case-insensitive: the modem upper-cases the whole line before dispatch
(`_strupr`, `HostInterface.c:263`; `ArdopHostTnc.cs:284`) `[doc]`. The original
casing survives **only** in the `ARQCALL`/`PING` echo (the modem echoes the
un-upper-cased copy, `cmdCopy`, `HostInterface.c:331`).

---

## 2. Reply / echo conventions

ARDOP's reply grammar has four shapes.

1. **Query** (parameter omitted): the modem replies `<CMD> <value>`.
   Example: `ARQTIMEOUT` → `ARQTIMEOUT 120` (`HostInterface.c:361` builds
   `"%s now %d"` for the set path; the query path prints `"%s %d"`).
2. **Set** (parameter present, valid): the modem echoes `<CMD> now <value>`.
   The literal token is **` now `** (lower-case, space-delimited). This is produced by
   the shared `DoTrueFalseCmd` helper (`snprintf(cmdReply, "%s now %s", …)`,
   `HostInterface.c:158`) and by every numeric/string setter individually
   (`"%s now %d"`, `"%s now %s"`). Pat keys on exactly this: it strips a
   case-insensitive `"now "` prefix when parsing echoes (`command.go:125-128`).
3. **Action** (no persistent value): the modem echoes the bare command token, e.g.
   `ABORT` → `ABORT`, `PURGEBUFFER` → `PURGEBUFFER`, `SENDID` → `SENDID`. A few
   actions produce no reply at all (see table).
4. **Fault**: `FAULT <text>` on any error (bad syntax, wrong state, unset MYCALL).
   Fault text is free-form; see §6 for the exact formats a grader must recognise.

A handful of commands break shape 2 — notably `ARQCALL`/`PING` (echo the original,
not `now`), `DISCONNECT` (replies `DISCONNECT NOW TRUE` / `DISCONNECT IGNORED`), and
`PURGEBUFFER` (replies via an async `BUFFER 0`, not `now`). These are flagged in the
table.

**Boolean value type.** Booleans are the literal words `TRUE` / `FALSE`
(case-insensitive on input; the modem emits upper-case). Pat parses
`strings.ToLower(value) == "true"` (`command.go:133`). `CWID` additionally accepts
`ONOFF` as a third truthy value (`HostInterface.c:564`, `ArdopHostTnc.cs:527`).

---

## 3. Command table

Direction legend: **S** = host→modem setter (echoes `now`); **Q** = host→modem query
(replies `<CMD> <value>`); **A** = host→modem action; **→H** = modem→host only
(see §4). Most get/set commands are both S and Q (query when parameter omitted).

Value-type legend: `bool` = TRUE/FALSE; `int` = decimal integer; `enum` = one of a
fixed set; `str` = free string; `call` = callsign; `none` = no argument.

Unless noted, the citation is to the ardopcf dispatcher `HostInterface.c` and/or the
M0LTE port `ArdopHostTnc.cs`, cross-checked against `[doc]`. Commands ardopcf actually
implements are listed first; the 2017-spec-only / legacy set follows.

### 3.1 Commands implemented by the deployed reference (ardopcf)

| Command | Arg | Dir | Reply on set / query | Notes & citation |
|---|---|---|---|---|
| `ABORT` (alias `DD`) | none | A | `ABORT` | Dirty-disconnect: clears buffer, forces DISC. `HostInterface.c:270`; `ArdopHostTnc.cs:384` |
| `ARQBW` | enum | S/Q | `ARQBW now 500MAX` / `ARQBW 500MAX` | Values `{200,500,1000,2000}×{MAX,FORCED}`. Change during a session → FAULT. `HostInterface.c:277`; enum `ARQ.c:70` |
| `ARQCALL` | `call int` | A | echoes original line (e.g. `ARQCALL W1AW 5`), **not** `now` | Starts ARQ connect; sends N CONREQ frames. Requires MYCALL set + PROTOCOLMODE ARQ. `HostInterface.c:308`; `ArdopHostTnc.cs:408` |
| `ARQTIMEOUT` | int | S/Q | `ARQTIMEOUT now 30` / `ARQTIMEOUT 120` | Idle seconds before auto-disconnect. Deployed range **30–240** (`ArdopHostTnc.cs:438`); 2017 spec says 30–600 `[spec-pdf]` |
| `AUTOBREAK` | bool | S/Q | `AUTOBREAK now FALSE` | Automatic IRS→ISS turnover. Default TRUE. `HostInterface.c:370` |
| `BREAK` | none | A | *(no reply)* | Manual link turnover request. `HostInterface.c:376`; `ArdopHostTnc.cs:451` |
| `BUFFER` | none | Q | `BUFFER <int>` | Outbound data bytes queued. Also emitted async (§4). `HostInterface.c:382` |
| `BUSYBLOCK` | bool | S/Q | `BUSYBLOCK now FALSE` | Reject inbound connects when channel busy. `HostInterface.c:395` |
| `BUSYDET` | int | S/Q | `BUSYDET now 6` | Busy-detector sensitivity 0–10 (0=off, higher=less sensitive). Default 5. `HostInterface.c:401` |
| `CALLBW` | enum | S/Q | `CALLBW now 500MAX` | Overrides ARQBW for ARQCALL; also accepts `UNDEFINED`. `HostInterface.c:427`; not in 2017 spec |
| `CAPTURE` | str | S/Q | `CAPTURE now plughw:1,0` | Audio capture device. Stored + echoed but audio not actually rerouted. `HostInterface.c:457` |
| `CAPTUREDEVICES` | none | Q | `CAPTUREDEVICES <name>` | Reports current capture device (not a real list in ardopcf). `HostInterface.c:473` |
| `CL` | none | A | *(async `BUFFER 0`)* | PTC/SCS emulator alias for PURGEBUFFER; slated for removal. `HostInterface.c:480` |
| `CLOSE` | none | A | *(no reply)* | Orderly shutdown of the TNC. `HostInterface.c:487` |
| `CMDTRACE` | bool | S/Q | `CMDTRACE now FALSE` | Log host commands to debug log. `HostInterface.c:493` |
| `CONSOLELOG` | int | S/Q | `CONSOLELOG 1` (**no `now`**) | Console verbosity 1–6 (lower=more). Query & set both print `<CMD> <n>`. `HostInterface.c:516` |
| `CWID` | enum | S/Q | `CWID now ONOFF` / `CWID FALSE` | Morse ID after ID frame. Values TRUE/FALSE/**ONOFF**. `HostInterface.c:541` |
| `DATATOSEND` | int(`0`) | S/Q | `DATATOSEND now 0` / `DATATOSEND <int>` | Query = queued bytes (like BUFFER); arg `0` clears the TX buffer. `HostInterface.c:587` |
| `DEBUGLOG` | bool | S/Q | `DEBUGLOG now FALSE` | Enable/disable debug log to disk. `HostInterface.c:615` |
| `DISCONNECT` | none | A | `DISCONNECT NOW TRUE` (in session) / `DISCONNECT IGNORED` (idle) | Graceful ARQ disconnect. `HostInterface.c:624`; `ArdopHostTnc.cs:578` |
| `DRIVELEVEL` | int | S/Q | `DRIVELEVEL now 50` | TX amplitude scale 0–100 (clamped to ≥1). Default 100. `HostInterface.c:637` |
| `ENABLEPINGACK` | bool | S/Q | `ENABLEPINGACK now FALSE` | Answer PING with PINGACK. Default TRUE. `HostInterface.c:667` |
| `EXTRADELAY` | int | S/Q | `EXTRADELAY now 10` | Extra RX→TX gap (ms) for long paths. `HostInterface.c:673` |
| `FASTSTART` | bool | S/Q | `FASTSTART now TRUE` | First ARQ data frame moderate (TRUE) vs most-robust (FALSE). Default TRUE. `HostInterface.c:705`; not in 2017 spec |
| `FECID` | bool | S/Q | `FECID now TRUE` | Send ID frame with each FEC transmission. `HostInterface.c:699` |
| `FECMODE` | enum | S/Q | `FECMODE now 8PSK.1000.100` | FEC frame type; see §3.3 for the value set. `HostInterface.c:711` |
| `FECREPEATS` | int | S/Q | `FECREPEATS now 5` | Repeat count 0–5 for FEC frames. `HostInterface.c:738` |
| `FECSEND` | bool | A | `FECSEND now TRUE` / `FECSEND now FALSE` | TRUE starts FEC send of buffered data; FALSE aborts. Not queryable. Requires MYCALL. `HostInterface.c:763` |
| `FSKONLY` | bool | S/Q | `FSKONLY now TRUE` | Use only FSK modulations for ARQ. `HostInterface.c:804` |
| `GRIDSQUARE` | str | S/Q | `GRIDSQUARE now CN87` | 4/6/8-char Maidenhead (also 2). Bad syntax → FAULT. Subsquare canonicalised lower-case. `HostInterface.c:810`; validation `ArdopHostTnc.cs:1085` |
| `INITIALIZE` | none | A | `INITIALIZE` | Clears queued state; first command a host should send. `HostInterface.c:833`; `ArdopHostTnc.cs:720` |
| `INPUTNOISE` | int | S/Q | `INPUTNOISE now 5000` | Diagnostic: add Gaussian noise (std-dev) to RX audio. `HostInterface.c` INPUTNOISE; `ArdopHostTnc.cs:727`; not in 2017 spec |
| `LEADER` | int | S/Q | `LEADER now 140` | Leader length ms, range 120–2500, rounded **up** to the next multiple of 10. Default 120 (ardopcf) / 160 (2017 spec). `HostInterface.c:844` |
| `LISTEN` | bool | S/Q | `LISTEN now FALSE` | Answer inbound ARQ/PING to MYCALL/MYAUX. Default TRUE. `HostInterface.c:878` |
| `LOGLEVEL` | int | S/Q | `LOGLEVEL now 1` | Debug-log verbosity 1–6. `HostInterface.c:888` |
| `MONITOR` | bool | S/Q | `MONITOR now FALSE` | Pass monitored (unaddressed) frames to host in DISC. Default TRUE. `HostInterface.c:911` |
| `MYAUX` | `call[,call…]` | S/Q | `MYAUX now K7AAA,K7BBB` / `MYAUX K7AAA` | Up to 10 aux callsigns, comma-separated. Invalid callsign clears the whole list + FAULT. `HostInterface.c:917` |
| `MYCALL` | call | S/Q | `MYCALL now K7CALL` | Station callsign. Invalid → FAULT. 3–7 A-Z/0-9 + optional `-SSID` (0–15 or A–Z). `HostInterface.c:965` |
| `PING` | `call int` | A | echoes original line, then may FAULT | Send N PING frames (needs MYCALL, DISC state). `HostInterface.c:994`; `ArdopHostTnc.cs:815` |
| `PLAYBACK` | str | S/Q | `PLAYBACK now plughw:1,0` | Audio playback device; stored/echoed, audio not rerouted. `HostInterface.c:1026` |
| `PLAYBACKDEVICES` | none | Q | `PLAYBACKDEVICES <name>` | Current playback device. `HostInterface.c:1042` |
| `PROTOCOLMODE` | enum | S/Q | `PROTOCOLMODE now FEC` / `PROTOCOLMODE ARQ` | `ARQ`\|`FEC`\|`RXO`. Any mode change forces state DISC. `HostInterface.c:1049` |
| `PURGEBUFFER` | none | A | *(async `BUFFER 0`)* + `PURGEBUFFER` | Empties outbound buffer. `HostInterface.c` PURGEBUFFER; `ArdopHostTnc.cs:889` |
| `RADIOFREQ` | str | S | *(no host reply; GUI only)* | Sets GUI freq field; missing arg → FAULT `RADIOFREQ command string missing`. `HostInterface.c` RADIOFREQ; `ArdopHostTnc.cs:894` |
| `RADIOHEX` | str | S | *(ignored)* | Would send CAT hex; no CAT device → silently ignored. `ArdopHostTnc.cs:903` |
| `RADIOPTTON` / `RADIOPTTOFF` | str | S | FAULT `… CAT Port not defined` | CAT PTT stubs. `ArdopHostTnc.cs:912` |
| `RXLEVEL` / `TXLEVEL` | int | S/Q | (embedded ADC volume) | Embedded-platform audio level; largely inert on desktop. `[doc]` |
| `SENDID` | none | A | `SENDID` | Transmit one ID frame with MYCALL. Needs MYCALL + DISC. `HostInterface.c` SENDID; `ArdopHostTnc.cs:919` |
| `SQUELCH` | int | S/Q | `SQUELCH now 6` | Leader-detector threshold 1–10. Default 5. `HostInterface.c:936` |
| `STATE` | none | Q | `STATE DISC` | Current protocol state (§4 state list). `HostInterface.c` STATE; `ArdopHostTnc.cs:954` |
| `TRAILER` | int | S/Q | `TRAILER now 40` | Trailer tone ms, 0–200, rounded **up** to the next multiple of 10. Default 20. `HostInterface.c` TRAILER; `ArdopHostTnc.cs:966` |
| `TUNINGRANGE` | int | S/Q | `TUNINGRANGE now 110` | ± Hz from 1500 an off-tune signal can decode, 0–200. `ArdopHostTnc.cs:987`; not in 2017 spec |
| `TWOTONETEST` | none | A | `TWOTONETEST` | 5 s two-tone burst at leader amplitude. Needs DISC → else FAULT `Not from state …`. `HostInterface.c` TWOTONETEST; `ArdopHostTnc.cs:970` |
| `USE600MODES` | bool | S/Q | `USE600MODES now TRUE` | Enable 4FSK.2000.600(S) for FM/2 m. Default FALSE. `ArdopHostTnc.cs:1000`; not in 2017 spec |
| `VERSION` | none | Q | `VERSION ardopcf_1.0.4.1.2` | Program name + version. `HostInterface.c` VERSION; `ArdopHostTnc.cs:1008` |

### 3.2 Legacy / 2017-spec commands (Pat knows them; ardopcf mostly does not)

Pat's command constants (`command.go:16-90`) include a large ARDOP_Win legacy set that
the deployed ardopcf/M0LTE modem does **not** implement (the dispatcher falls through to
the "not recoginized" fault, §6). A conformance grader targeting ardopcf should treat
these as unknown; a grader targeting a 2017 ARDOP_Win TNC should accept them.

| Command | Arg | Notes |
|---|---|---|
| `CODEC` | bool | Start/stop sound card. **Commented out in ardopcf** — "using it causes a segfault … disable this command" (`HostInterface.c:500-514` `[code]`). Pat sends it only if STATE==Offline (`tnc.go:135`), which ardopcf never is (boots DISC), so in practice unused. Was a normal command in 2017 `[spec-pdf]`. |
| `DISPLAY` | int(kHz) | Set waterfall dial-freq display. 2017 spec `[spec-pdf]`; absent in ardopcf. |
| `RESTOREBUFFER`/`RESTORBUFFER` | none | Undo a DATATOSEND 0 / PURGEBUFFER. 2017 spec `[spec-pdf]`; absent in ardopcf (buffer clear is irreversible there). |
| `RADIOANT`, `RADIOCTRL`, `RADIOCTRLBAUD`, `RADIOCTRLDTR`, `RADIOCTRLPORT`, `RADIOCTRLRTS`, `RADIOFILTERBW`, `RADIOICOMADD`, `RADIOISC`, `RADIOMENU`, `RADIOMODE`, `RADIOMODEL`, `RADIOMODELS`, `RADIOPTT` | various | Radio-control command family (2017 spec, red text `[spec-pdf]`). ardopcf implements only the `RADIOFREQ`/`RADIOHEX`/`RADIOPTTON`/`RADIOPTTOFF` stubs above; the rest are unknown. |
| `NEGOTIATEBW`, `SETUPMENU`, `TUNERANGE` (sic) | — | Referenced in Pat's constants / commented out in ardopcf (`HostInterface.c:988`). |

### 3.3 Value sets

**ARQ / CALL bandwidth** (`ARQBW`, `CALLBW`) — exactly these tokens
(`ARQBandwidths[]`, `ARQ.c:70` `[code]`; `ArdopHostTnc.cs:70`):

```
200FORCED 500FORCED 1000FORCED 2000FORCED
200MAX    500MAX    1000MAX    2000MAX     UNDEFINED
```

`ARQBW` accepts indices 0–7 (not `UNDEFINED`); `CALLBW` accepts all 9. Note the
suffix is **`FORCED`**, not `FORCE` (the `[doc]` prose says "MAX/FORCE" — a doc error;
see §6). Pat's bandwidth strings match: `<n>MAX`/`<n>FORCED`, defaulting to `MAX` when
the suffix is omitted (`ardop.go:72-100`).

**FEC modes** (`FECMODE`) — the host-selectable frame types
(`strAllDataModes`, `ArdopHostTnc.cs:57`; `[doc]`):

```
4FSK.200.50S   4PSK.200.100S  4PSK.200.100  8PSK.200.100  16QAM.200.100
4FSK.500.100S  4FSK.500.100   4PSK.500.100  8PSK.500.100  16QAM.500.100
4PSK.1000.100  8PSK.1000.100  16QAM.1000.100
4PSK.2000.100  8PSK.2000.100  16QAM.2000.100
4FSK.2000.600  4FSK.2000.600S
```

The `16QAM.*` modes are ardopcf additions; the 2017 spec `[spec-pdf]` lists the same
set without QAM. Naming: `<modulation>.<bandwidth Hz>.<baud>`, trailing `S` = short
frame; the two `600` modes are FM-only.

---

## 4. Asynchronous notifications (modem → host)

These arrive unsolicited on the **command socket** at any time; a host must read
continuously and tolerate interleaving with command replies. All are CR-terminated
ASCII.

| Notification | Format | Meaning & citation |
|---|---|---|
| `NEWSTATE <state>` | `NEWSTATE ISS ` (**note trailing space**) | Protocol state changed. `ARQ.c:338` builds `"NEWSTATE %s "` `[code]`. State token from `ARDOPStates[]`. |
| `STATE <state>` | `STATE DISC` | Reply to a `STATE` query (not truly async, but same state vocabulary). |
| `BUFFER <int>` | `BUFFER 1024` | Outbound TX-buffer byte count; emitted whenever the host loads data. `HostInterface.c:112` (`"BUFFER %d"`) |
| `PTT <bool>` | `PTT TRUE` / `PTT FALSE` | Keydown/keyup, ≤50 ms before the radio should key. Sent via the "quiet" (unlogged) path. `ALSASound.c:1916-1918`, `Waveout.c:954-956` `[code]` |
| `BUSY <bool>` | `BUSY TRUE` / `BUSY FALSE` | Channel busy-state change (only meaningful in DISC). `ARDOPC.c:2108,2126` `[code]` |
| `CONNECTED <call> <bw>` | `CONNECTED W1ABC 500` | ARQ session established; remote call + negotiated session bandwidth in Hz. `ARQ.c:1420,2027,2160` (`"CONNECTED %s %d"`) `[code]` |
| `DISCONNECTED` | `DISCONNECTED` | Session ended or connect failed (many causes, §QUIRKS). `ARQ.c:376,1496,…` `[code]` |
| `PENDING` | `PENDING` | A CONREQ/PING header was detected — early warning to pause scanning. `SoundInput.c:2402` `[code]` |
| `CANCELPENDING` | `CANCELPENDING` | The prior PENDING was not for us / mis-decoded — resume scanning. `ARDOPC.c:2317`, `ARQ.c:1292`, `SoundInput.c:2918,3023,3051` `[code]` |
| `TARGET <call>` | `TARGET K7CALL` | Target call of an inbound connect (== MYCALL or a MYAUX). Always precedes `CONNECTED` for inbound. `ARQ.c:1233` `[code]`; consumed by Pat's listener `listen.go:90` |
| `STATUS <text>` | `STATUS CONNECT TO LA3F FAILED!` | Human-readable status/progress; free-form (see §6 for the catalog). `ARQ.c`, `ARDOPC.c` many sites `[code]` |
| `FAULT <text>` | `FAULT Syntax Err: MYCALL 12` | Command error. §6. `HostInterface.c:1548` |
| `FREQUENCY <hz>` | `FREQUENCY 14105000` | Dial-frequency change (only when the TNC has radio control). Pat parses it as int (`command.go:162`). Not emitted by desktop ardopcf (no CAT). |
| `INPUTPEAKS <min> <max>` | `INPUTPEAKS -1234 5678` | Audio input level peaks (diagnostic, unlogged path). `ALSASound.c:1670`, `Waveout.c:758` (`"INPUTPEAKS %d %d"`) `[code]` |
| `PING <caller>>​<target> <snr> <q>` | `PING N7CAII>K6CALL 10 95` | A PING frame was decoded in DISC: caller>target, S:N dB, decode quality. `SoundInput.c:3039` (`"PING %s>%s %d %d"`) `[code]` |
| `PINGACK <snr> <q>` | `PINGACK 10 95` | Our PING was answered: S:N dB + quality. `SoundInput.c:3430` (`"PINGACK %d %d"`) `[code]` |
| `PINGREPLY` | `PINGREPLY` | We transmitted a PINGACK in reply to a PING addressed to us. `ARDOPC.c:2286` `[code]` |
| `REJECTEDBW <call>` | `REJECTEDBW W1ABC` | Connect rejected — incompatible bandwidth. `ARQ.c:1275,1362,…` `[code]` |
| `REJECTEDBUSY <call>` | `REJECTEDBUSY W1ABC` | Connect rejected — channel busy. `ARQ.c:1218,1971,…` `[code]` |

**Protocol state vocabulary** — `NEWSTATE`/`STATE` carry exactly these tokens
(`ARDOPStates[8]`, `ARDOPC.c:183` `[code]`):

```
OFFLINE  DISC  ISS  IRS  IDLE  IRStoISS  FECSEND  FECRCV
```

- `OFFLINE` — sound card released, not listening.
- `DISC` — initialised, listening, no session.
- `ISS` — Information Sending Station (transmitting data).
- `IRS` — Information Receiving Station.
- `IDLE` — connected, neither side sending.
- `IRStoISS` — mid-turnover from receiving to sending (transient).
- `FECSEND` — sending a FEC broadcast.
- `FECRCV` — receiving FEC data.

Pat additionally understands `FECRcv`/`FECSend` mixed-case keys (`ardop.go:123-124`)
and blank→`Unknown`; but ardopcf emits the **upper-case** forms above — see §6 for the
resulting case mismatch.

---

## 5. Connection / dial flow

**Host initialisation** (Pat's `open()`/`init()` sequence, `tnc.go:91-168` `[code]`,
mirrored by the ardopcf `[doc]` "typical ARQ init"):

1. `INITIALIZE` — clear queued state (first command).
2. `STATE` (query) — read current state.
3. `CODEC TRUE` — **only if** state == Offline (skipped against ardopcf, which boots
   DISC; and ardopcf has CODEC commented out anyway).
4. `PROTOCOLMODE ARQ`.
5. `ARQTIMEOUT <seconds>` — Pat's default is 90 (`ardop.go:18`); ardopcf accepts 30–240.
6. `LISTEN FALSE` — Pat disables inbound answering until it explicitly listens
   (`tnc.go:156`); enables `LISTEN TRUE` when a listener is opened (`listen.go:60`).
7. `MYCALL <call>`.
8. `GRIDSQUARE <grid>`.
9. (optional) `FSKONLY TRUE` behind an env flag (`tnc.go:161`).

ardopcf's own documented order additionally sets `ARQBW`, and for FEC mode substitutes
`BUSYDET`/`FECREPEATS`/`FECMODE` `[doc]`.

**Dialling** — `ARQCALL <target> <repeat>`:

- `<target>` is a callsign (or `CQ` per 2017 spec `[spec-pdf]`).
- `<repeat>` is the number of CONREQ frames to send. Pat's default is
  `DefaultConnectRequests = 10` (`dial.go:18`); the 2017 spec and `[doc]` say 2–15;
  the deployed parser only requires ≥1 (`parse_station_and_nattempts`,
  `HostInterface.c:185`; `ArdopHostTnc.cs:340`).
- The bandwidth used is `CALLBW` if set, else `ARQBW` (`[doc]`,
  `ArdopHostTnc.cs:428`).

**`ardop://` URL → command mapping** (Pat, `dial.go:25-48`):

- `?bw=<n>[MAX|FORCED]` → the bandwidth is applied via `ARQBW` **temporarily** for the
  dial and reverted on error or `conn.Close()` (`dial.go:82-101`). `<n>` ∈
  {200,500,1000,2000}; suffix defaults to `MAX`.
- `?connect_requests=<n>` → the `ARQCALL` repeat count (default 10).

**Dial outcome** — Pat's `arqCall` (`tnc.go:614-637`) resolves on:
`FAULT` → error; `NEWSTATE` DISC → connect timeout; `CONNECTED` → success. An inbound
connection is signalled `TARGET` then `CONNECTED` (Pat's listener relies on this order,
`listen.go:92-96`).

**Data path.** Payload flows on the data socket. Host→modem: enqueue bytes; the modem
answers with an async `BUFFER <n>`. Modem→host: tagged blocks (§6 framing) with a
3-char type — `ARQ` (connected data), `FEC` (FEC data), `ERR` (failed/uncorrected
data), `IDF` (decoded ID frame). Pat routes on these tags (`frame.go:23-26`,
`tnc.go:221-240`). During `INITIALIZE`, data-to-host is suppressed but command traffic
still flows (`blnInitializing`, `ArdopHostTnc.cs:1463`).

---

## 6. Single-serial multiplexed framing (serial transport; documented for completeness)

Over a **single serial link** (RS-232/USB/Bluetooth) ARDOP multiplexes the two logical
streams onto one wire. Over **TCP the two sockets carry the streams natively**, and the
serial prefix and CRC are **dropped**. Pat implements both; the framing is in
`frame.go` `[code]`.

**Serial frame envelope** (`readFrameOfType`, `frame.go:49-104`):

```
'*'  <type-byte>  ';'  <frame-body>  <CRC16-BE>
```

The leading `*` and `;` are the multiplex marker; `<type-byte>` selects the body form:

- **`c` — command frame:** the body is CR-terminated ASCII (a command line or a
  notification). On the wire, host→modem command frames are additionally prefixed
  `C:` before the payload (`writeCtrlFrame`, `frame.go:32-47`).
- **`d` — data frame:** the body is `<2-byte BE length><payload>`. For modem→host data
  the payload begins with a **3-char type tag** (`ARQ`/`FEC`/`ERR`/`IDF`) inside the
  counted length; host→modem data is prefixed `D:` on the wire and the length covers
  the raw payload (`conn.go:83-100`). The two-byte big-endian field counts everything
  after it in the block — tag plus payload modem→host, bare payload host→modem — and
  never itself, so a reader takes `2 + length` bytes off the stream per data frame
  (`frame.go:68`).

**CRC16** (serial only; `crc16.go` `[code]`):

- Polynomial **`0x8810`** (CRC-16-CCITT, reversed-reciprocal form).
- Initial seed **`0xFFFF`**.
- Processed **MSB-first**, one bit at a time, over the payload bytes (for a command
  frame, over the `C:`-prefixed payload including the trailing CR; for a data frame,
  over the length+payload but **excluding** the `D:` prefix — `conn.go:98`).
- Transmitted **big-endian** after the body. A mismatch yields
  `ErrChecksumMismatch` and, on the wire, a `CRCFAULT` prompt to resend
  (`command.go:18`, `conn.go:121`).

**Over TCP:** none of the above applies — no `*`/`;`/`C:`/`D:` prefixes, no CRC. The
command socket carries bare `<line>\r`; the data socket carries bare
`<2-byte BE length><payload>` (with the 3-char tag inside modem→host blocks). This is
exactly what `ArdopHostServer` implements (`ServeCommandAsync`,
`ArdopHostServer.cs:223`; `ServeDataAsync`/`SendTaggedData`,
`ArdopHostServer.cs:267,363`).

---

## 7. QUIRKS — where the deployed reference diverges from the 2017 spec

A conformance grader (or `besra`) must special-case the following. Provenance marked.

**Fault-string formats.**

- Unknown command → `FAULT CMD <cmd> not recoginized` — **"recoginized" is
  misspelled in the source** and is load-bearing for byte-exact matching
  (`HostInterface.c:1544` `[code]`; deliberately preserved in `ArdopHostTnc.cs:1014`).
- Syntax errors → `FAULT Syntax Err: <CMD> <param>` (`HostInterface.c` `strFault`
  sites; `ArdopHostTnc.cs` `Reply($"FAULT {fault}")`). Note "Err", not "Error".
- Wrong-state faults use two spellings: `Not from state <STATE>` (TWOTONETEST,
  lower-case "state") vs `Not from State <STATE>` (SENDID, capital "State") vs
  `No PING from state <STATE>` — inconsistent casing, ported verbatim
  (`ArdopHostTnc.cs:839,931,982`).
- Missing MYCALL → `FAULT MYCALL not set`. Mode faults → `FAULT Not from mode FEC` /
  `Not from mode RXO` (`ArdopHostTnc.cs:414-425`).

**`NEWSTATE` trailing space.** Every `NEWSTATE` (and some `STATUS`) line ends with a
trailing space because the format string is `"NEWSTATE %s "` (`ARQ.c:338` `[code]`).
Pat explicitly works around this: `parseCtrlMsg` calls `strings.TrimSpace` first, and a
code comment names it "Work around for ARDOPc trailing space in NEWSTATE"
(`command.go:115-116` `[code]`). A grader must not require the trailing space but must
tolerate it.

**State-token case mismatch.** ardopcf emits `FECSEND`/`FECRCV` (upper-case,
`ARDOPC.c:183`), but Pat's `stateMap` keys are mixed-case `FECSend`/`FECRcv`
(`ardop.go:123-124`) looked up after `ToUpper` — so those two never match in Pat and
resolve to `Unknown`. Harmless for Winlink (Pat only acts on DISC/ISS/IRS) but a real
inconsistency between the two open implementations.

**`ARQBW` suffix.** The value token is `FORCED` (`ARQ.c:70`, `ardop.go`,
`ArdopHostTnc.cs:70`), but ardopcf's own prose doc says "MAX/**FORCE**"
(`Host_Interface_Commands.md` ARQBW/CALLBW `[doc]`). Trust the code: `FORCED`.

**`ARQTIMEOUT` range.** 2017 spec: 30–600 s `[spec-pdf]`. Deployed: 30–240 s
(`ArdopHostTnc.cs:438`, matching ardopcf). Pat defaults to 90 s.

**`ARQCALL`/`PING` repeat range.** 2017 spec + `[doc]`: 2–15. Deployed parser: any
integer ≥ 1 (`HostInterface.c:185`). Pat sends 10 by default.

**`PROTOCOLMODE` accepts anything.** The 2017 spec lists only `ARQ|FEC`; ardopcf adds
`RXO`. Moreover ardopcf's validation guard is dead code —
`if (strcmp(p,"ARQ")!=0 && strcmp(p,"RXO")==0 && strcmp(p,"FEC")==0)` can never be true
(`HostInterface.c:1065-1066` `[code]`) — so **any** parameter is accepted and anything
other than `RXO`/`FEC` falls back to ARQ (`setProtocolMode`), while the echo parrots
the raw parameter (`PROTOCOLMODE now <whatever>`). Ported deliberately
(`ArdopHostTnc.cs:876-885`).

**`CODEC` removed.** A normal 2017 command `[spec-pdf]`, but commented out in ardopcf
because it segfaults ("disable this command until it is better understood" — LaRue, May
2024, `HostInterface.c:500-502` `[code]`). Sending it yields the "not recoginized"
fault. Pat only sends it from the Offline state, which ardopcf never enters.

**`CONSOLELOG` echoes without `now`.** Unlike its siblings, the set path prints
`CONSOLELOG <n>` (no `now`) — the reply mirrors `ardop_log_get_level_console()` via
`"%s %d"` (`HostInterface.c:529` `[code]`; `[doc]` shows `CONSOLELOG 1` → `CONSOLELOG 1`).

**`DISCONNECT` reply shape.** Not a `now` echo: `DISCONNECT NOW TRUE` (upper-case NOW)
when acting, `DISCONNECT IGNORED` when idle (`ArdopHostTnc.cs:583,587`; `[doc]`).

**`PURGEBUFFER`/`DATATOSEND 0`/`CL` clear via async `BUFFER 0`.** The buffer clear is
reported by an out-of-band `BUFFER 0` notification rather than (or in addition to) a
direct echo (`ClearDataToSend`, `HostInterface.c:112` path). The 2017 spec's
`RESTOREBUFFER` undo does not exist in ardopcf — a clear is irreversible.

**`DISCONNECTED` has many triggers.** Emitted on: ARQTIMEOUT reached; DISC frame sent 5×
with no response; either station sending a DISC or END frame (`ARQ.c` disconnect sites
`[code]`; `[doc]`). It is **not** solely a "connect failed" signal — Pat treats both
`DISCONNECTED` and `NEWSTATE DISC` as session-end (`tnc.go:254,263`).

**`STATUS` string catalog** (free-form; a grader should prefix-match, not equality-match).
Representative forms `[code]`/`[doc]`:

```
STATUS CONNECT TO <call> FAILED!
STATUS END ARQ CALL
STATUS ARQ CONNECTION ESTABLISHED WITH <call>, SESSION BW = <n> HZ
STATUS ARQ CONNECTION FROM <call>: SESSION BW = <n> HZ
STATUS ARQ CONNECTION ENDED WITH <call>
STATUS ARQ CONNECTION REJECTED BY <call>[, INCOMPATIBLE BW. | , REMOTE STATION BUSY.]
STATUS ARQ CONNECTION REQUEST FROM <call> REJECTED, CHANNEL BUSY.
STATUS ARQ Timeout from Protocol State:  <state>          (note the double space)
STATUS ARQ CONNECT REQUEST TIMEOUT FROM PROTOCOL STATE: <state>
STATUS END NOT RECEIVED CLOSING ARQ SESSION WITH <call>
STATUS QUEUE BREAK new Protocol State IRStoISS
STATUS BREAK received from Protocol State <state>, new state IRS
STATUS [RXO [<SessionIDByte>]] <FrameType> frame received OK. | frame decode FAIL.
STATUS INITIATING ARQ DISCONNECT
```

**Bind address.** ardopcf listens on all interfaces (`0.0.0.0`) by default — a security
note for `besra`, which should default to loopback (`ArdopHostServer.cs:54`) unless a
bind address is configured.

**Server tolerance.** The command reader reassembles CR-terminated lines across TCP
segments and processes any number per segment (`ProcessReceivedControl`,
`ArdopHostServer.cs:223`); blank lines and a leading `RDY` ACK are swallowed
(`ArdopHostTnc.cs:226`). A grader/`besra` must not assume one-command-per-packet.
