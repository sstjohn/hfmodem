<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Over index: high speed capable VARA bench corpus

`cut_overs.py` reads every bench tape (5 manifest sessions, 57 sweep ladders) and writes
`overs.jsonl`: 1979 keyings, one record per `PTT ON`/`PTT OFF` pair a station reported, plus a `call` field (A is
the station on port 8500; it is W9SSJ in the W9SSJ-caller tape). The script reads the tapes only to measure them.
Nothing here copies audio out of them. Rerun it to regenerate the tables below; the text above the marker is kept.

## Fields

- `sample_on/off`: the transcript PTT times mapped into the transmitting station's wav (A→`__a2b`, B→`__b2a`).
  `null` where the transcript has no edge: a PTT OFF with no ON at the head of a ladder (the previous session's
  B disconnect tail, `kind: unknown`), or B's last keying, whose PTT OFF lands in the next ladder's transcript.
- `measured_on/off`: `[measured_on, measured_off)` is the burst. Each edge is where a 5 ms power window (behind the
  sample for the rise, ahead of it for the fall) crosses the dB midpoint between the floor on that side of the edge
  and the burst power just inside it. The floor is clamped to 40 dB below the burst, so on a clean leg the cut falls
  at the foot of VARA's ramp. On the clean manifest legs the rise is a median 4 samples after the first non-zero
  sample and the fall 3 samples after the last, within 30 and 37 samples for 90% of keyings. Each edge uses its own
  side's floor because the injected noise switches on and off independently.
- `floor_rms`: RMS in [-400, -100] ms before PTT ON (after PTT OFF when there is no room before). `burst_rms`:
  RMS of [measured_on, measured_off). Both in dBFS; **-200 means digital silence**.
- `kind`: before the first `CONNECT` command → `unknown`. From the `DISCONNECT` command on → `disconnect`. Before
  the caller prints `CONNECTED` → `link_setup` (caller) or `connect_answer` (callee). After that → `data` for the
  station that sent the payload (`data_tx`) and `ack` for the other. `payload_len` = bytes the IRS delivered
  (`data_rx`) between that keying's PTT OFF and the ISS's next keying, so `0` marks a frame that was keyed but not
  delivered (74 are full-length data frames: retries under noise), or an idle frame.
- `level_tx` is the station's latest `BITRATE (N) … TX` at or before PTT ON. `level_rx_reported` is the peer's
  `BITRATE (N) … RX` before the station's next keying. Where both exist they never disagree.
- `session_ok`: the reference decoder's end note (`ok` and `byte_exact`) on manifest tapes. On ladders it is true only when
  the `data_rx` chunks add up to the `data_tx` length and each chunk starts at the right counter offset. False for
  `N0CALL-W9SSJ-bw500-counter65536` (the reference decoder recorded 49174 of 65536 received, not byte-exact) and for
  `sweep-bw2750-quiet/…060706Z`, which ends mid-transfer (14471/32768 bytes; its wav header has a zero data size).

## Mapping

Manifest tapes use the manifest `t_start`: sample = (t − t_start)·48000. Ladders have no manifest and no
capture marks, so each wav is placed in two steps. First, a 10 ms lag correlation of A's PTT spans against the
a2b frame power gives a coarse fit (the a2b leg is always clean). Second, each wav gets its own shift so that its
anchor keyings land where the same frames land on the manifest tapes: A `link_setup` (on +2202, off +1012
samples) and B `connect_answer` (on +841, off +116). On every ladder the b2a leg comes out **+1547 samples
(~32 ms; p10–p90 1334–1771)** from where a shared start would put it. So the two wavs of a ladder capture do not
start together, and a single t0 for both would be off by that much.

Pooled residuals (measured − mapped, samples, p10/median/p90, excluding disconnect/unknown):

| legs | n | on | off |
|---|---|---|---|
| manifest A | 142 | 1935 / 2391 / 2697 | 557 / 854 / 1465 |
| manifest B | 132 | −273 / 573 / 1211 | −97 / 148 / 357 |
| ladder A | 699 | 1170 / 2072 / 2403 | 890 / 1304 / 1573 |
| ladder B | 694 | 730 / 1187 / 1996 | −1273 / −434 / 215 |

The spread is VARA's own lag between reporting `PTT ON` and making sound. It runs from −6 to +56 ms and depends on
the station and the frame type. On top of that comes jitter on the transcript's command lines: a few keyings are logged up to 60 ms after their audio
starts. A mapped edge is good to ±50 ms. Use `measured_*` to cut.

Checked by hand:
- `N0CALL-W9SSJ-bw2300-counter4096` A data, level 8, t 48.074: first non-zero sample at +2406, but the first 10 ms
  frames run −51, −25, −16 dBFS, so the rise is cut at +3163. Last non-zero at +355 after PTT OFF, fall at −400.
- The same tape, B disconnect at t 67.325: the wav stays digital silence for **3.007 s after PTT ON** (first non-zero
  +144340, rise +144343). Every tape shows it (21 cases): B keys up while A is still sending its disconnect, and holds its
  audio for 3 s.
- `sweep-bw2750-quiet/…060010Z` B data, level 13: noise at −30.5 dBFS on both sides, burst −16.8; rise +982, fall −689.
- `sweep-bw2300/…051938Z` B data, level 8: the floor window is silent although this is the noisiest tape; first
  non-zero sample +1557, rise +2353; last non-zero −354, fall −1112.

Nothing is left unpinned: every ladder measured both anchor keyings on both legs. Unmeasured keyings: 27 with a
missing PTT edge (B's last disconnect keying runs past the end of its wav, plus the truncated ladder), and 17 B data
keyings on the noisiest b2a legs. Another 23 noisy B data keyings have an edge well outside the usual residual range.
In those, noise switching on or off next to the edge takes the crossing with it, so treat their `measured_*` as
approximate.

## Noise

Noise sits on the **b2a leg of the ladders only** (B→A; B is the ISS there). It is band-limited to about 300–2700 Hz
and **stepped 1 dB per ladder**: −26.4 → −10.4 dBFS on sweep-bw2300, sweep-bw2750 and sweep-bw500, and −32.4 →
−26.4 on `sweep-bw2750-quiet`. "Quiet" is the low end of the ladder, not a noise-free run. VARA bursts sit at about
−11 to −17 dBFS, so the noisiest ladders are near 0 dB wideband SNR. The noise is **gated**: it switches on and off
in stretches of seconds, including in the middle of a burst. So a keying whose `floor_rms` is −200 on a noisy tape
may still carry noise. Judge a tape by its leg's floor in the table below, not by one keying's floor. Fully clean:
all five manifest tapes (both legs) and the a2b leg of every ladder. Everything A keyed is clean.

Frame lengths worth knowing (samples): the caller's first link_setup is 83960 (no level yet), then 208365 at
level 4. Acks are 16376 or 22521. Data frames run ~176k–208k, and the top levels also key double-length frames of
~360.5k (bw500 level 13, bw2300 level 16, bw2750 level 17).

<!-- generated by cut_overs.py below this line -->

## Keyings by bandwidth, level and kind

Durations are measured_off - measured_on in samples at 48 kHz; `measured` counts keyings whose envelope edges were found. `data (no payload)` is an ISS keying after which the IRS delivered no bytes. `cleanest` is the tape whose leg carrying this keying has the lowest noise floor.

| bw | level_tx | kind | n | measured | min | median | max | cleanest (leg floor dBFS) |
|---|---|---|---|---|---|---|---|---|
| 500 |  | ack | 58 | 58 | 16376 | 16376 | 22520 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 |  | connect_answer | 36 | 36 | 20376 | 40640 | 47181 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 |  | disconnect | 2 | 2 | 22520 | 76788 | 131056 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 |  | link_setup | 18 | 18 | 83959 | 83959 | 83959 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 |  | unknown | 17 | 17 | 14177 | 26497 | 32583 | sweep-bw500/20260926T052115Z-ladder (-26.4) |
| 500 | 4 | ack | 210 | 210 | 16376 | 16376 | 65782 | sweep-bw500/20260926T052115Z-ladder (silent) |
| 500 | 4 | data | 3 | 3 | 205038 | 205387 | 205389 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 | 4 | data (no payload) | 6 | 6 | 32759 | 65527 | 65527 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 | 4 | disconnect | 34 | 34 | 22520 | 122484 | 223215 | sweep-bw500/20260926T052115Z-ladder (silent) |
| 500 | 4 | link_setup | 18 | 18 | 204873 | 205389 | 205391 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 | 5 | data | 24 | 24 | 203964 | 205639 | 205905 | sweep-bw500/20260926T052812Z-ladder (-18.4) |
| 500 | 5 | data (no payload) | 2 | 2 | 34120 | 34422 | 34724 | sweep-bw500/20260926T053832Z-ladder (-10.4) |
| 500 | 5 | disconnect | 2 | 1 | 34781 | 34781 | 34781 | sweep-bw500/20260926T053832Z-ladder (-10.4) |
| 500 | 6 | data | 19 | 19 | 194959 | 196038 | 196446 | sweep-bw500/20260926T052115Z-ladder (-26.4) |
| 500 | 7 | data | 11 | 11 | 174974 | 176136 | 177035 | sweep-bw500/20260926T052707Z-ladder (-19.3) |
| 500 | 7 | data (no payload) | 11 | 10 | 14122 | 90728 | 176184 | sweep-bw500/20260926T052917Z-ladder (-17.3) |
| 500 | 7 | disconnect | 2 | 2 | 22380 | 83802 | 145225 | sweep-bw500/20260926T052917Z-ladder (-17.3) |
| 500 | 8 | data | 20 | 19 | 176052 | 176206 | 181464 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 | 8 | data (no payload) | 21 | 20 | 15683 | 22520 | 181339 | sweep-bw500/20260926T052607Z-ladder (-20.4) |
| 500 | 8 | disconnect | 12 | 8 | 18401 | 22400 | 105541 | sweep-bw500/20260926T052607Z-ladder (-20.4) |
| 500 | 9 | data | 14 | 14 | 174324 | 176125 | 176214 | sweep-bw500/20260926T052202Z-ladder (-25.4) |
| 500 | 9 | data (no payload) | 14 | 13 | 13688 | 174355 | 179049 | sweep-bw500/20260926T053026Z-ladder (-16.4) |
| 500 | 9 | disconnect | 2 | 1 | 22446 | 22446 | 22446 | sweep-bw500/20260926T053026Z-ladder (-16.4) |
| 500 | 10 | data | 11 | 11 | 176055 | 176205 | 176217 | sweep-bw500/20260926T052115Z-ladder (-26.4) |
| 500 | 10 | data (no payload) | 5 | 5 | 10808 | 22520 | 179176 | sweep-bw500/20260926T053246Z-ladder (-14.3) |
| 500 | 10 | disconnect | 2 | 1 | 22343 | 22343 | 22343 | sweep-bw500/20260926T053246Z-ladder (-14.3) |
| 500 | 11 | data | 10 | 10 | 176056 | 176182 | 176227 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 | 11 | data (no payload) | 22 | 22 | 22417 | 96277 | 176031 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 | 11 | disconnect | 11 | 10 | 18367 | 22435 | 173295 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 | 12 | data | 16 | 16 | 183093 | 183122 | 183131 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 | 12 | data (no payload) | 11 | 9 | 22464 | 181684 | 183012 | sweep-bw500/20260926T052337Z-ladder (-23.4) |
| 500 | 12 | disconnect | 2 | 1 | 22413 | 22413 | 22413 | sweep-bw500/20260926T052337Z-ladder (-23.4) |
| 500 | 13 | data | 52 | 49 | 183100 | 360518 | 360542 | N0CALL-W9SSJ-bw500-counter65536 (silent) |
| 500 | 13 | data (no payload) | 6 | 5 | 22482 | 181950 | 188005 | sweep-bw500/20260926T052202Z-ladder (-25.4) |
| 500 | 13 | disconnect | 2 | 2 | 18368 | 20418 | 22468 | sweep-bw500/20260926T052202Z-ladder (-25.4) |
| 2300 |  | ack | 43 | 43 | 16375 | 16376 | 22521 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 |  | connect_answer | 40 | 40 | 21236 | 40806 | 47286 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 |  | disconnect | 6 | 6 | 22521 | 76788 | 155633 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 |  | link_setup | 20 | 20 | 83960 | 83960 | 84216 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 |  | unknown | 17 | 17 | 13329 | 26598 | 32614 | sweep-bw2300/20260926T050709Z-ladder (-26.4) |
| 2300 | 4 | ack | 125 | 125 | 16375 | 22521 | 65526 | sweep-bw2300/20260926T050709Z-ladder (silent) |
| 2300 | 4 | data | 15 | 15 | 206448 | 208253 | 208487 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 | 4 | data (no payload) | 18 | 18 | 32759 | 65528 | 65528 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 | 4 | disconnect | 34 | 34 | 22521 | 122484 | 223215 | sweep-bw2300/20260926T050709Z-ladder (silent) |
| 2300 | 4 | link_setup | 20 | 20 | 208365 | 208365 | 208621 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 | 5 | data | 15 | 15 | 189234 | 195450 | 196348 | sweep-bw2300/20260926T051002Z-ladder (-22.3) |
| 2300 | 6 | data | 15 | 15 | 188152 | 190341 | 190796 | sweep-bw2300/20260926T050709Z-ladder (-26.4) |
| 2300 | 6 | data (no payload) | 1 | 1 | 183546 | 183546 | 183546 | sweep-bw2300/20260926T051938Z-ladder (-10.3) |
| 2300 | 7 | data | 5 | 5 | 185046 | 185337 | 185786 | sweep-bw2300/20260926T051002Z-ladder (-22.3) |
| 2300 | 7 | data (no payload) | 2 | 2 | 20375 | 21032 | 21690 | sweep-bw2300/20260926T051938Z-ladder (-10.3) |
| 2300 | 7 | disconnect | 2 | 1 | 25773 | 25773 | 25773 | sweep-bw2300/20260926T051938Z-ladder (-10.3) |
| 2300 | 8 | data | 9 | 9 | 183591 | 184060 | 184103 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 | 9 | data | 11 | 10 | 182117 | 182295 | 182538 | sweep-bw2300/20260926T050709Z-ladder (-26.4) |
| 2300 | 10 | data | 14 | 12 | 173276 | 176202 | 176465 | sweep-bw2300/20260926T050709Z-ladder (-26.4) |
| 2300 | 10 | data (no payload) | 20 | 19 | 15249 | 22352 | 174958 | sweep-bw2300/20260926T050709Z-ladder (-26.4) |
| 2300 | 10 | disconnect | 16 | 10 | 18367 | 21876 | 22482 | sweep-bw2300/20260926T050709Z-ladder (-26.4) |
| 2300 | 11 | data | 5 | 5 | 176151 | 176212 | 176213 | sweep-bw2300/20260926T050752Z-ladder (-25.4) |
| 2300 | 11 | data (no payload) | 8 | 8 | 22520 | 22520 | 22776 | sweep-bw2300/20260926T050835Z-ladder (-24.4) |
| 2300 | 11 | disconnect | 8 | 8 | 18355 | 22418 | 106381 | sweep-bw2300/20260926T050835Z-ladder (-24.4) |
| 2300 | 12 | data | 7 | 6 | 176196 | 176214 | 176472 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 | 12 | data (no payload) | 8 | 8 | 20494 | 22520 | 22617 | sweep-bw2300/20260926T051002Z-ladder (-22.3) |
| 2300 | 12 | disconnect | 8 | 7 | 20733 | 22450 | 114852 | sweep-bw2300/20260926T051002Z-ladder (-22.3) |
| 2300 | 13 | data | 2 | 2 | 176213 | 176214 | 176216 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 | 13 | data (no payload) | 2 | 2 | 22520 | 22520 | 22521 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 | 13 | disconnect | 2 | 2 | 148718 | 161006 | 173295 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 | 14 | data | 3 | 3 | 176204 | 176211 | 176217 | N0CALL-W9SSJ-bw2300-counter4096 (silent) |
| 2300 | 15 | data | 2 | 2 | 183124 | 183126 | 183128 | N0CALL-W9SSJ-bw2300-counter65536 (silent) |
| 2300 | 15 | data (no payload) | 1 | 1 | 22521 | 22521 | 22521 | N0CALL-W9SSJ-bw2300-counter65536 (silent) |
| 2300 | 15 | disconnect | 1 | 1 | 173295 | 173295 | 173295 | N0CALL-W9SSJ-bw2300-counter65536 (silent) |
| 2300 | 16 | data | 12 | 12 | 183113 | 271830 | 360544 | N0CALL-W9SSJ-bw2300-counter65536 (silent) |
| 2750 |  | ack | 21 | 21 | 16376 | 16376 | 22520 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 |  | connect_answer | 48 | 48 | 21292 | 34700 | 47092 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 |  | disconnect | 2 | 2 | 22520 | 76788 | 131057 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 |  | link_setup | 24 | 24 | 83707 | 83960 | 84216 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 |  | unknown | 21 | 21 | 12694 | 26880 | 35488 | sweep-bw2750-quiet/20260926T060010Z-ladder (-30.4) |
| 2750 | 4 | ack | 250 | 250 | 16375 | 16377 | 65783 | sweep-bw2750-quiet/20260926T055838Z-ladder (silent) |
| 2750 | 4 | data | 10 | 10 | 207894 | 208175 | 208364 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 | 4 | data (no payload) | 6 | 6 | 32759 | 65528 | 65528 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 | 4 | disconnect | 44 | 44 | 22521 | 122484 | 223215 | sweep-bw2750-quiet/20260926T055838Z-ladder (silent) |
| 2750 | 4 | link_setup | 24 | 24 | 208364 | 208364 | 208364 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 | 5 | data | 12 | 11 | 194652 | 195823 | 196328 | sweep-bw2750/20260926T045941Z-ladder (-15.4) |
| 2750 | 6 | data | 32 | 32 | 188624 | 190301 | 193497 | sweep-bw2750-quiet/20260926T055838Z-ladder (-32.3) |
| 2750 | 6 | data (no payload) | 2 | 2 | 188301 | 188390 | 188478 | sweep-bw2750/20260926T050512Z-ladder (-10.4) |
| 2750 | 7 | data | 5 | 5 | 185207 | 185619 | 185645 | sweep-bw2750/20260926T045727Z-ladder (-18.4) |
| 2750 | 7 | data (no payload) | 8 | 8 | 12405 | 183744 | 188780 | sweep-bw2750/20260926T050204Z-ladder (-12.4) |
| 2750 | 7 | disconnect | 2 | 1 | 22624 | 22624 | 22624 | sweep-bw2750/20260926T050512Z-ladder (-10.4) |
| 2750 | 8 | data | 5 | 5 | 183775 | 184092 | 184106 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 | 8 | data (no payload) | 4 | 4 | 15828 | 22520 | 180339 | sweep-bw2750/20260926T050204Z-ladder (-12.4) |
| 2750 | 8 | disconnect | 2 | 1 | 22119 | 22119 | 22119 | sweep-bw2750/20260926T050204Z-ladder (-12.4) |
| 2750 | 9 | data | 9 | 9 | 182102 | 182177 | 182568 | sweep-bw2750/20260926T045304Z-ladder (-24.3) |
| 2750 | 9 | data (no payload) | 3 | 3 | 14512 | 182451 | 182598 | sweep-bw2750/20260926T050512Z-ladder (-10.4) |
| 2750 | 10 | data | 22 | 21 | 182201 | 182418 | 182575 | sweep-bw2750-quiet/20260926T055838Z-ladder (-32.3) |
| 2750 | 10 | data (no payload) | 11 | 11 | 13980 | 22520 | 186499 | sweep-bw2750/20260926T045139Z-ladder (-26.4) |
| 2750 | 10 | disconnect | 8 | 6 | 18367 | 22446 | 22476 | sweep-bw2750/20260926T045139Z-ladder (-26.4) |
| 2750 | 11 | data | 13 | 13 | 176088 | 176205 | 176460 | sweep-bw2750-quiet/20260926T060325Z-ladder (-28.4) |
| 2750 | 11 | data (no payload) | 16 | 16 | 21779 | 22520 | 175990 | sweep-bw2750/20260926T045304Z-ladder (-24.3) |
| 2750 | 11 | disconnect | 14 | 11 | 18368 | 22466 | 107177 | sweep-bw2750/20260926T045304Z-ladder (-24.3) |
| 2750 | 12 | data | 13 | 13 | 176069 | 176196 | 176214 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 | 12 | data (no payload) | 9 | 9 | 21151 | 22520 | 174362 | sweep-bw2750/20260926T045513Z-ladder (-21.4) |
| 2750 | 12 | disconnect | 8 | 6 | 22093 | 22444 | 118173 | sweep-bw2750/20260926T045513Z-ladder (-21.4) |
| 2750 | 13 | data | 11 | 11 | 176089 | 176194 | 176433 | sweep-bw2750-quiet/20260926T055838Z-ladder (-32.3) |
| 2750 | 13 | data (no payload) | 5 | 4 | 22465 | 22510 | 22520 | sweep-bw2750-quiet/20260926T055838Z-ladder (-32.3) |
| 2750 | 13 | disconnect | 4 | 4 | 18375 | 22512 | 65362 | sweep-bw2750-quiet/20260926T055838Z-ladder (-32.3) |
| 2750 | 14 | data | 13 | 13 | 176045 | 176169 | 176380 | sweep-bw2750-quiet/20260926T055838Z-ladder (-32.3) |
| 2750 | 14 | data (no payload) | 3 | 3 | 22472 | 22475 | 176083 | sweep-bw2750-quiet/20260926T060325Z-ladder (-28.4) |
| 2750 | 14 | disconnect | 2 | 2 | 18402 | 20442 | 22481 | sweep-bw2750-quiet/20260926T060325Z-ladder (-28.4) |
| 2750 | 15 | data | 17 | 17 | 175972 | 176204 | 176224 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 | 15 | data (no payload) | 4 | 4 | 22487 | 22520 | 22520 | sweep-bw2750-quiet/20260926T060010Z-ladder (-30.4) |
| 2750 | 15 | disconnect | 4 | 4 | 18658 | 22505 | 153341 | sweep-bw2750-quiet/20260926T060010Z-ladder (-30.4) |
| 2750 | 16 | data | 13 | 13 | 182988 | 183111 | 183375 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 | 16 | data (no payload) | 6 | 6 | 22520 | 183014 | 183102 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 | 16 | disconnect | 1 | 1 | 174063 | 174063 | 174063 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 | 17 | data | 19 | 19 | 182996 | 183118 | 360540 | N0CALL-W9SSJ-bw2750-counter65536 (silent) |
| 2750 | 17 | data (no payload) | 6 | 6 | 183018 | 183046 | 183104 | sweep-bw2750-quiet/20260926T055838Z-ladder (-32.3) |

## Mapping error and floor per tape

Residual = measured - mapped edge in samples, median [p10, p90], over keyings other than disconnect/unknown. Floor = median floor_rms (dBFS) over the leg's keyings whose floor window was not all zeros; `silent` = share of keyings whose floor window was all zeros.

| tape | pinned by | keyings | measured | on residual | off residual | floor a2b | silent a2b | floor b2a | silent b2a | session_ok |
|---|---|---|---|---|---|---|---|---|---|---|
| N0CALL-W9SSJ-bw2300-counter4096 | manifest | 29 | 29 | 2166 [472, 2467] | 518 [-102, 1425] | silent | 100% | silent | 100% | yes |
| N0CALL-W9SSJ-bw2300-counter65536 | manifest | 55 | 55 | 1628 [-335, 2567] | 397 [-34, 1342] | silent | 100% | silent | 100% | yes |
| N0CALL-W9SSJ-bw2750-counter65536 | manifest | 51 | 51 | 1646 [-440, 2503] | 542 [-3, 1351] | silent | 100% | silent | 100% | yes |
| N0CALL-W9SSJ-bw500-counter65536 | manifest | 125 | 125 | 1534 [148, 2599] | 554 [-76, 1023] | silent | 100% | silent | 100% | NO |
| W9SSJ-N0CALL-bw2300-counter4096 | manifest | 29 | 29 | 1893 [366, 2382] | 299 [-48, 1328] | silent | 100% | silent | 100% | yes |
| sweep-bw2300/20260926T050709Z-ladder | B-A +1553 | 21 | 21 | 1585 [1063, 2273] | 331 [-544, 1366] | silent | 100% | -26.4 | 27% | yes |
| sweep-bw2300/20260926T050752Z-ladder | B-A +1527 | 21 | 21 | 1450 [936, 2423] | 574 [-168, 1630] | silent | 100% | -25.4 | 27% | yes |
| sweep-bw2300/20260926T050835Z-ladder | B-A +1356 | 21 | 21 | 1844 [1306, 2364] | 404 [-582, 1330] | silent | 100% | -24.4 | 27% | yes |
| sweep-bw2300/20260926T050919Z-ladder | B-A +1552 | 21 | 21 | 1726 [593, 2247] | 401 [-574, 1323] | silent | 100% | -23.4 | 27% | yes |
| sweep-bw2300/20260926T051002Z-ladder | B-A +1277 | 21 | 21 | 1635 [1028, 2202] | 496 [-738, 1442] | silent | 100% | -22.3 | 27% | yes |
| sweep-bw2300/20260926T051045Z-ladder | B-A +1663 | 21 | 21 | 1719 [836, 2329] | 468 [-872, 1524] | silent | 100% | -21.3 | 27% | yes |
| sweep-bw2300/20260926T051128Z-ladder | B-A +1572 | 21 | 21 | 1574 [936, 2254] | 168 [-816, 1274] | silent | 100% | -20.3 | 27% | yes |
| sweep-bw2300/20260926T051211Z-ladder | B-A +1547 | 21 | 21 | 1986 [1115, 2342] | 359 [-789, 1500] | silent | 100% | -19.3 | 27% | yes |
| sweep-bw2300/20260926T051254Z-ladder | B-A +1380 | 21 | 21 | 1842 [904, 2232] | 410 [-1077, 1237] | silent | 100% | -18.3 | 27% | yes |
| sweep-bw2300/20260926T051337Z-ladder | B-A +1243 | 23 | 22 | 1940 [756, 2201] | 279 [-510, 1358] | silent | 100% | -17.4 | 16% | yes |
| sweep-bw2300/20260926T051425Z-ladder | B-A +1629 | 23 | 22 | 1231 [813, 2153] | 300 [-884, 1214] | silent | 100% | -16.4 | 16% | yes |
| sweep-bw2300/20260926T051513Z-ladder | B-A +1637 | 23 | 22 | 1688 [799, 2355] | 118 [-820, 1446] | silent | 100% | -15.4 | 16% | yes |
| sweep-bw2300/20260926T051601Z-ladder | B-A +1449 | 23 | 21 | 1458 [986, 2286] | 174 [-492, 1287] | silent | 100% | -14.4 | 16% | yes |
| sweep-bw2300/20260926T051649Z-ladder | B-A +2145 | 23 | 21 | 1832 [818, 2445] | 717 [-1219, 1695] | silent | 100% | -13.4 | 16% | yes |
| sweep-bw2300/20260926T051737Z-ladder | B-A +1429 | 23 | 21 | 1621 [883, 2277] | 257 [-799, 1278] | silent | 100% | -12.3 | 16% | yes |
| sweep-bw2300/20260926T051826Z-ladder | B-A +1644 | 33 | 32 | 1910 [954, 2809] | 590 [-1322, 1591] | silent | 100% | -11.4 | 23% | yes |
| sweep-bw2300/20260926T051938Z-ladder | B-A +1144 | 44 | 41 | 2188 [1203, 2664] | 473 [-1053, 1441] | silent | 100% | -10.3 | 26% | yes |
| sweep-bw2750-quiet/20260926T055838Z-ladder | B-A +1517 | 42 | 42 | 1048 [676, 1865] | 372 [-612, 1375] | silent | 100% | -32.3 | 42% | yes |
| sweep-bw2750-quiet/20260926T060010Z-ladder | B-A +1534 | 43 | 43 | 1106 [439, 2003] | 489 [-609, 1519] | silent | 100% | -30.4 | 40% | yes |
| sweep-bw2750-quiet/20260926T060143Z-ladder | B-A +1525 | 47 | 47 | 1166 [610, 2111] | 414 [-601, 1501] | silent | 100% | -29.4 | 41% | yes |
| sweep-bw2750-quiet/20260926T060325Z-ladder | B-A +1271 | 49 | 49 | 1216 [582, 2056] | 552 [-425, 1400] | silent | 100% | -28.4 | 32% | yes |
| sweep-bw2750-quiet/20260926T060511Z-ladder | B-A +1641 | 53 | 53 | 1142 [351, 2128] | 422 [-669, 1514] | silent | 100% | -27.4 | 33% | yes |
| sweep-bw2750-quiet/20260926T060706Z-ladder | B-A +1685 | 29 | 28 | 1065 [-192, 2108] | 496 [-834, 1557] | silent | 100% | -26.4 | 26% | NO |
| sweep-bw2750/20260926T045139Z-ladder | B-A +1968 | 20 | 20 | 1746 [747, 2362] | 305 [-814, 1578] | silent | 100% | -26.4 | 30% | yes |
| sweep-bw2750/20260926T045221Z-ladder | B-A +1507 | 21 | 21 | 1492 [778, 2277] | 207 [-862, 1358] | silent | 100% | -25.4 | 27% | yes |
| sweep-bw2750/20260926T045304Z-ladder | B-A +1301 | 21 | 21 | 1568 [1118, 2231] | 456 [-676, 1272] | silent | 100% | -24.3 | 27% | yes |
| sweep-bw2750/20260926T045347Z-ladder | B-A +1600 | 21 | 21 | 1592 [713, 2319] | 393 [-706, 1458] | silent | 100% | -23.4 | 27% | yes |
| sweep-bw2750/20260926T045430Z-ladder | B-A +1665 | 21 | 21 | 1607 [941, 2548] | 475 [-789, 1468] | silent | 100% | -22.3 | 27% | yes |
| sweep-bw2750/20260926T045513Z-ladder | B-A +1478 | 21 | 21 | 1828 [969, 2290] | 286 [-839, 1439] | silent | 100% | -21.4 | 27% | yes |
| sweep-bw2750/20260926T045556Z-ladder | B-A +1539 | 23 | 22 | 1693 [1028, 2312] | 262 [-968, 1391] | silent | 100% | -20.4 | 25% | yes |
| sweep-bw2750/20260926T045644Z-ladder | B-A +1719 | 21 | 21 | 1614 [723, 2440] | 292 [-784, 1440] | silent | 100% | -19.4 | 27% | yes |
| sweep-bw2750/20260926T045727Z-ladder | B-A +1495 | 21 | 21 | 1904 [761, 2346] | 383 [-630, 1632] | silent | 100% | -18.4 | 27% | yes |
| sweep-bw2750/20260926T045810Z-ladder | B-A +1421 | 21 | 20 | 2084 [1142, 2346] | 515 [-1132, 1322] | silent | 100% | -17.4 | 27% | yes |
| sweep-bw2750/20260926T045853Z-ladder | B-A +1524 | 23 | 22 | 1782 [516, 2362] | 422 [-755, 1457] | silent | 100% | -16.5 | 25% | yes |
| sweep-bw2750/20260926T045941Z-ladder | B-A +1270 | 23 | 22 | 1692 [1206, 2210] | 363 [-474, 1144] | silent | 100% | -15.4 | 16% | yes |
| sweep-bw2750/20260926T050029Z-ladder | B-A +1548 | 23 | 22 | 1734 [897, 2326] | 171 [-844, 1319] | silent | 100% | -14.4 | 16% | yes |
| sweep-bw2750/20260926T050116Z-ladder | B-A +1391 | 23 | 22 | 1918 [892, 2202] | 340 [-611, 1400] | silent | 100% | -13.4 | 16% | yes |
| sweep-bw2750/20260926T050204Z-ladder | B-A +1851 | 42 | 41 | 1953 [794, 2631] | 191 [-1535, 1636] | silent | 100% | -12.4 | 36% | yes |
| sweep-bw2750/20260926T050336Z-ladder | B-A +1420 | 44 | 42 | 1898 [1314, 2758] | 462 [-1900, 1420] | silent | 100% | -11.4 | 26% | yes |
| sweep-bw2750/20260926T050512Z-ladder | B-A +1777 | 53 | 51 | 1636 [1047, 2532] | 580 [-1285, 1738] | silent | 100% | -10.4 | 35% | yes |
| sweep-bw500/20260926T052115Z-ladder | B-A +1767 | 23 | 23 | 1966 [710, 2407] | 550 [-469, 1611] | silent | 100% | -26.4 | 25% | yes |
| sweep-bw500/20260926T052202Z-ladder | B-A +1814 | 23 | 23 | 1758 [818, 2273] | 322 [-748, 1399] | silent | 100% | -25.4 | 25% | yes |
| sweep-bw500/20260926T052250Z-ladder | B-A +1597 | 23 | 23 | 1622 [847, 2202] | 268 [-740, 1268] | silent | 100% | -24.4 | 25% | yes |
| sweep-bw500/20260926T052337Z-ladder | B-A +1667 | 23 | 22 | 1621 [748, 2260] | 560 [-425, 1455] | silent | 100% | -23.4 | 25% | yes |
| sweep-bw500/20260926T052424Z-ladder | B-A +1500 | 25 | 25 | 1915 [869, 2265] | 397 [-612, 1467] | silent | 100% | -22.4 | 38% | yes |
| sweep-bw500/20260926T052516Z-ladder | B-A +1475 | 25 | 25 | 1906 [1222, 2450] | 556 [-277, 1407] | silent | 100% | -21.4 | 38% | yes |
| sweep-bw500/20260926T052607Z-ladder | B-A +1720 | 29 | 29 | 1602 [909, 2216] | 670 [-712, 1407] | silent | 100% | -20.4 | 46% | yes |
| sweep-bw500/20260926T052707Z-ladder | B-A +1381 | 31 | 31 | 1650 [1048, 2187] | 362 [-655, 1197] | silent | 100% | -19.3 | 37% | yes |
| sweep-bw500/20260926T052812Z-ladder | B-A +1610 | 31 | 30 | 1528 [906, 2288] | 396 [-658, 1447] | silent | 100% | -18.4 | 31% | yes |
| sweep-bw500/20260926T052917Z-ladder | B-A +1752 | 33 | 33 | 1698 [935, 2369] | 398 [-955, 1503] | silent | 100% | -17.3 | 41% | yes |
| sweep-bw500/20260926T053026Z-ladder | B-A +1702 | 33 | 31 | 1715 [1028, 2268] | 620 [-872, 1521] | silent | 100% | -16.4 | 35% | yes |
| sweep-bw500/20260926T053136Z-ladder | B-A +1598 | 33 | 31 | 1685 [1066, 2270] | 692 [-514, 1411] | silent | 100% | -15.3 | 35% | yes |
| sweep-bw500/20260926T053246Z-ladder | B-A +1366 | 33 | 31 | 1623 [1021, 2160] | 821 [-809, 1299] | silent | 100% | -14.3 | 35% | yes |
| sweep-bw500/20260926T053357Z-ladder | B-A +1862 | 43 | 41 | 1358 [777, 2221] | 833 [-1105, 1427] | silent | 100% | -13.4 | 36% | yes |
| sweep-bw500/20260926T053528Z-ladder | B-A +1624 | 43 | 40 | 1560 [940, 2278] | 630 [-968, 1382] | silent | 100% | -12.3 | 36% | yes |
| sweep-bw500/20260926T053700Z-ladder | B-A +1399 | 43 | 42 | 1591 [961, 2250] | 860 [-916, 1398] | silent | 100% | -11.4 | 36% | yes |
| sweep-bw500/20260926T053832Z-ladder | B-A +1538 | 87 | 82 | 1994 [1203, 2657] | 388 [-1845, 1459] | silent | 100% | -10.4 | 39% | yes |
