<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Logbook and QSO rehear inspector

The logbook opens with fixed-size map dots. **Size dots by attempts** enables
bounded scaling (maximum radius 16 px); the browser remembers this choice.

**Summary** compares peers by band and observed connection success rate, with
attempt counts alongside rates. Band, protocol, bandwidth, callsign, date and
hour filters apply to both views. Hour windows include the ending hour and may
wrap midnight; choose UTC or the browser's local time zone. Date bounds use UTC.
A peer combines callsign SSIDs (for example W1AW and W1AW-10).

A success reached `connect`. An ended session without connect is a failure;
a session without teardown is pending unless it already connected. Replies
without connect count as answered, but not connected. Success rates exclude
pending sessions and imported QSOs, because imported logs generally lack failed
attempts. Each recorded session counts once; retries inside a session are not
separate attempts. Compare sample counts when interpreting peer rates.

The summary also shows unique peers by band, protocol, operating preset,
highest link speed and deepest milestone, hourly connection/failure counts,
mail totals, mean successful session duration and farthest successful contact.
**Export summary CSV** exports the filtered tables. ADIF export follows the same
filters. Static exports include the summary script.

## QSO rehear inspector

The waterfall Frequency selector switches between the full retained spectrum
and the nominal signal band centered at 1500 Hz. New artifacts retain 0–6000 Hz
to include receiver audio above 3 kHz; older artifacts show their stored extent.
This display range does not assert the radio’s filter width. Packet timing and
the dBFS scale stay fixed when zooming.


The station service includes the logbook and VARA, PACTOR and ARDOP replay inspector
on its configured port (8737 by default). The desk itself opens no radio devices.

The default data root is `HFMODEM_DATA_DIR` (or `operator-data` in the checkout).
Visit http://127.0.0.1:8737, expand a VARA, PACTOR or ARDOP QSO, and click **Inspect rehear**.

Offline exports need neither a radio nor a running station:

```sh
hfmodem console logbook --root PATH --export DIR
hfmodem console logbook --root PATH --adif imported.adi --export DIR
hfmodem console logbook --imap ACCOUNT
```

`--adif PATH` imports additional contacts and can be repeated. For live imports,
set `[desk] adif_dir = "imported-logs"` in the station file (relative to that file).
Its `.adi` files appear in the live logbook and `/api/log.adi`, the LoTW export.

Attended contacts use the same `working/*/arms/*/provenance.json` format as
mail links. VarAC saves one arm per QSO at link-up and closes it at disconnect,
with its own `receive.wav` cutout and `arm.log`. Cutouts retain the session's PCM
without resampling; transmit sample positions are rebased to the cutout clock.
Saving runs behind pending PCM on the recorder thread, away from modem and audio
callbacks. Original session recordings and journals remain intact. A failed
cutout is recorded in provenance and does not discard the contact or stop capture.
Both console and station logbooks read these arms, export VARA HF ADIF, and use
the ordinary VARA/PACTOR/ARDOP rehear and inspector paths. Incoming contacts carry
the local protocol role so TX/RX labels remain correct.

Backfill older VarAC sessions once with:

```sh
python -m hfmodem.station.contactlog --sessions /path/to/varac-session-journals
```

Backfill preserves source files and skips finished cutouts. Older journals lack
sample counters; their cutouts use UTC relative to the first host event, with the
unmeasured audio startup latency explicitly recorded as an approximation. New
contacts use callback sample positions. Incomplete contacts have no cutout.

`--imap ACCOUNT` refreshes delivery evidence once; the service never polls IMAP.
The release ships the desk without its tests. Of this directory's tests it ships
the portable ones: `test_link_state.mjs`, `test_playback.mjs`, `test_tx_timing.py`,
`test_vara_packets.py` and `test_vara_timing.py`.

Click **Rehear this QSO** to build it. The page displays queued/running status
and elapsed time, then loads the result automatically. Requests are deduplicated
per QSO and processed serially; eight requests may be pending. Errors appear in
the page. Reloading the same tab resumes status polling. Restarting the server
loses job status, but completed artifacts remain available.

The first supported view has a recording-time waterfall, selectable event
spans, actual received tone-QPSK/OFDM decision points, per-carrier selection,
latest-frame and rolling-window modes, playback speed, seeking and optional
audio. Amber edges and dimmed surroundings identify the selected waterfall
interval. Selecting an event pauses playback; **Follow playback** returns to it.

The builder can also be run directly with the same Python interpreter:

```sh
.venv/bin/python tools/logbook_view/vara_view.py \
  --root operator-data --out operator-data/replay --arm ARM_NAME
```

Use `--bw 500|2300|2750` if the logbook lacks the bandwidth, or
`--recording /file/...` to choose an exact recording URL listed for the QSO.
The primary clock and waterfall prefer rig RX, rig tap, then remote witness.
Rehear also discovers the other RX/tap/SDR recordings attached to the QSO.
Aligned sources are independently decoded and displayed side by side for the
selected packet. The preferred TX inspector uses the local monitor tap; the
preferred RX inspector uses the receiver recording. The source selector can
inspect either, and View switches between Auto, Tone map, and Spectrum.

All events have a measured tone map, RMS envelope and spectrum where audio
exists, including control/MFSK and base/index modulation. Auto displays real
unsliced decision symbols when acquired. Tone maps label their per-packet detail
gain; spectra and RMS envelopes use fixed dBFS scales. Input gain can differ
between recordings. Local TX monitor constellations describe our monitor audio,
not reception at the remote station.

## Artifact and telemetry contract

Derived data lives at `VIEWER_ROOT/ARM/FINGERPRINT/`. A manifest describes the
recording hash, resampling, channel, options, exact decoder source hashes,
loaded-module residency, parent Git revision, event schema and waterfall tiles.
Source hashes identify uncommitted instrumentation as well as committed code;
a revision alone is not claimed to describe a dirty checkout. A source/audio
change during a build refuses publication. Identical builds reuse their result.
Incomplete builds remain in hidden staging directories and are never served as
finished. QSO provenance and audio are never overwritten. Public logbook exports
omit inspector links and do not include replay artifacts.

`events.json` supplies source-local attempt IDs and intervals in seconds from
the selected recording's first sample. Each constellation is a separate JSON
file with I/Q estimates, sample times, carrier frequencies, ideal points, receive
stage and normalization. `SymbolFrame` is emitted by optional `on_symbols`
callbacks on the existing tone and OFDM decoders; the default path does not copy
symbol arrays. Failed CRC attempts retain estimates and label the level as a
hypothesis. A valid tone decode takes precedence over a failed OFDM hypothesis.
FFT analysis is batched; waterfall tiles and symbol batches are fetched on
demand and cached with fixed limits in the browser. The replay clock drives
age and expiry. A future live producer can supply the same events and batches.

## Source and direction

The page names the exact recording and whether it is rig RX, monitor tap, or
remote SDR. The local station is assumed to be the caller. For rig RX files
identified by the transcript, successful host PTT events are mapped to the
recording using nearby capture-sample counters. Caller/TX and called/RX lanes
use those intervals; handshake roles can also establish direction. Participating-QSO events outside local TX intervals belong to the peer;
observations outside the session remain separate channel observations. These are software
PTT estimates, not measured RF edges; the guard is not an accuracy guarantee.
Original TX descriptions (including explicit retries) are attached to the logged intervals, separately from
audio-decoded observations.

Short receive windows between logged TX intervals are decoded independently of
the energy gate; only CRC-clean results supplement its detections. This recovers
whole frames the gate can miss when it learns its floor during traffic. Base
index-modulated data is also decoded into CRC-graded timeline events, with the
receive-window bounds labelled as approximate rather than exact frame edges. The
page compares original logged incoming overs with replayed CRC-valid received
data frames. These counts use different units and do not claim exhaustive
coverage. Logbook mail/delivery evidence remains independent of replay.

## Current limits

- VARA symbol views cover tone levels 5–6/9/10 and OFDM levels 7–13,
  10–16, 11–17 for BW500, BW2300, BW2750 respectively. Base index/MFSK and
  control bursts have measured tone/frequency views instead of a PSK/QAM constellation. Base data recovered
  from logged receive windows is CRC-graded even without symbol telemetry.
- Burst segmentation and acquisition are imperfect. These are replay
  observations, not a complete transcript of every transmitted frame.
  Overlapping PTT, handshake and classifier evidence is grouped under its physical packet; distinct decoded frames within an over remain selectable.
- Direction outside recovered TX intervals assumes traffic belongs to the
  called station in this QSO; unrelated channel traffic is still possible.
  Replays without usable session timing may retain unattributed channel observations. Tap
  and SDR clocks are aligned by matching three or more audio envelopes, or by
  matching two or more unique CRC-valid packet payloads with independent frame
  timestamps. Repeated packets and approximate receive-window bounds cannot
  serve as decoded timing anchors. Three matching named handshake scanner locks
  can also align recordings, provided their offset cluster beats competing
  repetition aliases; segmenter edges are not used as handshake anchors. Fits that disagree by more than 80 ms are
  rejected. Additional matching payload hashes verify audio-based alignment;
  unmatched sources are excluded from synchronized comparison. No ACK/retry relationships
  are inferred; explicit original TX retry descriptions remain available.
  The original contact verdict is unchanged.
- The clock starts at the recording, not the QSO's logged start. Different
  recordings are never overlaid on an assumed common zero.
- At most 30 minutes of audio and 10,000 events per build. The builder loads
  the aligned recordings in memory; jobs run one at a time with a 30-minute wall timeout.
  Listening requires browser support for the source WAV/FLAC encoding.
- The local preview binds to loopback. There is no live audio attachment yet.

## Validation

```sh
python -m pytest -q tools/logbook_view
python -m pytest -q packages/hfmodem/hfmodem/tests/kestrel/test_vara_view_symbols.py
```

Tests cover telemetry-enabled/disabled decode equality, actual unsliced points,
carrier/time mapping, CRC failure, no acquisition, waterfall calibration,
artifact containment, and replay job serialization/deduplication/failure.
Browser checks exercise real KB8AY audio, frame selection, playback, rolling
history, per-carrier selection, and desktop/mobile layout.

The KC9GHZ and K7EK-10 examples have receiver/tap audio offsets of +22.30 s.
TX tone frames are recovered from the tap by replaying logged transmit windows,
including frames the energy gate misses. Regression tests cover calibrated tone
maps, three-window audio alignment, decoded-packet alignment and rejection of
ambiguous/repeated anchors, and TX-window symbol/CRC recovery.

VE3WLR's SDR failed envelope alignment but is registered by two named connect
requests and their response: +1.719 s, 12 ms anchor spread. `CR` is the audio
scanner's connect-request identification, while `tx connect-request-retry` is
original host-log evidence about a retry. They are shown as one packet with
supporting observations in Inspection, rather than duplicate timeline rows.

## PACTOR and link state

PACTOR uses the same offline build queue, timeline, source comparison and packet
signal views. P1 frames and control words get tone/spectrum/envelope views;
P2 single-copy decodes expose unsliced differential PSK inputs; P3 SL1–6 expose
the accepted carrier clocks and differential phase. P3's receiver uses phase
only, so its points have unit magnitude. Memory-combined CRC recovery is labelled
and does not claim that one transmission independently decoded.
<!-- site:omit:start -->
P4 is not yet supported by this adapter.
<!-- site:omit:end -->

The link panel follows playback and selection. It shows the evidence-derived
phase, current transmission, modulation/speed level and separate last TX/RX
observations. Entry packets remain entry attempts. Signal candidates cannot
establish a connection or speed level. Between observed transmissions the last
phase carries its evidence time and age, rather than claiming the link is still
alive. Level and baud describe the waveform, not delivered payload throughput.

PACTOR TX timing comes from the session's exact stream, master grid, slot numbers
and logged RF-start offsets. Hold slots logged after RF belong to that keying;
missing entry slots are interpolated only between adjacent unambiguous slots.
These remain software timing estimates. RX/tap/SDR clocks use the shared audio,
CRC-payload and named-control alignment rules. Repetitive answers with equally
plausible cycle offsets remain unaligned.

Validated real example: `pactor-current-kb5lzk-30-pounce-20260923T230205Z-robust-then-p3`:
15 locally transmitted P3 entry frames and eight peer robust answers, monitor tap
aligned +36.076 seconds. Its SDR has two equally plausible answer-cycle offsets
and is explicitly unavailable for synchronized comparison.

Additional checks:

```sh
python -m pytest -q tools/logbook_view/test_pactor_view.py
node --test tools/logbook_view/test_link_state.mjs
```

Playback opens a packet at its recorded start, labels its decode result as
hindsight, and adds constellation points by sample time. Tone maps show an audio
playhead; spectrum views remain whole-packet averages. A frame can be looped with
short margins. The listening source can be switched among aligned recordings.
See `working/the replay telemetry contract` for the incremental live contract,
including open bursts and separate capture/availability times. The current
viewer remains offline; live telemetry is not yet attached.

New PACTOR runs emit versioned `tx_event` JSON lines after confirmed transmission,
with the TX ID, recording path and host capture-sample positions. P3 waveform
records add the actual transmit path's mode, carrier count and symbol rate.
The viewer prefers these explicit positions and retains legacy grid parsing for
older QSOs. Dry-run waveform records are marked and ignored as transmit evidence.
These changes are in the viewer worktree; station processes must run that code to
produce the new records. Logging does not measure RF reception at the peer.


## ARDOP

ARDOP uses the modem's rolling receiver, with the original session ID and
memory-reset schedule once TX timing aligns the session log to the recording.
PSK and QAM views contain measured differential phases and magnitudes from each
carrier's accepted sample clock. FSK, leaders and controls retain measured tone,
spectrum and envelope views. Frames recovered with retransmission memory say so;
the plotted symbols always describe the selected transmission.

Bare controls have no payload integrity check. The inspector distinguishes them
from header candidates, failed bodies, checked control payloads and CRC-valid
data. Header timestamps exclude the preceding leader. Frame-type names provide
modulation, carrier count and symbol rate without inventing a speed-level number.

## Decoded bytes

Rehear revisions with `byte_schema: 1` retain checked decoder bytes. The Decoded
bytes panel follows the selected signal recording, offers hex/ASCII and escaped
readable text, and downloads each checked block without changing its bytes.
Selecting a packet shows its whole result; during playback bytes wait until
result availability (frame end when decoder latency was not recorded).

The traffic list follows the playhead and keeps repeated packets. Its byte count
is decoded traffic, not acknowledged goodput. VARA blocks retain protocol fields;
PACTOR payloads can contain source coding and padding; ARDOP exposes frame
payloads. This is not application-stream reassembly or decompression. CRC failures
and bare control detections do not produce validated payload bytes. Older replay
revisions need a new rehear; their waveforms and existing metadata still work.

## Exchange and packet structure

The exchange ladder defaults to evidence at the playhead. A separate whole-recording
view exposes future results for inspection. Rows select the same packet as the
waterfall, audio and byte viewer. Arrows indicate observed direction, not confirmed
receipt. Matching checked packet hashes are labelled as repeated bytes; no retry
or ACK relationship is inferred from timing alone. Pending received signals do
not expose their later-decoded identity.

The anatomy strip follows the packet clock and can seek to exported sections.
ARDOP exports header/body geometry; supported PACTOR/VARA adapters export their
known word or payload spans. Nominal spans and pre-data context are named as such.
Missing internal boundaries remain an explicitly undivided interval. Older artifacts
work unchanged; rehear to add new section metadata. Open live intervals have no
completion percentage. Recordings intentionally show known future section boundaries.
