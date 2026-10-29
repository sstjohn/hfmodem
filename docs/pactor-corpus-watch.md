<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# PACTOR corpus watch

Preview the channels, then start the receive-only collector:

```sh
tools/console pactor collect --preview
tools/console pactor collect --execute
```

The receiver locator defaults to **EN63bc**, and the CAT port defaults to the
console's configured station port; no port lookup or environment setup is needed.
`--cat-port` overrides `CAT_PORT`, which overrides the console default.
The station uses a Yaesu FTDX10. The collector takes its defaults from the
selected rig profile; select `HFMODEM_RIG=ftdx10` for that station. The FT-891
profile remains supported with an external `USB Audio Device`. Override these with
`--cat-port`, `--model`, `--baud`, and `--device`. `--rigctld host:port` uses an
existing daemon while still requiring the same CAT-port ownership claim.
The old `tools/daywatch.sh` and `tools/nightwatch.sh` now launch this collector;
both accept these options and `--preview`. They no longer use static day/night
frequency lists or kill an existing rigctld.

The collector claims the same per-CAT-port lock as the transmitting tools. It
opens an input stream only, checks PTT is inactive before tuning, and checks the
USB dial and passband readback. Its own rigctld uses a private local port and
CAT PTT readback enabled (`ptt_type=RIG`); disabling it also disables the
transmit-state query. The collector's command allowlist permits only frequency,
mode and PTT-state reads plus receive tuning, never PTT writes. Shutdown stops
only that child. It does not change codec gain.

## Choosing channels

The source is Winlink's completed-session propagation feed, the same source
used by the station’s earlier channel-ranking tool. The RMS Activity Reporter website is not
scraped, and gateway heartbeats are not treated as completed sessions. The
[Winlink propagation announcement](https://winlink.org/content/new_experimental_hf_propagation_map)
describes the feed's missing-grid and gateway reporting limitations.

Ranking uses the last 24 hours of PACTOR sessions, a two-hour recency decay,
distance to the farther endpoint and generation weighting.
Reported PACTOR-4 sessions receive double weight.
Both endpoints need usable coordinates within 4200 km by default. Distance is
a heuristic, not a propagation forecast. Rankings retain supporting session
times, modes, calls, distances and filter counts. Frequencies remain **exact
centers**, never clusters spanning neighboring channels; USB dial = center −
1500 Hz. Recent traffic is evidence of a candidate, not an ongoing session.

The station's existing public-feed configuration is reused. For another
installation, set `WINLINK_API_KEY`, or pass `--feed-file saved.json` with the
feed's `PropagationRecords` object. No key is logged. `--centers 7103500 7102000`
uses an explicit list without a network request. `--min-hz`, `--max-hz`,
`--reach-km`, `--hours` and `--channels` narrow the search. Refreshes happen
between rounds, never while following an exchange. A refresh failure is
reported and saved, and the previous ranking remains in use. Initial feed
failure or an empty ranking stops before opening the radio.

## Staying or moving

The default initial dwell is 60 seconds. A fitted on-center FSK pair can buy one
15-second candidate window, but energy or shape cannot renew a hold.
A first credible P4 header gets its own window even if an earlier FSK shape expired.
P1 setup,
zero-error P1 controls, CRC-valid packets and P3 decoded headers/controls are
readable evidence. A two-tone frequency fit rejects P1 events more than 60 Hz
off center; an unresolved fit is not accepted as zero offset. VARA, ARDOP and
unidentified shapes do not renew the hold. There is no automatic retuning to
an adjacent signal.

P4 entry chirps use a separate blind two-lane header search, followed by the
existing full-field CRC decoder. Partial strong headers earn only a candidate
window. Robust P4 Chu headers remain **tentative**, even when well matched;
two distinct candidates may extend an exchange with recent readable PACTOR
evidence. Neither a spreading comb nor a header correlation is called a
decoded ACK or a working mail connection. Header searches run near nominal
frequency; they do not chase an off-channel chirp.

The default quiet release is 120 seconds without accepted evidence, preserving
more material following an upgrade. Override it with `--quiet`.
Normal P4 payloads are not blindly acquired by this watch.
A session can be cut short once it stops producing recognizable evidence.
`--max-hold 1800` imposes an
absolute limit on each visit and labels that ending as potentially incomplete.
These thresholds need field calibration; the tests establish acceptance on
specified recordings and synthetic cases, not universal RF performance.

## Artifacts and checks

Each visit starts recording **before detection**, preserving the opening call
and all audio up to its release. It writes sample-counted 48 kHz PCM16 WAVs in
ten-minute segments, `session.json`, and `events.jsonl`. Metadata includes the
center/dial, filter readback, wall time, sample counts, ADC gaps, audio status,
detector drops, detector shutdown state and every hold/release reason. The run
manifest also records collector source hashes and options. Decoder
work runs in a separate process. A dead or overloaded detector is an explicit
failure; it must not masquerade as a quiet channel. Audio captured before a
failure is retained. Before an idle hop the collector keeps recording while
the decoder finishes the proposed boundary, with a 12-second catch-up limit;
a late setup result can cancel that hop. A visit is not a promise that both
ends were audible.

No recordings are silently discarded. Expect about 346 MB/hour, including idle
visits. Collection stops at `--min-free-gb 2`, on Ctrl-C, or after `--visits N`.
Run directories are chosen automatically under
the repository’s local work-data directory, in `pactor-collect/runs/` with
a UTC timestamp. The `pactor-collect/latest.json` pointer identifies the latest run. Recordings and generated logs there are
Git-ignored. No output path is required; `--out` optionally selects a new directory.

Ctrl-C prints a capture inventory after closing the current recording and
stopping the receiver daemon. The run directory contains `inventory.txt` and
`inventory.json`, refreshed after every visit and on exit (including errors).
They list channel visits, recorded hours, WAV paths, readable PACTOR activity
and capture/analysis issues.
The inventory also lists CRC-valid P4 entry chirps and tentative P4 headers,
with P4 evidence first in the terminal.
The terminal shows up to 20 visits; the files list all visits.
Detection counts are not counts of complete sessions, and a candidate
header may later produce a CRC-valid chirp in the same recording.

For an already-running older collector, or to inspect a previous night:

```sh
tools/console pactor collect --inventory
```

This finds the latest run and rebuilds its inventory without tuning or opening
audio. It also discovers older collector runs under `captures/`. To inspect a
specific run, pass its directory after `--inventory`.

Exercise the detector and hold policy without the radio:

```sh
tools/console pactor collect --replay capture.wav --out /tmp/pactor-replay
.venv/bin/python -m pytest tools/corpuslib/test_watch.py -q
```

Replay requires mono PCM16 at 48 kHz and analyzes the entire file even after a
hypothetical hop. It reports protocol detections and hold decisions; it does
not emulate the live provisional energy screen or CAT/PortAudio timing.
