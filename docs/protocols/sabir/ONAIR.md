# Sabir radio validation

The recorded radio evidence is one-way reception of narrow-tone beacons.
No two-station Sabir ARQ exchange has been demonstrated. The DATA profiles,
capability negotiation, turn-taking and recovery are implemented and tested in
simulation; those results do not establish live-radio performance.

`hfmodem.station.air.StationAir` binds Sabir's modem execution port to station
audio, burst detection and the transmit arbiter. Tests drive that path using
recorded or synthetic audio. Hardware timing, PTT turnaround, receiver AGC,
frequency drift and repeated recovery on a real RF path still need validation.

## Beacon tools

`hfmodem.sabir.offair` renders and decodes WAV files. `presence` uses the floor
MFSK waveform and carries identity and capabilities. `wspr` is Sabir's
narrow-tone beacon format with callsign and grid; it is not a WSPR-compatible
transmission.

```
python -m hfmodem.sabir.offair presence beacon.wav --callsign YOURCALL
python -m hfmodem.sabir.offair wspr beacon-deep.wav --callsign YOURCALL --grid FN31 --gear beacon_deep
python -m hfmodem.sabir.offair decode capture.wav
```

Sabir's nominal audio center is 1500 Hz. With USB, the intended RF center is
`dial + 1500 Hz`. Carrier placement and occupied spectra are specified in
[SPEC.md](SPEC.md). A tone at the center checks the audio route; it does not
measure the occupied bandwidth, spectral skirts or suitable OFDM drive level.

`hfmodem.sabir.onair` can rehearse a beacon without opening a radio or keying:

```
python -m hfmodem.sabir.onair --beacon presence --mycall YOURCALL \
    --channel 14100000 --wav rehearsal.wav
```

Actual transmission requires `--transmit`, a selected channel or dial,
explicit audio input/output devices, and the CLI's station and regulatory
settings. The current beacon CLI requires line PTT; rigctld or serial CAT tunes
and reads back the dial. `--gain` controls audio drive. An audio level suitable
for a constant-envelope beacon must not be assumed suitable for OFDM.

Before a beacon, the CLI reads the tuned frequency, senses the channel and
applies a software keyed-time ceiling. These checks are implementation
features, not evidence that a specific installation meets its emission limits.
The station session interface is another possible transmit path, so this CLI
is not the only code capable of keying a station.

## Capturing a beacon

A receive-only WebSDR can supply a one-way capture; it cannot complete ARQ.
A recording must cover the whole beacon. Disable lossy audio coding when
possible, record the sample rate, and preserve sufficient margin to avoid
clipping. The decoder resamples the WAV to its working sample rate.

The shared `../offair` capture tool is an above-noise burst detector. Use a
continuous recording for sub-noise narrow-tone beacons. Decode memory grows
with recording duration because processing includes the full complex audio.
For long recordings, use overlapping pieces with overlap greater than the
longest beacon being searched.

`tests/sabir/test_offair.py` exercises render, frequency offset, noise,
sample-rate conversion, integer WAV storage and decode. It does not reproduce
receiver AGC, lossy WebSDR audio, adjacent-signal interference or an actual
sample-clock mismatch.

## Receive-only monitoring

`hfmodem.sabir.monitor` reports decoded beacons and session blocks. It never
transmits:

```
python -m hfmodem.sabir.monitor capture.wav
python -m hfmodem.sabir.monitor --device "USB Audio Device"
python -m hfmodem.sabir.monitor --stdin
```

The stdin format is raw signed-16-bit mono audio at 48 kHz. The live input uses
ffmpeg. Both file and stream paths use the same control decoder and an adaptive
energy-gate segmenter. The gate is suitable for above-noise traffic;
sub-noise narrow-tone beacons require the offline decoder.

## Current DATA recording proof

`hfmodem.sabir.proof` prepares bounded DATAGRAM transmissions and verifies
recordings using the normal receiver. Each trial has a distinct message ID and
known payload. Preparation writes the identified WAV, expected payload files,
a source-hash inventory and a manifest. It opens no radio or network connection.

```
python -m hfmodem.sabir.proof prepare proof-run --callsign YOURCALL \
    --profiles workhorse fast --bytes 256
python -m hfmodem.sabir.onair --proof-manifest proof-run/manifest.json \
    --mycall YOURCALL --channel 14100000
python -m hfmodem.sabir.proof verify proof-run/manifest.json receiver.wav
```

The second command is an unarmed rehearsal. Prepared transmissions are limited
to 90 seconds including Morse identification, with each packet under 25 seconds.
This tool supports the OFDM DATA profiles. The transmitter checks the waveform
hash, current source hashes, duration and station identity before accepting the
artifact. Prepare a fresh manifest after changing the transmitter.

Verification scans the recording without expected-payload timing or waveform
correlation. It checks the decoded header's profile, object identity, full
payload bytes and SHA-256. It handles WAV sample-rate conversion, including
12 kHz receiver captures. Reports record the raw capture hash, transmit sources
and decoder sources; later decoder changes can re-examine an older recording.

A successful local WAV decode is labelled offline evidence. Receiver identity,
frequency, capture readiness and independent provenance must accompany an RF
claim. The decoder itself never claims radio proof. A successful independent
recording establishes one-way DATAGRAM reception, not ARQ or live session
recovery. Noise, missing headers, truncated bodies, wrong payloads under the
correct message ID and mismatched profiles are deterministic negative tests.

## Staged radio experiment

This section records the development station's orchestration workflow. Its
`tools/console.py` launcher and automatic receiver placement do not ship in the
public distribution. For the packaged prepare, rehearse and verify commands,
use the module commands in the preceding section.

Start with `workhorse` and preserve every attempted trial. Prepare repeated
profiles to obtain independent message IDs and payloads within one bounded
burst; omit `--seed` for fresh IDs. Use separate manifests for additional
conditions. The console supports offline preparation, an explicit dry run and
offline verification:

```
python tools/console.py sabir prepare proof-run --callsign YOURCALL \
    --profiles workhorse workhorse workhorse --bytes 256
python tools/console.py sabir proof proof-run/manifest.json \
    --sense --dry-run
python tools/console.py sabir verify proof-run/manifest.json receiver.wav \
    --report verification.json
```

To select gears directly for a flight:

```
python tools/console.py sabir proof proof-run/manifest.json \
    --gears workhorse fast max --repeat 2 --sense --execute
```

This sends the sequence `workhorse, fast, max` twice within one identified burst
and records it at one receiver. Each packet has a fresh identifier and a
256-byte payload by default; `--bytes` changes the payload size. `--gears fast`
selects just one gear. Fresh preparation uses the input manifest's callsign and
saves artifacts under its sibling `waveforms/` directory without changing the
input manifest or earlier runs. `--dry-run` also saves this local preparation
but starts no receiver or transmitter. Without `--gears`, the prepared waveform
is used unchanged. Decode summaries include counts per gear.

The available OFDM gears are `robust`, `workhorse`, `workhorse34`, `fast`, `max`,
`doppler`, `sparse34` and `wide256`. A burst is limited to eight packets and
90 seconds, with at most 25 seconds per packet. Longer combinations are refused
before receiver placement. `--repeat` and `--bytes` on `proof` require `--gears`.

The console reads the rig's current dial when `--channel` is omitted and adds
1500 Hz to obtain the RF center. It queries the existing rigctld, or direct CAT
when no daemon is listening. It does not guess a frequency if CAT fails. Even a
dry run performs this read; providing `--channel RF_CENTER_HZ` keeps the preview
entirely offline. If the dial changes during receiver placement, execution
stops before launching the transmitter.

Without `--kiwi`, the recorder ranks available nearby receivers and selects one
that supplies usable audio. An explicit `--kiwi HOST:PORT` requires that receiver.
Without `--gain`, the station's shared TX drive is used: `[audio] tx_drive` from
`HFMODEM_STATION` or repository `station.toml`, or the shipped 0.6 peak default
when no drive is configured. `--gain VALUE` overrides it.
The chosen drive and actual receiver are retained in the attempt evidence;
automatic placement does not assert that a path has already decoded Sabir.

The launcher's CAT power readback is the rig's configured power setting, not a
measurement of emitted power. OFDM has a varying envelope: average RF power and
peak envelope power differ, and a meter's response affects its displayed value.
Audio drive is a peak sample level, not watts. Record measured average power or
PEP with the meter and its mode when characterizing a transmission.

The console runs an operator-controlled test without a regulatory policy gate.
It does not require a licence class or power declaration. `--power WATTS` is
optional metadata and does not set rig power. The console respects the transmit
enable in `HFMODEM_STATION` or repository `station.toml`, when present, and
checks that file for changes during receiver placement. The station file is
optional; execution always requires `--execute`. Rig readback, channel sensing,
receiver readiness, bounded keying and PTT cleanup still apply.

Only an explicit
`--execute` in place of `--dry-run` starts recording and the guarded transmitter.
The receiver must supply audio and matching metadata before transmission.
With `--sense`, an occupied-channel result stops the attempt before PTT; the
console sets no waiting budget. The launcher refusal appears in the console
error and `provenance.json`, with the measurements in `transmit.log`. `--force`
is the explicit operator override for this gate. An aborted launch runs the
PTT-down check with the same CAT/PTT ports used by the launcher.
The requested USB passband defaults to 0–3000 Hz with fixed gain 50; both are
recorded as requests, not measured receiver response. `--passband LOW HIGH`
changes the request. The console records source and artifact hashes, receiver
identity, settings, launch times and raw capture provenance. A copied local
reference WAV cannot establish radio proof.

The console displays receiver placement, channel checks, PTT events and decoding
progress, followed by exact packet counts or the reason the attempt stopped.
Detailed evidence is saved in `provenance.json`, `verification.json` and the
launcher and receiver logs. Offline `verify` prints a readable result and saves
JSON when `--report` is supplied.

Use the following initial campaign. Ten trials characterize repeatability;
they do not establish a low error-rate bound. Keep each prepared transmission
within the tool's duration limits and use fresh IDs for every trial.

| Stage | Trials | Conditions and recorded result |
|---|---:|---|
| Baseline | 10 workhorse, 256-byte payloads | One receiver and fixed drive/power/filter settings; report exact payload successes out of all ten attempts. Investigate any failure before increasing modulation order. |
| Drive/power | 10 workhorse at each of three operator-selected operating points | Hold receiver settings and payload size fixed; change one setting per point and record actual settings, clipping evidence and measured receive SNR. |
| Profiles | 10 each for selected supported profiles | Interleave workhorse reference trials in the same recording period. Qualify the passband before fast/max/wide256; report each profile separately. |
| Repeatability | Repeat the baseline at a second receiver or later period | Preserve path, time and receiver differences; do not pool unlike conditions into one success rate. |

For each condition report decode successes, decode failures and recording
failures separately. Count unique expected IDs, not duplicate received copies.
Raw receiver noise before/after transmission and a manifest from a different
trial supply independent negative controls. Every selected profile is the
current implementation; these trials do not compare Sabir versions.

1. Rehearse the prepared WAV through the ordinary offline decoder, including
   a sample-rate-converted copy. Verify payload bytes and all selected IDs.
2. Qualify the actual transmit drive and receive path for workhorse: avoid
   clipping, preserve the nominal 1500 Hz center and record filter settings.
   A center tone alone cannot qualify the data waveform's spectral skirts.
3. At an authorized operating point, collect repeated independent trials.
   Retain failed attempts and complete raw recordings. Report decoded/attempted
   counts per profile, payload size and condition, with recording failures
   distinguished from byte decode failures. Record identification, headers,
   padding and all gaps when reporting delivered-byte throughput.
4. Repeat at multiple explicitly chosen transmit powers or receive SNR
   conditions, holding the payload, profile and receiver settings fixed where
   possible. Record the noise bandwidth and measurement method; receiver
   automatic gain or file normalization must not be treated as calibrated SNR.
   No particular channel, drive, power or receiver is selected by this procedure.
5. Test wider profiles only after qualifying the complete radio passband.
   `wide256` needs substantially more low-frequency response than workhorse;
   a 300 Hz high-pass can remove useful carriers. Requested passband settings
   do not establish the physical filter response. Preserve an ordinary-profile
   trial at comparable conditions as a reference.

Decode a receiver-only interval and a wrong-manifest capture as negative
controls. Offline truncated-header/body and wrong-ID tests provide additional
checks without extra transmissions. Keep raw recordings, manifests, expected
payloads, transmitter and decoder source hashes, sidecars and decode reports
together so later receiver changes can be assessed against the same evidence.

This experiment demonstrates current one-way DATA reception. A WebSDR cannot
acknowledge, negotiate, request retransmission or exercise turn-taking. Live
ARQ qualification requires a second Sabir station and complete exchanges,
including asymmetric loss, dropped controls, retries, duplicate suppression,
disconnect and recovery. Simulation coverage and one-way captures do not
substitute for those live exchanges.
