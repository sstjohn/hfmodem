<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->

# Offline Kestrel/Xara validation

This source validation kit connects the real Kestrel host/runtime and unchanged
Xara station through generated PCM. It checks exact payloads in both directions
and orderly disconnect without a radio, sound device, CAT, socket server or VM.
Xara is an independent peer; these checks do not establish VARA conformance.

The kit is included in the complete source release. It also runs from a Git
checkout. A wheel-only installation does not contain this source tool. Nothing
fetches Xara, installs Rust, builds the adapter or runs these checks during normal
package installation, builds, pytest or release qualification. All commands below
are explicit requests.

## Start in a clean environment

Extract the complete source release and change to its top-level directory, the
one containing `packages/` and `tools/`. Install Python 3.13 or later, Git, curl,
and a native compiler/linker: Xcode Command Line Tools on macOS, or a C build
toolchain such as build-essential on Debian/Ubuntu. First setup needs network
access to GitHub, Rust's distribution service and the Cargo registry. No station
configuration or hardware access is needed.

Create a new virtual environment and install the three matching local packages:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir ./packages/hfhost ./packages/creance ./packages/hfmodem
bash tools/interop/xara/check.sh setup
bash tools/interop/xara/check.sh session --bw 500 --caller kestrel \
  --out /tmp/hfmodem-xara-first
```

The shell entry uses `HFMODEM_INTEROP_PYTHON` when set, then this source tree's
`.venv`, then `python3`. A real Git worktree may also reuse its shared repository's
`.venv`; an extracted archive never borrows an enclosing checkout's environment.
For another virtual environment, set `HFMODEM_INTEROP_PYTHON` to its absolute
Python executable. `python3 tools/interop/xara/cli.py` is the direct entry using
that interpreter. Setup itself uses only the Python standard library. Sessions
use the installed runtime dependencies, including NumPy and SciPy.

Pytest is needed only for the explicit harness selftests:

```sh
.venv/bin/python -m pip install --no-cache-dir pytest
bash tools/interop/xara/check.sh selftest -q
```

These checks cover the sample scheduler, source/archive identity, cache and build
fingerprints, tag/hash/HEAD selection, invalid inputs and IPC deadlines/cleanup,
plus real-peer waveform and station checks. Selftest requires setup first and
remains outside default pytest collection. Channel/parser/classification checks
need neither setup nor Rust:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
nice -n 19 .venv/bin/python -m pytest tools/interop/xara/checks.py -k peer_free -q
```

## Select the peer explicitly

Default setup fetches Xara v0.1.1 commit
`2e96fcd1563c765a0f5731dae9cc9465c4b86ac6` from
<https://github.com/x0rro/xara>. It prepares a local minimal Rust 1.90.0 toolchain
and builds the offline adapter with the checked-in Cargo.lock and `--locked`.
It does not modify the shell profile or system Rust installation.

Tags, full hashes and remote HEAD remain available on request:

```sh
bash tools/interop/xara/check.sh setup --xara-ref v0.1.1
bash tools/interop/xara/check.sh setup --xara-ref 2e96fcd1563c765a0f5731dae9cc9465c4b86ac6
bash tools/interop/xara/check.sh setup --xara-ref HEAD
```

Setup resolves the request to an immutable commit, creates a separate detached
checkout, builds the adapter, and records the requested ref, resolved commit and
source URL in its receipt. Ordinary runs never fetch or follow a floating ref.
Only another explicit `setup --xara-ref HEAD` refreshes remote HEAD. Setup without
`--xara-ref` restores the tested default. A successful build activates the new
receipt atomically; a failed alternative build preserves the previous active
receipt and adapter. Alternative versions are experimental until tested.

`setup --xara-url /path/to/a/local/mirror` resolves the same explicit request
against that mirror. Existing source checkouts must be clean and match the
resolved commit. Dirty or different user checkouts are refused, never reset.
Xara retains its upstream MIT license in the fetched checkout. The harness is
AGPL-3.0-only. Neither upstream source nor prebuilt adapters are bundled with
this kit; keep upstream notices if redistributing a locally built adapter.

## Checks and evidence

Use a fresh `--out` directory on every run. The process exits nonzero on failure;
`--keep-going` records all independent cases while preserving failure status.
All commands request nice 19, set the six numerical thread caps to 1, and compile
with one Cargo job. A priority-setting failure is reported as a warning. Run one
heavy job at a time.

```sh
# All three HF bandwidths, both callers, 256 bytes each way.
bash tools/interop/xara/check.sh session --keep-going --record-audio \
  --out /tmp/hfmodem-xara-limited

# Registered TEXT, BW2300, both callers, 4096 bytes each way.
bash tools/interop/xara/check.sh session --bw 2300 --registered \
  --compression TEXT --bytes 4096 --record-audio --keep-going \
  --out /tmp/hfmodem-xara-text

# Host timing diagnostic; Xara caller at both wide bandwidths.
bash tools/interop/xara/check.sh session --bw 2300 --bw 2750 --caller xara \
  --application-gap 3 --keep-going --out /tmp/hfmodem-xara-delayed

# Independent cross-decoding at every normal HF level, plus QAM pairs.
bash tools/interop/xara/check.sh phy --pairs --out /tmp/hfmodem-xara-phy

# Optional Xara/Xara control sessions.
bash tools/interop/xara/check.sh control --out /tmp/hfmodem-xara-control
```

A session passes only on exact payload bytes both ways, no duplicate bytes after
close, and disconnect notifications from both endpoints. Results are JSON with
per-case traces; `--record-audio` adds directional PCM16 WAV recordings before
channel delay/noise. Successful PHY PCM is removed unless `--keep-audio` is set;
failed PCM remains. Retain the whole result directory when reporting a failure.

Provenance records the requested and resolved Xara version, source URL, adapter
and harness hashes, interpreter/platform, case arguments, process nice value and
thread caps. A Kestrel Git checkout additionally reports its actual commit and
working status. An archive reports `kestrel_source_kind = "archive"`, null Git
fields and a deterministic content identity instead of inventing a commit.
`kestrel_source_identity` lists the three package versions and hashes their
runtime package trees and pyproject.toml files, excluding tests and Python/test
caches. This scoped identity includes runtime assets; it is not a whole-release
archive digest or a release-qualification receipt. Harness inputs are identified
separately by `harness_sha256`. Package versions and runtime declarations must agree.

## Cache and source paths

Downloaded sources, local toolchain, generated manifests and adapter binaries go
under `${XDG_CACHE_HOME:-$HOME/.cache}/hfmodem/xara-interop/<source-id>`. The source
ID namespaces the cache by the resolved Kestrel source path, so separate kits do
not silently change each other's active peer. Nothing is written into installed
package resources. Every run validates the exact clean Xara commit and adapter
receipt against current Rust source/manifest/lock hashes and the binary digest.
Moving the kit or changing adapter build inputs requires explicit setup again.

| Variable | Purpose |
| --- | --- |
| `HFMODEM_INTEROP_ROOT` | Select another complete Kestrel source tree; imported package origins are checked |
| `HFMODEM_INTEROP_CACHE` | Override the generated-product cache |
| `HFMODEM_INTEROP_XARA` | Use a clean checkout matching the explicitly resolved peer commit |
| `HFMODEM_INTEROP_TOOLCHAIN` | Reuse a local toolchain directory containing `cargo/` and `rustup/` |
| `HFMODEM_INTEROP_PYTHON` | Select the interpreter for check.sh |

Use absolute override paths. The generated Cargo manifest and unchanged Xara
station-module inclusion use absolute paths, independent of the calling directory.
Missing source packages or preloaded modules from another source tree are refused.

## Scope

The channel is 48 kHz in 480-sample blocks with 50 ms delay each way. Kestrel's
synchronous TX occupies its sample interval and blocks its own receive cadence.
Its codec output is normalized to 0.12 RMS; Xara uses native -5 dB output drive.
Immediate application writes follow both CONNECTED reports; reverse writes follow
forward delivery, and five seconds precede DISCONNECT. `--noise` has been removed:
it fails with migration guidance. Its old receiver RMS value cannot be interpreted
as dB. Select `--snr clean`, `--snr inf`, or `--snr DB` instead.

Bootstrap targets macOS arm64/x86_64 and Linux aarch64/x86_64. The kit does not
validate native Xara CLI/audio-device behavior, real-time CPU deadlines, RF, FM,
or every loss/retry/compression permutation. Broader release qualification remains
separate and requires both macOS and Linux. Running this kit does not qualify a
release, operate a station, publish artifacts or select another peer implicitly.

## Impaired channels and the ladder

SNR2500 is signal power while keyed divided by noise power in 2500 Hz, expressed
in dB: `10 log10(Pkeyed / (N0 * 2500))`. This is the convention VARA and Winlink
figures use. The reference interval is 250–2750 Hz. Silence between transmissions
does not count as signal power. QRN is additional interference and is excluded
from the AWGN SNR. Requested SNR is before fading; the channel never raises the
noise gain to follow a fade.

For finite SNR the reference-band noise is fixed at 0.05 RMS. Keyed signal RMS is
`0.05 * 10**(SNR/20)`. `inf` keeps 0.12 RMS signal with no AWGN. Both implementations
measure keyed extent from the first through last nonzero sample. Kestrel scales
its complete rendered burst. Xara retains its native -5 dB drive inside the
unchanged station; the harness measures a native RMS table with the adapter's
data and control render commands, then applies the matching scalar before channel
processing. Calibration entries are cached within a case. CW ID uses the mean
native data-level RMS and records that fallback. PTT completion bounds fallback
calibration; its first-to-last-nonzero extent also supplies channel evidence.
Calibration errors invalidate evidence.

The impaired path mixes at 1500 Hz with analytic gain 2 and a causal image filter,
then applies Sabir's continuous Watterson taps, frequency rotation, a persistent
32-tap sinc clock interpolator, AWGN/QRN, and the SSB receive filter. There is no
AGC or clipping. Fixed 480-sample processing partitions make the realization
independent of caller chunking. Good/moderate/poor use Sabir's two equal-power
Rayleigh taps: (0, 0.5 ms)/0.1 Hz, (0, 1 ms)/0.5 Hz, (0, 2 ms)/1 Hz. Quoted Doppler
spreads are two-sided 2-sigma Gaussian widths. Tap traces always cover 3600 seconds
plus guards; delayed input history persists across blocks.

The image filter passes |f| ≤1450 Hz and stops at 1575 Hz. The receive filter
passes 60–3150 Hz within 0.1 dB and rejects below 20 Hz and above 3500 Hz by at
least 60 dB. The image filter is a 1055-tap equiripple linear-phase FIR. The
receive filter is a minimum-order elliptic bandpass IIR (18 poles, nine
second-order sections), avoiding the long delay of a narrow-transition linear-phase
receive FIR. Its group delay at 1500 Hz is about 15 samples and varies with
frequency. JSON records coefficient hashes, coefficient counts, nominal group
delay and equivalent noise bandwidth; coefficient arrays stay internal. Real
noise is filtered together with
the signal; reference gain corrects its raw variance. Separate signal, AWGN and
QRN measurements retain the wider total receive-noise power.

Nominal end-to-end delay is 2400 samples (50 ms), including filter group delay
at 1500 Hz, one 480-sample partition, and clock buffering. Propagation padding
shrinks to fit those costs. Zero ppm reserves only the 16-sample interpolator
margin. Nonzero ppm additionally reserves `ceil(abs(ppm)*1e-6*horizon*48000)+480`
samples; only when processing exceeds 2400 samples does the nominal delay grow.
JSON records the resulting delay, propagation padding and clock headroom.
The frozen inventory's longest session horizon sizes clock headroom. At 256 bytes
and zero application gap, BW500's 18.1 bps lowest normal rate gives a 399.45 s
transfer deadline and a 944.90 s worst-case horizon, including warm-up, connect,
two transfers and close observation. Scheduler rounding is included. Realized
burst windows invert the clock mapping, with the image delay before the warp and
receive delay after it. Kestrel additionally accepts
the production 4800-sample receive buffering latency. Clock error is limited to
±50 ppm and horizons to 3600 seconds. Positive ppm advances source time; positive
offset/drift/clock flags apply A→B and their negatives B→A. Frequency phase is
`2*pi*(offset*t + drift*t*t/2)`; clock-induced audio shift is separate.

QRN uses 20 Poisson arrivals/s and log-uniform 0.05–0.3 ms Gaussian slabs, with
+25 dB slab power relative to keyed signal power in the reference band after the
receive filter. Slabs overlap additively and retain boundary tails. JSON records
the event count and realized energy. Seed 20261025 is the default. A versioned
SHA-256 encoding derives separate fade, AWGN, arrival, duration and amplitude
streams from master seed, bandwidth, caller, condition, replicate and direction.
Implementation identity is excluded, so paired controls use the same environment.
Neither idle time nor packet boundaries restart it. Different ARQ timing samples
different portions of that environment.

`clean` preserves the original gains, 50 ms cable, and Kestrel receive segmentation.
It rejects other impairments. `inf` exercises the filters and production RX path
without noise. Impaired Kestrel reception uses the actual `StationTransport`
receive queue, float32, 4800-sample pieces, stream recognition, `Segmenter`, sample
cutoffs and generation resets. Its gate is unchanged. It can miss negative-SNR
data even when the demodulator could decode it.

Every Kestrel case also replays its received PCM and application commands into
a `ptt-replay` diagnostic, using known peer PTT intervals with filter support and
pre-roll. Diagnostic transmissions are discarded. It records connect, outgoing
payload confirmation, incoming payload delivery and local disconnect recovery
separately. It cannot make primary delivery pass. An RX-gate shortfall requires
recovery of the primary's failed receive phase and a passing matching control;
reverse payload delivery alone cannot explain a failed close or connect.

```sh
# One custom channel, both callers and all bandwidths.
bash tools/interop/xara/check.sh session --snr 3 --fading moderate --seed 20261025 \
  --keep-going --out /tmp/hfmodem-xara-moderate
bash tools/interop/xara/check.sh control --snr 3 --fading moderate --seed 20261025 \
  --keep-going --out /tmp/hfmodem-xara-moderate-control

# Paired default and full ladders, including independent PHY controls.
bash tools/interop/xara/check.sh ladder --ladder default --keep-going \
  --out /tmp/hfmodem-xara-ladder
bash tools/interop/xara/check.sh ladder --ladder full --keep-going \
  --out /tmp/hfmodem-xara-full

# Explicit channel flags select one custom paired condition, instead of --ladder.
bash tools/interop/xara/check.sh ladder --snr 10 --offset 50 --drift 0.1 \
  --clock-ppm 50 --qrn standard --bw 2300 --keep-going \
  --out /tmp/hfmodem-xara-custom

# Standalone impaired PHY, levels 1 and 4, with Xara-TX/Xara-RX controls.
bash tools/interop/xara/check.sh phy --level 1 --level 4 --snr 3 \
  --out /tmp/hfmodem-xara-phy-noise
```

The default ladder has 11 conditions: clean, inf, AWGN +10/+3/0/-5/-10/-15,
moderate/+3, poor/+3, and moderate/+10 with +50 Hz/+0.1 Hz/s/+50 ppm. It uses one
seed, limited mode and 256 bytes/OFF: 66 sessions and 66 controls before filters.
Default PHY probes levels 1 and 4 at clean and AWGN/+3, with directional controls.
The full ladder adds good/+3/-5/-10, moderate/-5/-10, poor/-5/-10, independent
±50 Hz, ±0.1 Hz/s and ±50 ppm at +10, plus AWGN/0 with QRN. Clean runs once;
other conditions use three independent seed replicates. Registered mode is added
only at clean, inf, AWGN/+10/+3 and good/+3. Full PHY uses every normal level,
QAM pairs, clean/inf, the AWGN waterfall and moderate/+3/-5/-10, one seed. Short
registered transfers report actual levels and do not prove sustained high speed.

A and B are fixed positions, independent of caller. A is Kestrel in sessions and
Xara in controls; B is always Xara. Both callers send A→B then B→A using identical
payloads, deadlines, channel and seeds. Warm-up is one second; connect allows 75 s;
each transfer allows `60 + 3*8*bytes/min_bps`, using the bandwidth's lowest normal
inventory rate. Five seconds precede A's disconnect, which allows 60 s. A further
five seconds observes late or duplicate delivery. IPC has a separate wall watchdog.

Exact payloads and orderly close are required. Each prefix is checked immediately.
Timeout, early close and missing close are ordinary failures; crash, IPC, harness
or calibration errors are invalid evidence. Corruption at Kestrel is unconditional;
corruption at Xara also seen at either Xara receiver in the matching control is
a peer finding; receiver and direction remain in JSON. Cells
report both pass, both fail cleanly, a shortfall, or Kestrel-only pass. Both failing
does not prove interop. Multi-seed thresholds are grouped by bandwidth/profile/mode:
the lowest rung passing every distinct caller/seed/replicate identity must be
within one rung of Xara's. Missing or duplicate identities make coverage incomplete.
Single-seed thresholds are smoke observations. Ladder exits nonzero for corruption,
invalid/incomplete evidence or a threshold shortfall. Standalone session/control
exit nonzero for any failed delivery. Stop-on-failure completes the current pair;
`--keep-going` is needed for a complete waterfall.

The inventory is printed before execution and retained in `result.json` alongside
provenance, raw traces, calibration, channel seeds, payload lengths/hashes,
transmitted/decoded highest and final levels, simulated timing and wall time.
`summary.txt` is a fixed-column table with controls beside each direction,
threshold blocks per bandwidth/profile/mode, and trace paths below the table. Connect
time ends at both CONNECTED reports; transfer goodput includes retries/turnaround.
Kestrel's synchronous `driver_seconds` is separate. Unavailable retry identities
are null with a reason, rather than inferred from repeated control codes.

Measured on an Apple Silicon Mac at nice 19: the default ladder takes about 20 minutes.
That covers 66 paired cases plus 24 PHY probes. The slowest Kestrel session, PTT replay
included, took 26 s at BW500 and 32 s at BW2300 and BW2750. The slowest Xara/Xara control
took 14 s at BW500 and 21 s at the wide bandwidths. The full ladder runs 516 paired cases,
about eight times the default. Expect around three hours, plus the PHY waterfall.
Run one heavy job at a time; all commands remain explicit and outside normal builds.
