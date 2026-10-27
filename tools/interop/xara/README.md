<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->

# Offline Kestrel/Xara validation

This source validation kit connects the real Kestrel host/runtime and unchanged
Xara station through generated PCM. It checks exact payloads in both directions
and orderly disconnect without a radio, sound device, CAT, socket server or VM.
Xara is an independent peer, not a VARA conformance oracle.

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
remains outside default pytest collection.

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
forward delivery, and five seconds precede DISCONNECT. `--noise` adds independent
Gaussian receiver noise but does not establish noise performance.

Bootstrap targets macOS arm64/x86_64 and Linux aarch64/x86_64. The kit does not
validate native Xara CLI/audio-device behavior, real-time CPU deadlines, RF, FM,
or every loss/retry/compression permutation. Broader release qualification remains
separate and requires both macOS and Linux. Running this kit does not qualify a
release, operate a station, publish artifacts or select another peer implicitly.
