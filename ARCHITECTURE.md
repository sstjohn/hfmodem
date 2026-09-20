# Why things are where they are

Four modems, one station. This file records the decisions that are easy to
mistake for accidents, so they do not get undone.

## Three distributions

    packages/hfmodem    the station and all four protocols
    packages/hfhost     host-protocol clients — stdlib only, forever
    packages/creance    the conformance bench

`hfhost` has to run beside a commercial modem on a machine with no numpy, and its
independence from these implementations is what makes creance's grading mean
anything: a client written from the same source as the server grades nothing.
Both halves of that are enforced — `tests/gates/test_import_direction.py` against
the tree, `test_packaging.py` against the built wheel.

`creance` imports no modem, with one named exception: `creance/monitor/runners/`
holds decode adapters spawned as subprocesses. They are not on the grading path,
which goes through `hfhost.Link` over a socket. An unnamed exception would be
indistinguishable from a rule nobody enforces.

## What is shared, and what deliberately is not

`hfmodem/core/` holds what is true of the radio rather than of a waveform. The
test for admission is not "could this be shared" but **a fact that must be true in
more than one place, or code whose divergence has already cost us.**

Not shared, each for its own reason:

- **No unified `ModemCore`/`ModemObserver`.** Protocols' event vocabularies, not
  drafts of one interface. `hfhost.Link` normalises client operations; the
  modem host protocols remain distinct. Sabir uses its native framed-CBOR
  interface rather than the VARA-style two-socket command grammar.
- **No shared ARQ core.** Four state machines and four gearshift rules, the
  smallest around 600 lines and the largest past 1400. The pattern is
  documented; the code is not hoisted.
- **No shared CRC implementation.** Seven parameterisations on demodulator hot
  paths, and besra's is not a standard CRC. A `CRCSpec` record and a generic
  reference used as a test oracle — share the oracle, not the implementation.
- **No unified virtual air.** besra's `StepAir` is a synchronous test substrate;
  sabir's `LiveAir` is a threaded realtime backend creance drives over TCP. They
  share a channel-function signature and a sample-rate contract, nothing else.
- **Each mode keeps its own sample rate.** ARDOP's 12 kHz is normative to ARDOP;
  the sound card is the only shared boundary. Resampling happens at the edges.

## The station owns the radio

One process owns the CAT port, the PTT line and the sound card, and hosts
whichever protocols are enabled — each still serving its own host interface.
That is what makes contention impossible by construction rather than by operator
discipline, and it is the point: a transmitting radio mutes its own receiver, and
one modem transmitting while another recorded has already invalidated eight
on-air sessions reported as a silent band.

**No on-air session has yet run this way.** Every one has been a per-modem
process launched on its own, so the exclusion above is a property of the station
process and not of how this station has actually operated. Two modems started by
hand still contend, and the interlock is a lock inside one process — it could not
have prevented those eight sessions either.

Threads, not asyncio: a PortAudio callback cannot `await`, all four ARQ machines
are synchronous, and every timing budget here is measured against `time.monotonic()`
and sample counters.

## Station policy profiles

`core/band.py` supplies dial arithmetic. `core/regulatory/` defines the
`Profile` interface used to check configured emissions. The `part97` profile
uses band tables and station settings; `unregulated` bypasses those checks and
requires a stated reason. The configuration requires an explicit selection.

An `Emission` carries the modulator's audio passband. `core.occupied` records
the 26 dB span measured from the modulators; channel sensing, receiver filters
and profile checks read that value. A nominal bandwidth name, a 99%-power span
and a 26 dB span can give different numbers for the same signal, so the software
keeps the measurement definition attached to the value.

## Facts with exactly one home

Ratcheted in `tests/gates/test_singleton_facts.py`, keyed on the *named setting*
rather than the literal, because both numbers are overloaded:

- **the codec input gain**, `core/levels.py`. It was simultaneously 0.51, 0.118,
  0.50 and 0.18 elsewhere. `0.04` is also the FT-891's PTT settle time.
- **the dial offset**, `core/band.py`. `1500` is also five modulators' audio
  passband centre — a different fact that shares a number, and collapsing them
  would couple a waveform change to the dial convention.

## What is not here, and why the boundary is load-bearing

The record this was built from — the analysis, the oracle harnesses, the alignment
notes and the off-air capture corpus — is kept outside this tree, and nothing here
imports from any of it. Some tests read fixtures out of it and skip visibly when it
is absent, which is the section below.

The boundary carries a licensing weight as well as a tidiness one. The oracles this
project validates against — PMON, `vara.exe`, and another implementation's
ground-truth renders — cannot be redistributed, and several hundred files derived
from them are in the same position. So this tree is an allowlist of what was cleared
to leave rather than a copy with things deleted out of it: an absent module is a
decision, and a mistake at that boundary is a licensing violation.

What is here instead is the evidence a reader can re-run — the suites, four
off-air ARDOP fixtures, the VARA mail-session recording under `evidence/`,
and the normative protocol descriptions.

## Tests say when they cannot run

Never a silent deselect. Every gate produces a visible skip naming the variable
that turns it on: `HFMODEM_CORPUS`, `ARDOPCF`, `HFMODEM_AUDIO`/`HFMODEM_RIG`. A
suite that quietly omits a test looks identical to one where it passed, and this
project has been bitten by that twice — a conformance selftest returning 0 with an
entire modem's stages skipped, and a lint gate that passed for weeks by shelling
out to a tool that was not installed.
