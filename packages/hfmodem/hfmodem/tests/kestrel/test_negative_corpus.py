# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A labelled negative that holds traffic understates every floor measured on it.

The false-accept figures in ``vara_arq`` are all of one shape: score the whole of
some off-air audio against callsigns that are not on it, and report the best a
wrong one reaches. If real traffic is sitting in that audio, some of what the
measurement scored as a wrong callsign was right, and the floor comes out lower
than the search actually faces — by an amount nobody can recover afterwards,
because the measurement kept no record of which alignments it counted.

Which is not hypothetical: 164 s of `ONAIR_GATEWAY_GREETING` were called band
noise from 2026-08-06 until 2026-08-19, and two of those seconds are a third
station calling two gateways in turn (`corpora.ONAIR_STRANGER_REQUESTS`). No
floor rested on that stretch, so nothing had to be re-measured — but nothing had
noticed either, and a comment cannot be run.

So the negatives are declared in `corpora.QUIET_STRETCHES` and scanned here with
the monitor's own handshake scanner, against the whole published gateway panel.
Anything it names that the declaration does not is a corpus defect, not a test
failure to relax.

A check that fires on clean audio is worse than none, because the cure for a
contaminated negative is to stop trusting it — so the scanner's own false-accept
rate is measured on audio somebody else labelled empty, not on the stretches
below. Over 3.9 h in 336 recordings — the 314 of `rf-corpus` whose transcripts say
nothing was heard, the seven `neg_*` regression fixtures, the fourteen files of
the ten 80 m monitor sessions `shrike.rxfront` counts as quiet, and the listen
window kept from 7108.5 kHz — it names nothing at all. It is exact
about what it looks for: a handshake burst's every tone is fixed by its kind and
the callsign it addresses, and each of the two found here stands about twenty
tones clear of the next-best of the published panel.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_mfsk as MK

FS = MK.FS

#: How far a detection may sit from a declared burst's start and still be it. The
#: scanner reports a burst from the lattice point its plateau ranks best, which is
#: within a symbol of where the burst begins.
_SLACK_S = 0.2


def _panel() -> list[str]:
    gateways = corpora.VARA_GATEWAY_PANEL
    if not gateways.exists():
        pytest.skip("the published gateway list is not in this tree")
    panel = sorted({r["Callsign"].strip().upper()
                    for r in csv.DictReader(gateways.open(newline=""))})
    assert len(panel) > 300, f"only {len(panel)} callsigns — the list has shrunk"
    return panel


def _declared(path) -> tuple[tuple[float, float, str], ...]:
    """Traffic this recording is on record as holding."""
    if path == corpora.ONAIR_GATEWAY_GREETING:
        return corpora.ONAIR_STRANGER_REQUESTS
    return ()


@pytest.mark.parametrize("path,t0,t1,calls", corpora.QUIET_STRETCHES,
                         ids=[f"{p.parent.name}/{p.stem}"
                              for p, *_ in corpora.QUIET_STRETCHES])
def test_a_declared_quiet_stretch_holds_no_handshake_burst(path, t0, t1, calls):
    if not path.exists():
        pytest.skip(f"{path} not present")
    vm = corpora.harness("vara_monitor")
    panel = _panel()
    x = corpora.wav_mono(path)[int(t0 * FS):int(t1 * FS)]
    x = x / (np.abs(x).max() or 1.0)

    # Our own callsign and the one we were calling join the panel: KD9USW is not a
    # published gateway, so without them this station's own eight requests read as
    # "a station not on the candidate list" and the check fails on itself.
    scanner = vm.HandshakeScanner([*panel, *calls])
    ours = {c.split("-")[0] for c in calls}
    unexpected = []
    for i in range(0, len(x), vm.CHUNK):
        for start, r in scanner.push(x[i:i + vm.CHUNK]):
            at = t0 + start / FS
            if r.gateway and r.gateway.split("-")[0] in ours:
                continue
            if any(abs(at - s) <= _SLACK_S for s, _e, _c in _declared(path)):
                continue
            unexpected.append(f"{at:.3f} s {r.kind} {r.info} ({r.quality})")

    assert not unexpected, (
        f"{path.name} is declared quiet from {t0} to {t1} s and the handshake "
        f"scanner names {len(unexpected)} burst(s) in it: {unexpected}. Either the "
        f"recording carries traffic — in which case every floor measured over it is "
        f"understated and the stretch has to be re-declared — or the scanner has "
        f"started inventing them, which the corpus-wide zero in this module's "
        f"docstring would also have to stop being true for")


@corpora.requires_onair_gateway_greeting
def test_the_two_strangers_are_still_where_the_corpus_says():
    """The other direction. A negative stays honest by being scanned, and the scan
    is worth what it finds — so the two bursts that turned this recording into a
    positive as well are asserted where the corpus says they are.

    Each window is read against the whole panel, joined by the callsign the corpus
    declares: what makes these evidence is not that they match, it is that nothing
    else on the panel comes within 20 tones of either of them.

    The join is what the panel being a live roster costs. N3HYM-10 published a VARA
    gateway on 7103.5 kHz when this burst was recorded and had left the list by
    2026-08-28, and its preamble arrived 7 of 10 — a tone under the lock the scanner
    reports an unlisted station on — so its only route to a detection is the one
    that regenerates a candidate callsign, and the roster had stopped offering one.
    W9FE locks 10 of 10 and is named either way, which is why one stranger went
    quiet and the other did not. Drawing the name from the corpus rather than from
    the roster puts the specimen back under the detector it is meant to measure.
    """
    vm = corpora.harness("vara_monitor")
    panel = _panel()
    whole = corpora.wav_mono(corpora.ONAIR_GATEWAY_GREETING)
    whole = whole / (np.abs(whole).max() or 1.0)

    for at, end, call in corpora.ONAIR_STRANGER_REQUESTS:
        x = whole[int((at - 2.0) * FS):int((end + 2.0) * FS)]
        scanner = vm.HandshakeScanner(sorted({*panel, call}))
        named = [(at - 2.0 + s / FS, r) for i in range(0, len(x), vm.CHUNK)
                 for s, r in scanner.push(x[i:i + vm.CHUNK])]
        assert len(named) == 1, f"{call} at {at} s: {len(named)} detections, {named}"
        t, r = named[0]
        assert abs(t - at) <= _SLACK_S and r.gateway == call, (
            f"the burst at {at} s now reads as {r.info!r} at {t:.3f} s, not a "
            f"request to {call} — the specimen has changed, or the panel has")
