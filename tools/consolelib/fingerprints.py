# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Three measurements that ride free on every arm, and one reader that takes them.

An arm's transcript already holds them: where on the raster the peer answered,
how long the peer's cycle ran in our own samples, and which codeword arrived
beside which. Nothing new is recorded to get them and nothing on disk is
rewritten -- `arm.log` is the record, exactly as `provenance.json` is the record
`logbook.py` reads, and this is the second reader over the same material.

`peers.json` is hand-compiled and carries a provenance; a measured block edited
by a tool would make that file two things. So the output is a derived
`fingerprints.json` beside the findings, regenerable from the corpus at any time
and authoritative about nothing the transcripts do not already say.

THE THREE READINGS, AND WHAT EACH IS MEASURED AGAINST.

`d` is the peer's answer measured from the end of our own audio, which is what
`_MasterGrid` steers the receive window by and what `--tx-offset` guesses. It is
NOT the constant: the constant is the instant on the 1.25 s raster, and on the
PACTOR-1 leg the two differ by the fixed 960 ms packet. An arm that keys a
shorter packet moves `d` and does not move the peer, which is why the census
keeps the leg beside every reading.

The clock is fitted to the peer's burst ONSETS in our stream's own samples. A
witness cannot be used for it -- one has been measured 1.65 % low and its lag
aliases on our raster -- so the only clock here is the converter that wrote the
arm's own audio, and a stream the driver spliced is dropped rather than fitted.

The codeword context is the arm's own order: what preceded a word, what followed
it, and what our packet counter did across it.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

from . import station

FS = 48000

#: Every tree an arm transcript has ever been written into. One walk covers the
#: console's own `<mode>/arms/<name>/arm.log` as well, because that lives here
#: too; the duplicates a launcher leaves in two places collapse on content.
TRANSCRIPTS = ('working', 'captures', 'logs')

#: What `d` is a gap from, in milliseconds: our PACTOR-1 data end, which
#: `_MasterGrid.rx_ref_n` pins and holds across an upgrade for exactly the reason
#: this census exists. So the instant on the raster is this plus `d` on both
#: legs, and the PACTOR-3 leg's own reading is a check on the pin rather than a
#: second measurement. `spec.P1_PACKET_S`.
REFERENCE_MS = 960.0

#: `_MasterGrid._track` prints the FILTERED turnaround and the raw error beside
#: it, so the reading that cycle actually took is recoverable. `TIMING_GAIN`.
GAIN = 0.125

ANSI = re.compile(r'\x1b\[[0-9;]*m')
HEADER = re.compile(r'== PACTOR-\d+ ([A-Z0-9]+) -> ([A-Z0-9/\-]+), centre (\d+)')
DXCALL = re.compile(r'--dxcall[= ]([A-Z0-9/\-]+)')
BAND = re.compile(r'--band[= ](\d+)')
CENTRE = re.compile(r'--center[= ](\d+)')
REPLAY = re.compile(r'--replay[= ](?!None)\S')
OUTDIR = re.compile(r'--outdir[= ](\S+)')
TRANSMITTED = re.compile(r'--transmit=True')
ANCHOR = re.compile(r'master grid: anchor @ sample (\d+), cycle ([\d.]+) s')
BUDGET = re.compile(r'holds an answer whole for d (\d+)-(\d+) ms, of which (\d+)-(\d+) can '
                    r'be read.*?TX offset ([\d.]+) assumes d (\d+) ms')
CLOCK = re.compile(r'capture clock: (\d+) samples \([\d.]+ s\), (\d+) xruns, \d+ TX '
                   r'underruns, ([+-][\d.]+) ppm')
TRACK = re.compile(r'\[grid\] d (-?[\d.]+) ms @ sample (\d+): error ([+-][\d.]+) ms')
ACQUIRE = re.compile(r'\[grid\] (?:turnaround CANDIDATE|TURNAROUND ACQUIRED) @ sample '
                     r'(\d+): d = (-?[\d.]+) ms')
ENTRY = re.compile(r'\[entry\] ENTRY ANSWER ([A-Z ]+?): (-?[\d.]+) ms after our slot '
                   r'boundary @ sample (\d+)')
# The instrument's own raster-absolute reading, and the only one in the
# transcript that is stated rather than derived: where this peer was answering
# in the cycle BEFORE our packet changed length.
BEFORE = re.compile(r'off where it answered before the entry \(([\d.]+)\)')
FLOOR = re.compile(r'(\d+) UNDER THE FLOOR, nearest ([+-][\d.]+) ms')
OUTSIDE = re.compile(r'\d+ burst\(s\) heard, nearest at ([+-]\d+) ms where an answer has '
                     r'to fall between (\d+) and (\d+)')
RASTER = re.compile(r"the peer's raster re-read on a burst nothing decoded: [\d.]+ ms at "
                    r'sample (\d+),')
KEYED = re.compile(r'TX\[\d+\] (\S+)(?: (\S+))?')
COUNTED = re.compile(r'pkt#(\d+)')
P1_PACKET = re.compile(r'TX\[\d+\] P1 pkt#')

#: What one keying of ours WAS, which is the other half of a codeword's context.
#: Only the first of these carries the packet counter the peer acknowledges, so
#: the rest are named rather than folded into a missing value.
KEYINGS = dict(P1='PACTOR-1 packet', P3='PACTOR-3 packet', SL1='PACTOR-3 SL1',
               SL2='PACTOR-3 SL2', SL3='PACTOR-3 SL3', SL4='PACTOR-3 SL4',
               CS1='CS1', CS2='CS2', CS3='CS3', CS4='CS4', CS5='CS5', CS6='CS6',
               connect='connect burst', ID='station ID')
WORD = re.compile(r'RX +[\d.]+ +(?:cs|unassigned) +(?:unassigned )?'
                  r'(CS\d|ACK|0x[0-9A-F]{3})/?([a-z0-9 \-]*?) *\((\d+) bit errors'
                  r'(?:, PACTOR-(\d), shift (normal|inverted))?')
SHIFT = re.compile(r'\[host\] rx (?:unassigned: )?(CS\d|ACK|0x[0-9A-F]{3})[^()]*'
                   r'\(\d+ bit errors, PACTOR-\d, shift (normal|inverted)\)')
UPGRADED = re.compile(r'\[host\] link upgraded to PACTOR-(\d)')
# WHAT WE ASKED FOR, which turned out to be the whole of the codeword census.
# Bits 4-5 of our announcement are the grant request; `--pactor1-only` is the
# arm that asks and then does not act on the answer.
ASKED = re.compile(r'--p1-status-bits45[= ](\d+)')
STAYED = re.compile(r'--pactor1-only=True')
CONNECTED = re.compile(r'\*\* CONNECTED')
PEER_P3 = re.compile(r'RX \(P3 changeover acquired\)|HOLD RX +[\d.]+ +packet +P3 ')

#: Words whose meaning the station has not settled, and the reason this census
#: exists. Everything else in the vocabulary is tabulated too; these two are
#: reported on their own because the grant question is asked about them.
UNSETTLED = ('0x59A', '0x6A9')


def transcripts(root=None):
    """Read arm transcripts with explicit new identity and unbound old bytes.

    Registered native/launcher mirrors share one capture; distinct old text
    never collapses merely because it names the same minute directory.
    """
    from hfmodem.core import capture_identity as ci
    from .operator import data_root
    root = Path(root or data_root())
    seen, registries = {}, {}
    for tree in TRANSCRIPTS:
        for path in sorted((root / tree).rglob("*.log")):
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            binding = ci.artifact_binding(path)
            if binding["status"] == "bound":
                if binding["kind"] != "arm":
                    continue
                role = binding["writer_role"]
                if role not in {"native-transcript", "launcher-transcript"}:
                    continue
                capture_root = next(p for p in path.parents
                                    if (p / ci.MANIFEST).exists())
                capture = ci.load(capture_root)
                if role == "launcher-transcript":
                    declared = capture.record["transcript"]
                    expected = capture.root(declared["root_role"]) / declared["path"]
                else:
                    expected = capture.root("native") / "arm.log"
                if path.absolute() != expected:
                    raise ci.IdentityError("Transcript is not its explicit registered reference")
                identity = binding["capture_id"]
                previous = registries.setdefault(identity, binding["registry_sha256"])
                if previous != binding["registry_sha256"]:
                    raise ci.IdentityError("Capture transcript registries disagree")
                key = ("bound", identity)
                preference = 0 if role == "native-transcript" else 1
            else:
                key, preference = None, 2
            try:
                text = ANSI.sub("", path.read_text(errors="replace"))
            except OSError:
                continue
            if "[grid] " not in text or not TRANSMITTED.search(text):
                continue
            if REPLAY.search(text) or "K7ABC" in text[:20000]:
                continue
            digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
            if key is None:
                key = ("legacy-unbound", digest)
            kept = seen.get(key)
            candidate = (preference, str(path))
            if kept is None or candidate < kept[3]:
                seen[key] = (path, text, digest, candidate)
    return sorted((row[:3] for row in seen.values()), key=lambda row: str(row[0]))


def _keying(line):
    """What one keying of ours was, and the packet counter it carried if any."""
    keyed = KEYED.search(line)
    if not keyed:
        return None
    head, tail = keyed.group(1).split('->')[0], keyed.group(2) or ''
    named = KEYINGS.get(head, head)
    what = ('PACTOR-3 entry' if tail == 'ENTRY' else
            f'{named} {tail}' if tail.startswith('CS') or tail == 'BREAK' else named)
    counter = COUNTED.search(line)
    return what, (int(counter.group(1)) if counter else None)


def _named(path):
    try:
        return str(path.relative_to(station.ROOT))
    except ValueError:
        return str(path)


def read(path, text, digest):
    """One arm, as the three censuses need it. Nothing here writes anything."""
    header, dx = HEADER.search(text), DXCALL.search(text)
    band, centre = BAND.search(text), CENTRE.search(text)
    anchor, budget, clock = ANCHOR.search(text), BUDGET.search(text), CLOCK.search(text)
    asked = ASKED.search(text)
    answers, onsets, leg = [], [], None
    outside, undecoded = [], []
    words, counter, order = [], None, []
    keyed_before = 'nothing yet'
    for line in text.splitlines():
        keyed = _keying(line)
        if keyed:
            keyed_before, counter = keyed
            leg = 'p3' if keyed_before.startswith('PACTOR-3') else 'p1'
            order.append(('tx', keyed_before, counter))
        for was, at, error in TRACK.findall(line):
            # The printed `d` has already been pulled an eighth of the way onto
            # this cycle's reading; the reading itself is what the census wants.
            answers.append(dict(d_ms=float(was) + (1 - GAIN) * float(error),
                                sample=int(at), leg=leg, how='tracked'))
            onsets.append(int(at))
        for at, was in ACQUIRE.findall(line):
            answers.append(dict(d_ms=float(was), sample=int(at), leg=leg, how='acquired'))
            onsets.append(int(at))
        for where, ms, at in ENTRY.findall(line):
            answers.append(dict(instant_ms=float(ms), sample=int(at), leg='p3',
                                how='entry', position=where.strip().lower()))
            onsets.append(int(at))
        for gap, low, high in OUTSIDE.findall(line):
            outside.append(dict(d_ms=float(gap), band_ms=[int(low), int(high)], leg=leg))
        for at in RASTER.findall(line):
            undecoded.append(int(at))
            onsets.append(int(at))
        for word, where, errors, level, shift in WORD.findall(line):
            words.append(dict(word=word, where=where.strip() or 'at anchor',
                              bit_errors=int(errors), pactor=int(level) if level else None,
                              shift=shift or None, leg=leg, counter=counter,
                              keyed_before=keyed_before))
            order.append(('rx', len(words) - 1, None))
        # The screen line the transcript keeps is clipped to a terminal width and
        # loses the shift; the host's own line behind it is not, and names the
        # same read. See `view.View`.
        spelled = SHIFT.search(line)
        if spelled and words and words[-1]['word'] == spelled.group(1):
            words[-1]['shift'] = spelled.group(2)
    for index, entry in enumerate(order):
        if entry[0] != 'rx':
            continue
        keyed = next((row for row in order[index + 1:] if row[0] == 'tx'), None)
        words[entry[1]].update(keyed_after=keyed[1] if keyed else 'we never keyed again',
                               counter_next=keyed[2] if keyed else None)
    span = round((float(anchor.group(2)) if anchor else 1.25) * FS)
    return dict(
        transcript=_named(path), sha256=digest,
        call=(dx.group(1) if dx else header.group(2) if header else None),
        band_m=int(band.group(1)) if band else None,
        centre_hz=int(centre.group(1)) if centre else
                  int(header.group(3)) if header else None,
        cycle_s=float(anchor.group(2)) if anchor else None,
        anchor_sample=int(anchor.group(1)) if anchor else None,
        hold_ms=[int(budget.group(1)), int(budget.group(2))] if budget else None,
        read_ms=[int(budget.group(3)), int(budget.group(4))] if budget else None,
        tx_offset_s=float(budget.group(5)) if budget else None,
        assumes_d_ms=int(budget.group(6)) if budget else None,
        capture_ppm=float(clock.group(3)) if clock else None,
        capture_xruns=int(clock.group(2)) if clock else None,
        spliced='THE STREAM IS SPLICED' in text,
        asked_bits45=int(asked.group(1)) if asked else None,
        stayed_pactor1=bool(STAYED.search(text)),
        connected=bool(CONNECTED.search(text)),
        p1_packets_keyed=len(P1_PACKET.findall(text)),
        upgraded=max((int(m) for m in UPGRADED.findall(text)), default=None),
        peer_p3=bool(PEER_P3.search(text)),
        under_floor=[float(ms) for _, ms in FLOOR.findall(text)],
        answered_before_entry_ms=sorted({float(ms) for ms in BEFORE.findall(text)}),
        answers=answers, outside=outside, undecoded=uncensored(answers, undecoded, span),
        onsets=sorted(set(onsets)), words=words)


#: How far the per-arm end-of-audio phase may scatter across the arm's own
#: decoded answers before the undecoded bursts cannot be placed against it.
PHASE_SD_MS = 10.0


def uncensored(answers, bursts, span):
    """Where the bursts NOTHING decoded fell, in the same `d` the decoded ones use.

    This is the half of the distribution the instrument cannot report as an
    answer, and the half that matters: the codeword search runs inside a band,
    so a census built only out of readings the search returned is censored by
    the window it is being used to judge. The arm's own decoded answers fix
    where our audio ended on the raster -- each is a sample and a `d` measured
    from that end -- and every other onset in the same arm is then placeable.

    An arm whose phase does not hold still is not placeable and returns nothing:
    a grid that moved mid-arm would otherwise turn into a spread of turnarounds
    that nobody transmitted.
    """
    phase = [(answer['sample'] - answer['d_ms'] * FS / 1e3) % span
             for answer in answers if 'd_ms' in answer]
    if len(phase) < 3:
        return []
    middle = statistics.median(phase)
    centred = [(value - middle + span / 2) % span - span / 2 for value in phase]
    if statistics.stdev(centred) / FS * 1e3 > PHASE_SD_MS:
        return []
    ends = (middle + statistics.fmean(centred)) % span
    return [round(((at - ends) % span) / FS * 1e3, 1) for at in bursts]


def records(root=None):
    return [read(*row) for row in transcripts(root)]


def _spread(values):
    values = sorted(values)
    return dict(n=len(values), median=round(statistics.median(values), 1),
                low=round(values[0], 1), high=round(values[-1], 1),
                sd=round(statistics.stdev(values), 1) if len(values) > 1 else None)


def answer_instants(rows):
    """Where each gateway answers on the raster, per station and per band.

    Reported as `d` because that is the quantity the receive window and
    `--tx-offset` are both spelled in, and as the raster instant beside it
    because that is the one that does not move when our packet length does.
    """
    census = {}
    for row in rows:
        if not row['call'] or not row['band_m']:
            continue
        seen = census.setdefault(f"{row['call']} {row['band_m']}m", dict(
            arms=0, legs={}, window=set(), assumes=set(), entry=[],
            undecoded=[], outside=[], stated=[], floor=[]))
        seen['arms'] += 1
        seen['undecoded'] += row['undecoded']
        seen['stated'] += row['answered_before_entry_ms']
        seen['floor'] += row['under_floor']
        seen['outside'] += [burst['d_ms'] for burst in row['outside']]
        if row['read_ms']:
            seen['window'].add(tuple(row['read_ms']))
            seen['assumes'].add(row['assumes_d_ms'])
        for answer in row['answers']:
            if answer['how'] == 'entry':
                seen['entry'].append(answer['instant_ms'])
            elif answer['leg']:
                seen['legs'].setdefault(answer['leg'], []).append(answer['d_ms'])
    out = {}
    for name, seen in census.items():
        legs = {leg: dict(_spread(found),
                          instant_ms=round(REFERENCE_MS + statistics.median(found), 1))
                for leg, found in seen['legs'].items() if found}
        window = sorted(seen['window'])
        out[name] = dict(arms=seen['arms'], d_ms=legs, window_ms=window,
                         tx_offset_assumes_d_ms=sorted(x for x in seen['assumes'] if x),
                         stated_instant_ms=_spread(seen['stated']) if seen['stated'] else None,
                         entry_instant_ms=_spread(seen['entry']) if seen['entry'] else None,
                         undecoded_d_ms=_spread(seen['undecoded']) if seen['undecoded'] else None,
                         nearest_when_none_in_band_ms=(_spread(seen['outside'])
                                                       if seen['outside'] else None),
                         under_the_floor_ms=_spread(seen['floor']) if seen['floor'] else None,
                         caught=caught(seen['legs'].get('p1'), window))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]['arms']))


def caught(found, window):
    """What share of a station's PACTOR-1 answers each flown window would read.

    The readable band, not the one that holds an answer whole: a codeword the
    search cannot finish reading before we key reaches no state machine however
    complete it arrived.
    """
    if not found or not window:
        return None
    return {f'{low}-{high} ms': round(
        100 * sum(1 for d in found if low <= d <= high) / len(found))
        for low, high in window}


#: How far a peer's burst onset may sit off the fitted line before the fit is
#: not a clock measurement. The onset detector places a burst on a 5 ms grid, so
#: a residual much past that is a reading of something else -- QRM, a repeat, or
#: a cycle index the fit guessed wrong.
ONSET_SD_MS = 8.0
CLOCK_MIN_ONSETS = 10
CLOCK_MIN_CYCLES = 20


def cycle_error(row):
    """The peer's cycle length in our own samples, as parts per million.

    The onsets are the peer's, the clock is ours, and the index each onset takes
    is read off our own cycle -- so an arm whose capture stream the driver
    spliced is refused outright rather than fitted through the splice.
    """
    onsets, cycle = row['onsets'], row['cycle_s']
    if row['spliced'] or not cycle or len(onsets) < CLOCK_MIN_ONSETS:
        return None
    span = round(cycle * FS)
    first = onsets[0]
    index = [round((at - first) / span) for at in onsets]
    if index[-1] - index[0] < CLOCK_MIN_CYCLES or len(set(index)) != len(index):
        return None
    mean_k = statistics.fmean(index)
    mean_s = statistics.fmean(onsets)
    spread = sum((k - mean_k) ** 2 for k in index)
    slope = sum((k - mean_k) * (at - mean_s) for k, at in zip(index, onsets)) / spread
    intercept = mean_s - slope * mean_k
    residual = [at - (intercept + slope * k) for k, at in zip(index, onsets)]
    sd = statistics.stdev(residual) if len(residual) > 2 else 0.0
    if sd / FS * 1e3 > ONSET_SD_MS:
        return None
    return dict(ppm=round((slope - span) / span * 1e6, 1),
                sigma_ppm=round(sd / spread ** 0.5 / span * 1e6, 1),
                onsets=len(onsets), cycles=index[-1] - index[0],
                residual_ms=round(sd / FS * 1e3, 2),
                our_capture_ppm=row['capture_ppm'])


def clocks(rows):
    """Every gateway's cycle-length error, per station, over the arms that fit."""
    census = {}
    for row in rows:
        fit = cycle_error(row) if row['call'] else None
        if fit:
            census.setdefault(row['call'], []).append(dict(fit, arm=row['transcript'],
                                                           band_m=row['band_m']))
    out = {}
    for call, fits in census.items():
        found = [fit['ppm'] for fit in fits]
        ours = [fit['our_capture_ppm'] for fit in fits if fit['our_capture_ppm'] is not None]
        out[call] = dict(_spread(found), arms=len(fits), pooled_ppm=_pool(fits),
                         our_capture_ppm=_spread(ours) if ours else None,
                         per_arm=sorted(fits, key=lambda fit: fit['arm']))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]['arms']))


def _pool(fits):
    """The station's figure, weighted by how well each arm resolved it.

    An arm's lever arm is its own -- twenty cycles and eighty resolve the same
    quantity to very different precision -- so the arms are combined on their
    own standard errors rather than counted equally. `scatter` is the check on
    that: an inverse-variance mean whose arms disagree by more than their errors
    allow is reporting a precision the corpus does not have, and says so here
    instead of in a footnote.
    """
    # A sigma that rounds to zero is below the tenth of a ppm the fits are
    # reported to, not absent; the floor keeps such an arm in the weighting
    # rather than dropping the best-resolved arm on the roster.
    weight = [1 / max(fit['sigma_ppm'], 0.05) ** 2 for fit in fits]
    mean = sum(w * fit['ppm'] for w, fit in zip(weight, fits)) / sum(weight)
    chi = (sum(w * (fit['ppm'] - mean) ** 2 for w, fit in zip(weight, fits))
           / max(len(fits) - 1, 1))
    return dict(ppm=round(mean, 1), sigma_ppm=round(sum(weight) ** -0.5, 2),
                arms=len(fits), scatter=round(chi ** 0.5, 2))


def contexts(rows):
    """What sits either side of every codeword, and what our counter did across it.

    Every word in the vocabulary, not only the unsettled two: a distribution is
    informative against the alternatives it did not take, and `0x59A`'s counter
    column says nothing until CS1's and CS3's are beside it.
    """
    vocabulary, per = Counter(), {}
    for row in rows:
        heard = row['words']
        for index, word in enumerate(heard):
            vocabulary[word['word']] += 1
            seen = per.setdefault(word['word'], dict(
                n=0, stations=Counter(), before=Counter(), after=Counter(),
                shift=Counter(), leg=Counter(), bit_errors=Counter(), keyed=Counter(),
                counter=Counter(), runs=[], arms=set(), also_p3=0, alternated=Counter()))
            seen['n'] += 1
            if index and heard[index - 1]['word'] == word['word'] and word['shift'] \
                    and heard[index - 1]['shift']:
                seen['alternated']['flipped' if heard[index - 1]['shift'] != word['shift']
                                   else 'held'] += 1
            seen['keyed'][f"{word['keyed_before']} -> {word['keyed_after']}"] += 1
            seen['stations'][row['call']] += 1
            seen['before'][heard[index - 1]['word'] if index else 'start of arm'] += 1
            seen['after'][heard[index + 1]['word']
                          if index + 1 < len(heard) else 'end of arm'] += 1
            seen['shift'][word['shift'] or 'unstated'] += 1
            seen['leg'][word['leg'] or 'unstated'] += 1
            seen['bit_errors'][word['bit_errors']] += 1
            seen['counter'][_counter_did(word)] += 1
            seen['arms'].add(row['transcript'])
            seen['also_p3'] += bool(row['peer_p3'])
        for word in {one['word'] for one in heard}:
            per[word]['runs'] += _runs(heard, word)
    return dict(vocabulary=dict(vocabulary.most_common()),
                by_announcement=announced(rows),
                words={word: _spell_context(seen) for word, seen in
                       sorted(per.items(), key=lambda kv: -kv[1]['n'])})


def announced(rows):
    """Every word against the announcement the arm that heard it went out with.

    Bits 4-5 of our own announcement are the only thing this station varies that
    a gateway could be answering, so a word's distribution across them is the one
    cut that separates what the far end volunteers from what it was asked for.
    """
    census = {}
    for row in rows:
        if not row['connected']:
            continue
        asked = f"bits45={row['asked_bits45']}" + (' and we stayed at PACTOR-1'
                                                   if row['stayed_pactor1'] else '')
        seen = census.setdefault(asked, dict(arms=0, p1_packets=0, words=Counter(),
                                             arms_with=Counter()))
        seen['arms'] += 1
        seen['p1_packets'] += row['p1_packets_keyed']
        for word in {one['word'] for one in row['words']}:
            seen['arms_with'][word] += 1
        for one in row['words']:
            seen['words'][one['word']] += 1
    return {asked: dict(seen, words=dict(seen['words'].most_common()),
                        arms_with=dict(seen['arms_with'].most_common()))
            for asked, seen in sorted(census.items())}


def _runs(heard, word):
    found, run = [], 0
    for one in heard:
        if one['word'] == word:
            run += 1
        elif run:
            found.append(run)
            run = 0
    return found + ([run] if run else [])


def _counter_did(word):
    """Our packet counter across one received word: advanced, repeated, or unsaid.

    The counter is the PACTOR-1 packet alternation and it is the peer's own
    acknowledgement made visible at our end: it advances when the far end took
    the packet and holds when it did not. A word that arrives while the counter
    advances arrived on a cycle the peer ACCEPTED.
    """
    before, after = word['counter'], word.get('counter_next')
    if before is None or after is None:
        return 'not between two counted packets'
    return 'advanced' if after != before else 'repeated'


def _spell_context(seen):
    return dict(n=seen['n'], arms=len(seen['arms']),
                arms_where_peer_p3_decoded=seen['also_p3'],
                stations=dict(seen['stations'].most_common()),
                preceded_by=dict(seen['before'].most_common()),
                followed_by=dict(seen['after'].most_common()),
                shift=dict(seen['shift'].most_common()),
                shift_across_a_repeat=dict(seen['alternated'].most_common()),
                our_leg=dict(seen['leg'].most_common()),
                bit_errors=dict(sorted(seen['bit_errors'].items())),
                we_keyed=dict(seen['keyed'].most_common()),
                our_counter=dict(seen['counter'].most_common()),
                runs=_spread(seen['runs']) if seen['runs'] else None)


def fingerprints(root=None):
    rows = records(root)
    return dict(
        corpus=dict(transcripts=len(rows),
                    attributed=sum(1 for row in rows if row['call'] and row['band_m']),
                    spliced=sum(1 for row in rows if row['spliced']),
                    answers=sum(len(row['answers']) for row in rows),
                    words=sum(len(row['words']) for row in rows)),
        answer_instant=answer_instants(rows),
        clock=clocks(rows),
        codeword_context=contexts(rows))


#: Where the derived file goes by default: beside the findings that read it, and
#: never beside `peers.json`, which is hand-compiled and carries a provenance of
#: its own. Nothing reads this back -- it is regenerated, not maintained.
from .operator import data_root
FOUND = data_root() / 'fingerprints.json'


def write(out=None, root=None):
    found = fingerprints(root)
    path = Path(out or FOUND)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, indent=2) + '\n')
    return found


def spell(found):
    corpus = found['corpus']
    yield (f"{corpus['transcripts']} transcript(s), {corpus['attributed']} attributed, "
           f"{corpus['answers']} answer(s), {corpus['words']} codeword(s).")
    yield ''
    yield 'ANSWER INSTANT -- d after our data ends, and the instant on the raster'
    for name, seen in found['answer_instant'].items():
        p1 = seen['d_ms'].get('p1')
        if not p1:
            continue
        yield (f"  {name:14} n={p1['n']:<5} median d {p1['median']:6.1f} ms "
               f"({p1['low']:.0f}-{p1['high']:.0f}, sd {p1['sd']}), "
               f"raster instant {p1['instant_ms']:.0f} ms; "
               f"window {seen['window_ms']}, offset assumes "
               f"{seen['tx_offset_assumes_d_ms']}")
    yield ''
    yield 'CLOCK -- the peer cycle against our own stream'
    for call, seen in found['clock'].items():
        pooled = seen['pooled_ppm'] or {}
        yield (f"  {call:10} {seen['arms']:3} arm(s)  median {seen['median']:+8.1f} ppm "
               f"({seen['low']:+.0f} to {seen['high']:+.0f}, sd {seen['sd']}); "
               f"pooled {pooled.get('ppm'):+.1f} +- {pooled.get('sigma_ppm')} "
               f"(scatter {pooled.get('scatter')})")
    yield ''
    yield 'CODEWORD CONTEXT -- every word against what our announcement asked for'
    for asked, seen in found['codeword_context']['by_announcement'].items():
        yield (f"  {asked:34} {seen['arms']:3} connected arm(s), "
               f"{seen['p1_packets']:5} PACTOR-1 packet(s) keyed")
        yield f"{'':38}{seen['words']}"
    yield ''
    yield 'CODEWORD CONTEXT -- what our counter did across each word'
    for word, seen in found['codeword_context']['words'].items():
        did = seen['our_counter']
        moved, held = did.get('advanced', 0), did.get('repeated', 0)
        yield (f"  {word:6} n={seen['n']:<6} {seen['arms']:3} arm(s)  "
               f"advanced {moved:<5} repeated {held:<5} "
               f"runs median {(seen['runs'] or {}).get('median', '-')}  "
               f"after {list(seen['followed_by'])[:3]}")
