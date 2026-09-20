# Evidence tiers

Every substantive claim in these documents carries a tag saying what stands
behind it. A reader should never have to guess whether a number came from a
published Recommendation, from a stranger's transmission, or from this station
on one evening — and should be able to tell those apart at a glance, because
they fail in different ways and are worth different amounts.

## The three tiers

**[P] — primary.** An off-air recording of somebody else's transmission, read by
an instrument that has been validated against something other than itself. The
station being measured had no idea it was being measured and no stake in the
result, which is what makes this the strongest tier available here. It is also
the scarcest: nobody can commission a recording of a protocol they are trying to
learn.

**[S] — secondary.** A published document, an independent implementation, or a
decoder written by somebody else. ITU-R Recommendation M.1798, the SCS protocol
papers, the 1990 PACTOR-1 description by DL6MAA and DF4KV, and Sailer HB9JNX's
GPL hf-pactor are the sources that carry this tier in the PACTOR documents. A
secondary source can be wrong, and where two of them disagree these documents
say so rather than picking a winner.

**[T] — tertiary.** This station's own sessions, its own transmissions, and
inferences drawn from them. A tertiary claim is a report about one station's
equipment, one path and one set of choices. It is not worthless — it is often
the only thing there is — but it cannot settle a question about the protocol,
and nothing in these documents should rest on it alone without saying so.

Most claims carry more than one tier, and the combination is the point:
`[P+S]` is a measurement that agrees with a published value, which is as good as
it gets here. `[T]` alone is a note to the next person, not a specification.

## A measurement tag states its population

`[P]` and `[T]` are qualified by how much they rest on, in the smallest honest
form:

    [P: 4 stations, 2 bands]      four independent transmitters
    [P: 1 clip, 79 s]             one recording of one session
    [T: 1 station, 2026-08-26]    this station, one evening

The count is not decoration. Four stations across two bands and a single
79-second clip were both written `[MEASURED]` in earlier revisions of these
documents, and that tag was retired because it made them look alike. A claim
that cannot state its population is not yet a measurement.

An **absence** — "no station has ever done X" — carries the same burden from the
other side, and additionally needs the sensitivity it was measured at. A zero
taken at a bar that could not have detected the thing is not a negative result;
it is an instrument saying nothing. Where these documents state a negative they
state what would have had to be there to see it.

## What "validated" means for the instrument

A primary measurement is only as good as the receiver that took it. The PACTOR-3
receive path that produced the `[P]` figures in `pactor3.md` was checked against
a third party's transmission before it was used as an instrument: 27 CRC-valid
fields recovered from a recording of a session neither end of which was ours,
agreeing with an independently written decoder, against 141,571 null trials at
zero accepts. `pactor2.md` §7.7 and §7.8 are the model for saying where an
instrument stops: a recording that does not demodulate is not evidence, and one
clip is a binding limit on everything derived from it.

## Where the tags are used

`pactor2.md`, `pactor3.md`, `pactor-connect-frames.md` and
`pactor-capability.md` tag their section headings and individual claims.
`pactor1-control-signals.md` states its sources inline instead, naming the 1990
description, ITU-R M.1798 and hf-pactor at each point and recording where they
disagree; that is the same discipline written out in prose, and it is the better
form where three sources have to be weighed in the same sentence.
