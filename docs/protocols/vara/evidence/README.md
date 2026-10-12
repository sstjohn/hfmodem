# VARA HF speed levels: bench evidence

This is a measurement record of the source and recordings examined for that
campaign. Implementation gaps described below refer to that source revision.
For current codec and link coverage, see the [protocol index](../README.md)
and [implementation evidence](../../../STATUS.md#vara).

The measurements the speed levels 5–17 in `01`, `03`, `04` and `06` were closed from. Every over was keyed by
high speed capable VARA HF 4.9.0 on both ends of a local audio bench and decoded by kestrel's own receivers. The
scripts that print these tables and the bench tapes they read stay in the private working tree; no audio ships.

- [`OVER-INDEX.md`](OVER-INDEX.md): the index of every over cut from the tapes.
- [`RASTER-bw2300.md`](RASTER-bw2300.md): symbol raster, carriers and timing of each BW2300 level.
- [`RASTER-bw2750.md`](RASTER-bw2750.md): the same at BW2750.
- [`RASTER-bw500.md`](RASTER-bw500.md): the same at BW500.
- [`RECORD-SEMANTICS.md`](RECORD-SEMANTICS.md): what a record row means and what a tape must supply, and [`RECORD-SEMANTICS-REVIEW.md`](RECORD-SEMANTICS-REVIEW.md), its review.
- [`KPA-bw2300-const.md`](KPA-bw2300-const.md): the BW2300 OFDM levels 10–16 closed byte-exact from known plaintext.
- [`KPA-bw2300-subband.md`](KPA-bw2300-subband.md): the BW2300 tone levels 5–9, closed the same way.
- [`KPA-bw2750-const.md`](KPA-bw2750-const.md): the BW2750 OFDM levels 11–17.
- [`KPA-bw500-const.md`](KPA-bw500-const.md): the BW500 OFDM levels 7–13 and the turbo group puncture.
- [`KPA-tone-bw2750-bw500.md`](KPA-tone-bw2750-bw500.md): the tone levels at BW2750 and BW500.
- [`LEAD-GENERATORS.md`](LEAD-GENERATORS.md): the session lead stream and its per-over draw counts.
- [`LEAD-CHECK-tone.md`](LEAD-CHECK-tone.md): the tone-family lead checked cell by cell.
- [`TAPER-WINDOW.md`](TAPER-WINDOW.md): the OFDM symbol's edge taper and receive window.
- [`HIGH-SPEED-MARKER.md`](HIGH-SPEED-MARKER.md): the two link-setup bytes that mark a high speed capable caller.
- [`ACK-LEVEL-REQUEST.md`](ACK-LEVEL-REQUEST.md): the level step each answer's frame index requests.
- [`GEARSHIFT-OBSERVED.md`](GEARSHIFT-OBSERVED.md): the gear-shift a high speed capable pair runs, clean and in noise.
