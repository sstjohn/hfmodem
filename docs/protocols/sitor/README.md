# Lanner: RTTY, SITOR-B and NAVTEX

Lanner implements the ITA2 alphabet and character framing, CCIR 476 seven-unit
codewords, collective SITOR-B time-diversity FEC, and NAVTEX bulletin parsing.
The [code tables and primary sources](code-tables.md) describe the wire codes.

## Operator entry points

Use the console for passive reception:

```sh
tools/console rtty monitor
tools/console sitor-b monitor
tools/console navtex monitor
```

These commands use the current dial and preserve the radio's sideband/filter.
An optional `--freq KHZ` tunes once; subsequent manual radio retuning is allowed.
Tone pairs can be acquired across receive audio up to 4 kHz. See the
[monitor guide](../../console-monitor.md) for source selection and recording.

RTTY defaults to 45.45 Bd, 170 Hz shift and US-TTY figures, with unshift on
space. `--no-rtty-usos` preserves figures across spaces. SITOR-B uses 100 Bd
and 170 Hz shift. NAVTEX uses that receive path and extracts completed bulletins;
no bulletin output can mean that reception has not reached a complete header
and end marker. Broadcast assigned frequencies and USB dial frequencies differ:
the SITOR default audio centre is 1700 Hz, so a 518 kHz assigned frequency
corresponds to a 516.3 kHz USB dial.

## Scope and evidence

The library includes waveform generation for RTTY and collective SITOR-B.
This does not turn a passive monitor into a transmitter. SITOR-A ARQ,
selective Mode-B addressing and a two-way console terminal are outside these
entry points. NAVTEX parsing does not certify a bulletin's operational accuracy.

`tests/lanner/` covers the character tables, modulation/demodulation, stream
boundaries, tone/sample offsets, noise, time diversity and bulletin assembly.
These are local software tests. No named independent on-air Lanner result is
claimed in [implementation evidence](../../STATUS.md#lanner-rtty-sitor-b-and-navtex).
