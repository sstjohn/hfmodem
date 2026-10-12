# The 16-tone DPSK mode (MIL-STD-188-110 appendix A)

`hfmodem.bateleur.tone16` sends and receives the 16-tone differential
phase-shift keying mode of MIL-STD-188-110B appendix A, 75 to 2400 bit/s. It
was written from the appendix. No other implementation was read, and no
recording of another sender has been found to test against.

## The waveform (appendix A)

| Part | Where | Paragraphs |
|---|---|---|
| 16 data tones, 935–2585 Hz, 110 Hz apart; a 13 1/3 ms signal element (75 baud, 128 samples at 9600) at every rate | `TONE_HZ`, `T`, `_element` | A.5.1, A.5.2, table A-I |
| An unmodulated Doppler tone at 605 Hz, 7 dB over a data tone; nothing at 825 Hz, the synchronization slot | `DOPPLER_HZ`, `SLOT_HZ` | A.5.4, A.5.5 |
| Four-phase at 2400 and 1200 bit/s: (odd, even) bits (0,1) +45°, (0,0) +135°, (1,0) +225°, (1,1) +315°, MARK being 1; two-phase below, MARK +315°, SPACE +135° | `changes`, `_DPHI4`, `_DPHI2` | A.5.3, table A-II |
| In-band diversity: at 1200 tones 9–16 repeat 1–8; at 600, 300, 150 and 75 each bit on 2, 4, 8 or 16 tones | `carried` | A.5.6, table A-I |
| Preamble: 605 Hz and 1705 Hz, the latter turned 180° each element, the 605 Hz tone 7 dB up and the pair at the data's rms; then one element of every tone, the phase reference | `transmit` | A.5.4 |

Where the text leaves a choice open:

- Each tone's phase is reckoned from the start of its own element. That is how
  the published recordings showed appendix B's modems to work (see
  [TONE39.md](TONE39.md)). No recording of this mode has confirmed it.
- Of a dibit, the earlier bit is the odd bit location, as in appendix B.
- The reference element uses Newman phases `pi*k*k/16`, reducing the crest
  factor and coherent low-rate skirts. The data is in phase changes; the
  receiver measures the transmitted reference.
- The preamble is 12 elements (160 ms), within A.5.4's 5 to 32.
- The Doppler tone runs unbroken from the preamble to the end.

**Bytes.** The appendix carries a bit stream: no coding, no framing, no
end-of-message. Bytes go out least significant bit first by default. The last
element is filled out with zero bits, which arrive as up to three NUL bytes
(`offair decode --trim` drops them). The message ends when the signal does.

## The receiver

`stream.Receiver16` streams, and `tone16.receive` runs it over a recording.
The signal does not carry its rate, so both ends set it beforehand, as for
STANAG 4285.

- **Finding it.** A spectrum of a few elements looks for the Doppler tone's
  line and energy near 1705 Hz. With each element's phase reckoned from its
  own start, the 1705 Hz reversals make one line 17.5 Hz below 1705, not two
  lines 37.5 Hz either side. The reversals, read the same way, confirm it and
  give a first timing.
- **The reference element** is where the data tones' power steps up, over
  four elements either side: at 0 dB one element's power swings as far as
  the step. The tones next to 1705 Hz are left out, since its reversals,
  read across an edge, spill into them.
- **Timing.** A window read across an element's edge mixes two elements'
  phases. Over the first 24 data elements, the receiver measures how
  tightly the changes, their diversity copies added, sit on the
  constellation (its fourth power), and fits the run of windows that lie
  wholly inside an element: a box of known width, placed where the inside
  beats the outside most. Where that has no contrast, what spills into the
  places where nothing is sent (the slot, 495, 2695 and 2805 Hz) serves
  instead. With the windows inside elements, the reference is then taken
  again as the first element standing over the midpoint between the two
  power levels. Every 32 elements the receiver moves a sample away from
  whichever side spills more, which follows a sender's clock.
- **Offset.** The reversals give the offset to a multiple of 75 Hz, which
  the spectrum settles. The Doppler tone, being unbroken, then refines it and
  tracks it through the message: its own change, less its frequency's known
  turn, is what the offset turns every tone by.
- **Deciding.** Each tone's change is turned back by the Doppler tone's
  estimate. The copies of a bit under diversity are added, each weighed by
  its strength, before the decision.
- **The end** comes when the data tones, averaged over eight elements, fall
  to the level of the empty places. Decisions are held back 12 elements, and
  the message ends at the last element that still read as signal.

## How it is tested

Against itself only, through `bateleur.sim`'s channels. Every rate round
trips. Offsets reach ±41 Hz. CCIR poor (2 ms, 1 Hz) is covered. A sender's
clock off by ±100 ppm is followed over a 20-second message. Two minutes of
noise start nothing. The streaming receiver, fed in odd-sized pieces, gives
what the whole recording gives. On the station, a message sent from the host
port keys through the arbiter and decodes back, and a replayed transmission
reaches the host port.

The earlier BER table used the old per-burst-normalized fading channel and
did not give exact per-trial error counts, seeds or a source revision. Those
figures are withdrawn. The corrected-channel harness recorded for the
serial and 39-tone waveforms does not measure this waveform.

This mode is uncoded. The regression tests include one 75 bit/s message at
0 dB; that single simulation does not establish an acquisition rate.
