# What marks a VARA HF link-setup as high speed capable

Corpus: the bench tapes, VARA HF high speed capable on both ends, calling as
N0CALL and as W9SSJ. Every link-setup below was decoded by
our own receiver (`rx.varahf2300.decode_over` at `BASE_LEVELS[bw]`,
`rx.varahf500.decode_burst_frames`), CRC clean, one keying per session between
the caller's connect request and `CONNECTED`. The responder printed
a high speed capability notification right after it in every session.

## Link-setup frames, high speed capable caller

| Session | bw | Caller | body[0:10] | body[7] | body[-2:] | CRC |
|---|---|---|---|---|---|---|
| 20260925-224457 (counter4096) | 2300 | N0CALL | `39b0c130c00000811409` | **81** | **06** 82 | b694 |
| 20260925-224835 (counter4096) | 2300 | W9SSJ | `5e44d3280000008114f6` | **81** | **06** 82 | 614c |
| 20260926-004204 | 2750 | N0CALL | `39b0c130c00000811409` | **81** | **06** 82 | b694 |
| 20260926-004659 | 500 | N0CALL | `39b0c130c00000811409` | **81** | **06** 82 | 2cfb |
| sweep-bw2300 050709Z, 050835Z | 2300 | N0CALL | same | **81** | **06** 82 | b694 |
| sweep-bw2750 045139Z | 2750 | N0CALL | same | **81** | **06** 82 | b694 |
| sweep-bw2750-quiet 060511Z | 2750 | N0CALL | same | **81** | **06** 82 | b694 |
| sweep-bw500 052115Z | 500 | N0CALL | same | **81** | **06** 82 | 2cfb |

Zero fill is 78 bytes at 2300/2750 and 32 at 500, as in ours.

## Against ours and against free builds

`link_setup_frame("N0CALL")` emits `39b0c130c00000 80 1409 …00… 04 82 3e0b`;
`link_setup_frame("W9SSJ")` emits `5e44d328000000 80 14f6 …00… 04 82 e9d3`.

Byte-by-byte, the high speed capable frames differ from ours in exactly two body
positions and nowhere else:

| Offset | Ours / every free build | High speed capable | Notes |
|---|---|---|---|
| body[7] | `0x80` | `0x81` | bit 0 set; same for both callsigns, all bandwidths |
| body[-2] (88 wide, 42 at 500) | `0x04` | `0x06` | the "ask" byte; `grade` byte `0x82` unchanged |
| CRC | follows | follows | recomputed over the changed body |

Rebuilding with those two bytes set (`body[7] = 0x81`, `ask=6`) and recomputing
CRC-16/GENIBUS reproduces all nine captured frames bit-exact: b694, 614c, 2cfb.

Free-build (speed limited) evidence, all `body[7] = 0x80` and ask `0x04` or lower:
`test_linksetup_frame.py` (AAAA1 at 2300; W9SSJ, W9SSJ-5, W9SSJ-10 heads; W9SSJ,
K5ABC, KB1A, W9SSJ-10, VE7QRP at 500) and all 14 frames in
`tests/kestrel/fixtures/kd7uhr-response-zero-0920/stock-setup-frames.json`
(levels 1-4, three bandwidths, W9SSJ, W9SSJ-10 and a third call). No free build
in the corpus has ever sent `0x81` or an ask above `0x04`.

## Hypotheses tested

- **Per-callsign token:** no. The callsign-dependent bytes (6-bit packed call,
  SSID, `14 <crc(call) >> 8>`) are identical to what our builder emits for both
  N0CALL and W9SSJ. The two changed bytes are the same value for both calls and
  constant across all nine sessions, three bandwidths and two callers.
- **Fixed flag:** yes, `body[7]` bit 0.
- **Level field:** yes, ask `06` where free builds cap at `04`. Doc 06 already
  records that a speed limited build caps at `BITRATE (4)`; the high speed
  capable sessions go on to `BITRATE (8)`..`(17)`.
- **Anything else:** no. See the handshake check below.

The corpus cannot say which of the two bytes the responder reads to print
a high speed capability notification: every high speed capable frame carries both changes and
every free frame carries neither. They co-vary in every frame on file.

## The other pre-connect keyings: identical to ours

Preamble-locked and demodulated off the same tapes, scored against our
generators (`vara_frames`):

| Session | Connect request (caller) | Connect response (responder) | Connected-ack tail (responder) |
|---|---|---|---|
| 2300, N0CALL calls | 31/31, position 0 | 15/15, position 17 (level 4) | 7/7 `control_bursts("N0CALL")[1]` |
| 2300, W9SSJ calls | 31/31, position 0 | 15/15, position 17 | 7/7 `control_bursts("W9SSJ")[1]` |
| 2750 | 31/31, position 0 | 15/15, position 3 | 7/7 |
| 500 | 31/31, position 0 | 15/15, position 24 | 7/7 |

A high speed capable responder answers with the same level-4 offer lattice
position a free build does, and its connected-ack is the generated one.
Nothing in the answer path marks high speed capability; no change is needed
there.

## Change to `link_setup_frame()`

To present as high speed capable for any callsign, the level-4 frame must
carry both measured differences:

```python
body[7] = 0x81          # was 0x80
body[-2] = 6            # ask; was 4 by default
```

`is_link_setup()` also checks `b[7] == 0x80` and would reject a high speed
capable peer's frame, so it needs `b[7] & 0xFE == 0x80` for a high speed
capable caller to be recognised when we answer.

Two cautions, both open:

1. Which byte is load-bearing is not separable from these captures. A four-cell
   bench run (`80/04`, `81/04`, `80/06`, `81/06`) against a high speed capable
   stock responder, reading a high speed capability notification/a speed limited link notification, settles it.
2. If the ask byte is the greeting's host level (as the builder's docstring
   reads it for free builds), asking `06` invites the called station to answer
   at a level above 4 that our receiver must then decode. Levels 1-3 of the
   high speed capable setup frame were not captured, so what `body[7]` and the
   ask hold there is unmeasured.
