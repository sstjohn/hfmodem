<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# What marks a VARA HF link-setup as high speed capable

A VARA HF link-setup says whether the calling station is high speed capable.
Every link-setup below was decoded by our own receiver
(`rx.varahf2300.decode_over` at `BASE_LEVELS[bw]`,
`rx.varahf500.decode_burst_frames`), CRC clean.

## Link-setup frames from high speed capable callers

| bw | Caller | body[0:10] | body[7] | body[-2:] | CRC |
|---|---|---|---|---|---|
| 2300 | N0CALL | `39b0c130c00000811409` | **81** | **06** 82 | b694 |
| 2300 | W9SSJ | `5e44d3280000008114f6` | **81** | **06** 82 | 614c |
| 2750 | N0CALL | `39b0c130c00000811409` | **81** | **06** 82 | b694 |
| 500 | N0CALL | `39b0c130c00000811409` | **81** | **06** 82 | 2cfb |

Zero fill is 78 bytes at 2300/2750 and 32 at 500.

## Against speed limited link-setups

A speed limited `link_setup_frame("N0CALL")` is `39b0c130c00000 80 1409 …00… 04 82 3e0b`;
`link_setup_frame("W9SSJ")` is `5e44d328000000 80 14f6 …00… 04 82 e9d3`.

The two forms differ in exactly two body positions:

| Offset | Speed limited | High speed capable | Notes |
|---|---|---|---|
| body[7] | `0x80` | `0x81` | bit 0 set; same for both callsigns, all bandwidths |
| body[-2] (88 wide, 42 at 500) | `0x04` | `0x06` | the "ask" byte; `grade` byte `0x82` unchanged |
| CRC | follows | follows | recomputed over the changed body |

Setting those two bytes (`body[7] = 0x81`, `ask=6`) and recomputing
CRC-16/GENIBUS reproduces every frame in the table bit-exact: b694, 614c, 2cfb.

Speed limited link-setups all carry `body[7] = 0x80` and an ask of `0x04` or
lower: `test_linksetup_frame.py` and the 14 frames in
`tests/kestrel/fixtures/kd7uhr-response-zero-0920/peer-setup-frames.json`
(levels 1-4, three bandwidths).

The callsign-dependent bytes (6-bit packed call, SSID, `14 <crc(call) >> 8>`)
are the same in both forms. Which of the two changed bytes a responder reads is
not separable: they always change together. The connect request, connect
response and connected-ack around the link-setup are the same in both forms.

## In this implementation

`link_setup_frame(..., high_speed=True)` sets both bytes, and
`link_setup_marks()` reads them back from a caller's frame. `is_link_setup()`
accepts either form.

Whether this station sends the high speed capable form is the operator's
setting: `high_speed` in `[protocols.vara]` (default `true`), or
`--low-speed-only` on the VARA console and launchers. A speed limited station
keys the `0x80`/`04` form and holds speed levels 1 to 4.

At levels 1-3 the high speed capable link-setup has not been observed, so what
`body[7]` and the ask hold there is unmeasured.
