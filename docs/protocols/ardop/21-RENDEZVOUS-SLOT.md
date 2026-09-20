# 21 — ARDOP rendezvous slot (receiver-source reading)

**Method.** This is a pure reading exercise against the open ARDOP receiver — no
capture, no probing. For each candidate field I traced the actual RX parse path
from demodulation to the decision that accepts or drops the frame, and read what
the code does with an out-of-range value. Where the reference C is terse I
cross-read the managed port, which is line-annotated back to the C, to confirm
the same behaviour (and to catch places where the two stacks *differ*, since a
rendezvous slot must survive both).

**Sources / licences.**
- `reference/ardopcf/src/common/` — the reference receiver (ardopcf lineage of
  ARDOP_C by Muething/Wiseman/LaRue). **MIT.** Primary citations below.
- `reference/M0LTE.Ardop/src/M0LTE.Ardop/` — a second, clearer reading of the
  same parse logic; a managed port whose doc-comments cite the C file:line.
  **AGPL-3.0.** Used only as a corroborating reading, not copied.
- `reference/ardopcf/docs/refs/ARDOP_Specification_20171127.pdf` — field semantics.

All frame-type constants: `ardopcf/src/common/ARDOPC.h:342-360`
(`IDFRAME 0x30`, `ConReq* 0x31-0x38`, `ConAck* 0x39-0x3C`, `PING 0x3E`,
`PINGACK 0x3D`).

---

## Background on the frame skeleton

Every 50-baud 4FSK control frame is `type` + `type^sessionID` carried in the
*tones* (with symbol parity), followed by a payload. The frame TYPE is what the
receiver validates — it min-distance matches the received tones against the fixed
table of *valid* types only (`SoundInput.c:2160` `ComputeDecodeDistance` over
`bytValidFrameTypes`; RXO path `RXO.c:120` `RxoMinimalDistanceFrameType`). Two
consequences that kill several "obvious" candidates up front:

- **No reserved-frame-type slot.** An undefined type byte is snapped to the
  nearest *defined* neighbour or rejected when the distance exceeds 0.30. You
  cannot smuggle a bit as a novel type value — it is either misread as a real
  frame or dropped. **Not tolerant.**
- **No free session-ID slot in ConReq.** ConReq's session ID is definitionally
  `0xFF` (`ARDOPC.c:1376`) and the tones 5-8 that carry `type^sessionID` feed the
  frame-type decode itself (`SoundInput.c:2166-2168` matches against `0xFF`).
  Perturbing it degrades the *type* decode. **Not free.**

Payloads split into two error-control regimes (`SoundInput.c:1155-1164`):
data frames get a length byte + 2 CRC bytes; **non-data control frames
(ID, ConReq, ConAck, Ping/PingAck) get `intDataLen + intRSLen` bytes and *no
CRC*.** This distinction is the whole game.

---

## Candidate A — ConAck third timing-copy byte  ★ BEST

**Location.** ConAck (`0x39-0x3C`), the answering station's first response in the
connect handshake. Geometry `intDataLen=3, intRSLen=0` (`ARDOPC.c:808-821`) — a
3-byte payload with **no RS and no CRC**. The encoder writes the measured
leader-timing byte *three times* for redundancy:
`bytreturn[2]=bytreturn[3]=bytreturn[4]=bytTiming` (`ARDOPC.c:1490-1503`,
`EncodeConACKwTiming`). Wire bytes of interest: the **3rd copy**
(`bytFrameData1[2]`), 8 bits.

**CRC coverage.** None. `intRSLen=0`, and control frames carry no CRC. The three
bytes are protected only by raw 4FSK symbol decode; frame *acceptance* does not
depend on them at all (see below).

**What the RX does with an out-of-range value.** The payload is consumed solely
by a 2-of-3 majority vote (`SoundInput.c:3059-3091`, `Decode4FSKConACK`):

```
if (bytFrameData1[0] == bytFrameData1[1])      Timing = 10 * bytFrameData1[0];
else if (bytFrameData1[0] == bytFrameData1[2]) Timing = 10 * bytFrameData1[0];
else if (bytFrameData1[1] == bytFrameData1[2]) Timing = 10 * bytFrameData1[1];
if (Timing >= 0) { *intTiming = Timing; ... return TRUE; }
```

If bytes 0 and 1 agree (the normal case), the third byte is **read but never
used** — the vote resolves on `[0]==[1]` before `[2]` is even compared. The
caller then rebuilds `bytData[0] = intTiming/10` (`SoundInput.c:3412`) and the ARQ
layer touches only that (`ARQ.c:1405`, `ARQ.c:1945`
`CalculateOptimumLeader(10 * bytData[0], …)`). The recovered value only tunes
leader length; it never gates the connection.

Robustness across both stacks: in ardopcf `Decode4FSKConACK` returns TRUE even
when all three bytes disagree (`Timing` stays 0, which is `>= 0`). The managed
port is *stricter* — all-three-disagree returns null → `Ok=false`, dropping the
ConAck (`ArdopFrameCodec.cs:193-201` `DecodeConAck`; `ArdopDemodulator.cs:1937-
1945` `Ok = timing is not null`). **Therefore the safe construction is to keep
`[0]==[1]` = true timing and place the flag in `[2]`** — that yields a clean
2-of-3 majority in *both* implementations, so both accept the frame, both recover
the correct leader timing, and both ignore `[2]`.

**Fail-safe verdict. Passes — cleanly.** Presence is a specific value in `[2]`
against a matched `[0]==[1]`. If `[2]` is corrupted on-air, a stock peer's timing
vote already resolved on `[0]==[1]` and is unaffected; a besra peer simply sees a
non-magic `[2]` and reads *absent*. Corruption can never manufacture a false
rendezvous, and it cannot drop a stock link (timing is advisory). Direction:
**answerer → caller.**

---

## Candidate B — PingAck third copy byte

**Location.** PingAck (`0x3D`), same 3-byte no-RS/no-CRC geometry
(`ARDOPC.c:808-821`), byte carries packed S:N+quality written three times
(`ARDOPC.c:1510-1524`). Decoded by the identical 2-of-3 majority
(`SoundInput.c:3095-3128` `Decode4FSKPingACK`; managed `ArdopFrameCodec.cs:218`).

**CRC coverage / RX behaviour / fail-safe.** Same as Candidate A: the 3rd copy is
ignored when the first two agree, no CRC, corruption reads as absent. **Passes.**
Ranked below A only because PingAck is a ping-response, not part of the connect
handshake — it fits "discover a peer on the air" but not "inside an ordinary
connect."

---

## Candidate C — ID-frame grid-square Packed6 (48 bits, tolerant)

**Location.** IDFrame (`0x30`), payload = callsign Packed6 (bytes 2-7) + grid
Packed6 (**bytes 8-13**) + RS(4) (`ARDOPC.c:1431-1460`, `Encode4FSKIDFrame`;
geometry `intDataLen=12, intRSLen=4` at `ARDOPC.c:788-806`). The grid is a full
6-byte / 48-bit Packed6 field.

**CRC coverage.** RS(4) covers all 12 data bytes including the grid (no CRC). So
the field is *error-corrected*: a value you choose is delivered intact, but you
must let the encoder compute RS over it — you cannot flip bits post-RS.

**What the RX does with an out-of-range value.** This is the strongly *tolerant*
path. `Decode4FSKID` (`SoundInput.c:3131-3185`) RS-corrects, decodes the callsign,
then `locator_from_bytes(&bytFrameData1[COMP_SIZE], grid)` at
`SoundInput.c:3172`. A grid that fails Maidenhead validation only **logs** and is
returned empty — it does **not** set `FrameOK=FALSE` (`SoundInput.c:3173-3175`;
contrast the callsign check at 3166-3170 which *does*). `Locator.c` reinforces
this: `locator_uncompress` maps an invalid grid to empty, and there is an
explicit *legacy tolerance* accepting two "No GS" byte sequences as unset
(`Locator.c:16-19`, `Locator.c:84-90`). So a stock RX accepts the ID regardless of
grid content and shows either the encoded Maidenhead locator or nothing.

**Fail-safe verdict. Passes.** An unrecognised grid decodes to *empty/absent*, and
the ID frame is accepted either way; corruption cannot fabricate a valid-looking
rendezvous. Ranked below A/B because the ID frame is a **beacon, not a connect** —
its association with a given connect attempt is loose, and 48 bits is far more
than a capability flag needs (over-provisioned, and a stock peer *displays* the
grid to the operator, so a non-Maidenhead payload is visible as a blank/odd grid
rather than truly invisible).

---

## Rejected — investigated, not usable

- **ConReq body (caller + target callsign).** ConReq payload is exactly two
  Packed6 callsigns (bytes 2-13) + RS(4) (`ARDOPC.c:1337-1392`). Packed6 is a
  dense 48-bit SIXBIT pack — 8 chars × 6 bits, **zero spare bits**
  (`Packed6.c`). Both callsigns are validated on decode
  (`SoundInput.c:2888-2898`, `Decode4FSKConReq`); any invalid field sets
  `FrameOK=FALSE` and fires `CANCELPENDING` (`SoundInput.c:2914-2918`). So the
  ConReq **fails safe but has no tolerant capacity** — every bit is load-bearing
  identity. This is the frame we'd most like to flag (caller → answerer
  direction), and it offers nothing: besra cannot signal
  from *inside* the ConReq without changing who the caller appears to be.
- **Callsign SSID space.** `stationid_ssid_unpack` (`StationId.c:200-212`) accepts
  SSID bytes `'0'..'?'` (0-15) **and** `'A'..'Z'`, so lettered SSIDs decode as
  "valid." Tempting as spare code-space, but the SSID is identity: changing it
  changes the callsign the peer sees and matches against `IsCallToMe`
  (`ARQ.c:541-556`). Not a free slot.
- **Frame-type reserved values / session ID.** Covered under "frame skeleton"
  above — not tolerant / not free.

---

## Summary (5 lines)

1. **Best slot: the ConAck's third leader-timing copy byte** (`bytFrameData1[2]`),
   an 8-bit field with no RS and no CRC (`ARDOPC.c:808-821`, `:1490-1503`).
2. It is read only by a 2-of-3 majority vote that resolves on `[0]==[1]` first, so
   with the true timing in `[0]==[1]` the third byte is never used
   (`SoundInput.c:3059-3091`), and the ARQ layer consumes only the recovered value
   (`ARQ.c:1405/1945`).
3. Confirmed in both stacks; keeping `[0]==[1]` gives a clean majority in the C
   *and* the stricter managed port (`ArdopFrameCodec.cs:193-201`), so neither
   drops the frame nor mis-times its leader.
4. **Fail-safe: yes.** Corruption of `[2]` reads as *absent* (stock timing already
   resolved on `[0]==[1]`; besra sees non-magic → no rendezvous); it can never
   forge a false positive or break a stock link.
5. Caveat: it signals **answerer → caller** only. The caller-side ConReq has *no*
   tolerant bits (dense, validated callsigns) — the ID-frame grid (Candidate C) is
   the fallback for caller-side signalling, at beacon-not-connect granularity.
