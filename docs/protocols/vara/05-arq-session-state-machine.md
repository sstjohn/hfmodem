# 05 — ARQ & session state machine

For current DATA-answer detection and lost-ACK/NAK recovery, see
[21 — DATA answer recovery](21-data-answer-recovery.md), including controlled
BW2300 and BW2750 missing-answer versus missing-DATA experiments.


The VARA HF v4.9.0 link layer: the session states, the transitions between them,
the connect and disconnect handshakes, the stop-and-wait data exchange, and the
timing each side keeps. The **initiator** opens the session and sends DATA overs;
the **responder** answers and sends ACK overs.

Timings, block sizes and host messages below describe a BW500 `P2P SESSION` link
unless the row says otherwise; §5.3.3 and the on-air turnaround row of §5.5 are
BW2300. Fields that ride inside the waveform and never reach the host interface
are marked where this document does not fix them.

## 5.1 States

| State (label)   | Meaning                          | Observable signature |
|-----------------|----------------------------------|----------------------|
| DISCONNECTED    | Idle; `LISTEN OFF`, not in a link | no PTT; `DISCONNECTED`/no `CONNECTED` outstanding |
| LISTENING       | Responder armed, awaiting inbound connect | after host `LISTEN ON`; `PENDING` on detect |
| CONNECTING      | Handshake in progress (connect-req/resp/setup bursts) | `BUSY ON`; connect-req/resp bursts; no `CONNECTED` yet |
| CONNECTED       | Link up; alternating ISS (TX turn) / IRS (RX turn) sub-roles | `CONNECTED src dst bw`; stop-and-wait over exchange; idle keepalive |
| DISCONNECTING   | Closing after `DISCONNECT` | one 32-symbol disconnect burst, unanswered (§5.6); `BUSY`-then-`BUSY OFF` |

> Within CONNECTED, ARQ alternates an **ISS** (information-sending station, holds
> the TX turn = keys DATA overs) and an **IRS** (keys short ACK overs). This is a
> role, not a separate host-visible state; it follows the strict DATA-over ⇄
> ACK-over alternation of §5.4.

## 5.2 Transitions

| From | Event / trigger | To | Emitted frame / host msg |
|------|-----------------|----|--------------------------|
| DISCONNECTED | host `LISTEN ON` (responder) | LISTENING | — |
| DISCONNECTED | host `CONNECT src dst` (initiator) | CONNECTING | keys connect-request burst (~1.73 s); `BUSY ON` |
| LISTENING | decodes connect-request | CONNECTING | `PENDING`, then keys connect-response burst (~1.0 s) |
| CONNECTING | handshake completes | CONNECTED | `CONNECTED src dst bw` (responder ~0.6 s before initiator) |
| CONNECTED (idle) | ~10–12 s idle timer | CONNECTED | on-air keepalive exchange (short + ~1.4 s burst) |
| CONNECTED (ISS) | host writes data | CONNECTED | `BUFFER n`; keys DATA over; peer keys ACK; `BUFFER` decrements |
| CONNECTED | host `DISCONNECT` (after `BUFFER 0`) | DISCONNECTING | keys the 32-symbol disconnect burst (~1.365 s), once |
| DISCONNECTING | disconnect handshake completes | DISCONNECTED | `DISCONNECTED` (both sides), `BUSY OFF` |

## 5.3 Handshake

Connect sequence; `CONNECT` → responder `CONNECTED` takes ≈ **7.4 s**:

| Step | Description | Burst (keyed by / dur) | Host msgs straddling |
|------|-------------|------------------------|----------------------|
| 1 | Initiator connect-request / acquisition | initiator / **1.73 s** (1.698–1.774) | after `CONNECT`; `BUSY ON` both |
| 2 | Responder connect-response | responder / **1.0 s** (0.992–1.003) | responder: `PENDING` before keying |
| 3 | (both) registration / encryption exchange | — | `REGISTERED`, `ENCRYPTION DISABLED` both |
| 4 | Initiator link-setup over | initiator / **4.31 s** | initiator: `UNENCRYPTED LINK`, `BITRATE (4) 88 bps TX` |
| 5 | Responder connected-ack ([`04 §4.2C`](04-frame-block-formats.md)) | responder / **0.446–0.492 s**, 11 two-tone symbols | responder: `SN`, `CONNECTED …`, `LINK UNREGISTERED` |
| 6 | Initiator confirms | initiator / ~0.70 s | initiator: `CONNECTED …`; then `IAMALIVE` both |

Frame *contents* of the connect bursts (callsigns, BW request, negotiated level)
ride the CONTROL waveform ([`02 §2.6`](02-preamble-sync.md)) and are **not**
host-visible; reading them requires demodulating the control burst.

### 5.3.1 Connect-response (step 2) — what the originator decodes

The step-2 connect-response burst is specified in full — waveform, standalone
predictor and recognizer — in [`04 §4.4.5`](04-frame-block-formats.md). Its role
in the handshake:

- The connect-response is a **23-tone MFSK burst (~1.005 s)**: 8 fixed preamble tones +
  **15 payload tones that are a deterministic function of the CALLED (answering) station’s
  callsign ALONE** (caller-invariant). Same PRNG-MFSK family
  as the CR, different seed (`seed=(crc+289)&0x7FFF`, pre-advance `mult+511`).
- **What the originator decodes to confirm acceptance:** the originator dialed a known
  `called` callsign, so it recomputes the expected 15 payload tones (`resp_payload_tones(called)`)
  and matches them against the demodulated response burst. A match means "my CR to `<called>` was
  accepted" and advances the CONNECTING state toward `CONNECTED`. The originator’s
  own callsign is **not** carried in the connect-response; it rides the later initiator
  link-setup over (step 4).
- **Discrimination:** the correct dialed callsign matches **15/15** payload tones;
  any other callsign matches **≤2/15**, with no false accept anywhere in a 12×12
  callsign matrix.

Step 4 is §5.3.2 and [`04 §4.4.6`](04-frame-block-formats.md); step 5 is
[`04 §4.2C`](04-frame-block-formats.md).

### 5.3.2 Link-setup (step 4) — the OFDM data frame that carries the caller

Full detail and closed forms in [`04 §4.4.6`](04-frame-block-formats.md).

- **Step 5 (responder connected-ack).** The burst a responder keys as it emits
  `CONNECTED` is **11 two-tone symbols (0.479 s) carrying no callsign**
  ([`04 §4.2C`](04-frame-block-formats.md)).

- **Step 4 (initiator link-setup) — the caller callsign rides the ordinary BW2300 rec3
  wideband OFDM burst.** The over reported as `BITRATE (4) 175 bps TX` is the same waveform,
  coding chain and 92-byte container as a DATA over: 371 data emission columns of 512 samples,
  one-hot over a 16-bin span, turbo-FEC coded, **keyed to the CALLER callsign**. It is the only
  burst carrying the caller's identity — the CR and response are keyed to the CALLED
  callsign only, and the connected-ack is keyed to nothing.
  - **How the responder learns the caller.** Recover the frame through
    the rec3 DATA chain → de-whiten (the BW2300 whitener PN) → **CRC-16/GENIBUS** → caller frame; the
    caller callsign is a 6-bit-packed field at the frame start, with any SSID in `body[5]`
    (see [`04 §4.2A`](04-frame-block-formats.md), [`03 §3.5.3`](03-coding.md)).
  - Every initiator transmission occupies 735–2350 Hz; the session carries no
    narrowband emission ([`04 §4.2A`](04-frame-block-formats.md)).

### 5.3.3 Post-connect session-setup bursts

Between the link-setup over and the first DATA over the initiator sends a **trio
of short bursts** (~0.70 s, ~1.38 s, ~1.38 s) and the responder answers each.
These are **not** OFDM data frames: they are further members of the **PRNG-MFSK
handshake family** of §4.2, keyed to the **called** callsign, and predicted
**byte-exact from the callsign alone**.

**Physical layer.** One tone per **2048-sample symbol** (23.4375 Hz grid), tone index in
the same `29 + parity + 14P + 2D` space as the CR and connect-response
bursts (33..98). Frame sizes: initiator **short = 16 tones**, initiator
**long = 32 tones** (sent twice, different content), responder **= 24 tones**. Each
frame opens with a **fixed preamble tone** — `74` for the initiator long frames, `62` for
the short and responder frames (`62` is also the first preamble tone of
`CONNECT_RESPONSE`/`CONNECTED_ACK`, §4.2.2) — followed by callsign-keyed payload tones.
There is no FEC and no interleaver in these frames: the tone sequence is drawn
straight from the VB6 `Rnd` LCG.

**Generators.** Identical law to §4.2.3 with per-burst constants;
`parity(k) = par0 if k even else 1-par0`:

| burst | when | preamble tone | payload tones | SEED_OFF | PREADV | par0 |
|---|---|---|---|---|---|---|
| session-confirm | step 6, initiator, ~0.70 s | 62 | 15 | 1551 | 91 | 1 |
| session-keepalive-a | initiator, ~1.38 s | 74 | 31 | 1550 | 187 | 1 |
| session-keepalive-b | initiator, ~1.38 s (2nd) | 74 | 31 | 60 | 1241 | 1 |
| session-drained | initiator, once its send queue drains | 74 | 31 | 60 | 807 | 1 |
| session-drained (responder) | responder, same occasion | 74 | 31 | 288 | 807 | 1 |
| responder answer | after each of the above | — | — | *is the ordinary connect-response burst* | | |

```
crc  = CRC-16/GENIBUS(called_callsign_ASCII)
seed = (crc + SEED_OFF) & 0x7FFF
mult = G(cs) + ((crc + 50) >> 15)                # same G and +50 carry as §4.2.3
s    = start(seed); advance (mult + PREADV) draws
per payload tone k:  P = floor(Rnd*5), D = floor(Rnd*7)
                     tone = 29 + parity(k) + 14P + 2D
```

Each `(SEED_OFF, PREADV)` pair is unique: of all 32768 seed offsets exactly one
is consistent across callsigns. From the callsign alone, with no other input, the
generators reproduce the frames exactly — **16/16, 32/32, 32/32 and 23/23 tones**
across five independent called callsigns.

**The responder's answer is not a new frame at all — it is the ordinary connect-response
burst (§5.3.1) re-sent**, 23/23 tones exact for all five callsigns. kestrel already
generates it.

**The 1.38 s frame is VARIABLE, not fixed.** The generators above hold for an
**idle** link. With data flowing from a gateway, the initiator answers every DATA over
with a 1.38 s / 32-tone burst whose content **changes per over**. In one live gateway
session the eleven 1.38 s bursts transmitted were four mutually distinct frames during
the data phase (t = 8.1, 17.4, 23.7, 30.0 s), then seven byte-identical frames once the
exchange went quiet (t >= 32.5 s). The burst at t = 17.4 s matched `SESSION_KEEPALIVE_A`
**32/32 exactly**.

So this burst is the initiator's **per-over ARQ response**, carrying session state, and the
"keepalive" generators pin the value it takes when that state is idle — one point of a
larger space, not the whole frame. `SESSION_KEEPALIVE_A/B` remain correct for the idle case
(20/20 byte-exact across five callsigns) and are what an idle link repeats. What the varying
field encodes — sequence, ACK/NAK, or requested speed level — is not fixed by this document.

**One more point of that space is now fixed: `session-drained`.** In two full
bidirectional real-VARA ⇄ real-VARA BW2300 sessions with different callsign pairs
(W9SSJ/W1AW, K5ABC/N0DX), 126 bytes each way, each station keys its own row above
on the turnaround after its host reports `BUFFER 0` — the initiator's twice, the
second time on the turnaround immediately before the peer begins transmitting, and
tone-identical to the first. It is therefore **not a grant answering a request**: the
same frame goes out when nobody has asked for anything.

What the recordings put around it is the peer's short answer arriving **late** —
0.811 s after the holder's burst ended, against 0.02–0.12 s on every other
turnaround in the session — and the peer's first DATA over following 0.029 s after
the holder's next burst. So the turn changes hands on the holder announcing an empty
queue, not on a request/grant pair, and the peer's late answer is the only carrier
of "I want to send" that these recordings expose. Its encoding is in the
**11-symbol two-tone control burst** (§4.2C), whose state symbols remain unspecified
at BW2300: none of the three BW500 tails matches any of them.

The two rows are singletons only across the two sessions — either alone leaves 11
`(SEED_OFF, PREADV)` pairs standing. Read as keyed to the sender or to the peer, no
pair is consistent across both; keyed to the **called** station, as every session
frame here is but the two turn frames, each is unique. What they mean beyond "this
station has nothing more queued" is not fixed by this document.

> **Implementation note.** An implementation that sends only `SESSION_KEEPALIVE_A/B`
> tracks an idle link and nothing else. That is a stopgap covering the idle case; it
> is not what VARA sends during a data exchange, and it cannot follow one.

**One response frame is callsign-INDEPENDENT (BW2300).** Across 17 recorded sessions
(182 responses, 36 distinct frames) a single 32-tone frame was sent **18 times to three
different gateways** and matches no callsign-keyed generator (<9/32 against all of them).
It is therefore pure session state, and it is the frame an idle-but-connected link
repeats — recorded as `SESSION_RESPONSE_2300` in `kestrel/vara/vara_frames.py`.

At **BW2750 the picture differs**: there the most-repeated frame of each session is
specific to that session's called callsign (four sessions, repeated 56, 17, 14 and 9
times, each its own), so callsign-independence is **not** established for 2750 and no
constant is recorded for it. Whether that is a real BW2750/BW2300 difference or follows
from which sessions carried data is unresolved.

The varying responses are **not** the same callsign-seeded LCG stream read at a different
advance. Every advance 0..9000 of the `seed=(crc+1550)&0x7FFF` stream recovers the idle
frame exactly (31/31 at `mult+187`) and leaves the four data-phase frames at 6/31, i.e.
chance. Whatever varies changes the seed itself, or the frames are not PRNG-drawn at all.

**Interop consequence.** These bursts need no new coding layer: they are three more
`BurstKind`s over the existing MFSK machinery (`SESSION_CONFIRM`, `SESSION_KEEPALIVE_A`,
`SESSION_KEEPALIVE_B` in `kestrel/vara/vara_frames.py`, regression-tested in
`tests/kestrel/test_session_bursts.py`). A byte-exact drop-in emits the
confirm after CONNECTED and the keepalives while idle, and recognises the peer's by
regenerating them for the called callsign — the same mechanism as the CR/response.

### 5.3.4 The responder's answer to an idle frame (BW2300)

The row above — *the responder's answer is the ordinary connect-response burst* — holds
for the confirm and the keepalives. It does **not** hold once the initiator has taken the
turn. Across two off-air BW2300 sessions with live Winlink gateways (W9SSJ calling NS0A
and KC9GHZ, 2026-07-24/25), the gateway answered the initiator's idle frame ten times —
five in each session, 0.112–0.150 s after the initiator's burst ended — with a further
member of the PRNG-MFSK family, keyed to the **called** callsign:

| burst | when | preamble tone | payload tones | SEED_OFF | PREADV | par0 |
|---|---|---|---|---|---|---|
| session-idle-response | responder, ~1.38 s | 74 | 31 | 288 | 1241 | 1 |

Each occurrence's 31 payload tones admit exactly one 24-bit generator state, and
`(288, 1241)` is the only offset/pre-advance pair that produces both sessions' bursts
from their two different callsigns. The seeding map is many-to-one — a pair fitted to a
single recording is not evidence — so the cross-callsign intersection is the whole
identification. The frame is bit-identical at all ten occurrences and carries no
variable field.

**What it means is not fixed by this document.** It answered the initiator's idle frame
and nothing else in either recording; in particular it did not answer either session's
turn-request, so it is not established as a turn grant. The three turnarounds it does not
account for (the first of the NS0A session, the first two of the KC9GHZ one) hold
gateway-strength bursts on the same symbol grid that are members of neither this family
nor the two-tone control burst of §4.2C, and occur once each. No logged VARA-to-VARA
session holds this frame: a responder's transmit path over 479 s answers keepalives,
turn-requests and overs with the two-tone control burst instead, so the frame may be
specific to BW2300, to a gateway, or to the turn-held state.

### 5.3.5 The session at BW500 — the same frames on the narrower alphabet

**A session frame's `(SEED_OFF, PREADV)` pair is the state and its alphabet is the
bandwidth's.** Every post-connect frame of §5.3.3 and §5.3.4 regenerates at BW500
from the descriptor it already has, with the 14-carrier alphabet of
[`04 §4.2.2`](04-frame-block-formats.md) substituted and nothing else changed —
same preamble, same `keyed_by`, same 2048-sample symbol grid. Only the handshake
pair carries a genuinely different descriptor there, and §4.2.2 already gives it.

Four stock BW500 sessions, two callsign pairs, four callsigns: the two loopbacks
of 2026-07-13 (`AAAA1` → `BBBB2`), the answered pair of 2026-08-30 and a 2026-09-02
bench run against a stock responder (both `W9SSJ` → `W1AW`). **41 keyings of twelve
frame kinds, 1233 of 1233 carriers exact**, every burst scored whole — preamble and
payload — against both alphabets and both callsigns:

| frame | keyed to | keyings | carriers |
|---|---|---|---|
| connect-request, BW500 | called | 3 | 41/41 |
| connect-response, BW500 | called | 3 | 23/23 |
| session-confirm | called | 3 | 16/16 |
| session-keepalive-a | called | 2 | 32/32 |
| session-keepalive-b | called | 6 | 32/32 |
| session-turn-request | **caller** | 2 | 32/32 |
| session-turn-request (responder) | **caller** | 11 | 32/32 |
| session-drained | called | 1 | 32/32 |
| session-drained (responder) | called | 1 | 32/32 |
| session-idle-response | called | 7 | 32/32 |
| session-turn-release | called | 1 | 17/17 |
| session-disconnect-final | called | 1 | 16/16 |

The BW2300 alphabet is the control on the same audio: it takes each burst's shared
fixed preamble and then **0 to 3 payload carriers of 15 to 31**, which is chance.
No BW500 handshake burst can emit an odd carrier, so a station reading these frames
on the wide alphabet is deaf to every one of them.

Two further 32-symbol caller frames appear in the 0713 `zeros256` session that
match no kind above (6 and 7 of 32 against the nearest). Both admit exactly one
generator state, and both sit on the 62-draw lattice this family's 32-symbol
members occupy — so they are further points of the same stream and not a different
generator. One session and one callsign pair is below the standard the descriptors
here are held to, and neither is on the delivery path, so this document names
neither.

**A BW500 delivery is opened by the responder's turn-request.** A stock responder
holding 172 bytes keyed `session-turn-request` — the responder's row, keyed to the
CALLER — eleven times over 127 s, 3.6–6.4 s apart and bit-identical, and started
nothing until it was answered. That is the frame an initiator must recognise
before a gateway at this bandwidth will send it anything.

**`session-over-response` is not a BW500 frame.** At BW500 the answer to a DATA
over is the two-tone burst of §4.2C — the 8-symbol one at every intermediate over
and the 11-symbol one at the last. 50 continues and 2 control bursts across the
0713 delivery, 4 and 1 across the 0830 one, and not one 32-symbol frame appears in
either delivery's over phase on either cable.

**An unanswered BW500 connect-request is retried at 32 symbols, not 41.** The
acquisition CR is the ordinary 41-symbol burst; the retries at 3.2 s carry the same
31 payload tones behind the single preamble tone `74`, the 10-tone acquisition
preamble dropped. A reader that names a burst by its symbol count will read one as
a session frame.

### 5.3.6 The session at BW2750 — the same frames on the wider alphabet

The rule of §5.3.5 holds at BW2750 with the alphabet of
[`04 §4.2.2`](04-frame-block-formats.md) substituted — 84 carriers on bins 22..105 —
and the handshake pair carrying its own descriptor, as at BW500. One stock
loopback session (AAAA1 → BBBB2, 2026-07-21), both cables, every burst scored whole
against both alphabets and both callsigns:

| frame | keyed to | alphabet | carriers | on BW2300's reading |
|---|---|---|---|---|
| connect-request, BW2750 | called | **BW2300's**, `(58, 1)` | 41/41 | 2/31 against `(50, 1)` |
| connect-response, BW2750 | called | BW2750's, `(579, 91)` | 23/23 | 0/15 against `(289, 511)` |
| session-confirm | called | BW2750's | 16/16 | 0/15 |
| session-keepalive-a | called | BW2750's | 32/32 | 0/31 |
| session-keepalive-b | called | BW2750's | 32/32 | 0/31 |

The BW2300 session the same modems keyed 50 s later on the same tape reads 41/41,
23/23 and 32/32 at BW2300 and 0–3 of 31 on the BW2750 alphabet. The session
holds no delivery — its harness died before the payload was queued — so the turn
and over frames are unmeasured at this bandwidth; they are keyed on the BW2750
alphabet on the strength of §5.3.5's twelve kinds and these three.

The responder's two-tone bursts are its own at this bandwidth too: the
connected-ack of [`04 §4.2C`](04-frame-block-formats.md) answered the link-setup
and the session-confirm, and a second tail answered the first keepalive.

The **8-symbol answer to an intermediate over** is measured here: two 2026-09-04
cable sessions of a link W9SSJ called, each delivery padded past one over so both
ends had an intermediate one to answer, gave ten copies — six from the caller,
tone for tone identical, and two further sets of seven from the responder. Symbol
0 is `(64, 67)` in every copy at every bandwidth and no other symbol is shared;
two of the responder's land on bins 22 and 104-105, which no BW2300 burst reaches,
so the seven behind the lead are drawn on this bandwidth's own alphabet
[`vara_frames.OVER_CONTINUE_CALLER_2750`].

The **11-symbol tails** are still unmeasured for a link this station called at
BW2750, so a session there keys the BW2300 pair, which sits inside its band and
which a BW2750 responder acts on: the stale-state burst answered a delivery's last
over and closed it  [`vara_frames.control_bursts`]. The two do not share that
fallback — a bandwidth with no continue frame of its own keys the generated
32-symbol answer rather than another bandwidth's copy, which three BW2750 fetches
established by taking one over each and nothing further.

## 5.4 ARQ mechanics

| Attribute                                   | Value |
|---------------------------------------------|-------|
| ACK / NAK scheme (stop-and-wait / SR / GBN) | **Stop-and-wait**: exactly one DATA over ⇄ one short ACK over, strictly alternating (never 2 DATA overs back-to-back except under retransmit) |
| Sequence number space / wrap                | Rides in the waveform, not host-visible; blocks are reassembled in order and duplicates suppressed, so a seq/ack field exists — this document does not fix its width or wrap |
| Retransmission trigger                      | Over repeats when it fails to advance `BUFFER` (decode-failure); seen once with collision-like overlapping PTT + down-shift, and unconfirmed beyond that single session |
| Memory-ARQ / soft combining                 | Not host-observable; not fixed here |
| Max retries / give-up                       | This document does not fix a retry limit; no give-up occurred on an error-free link |
| Flow control / window                       | `BUFFER n` on cmd port = **raw host bytes still queued**; decrements per ACKed block (43 B at level 4, 34 B at level 3). Window = 1 (stop-and-wait) |
| Blocks per over (2nd throughput axis)       | Grows **1 → 2 blocks/over** after several consecutive successes (DATA over 4.31 s → 8.51 s, ×1.94); independent of speed level |
| Speed level of the closing over             | **Drops one record** for the over that ends a delivery, and only that one: at BW2300 the close behind a full over whose field byte is `0x81` is keyed at record 2 (`BITRATE (3)`), 6 of 6 deliveries, and the two behind `0x89` stay at the base level; the next delivery's first over is back at the base level. A delivery that fits one short over closes at the base level. BW500 does the same, its 200-byte delivery closing on the 35-byte record-2 body  [`04 §4.2B`]. A close keyed at the base level behind `0x81` is answered with the continue burst and the delivery never completes. The byte alone does not fix the record: a stock responder closing to this station kept the base level behind `0x81` three times, read byte-exact each time |

## 5.5 Turnaround timing & tolerances

| Attribute                        | Value | Unit |
|----------------------------------|-------|------|
| TX→RX turnaround (min)           | ~85 (mean 88.5, range 52–122, sd 11, n=116; symmetric initiator↔responder) | ms   |
| RX→TX turnaround (min)           | ~85 (same distribution; not directionally distinguishable) | ms   |
| ACK timeout                      | Not fixed here — no ACK timed out on an error-free link; the host documentation gives a `P2P SESSION` retry cycle of ~4.6 s | ms   |
| Keepalive interval (on-air)      | ~10–12 (idle exchange: short + ~1.4 s burst) | s    |
| Keepalive interval (host `IAMALIVE`) | ~60 (Δ 60.0) | s    |
| Idle disconnect timeout          | Not fixed here — sessions were closed by the host | s    |
| **TX→RX turnaround, on air** | **91–98 (median 97, min 85, n=20)** — a live RMS gateway keys this long after an over ends, on two independent gateways at BW2300. The distribution above holds unchanged on real RF, so it is a property of the protocol and not of a virtual audio path. **Observed, not normative:** this is where two gateway implementations key, over 20 overs — neither is known to enforce it as a limit, and the ACK timeout one row above is left unfixed for exactly that reason. Answering within this window is what a peer has been seen to tolerate; answering outside it is untested. | ms |

**The window a per-over answer has to fit inside, at BW500.** The interesting bound
is not when the sender's next over may start but when it *does*: a peer that keys
its repeat while the answer is still going out never reads the answer. Measured off
the PTT ledger of the 2026-08-30 stock pair, over the five overs of a 200-byte
delivery — all times in seconds from the sender's unkey:

| the sender's over | answer keys at | answer lasts | answer's last sample | sender's next key-down |
|---|---|---|---|---|
| 1 | 0.107 | 0.355 | 0.462 | 0.550 |
| 2 | 0.083 | 0.357 | 0.440 | 0.520 |
| 3 | 0.097 | 0.355 | 0.452 | 0.556 |
| 4 | 0.097 | 0.374 | 0.471 | 0.544 |
| 5, the last | 0.172 | 0.481 | 0.653 | 0.741 |

So a BW500 sender re-keys **0.52–0.56 s** after its own unkey on an intermediate
over, and the stock answer clears it by 0.073–0.104 s. The window is narrower and
far steadier than BW2300's 0.47–1.69 s, and it is what makes the 8-symbol burst of
[`04 §4.2C`](04-frame-block-formats.md) the only answer that fits: the 32-symbol
`session-over-response` runs 1.366 s and its last sample would land 0.9 s inside
the sender's own next transmission.

## 5.6 Disconnect

| Attribute                     | Value |
|-------------------------------|-------|
| Graceful disconnect sequence  | Host `DISCONNECT` (after `BUFFER 0`) → **one 32-symbol burst, ~1.365 s, keyed to the called callsign, and nothing answers it**. Measured 2026-08-26 between two stock 4.9.0 instances on one cable: the closing station keys it immediately ahead of its own CW ident, and the only thing that follows is the peer's ident. It sits at `SEED_OFF 60`, `PREADV 125` on the family's 62-draw lattice — position 2, which no other 32-symbol member occupies |
| ~~3-burst graceful close~~    | **Withdrawn 2026-08-26.** The earlier reading — disc-req (~0.75 s), disc-ack (~0.49 s), final (~0.75 s), ~2.2 s total — came off a loopback tape and no bench session reproduces it. Its 0.75 s opening burst is the **turn release** (§5.6.1), which ends nothing: both bench sessions ran on for two further overs after it. A close described as three bursts is describing an older version or an artifact of that tape |
| Abort / forced disconnect     | `ABORT` is defined at the host interface as an immediate dirty close; its on-air sequence is not covered here |

### 5.6.1 Turn release

The 17-symbol / ~0.725 s MFSK burst that used to be read as a disconnect request.
It is what a station keys to hand the channel back, and the peer answers it by
**transmitting**. Both stations key it; it is keyed to the called callsign, from
the same handshake-tone family as the connect bursts
([`04 §4.2`](04-frame-block-formats.md)) at `preamble (62, 67)`, `N=15`,
`SEED_OFF 61`, `PREADV 391` — regenerated at 17 of 17 tones off two stock 4.9.0
instances, against 1 of 15 for the next best reading of the same burst.

The data phase is one shape repeated, in both directions:

    over (4.34 s) → the peer's two-tone ack (0.47 s) → THE RELEASE (0.725 s)
                  → the peer's own over (4.34 s)

with turnarounds of 0.16 s to the ack, 0.13–0.14 s from the ack's last sample to
the release, and 0.07–0.10 s from the release's last sample to the peer's next
over. The caller does not ask for the turn at the start: it holds it from the
connect, keys its link-setup over, and hands the channel over with the step-6
confirm, which is why the answering station speaks first.

The responder keys a 17-symbol release of its own, opening on the same `(62, 67)`
preamble and matching no frame this project holds at any alignment — recorded
rather than fitted, and not needed to originate a session.

## 5.7 Provisional facts and unfixed values

- **The data-ACK carries NO sequence number — it is a constant token.** Read
  through the control-burst demod ([`02 §2.6`](02-preamble-sync.md), DBPSK@1350),
  **48 of 51 data-ACKs** in a 48-block transfer are the bit-identical 32-symbol
  token `01110111111101110000100001111000` (the other 3 differ by a 1-column
  demod rotation and a mis-sliced 44-bit burst). It does not identify the block it
  acknowledges. That suits the strict stop-and-wait of §5.4: one
  block is ever outstanding, so "an ACK arrived" unambiguously means "my last
  block got through" — the sequence is **implicit in the alternation**, not carried
  on the wire. **Interop consequence:** an implementation ACKs a block by emitting this
  one fixed waveform and reads a VARA ACK by correlating against it — no field to
  compute.
- **NAK is a constant token too.** When an over fails to decode, the responder
  withholds its ACK and emits a distinct 32-symbol DBPSK
  token `01111000111101111000100010001111` — Hamming 0 across the failure
  bursts, Hamming 11 from the data-ACK, and it appears only in that case. Like the ACK it
  carries **no sequence field**: in stop-and-wait a NAK just means "resend the outstanding
  over." So both ARQ acknowledgements are seqless constant tokens.
  (`kestrel/vara/vara_control.py` "nak".)
- **Both of the above are BW500, and only BW500.** At BW2300 there is no DBPSK
  control token: the acknowledgement is the two-tone index-modulated burst of
  [`04 §4.2C`](04-frame-block-formats.md), the same waveform as the connected-ack, and
  whether it is seqless is not established because nothing held here separates an ACK
  from a NAK among its state symbols. See the retraction in
  [`02 §2.5`](02-preamble-sync.md).
- **Memory-ARQ and max-retries** still live inside the control-burst waveform for the
  cases an error-free link never exercises (NAK, down-shift).
- **Retransmit / down-shift path** rests on one session — a stuck `BUFFER` +
  overlapping PTT + level drop — and is unconfirmed. A deliberate link-impairment
  sweep is what would confirm the trigger and measure the **ACK timeout** and
  **idle-disconnect timeout**, neither of which this document fixes.
- **P2P vs WINLINK retry timing** (4.6 s vs 4.0 s cycle at the host interface) is
  not measured on air; the timings here are for `P2P SESSION`.
