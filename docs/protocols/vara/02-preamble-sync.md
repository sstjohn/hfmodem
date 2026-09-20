# 02 — Preamble, synchronization & acquisition

Transmission is **burst / PTT-keyed**: every over is one keyup→keydown burst.
After keyup the RF energy ramps to full amplitude over **~65–82 ms**
(keying/soundcard ramp) and then stays flat.

The BW500 symbol is **512 samples** (10.667 ms, 93.75 baud) on two sub-bands at
1350/1650 Hz, which is what recovers frames byte-exact — see
[`01 §500`](01-physical-layer.md) and [`03 §3.6.1`](03-coding.md). Several
acquisition metrics below are computed on a 256-sample stride; that stride is an
analysis window of half a symbol, not the symbol period.

There are **two burst populations** (see [`01` §1.5](01-physical-layer.md)):
long **DATA** bursts (initiator) and short **CONTROL/ACK** bursts (responder,
~0.34–0.68 s, distinct centre-carrier-dominated structure, carrying the ARQ
handshake / acquisition preamble — §2.5). DATA-burst onset shows a short
(~6–10 symbol) transient with a distinct k=−3 head before the steady
alternating-tone body; whether that head is a fixed SOF/training marker is not
settled here.

| Attribute                              | Value |
|----------------------------------------|-------|
| Preamble present? (per frame/session)  | Per burst (PTT-keyed); no distinct low-power lead-in; short (~6–10 sym) onset transient at same power |
| Preamble duration                      | onset transient ~6–10 symbols (~32–53 ms) before steady alternating-tone body; energy ramp ~65–82 ms is keying |
| Preamble waveform type (chirp/tone/PN) | not specified here; the onset shows a k=−3 tone excursion |
| Known training symbol pattern          | not specified here |
| Repetition / count                     | DATA body is period-256 symbols with strict k=±1 tone alternation (a 512-sample super-period) |

## 2.2 Timing synchronization

| Attribute                          | Value |
|------------------------------------|-------|
| Symbol-timing recovery method      | de-rotation reference plus blind `c0` lock by turbo self-consistency, per [`03 §3.6.2`](03-coding.md) |
| Correlation peak shape / metric    | active/guard energy ratio ~115–137 at lock (sharp; degrades away from the true 256-sample stride alignment) |
| Acquisition tolerance (timing)     | integer-sample offset resolved within the 256-sample stride; SCO small over a burst |

## 2.3 Frequency synchronization

Signal centre is 1500.0 Hz (DC-null). Residual CFO relative to the estimated band
centre runs ~8–20 Hz — a synchronizer quantity (carrier/soundcard offset), not a
fixed waveform parameter.

| Attribute                              | Value |
|----------------------------------------|-------|
| Coarse frequency-offset acquisition    | CFO handling for the BW500 data waveform is [`03 §3.6`](03-coding.md); this document does not fix a coarse pull-in figure |
| Fine / residual CFO tracking           | residual per DATA burst observed between **±0.1 and ~11 Hz** (soundcard offset, not a waveform parameter); k=0 carrier is the phase/CFO reference |
| AFC pull-in range                      | not specified here |

## 2.4 Frame boundary detection

Frame = burst. ARQ cadence is very regular: initiator (A) DATA bursts run
**~4.35 s (~785 sym)** each (rising to ~8.53 s / ~1570 sym when more data is
queued), responder (B) CONTROL/ACK bursts run **~0.34–0.68 s (~65–130 sym)**, and
the TX↔RX turnaround gap is a near-constant **~85 ms** between every keydown
and the opposite station's next keyup.

| Attribute                        | Value |
|----------------------------------|-------|
| Start-of-frame marker            | PTT keyup; energy onset ~65–82 ms later. Modem SOF symbol not specified here |
| End-of-frame / burst tail        | PTT keydown ends the burst (power drops to silence) |
| Inter-frame gap                  | ~85 ms turnaround between opposite-station bursts |

## 2.5 Control / sync bursts (responder-sent; distinct waveform)

The responder's bursts are a **separate waveform** from the initiator's DATA
bursts and carry the ARQ handshake / acquisition (connect, ACK, keepalive,
disconnect). A 256-sample-stride FFT alternation metric (swing between the k=+1
and k=−1 bins each symbol; ≈1.0 = strict per-symbol tone alternation, ≈0 =
steady) cleanly separates the two populations:

| Burst (function)            | Duration | Spectral peak | Alternation metric |
|-----------------------------|----------|---------------|--------------------|
| DATA (initiator, ref)       | ~4.31 s  | off-centre (comb) | **0.99** (alternating) |
| data-ACK (responder)        | **~0.36 s** (0.31–0.38) | **1500 Hz (centre)** | **0.20** |
| connected / keepalive / over-ack | ~0.48–0.50 s | 1500 Hz | ~0.23 |
| connect-response            | ~1.0 s   | ~1640 Hz spread | 0.06 |
| connect-request (acquisition) | ~1.7 s | ~1780 Hz spread | 0.06 |

| Attribute | Value |
|-----------|-------|
| Control-burst family present? | Yes — every responder over is a control burst; **no k=±1 tone alternation** (metric ≈0.06–0.23 vs DATA ≈0.99) |
| Short data-ACK burst | **~0.36 s**, spectral peak at the **1500 Hz centre carrier**; punctuates every DATA over (one ACK per DATA over, stop-and-wait → [`05 §5.4`](05-arq-session-state-machine.md)) |
| Duration ↔ function taxonomy | ~0.36 s data-ACK · ~0.49 s connected/keepalive · ~0.70–0.75 s **turn release** — read as a disconnect request until 2026-08-26, when a two-VARA bench showed both sessions running on for two more overs after it ([`05 §5.6.1`](05-arq-session-state-machine.md)) · ~1.0 s connect-response · ~1.4 s idle-keepalive · ~1.37 s disconnect · ~1.7 s connect-request |
| Internal structure | Centre-carrier-dominated (short ACKs peak at 1500 Hz), **not** the alternating comb of DATA; onset/preamble sub-structure vs payload split not specified here |

**Control-burst modulation: raw DBPSK on the k=−1 (1350 Hz) sub-band.** Under the
BW500 front-end (down-convert 1350/1650, H=512), the k=−1 (1350 Hz) sub-band of a
responder control burst carries a clean **differential BPSK** stream —
constellation collapse `|E[e^{2jθ}]|` = **0.80–0.89** across 15+ bursts, against
0.15 for coherent BPSK and 0.28 for coherent QPSK. The differential is **raw**
(`cell[n]·conj(cell[n−1])`), **not** the grid480-scrambled differential of the DATA
path: a DATA over needs grid480 de-rotation first (raw diff ≈0.2) while the
control burst does not (raw diff ≈0.9). For reference on the same metric, a DATA
over scores **0.984** under grid480+BPSK and silence scores 0.0. The k=+1
(1650 Hz) sub-band is weaker and ambiguous (0.57–0.62); whether the burst also
modulates it, or 1350 is the sole payload carrier, is not settled here.

The bursts are **payload-independent** — bit-identical whatever message the
session carries — so they carry ARQ state (seq / ACK / NAK), not the message. The
exact symbol timing/rate and any CRC are not specified here.

**Raw DBPSK bits; the control burst is not turbo-coded.** The raw-differential bit
stream (`sign(Re(cell[n]·conj(cell[n−1])))`, best-onset) off the 1350 Hz sub-band
is **stable and repeatable** — identical ARQ positions demodulate byte-for-byte
identical, within and across sessions. Two frame lengths appear:

- **44-bit frame** (the ~0.489 s class): e.g. `1111111111111111 0000011110000 1111 00000000000`.
  Its bits are a **fixed template**; only positions **0 and 41–43**
  toggle between states (Hamming distance 4), the bulk (1–40) invariant. One distinct
  variant at higher Hamming (≈19) is a different control function (disconnect/idle class).
- **32-bit frame** (the ~0.361 s data-ACK class): e.g. `01110111111101110000100001111000`;
  likewise a template with a few varying positions.

The **small** bit-change per state is itself informative: a turbo-coded frame (as DATA
uses) would avalanche ~½ the coded bits on any info change, so the control burst is
**light- or un-coded** (plausibly repetition — note the 4–5-bit runs), fitting a tiny
ACK that needs robustness over rate.

**The control channel is a small vocabulary of fixed tokens, not parametric
frames.** Demodulating every responder burst shows no per-burst arithmetic field,
only a handful of canned waveforms distinguished by duration and a few flag bits:

| token | ~dur | symbols | bits (DBPSK@1350) | when |
|-------|------|---------|-------------------|------|
| **data-ACK** | 0.36 s | 32 | `01110111111101110000100001111000` (constant, 48/51) | after each DATA over |
| **connected-ack** | 0.49 s | 44 | `1111111111111110000011110000111100000000`+`0000` | at CONNECT setup |
| **idle-keepalive** | 0.49 s | 44 | …same 40-bit body…+`1111` (only the 4-bit tail differs) | connected & idle |
| connect-response | 1.0 s | — | (MFSK, spec 04 §4.2 — DBPSK demod fails q≈0.54, as expected) | responder answers CR |
| long idle | 1.4 s | 128 | (not yet a clean template) | long idle gap |

So the connected-ack and idle-keepalive are the **same 40-bit template** with a 4-bit
tail flag (`0000` vs `1111`); the data-ACK is its own constant token.

> **The 0.49 s rows are read as DBPSK here and as two-tone MFSK in
> [`04 §4.2C`](04-frame-block-formats.md); the readings contradict each other and
> §4.2C is the one measured against the spectra.** The two are the same burst:
> 44 x 512 samples is 11 x 2048, exactly. Every 2048-sample
> window of that burst holds **two sharp equal carriers** on the tone lattice, each
> with its own +/-1-bin Hann skirt and nothing between them — for pairs as far apart
> as bins 52 and 86, which no single 1350 Hz DBPSK carrier produces. The DBPSK
> templates recorded above are still usable as kestrel-to-kestrel tokens and are
> still what `vara_control` emits, but they are **not** a demodulation of what VARA
> keys at CONNECT setup, and the row's name should not be read as evidence about
> step 5. What the *idle-keepalive* row is, this document does not fix.

The construction is a canned token set rather than a coded frame with a sequence
counter — consistent with the stop-and-wait ACK carrying no seq (§5.7). Interop
needs only to emit and recognise this token set.

**Both sides use fixed tokens.** The initiator is the same story at
longer length: its idle-keepalive is a **128-symbol (~1.385 s) constant token** (bit-
identical across three idle gaps in one session), with a ~65-symbol connect-confirm and
distinct short bursts read at the time as disconnect requests. The responder answers
those by **reusing the `…1111` alive token** — the same waveform serves
keepalive-response and that answer, its meaning set by the initiator's state, not by a
distinct burst. *What those bursts mean is now open: at BW2300 the burst of that length
is the turn release and not a disconnect ([`05 §5.6.1`](05-arq-session-state-machine.md)),
and this BW500 token corpus has not been re-read against that. Take the token set as
measured and the names in this paragraph as provisional.* So the whole BW500 control
channel is a **small fixed-token vocabulary on both ends**, no sequence arithmetic
anywhere, matching the seq-less stop-and-wait ACK.

**The NAK token.** When the responder fails to decode an over — its input swamped
with noise until decoding fails — it emits a distinct 32-symbol DBPSK token,
`01111000111101111000100010001111`: Hamming 0 across the failing bursts,
Hamming 11 from the data-ACK, and appearing only while overs fail. Like the ACK it
is a **constant seqless token** ("resend the outstanding over"). So the full BW500
responder set is: **data-ACK, NAK, connected-ack, alive, ready** — all fixed
tokens.

**BW2300 control channel — RETRACTED 2026-08-23.** This section asserted until
2026-08-23 that BW2300's control channel is the same DBPSK-constant-token family as
BW500, and gave four band-specific bit patterns for it. **It is not, and they are
not.** No real VARA has been recorded keying a single-carrier DBPSK control token at
BW2300, and the four patterns match nothing:

- On a 2026-08-14 bench capture of VARA HF v4.9.0 against itself at BW2300 — **both
  directions on tape**, with the host-port event log beside them, so every burst is
  attributable to a side and to a protocol event — the session's thirteen short
  control bursts match none of the four patterns at Hamming 2, and a bit search over
  the whole of both sides at sixteen independent column phases matches none of them
  either.
- Off air: zero token lines across all four 2026-08-23 monitor runs, both successful
  connects included. A bit search over `offair/KC9GHZ_2300`, `offair/NS0A_2300` and the
  three 2026-07-24 confirmed QSOs — real VARA at both ends of every one — returns 1 to 2
  hits per recording at ±60 Hz, which is exactly what the same search returns on
  material that holds no responder burst at all (see below), and nothing clustered,
  repeated or in a turnaround.
- The same detector on the same bench at BW500 finds more than forty data-ACKs at
  Hamming 0 (`test_vara_control`), so it is the table that is absent, not the
  instrument.

**That floor is the whole of the original result.** At the same settings — Hamming 2,
sixteen column phases, ±60 Hz — the same search returns the same 1–2 hits per recording
on material that cannot hold a BW2300 token: **2** in the 73.7 s BW2300 capture that
carries only station A's DATA overs, **1** in 110 s of the BW500 control-burst session
read with the BW2300 table, and **0** in 30 s of clear-channel noise. The counts this
section used to cite — "9 data-ACK, 3 connected-ack, 2 alive and 2 NAK" over a whole
session — are that
same rate, and nothing in them separates a match from a chance one. How the patterns
were arrived at cannot be rechecked: the 2026-07-23 session they were read off is in no
corpus. The two-sided reference session is what settles it anyway, because there the
answer is not "few" but "none".

**What the burst actually is.** The 0.489 s BW2300 control burst — which both
stations key, not only the responder — is index-modulated, not DBPSK: **eleven two-tone MFSK symbols of four 512-sample columns
each** in a 0.489 s keying, opening on the four fixed preamble pairs that [`04 §4.2C`](04-frame-block-formats.md)
already specifies — `{64,67} {56,74} {64,69} {68,78}`, 1500/1570, 1312/1734, 1500/1617,
1594/1828 Hz — with seven state symbols behind them. Measured over all eleven instances
in the reference session, on both sides, the preamble is identical in every one.

The 0.361 s class the old table called `data-ACK`/`NAK` is the same modulation one size
down: 33 columns, **eight** two-tone symbols of four columns. It shares only the ack's
first symbol (1500/1570 Hz) and none of the other three preamble pairs, and the two
instances in the reference session — one from each side, at the turn exchange — share
nothing past that first symbol. The **durations** the old table gave were right; the
modulation it read them through was not.

This section used to carry a caveat beside the table — "not every 0.489 s BW2300
responder burst is a constant token", four instances demodding to three different DBPSK
bit strings. That is what it was seeing: three different *state fields* read through the
wrong modulation. §4.2C is the reading that reproduces, and the two sections were
describing the same burst and contradicting each other about it.

**The acknowledgement and the connected-ack are one waveform at BW2300.** In the
reference session the responder's CONNECT-time burst (`b2a` 8.567 s) and its first
per-over answer (`b2a` 9.943 s) correlate **1.00**. Three distinct state fields appear
on the responder side of that session and two on the initiator's, so there is a small
vocabulary here — but nothing held anywhere separates an ACK from a NAK among them.
**Settling that needs a two-sided BW2300 capture with an over deliberately corrupted**,
which is how the BW500 NAK was solved and what BW2300 has never had.

`vara_control._TOKENS_2300` is gone, along with the band parameter that dispatched to
it: BW500 is the only bandwidth with a token vocabulary, so there is nothing to select.
What used to index it — `vara_arq._send_token("nak")` — no longer runs at BW2300 at all,
and the reference session now holds the BW500 vocabulary to the same standard, resolving
none of its thirteen short bursts
(`test_this_vocabulary_names_nothing_in_a_real_vara_bw2300_session`). The short-burst
path at BW2300 reads `vara_arq._ack_plateau` instead, in the live receiver and in
`kestrel/arq/phy.detect_token` alike.

**Token synthesis in this implementation.** BW500 only; there is no BW2300 token to
synthesise. A real BW500 data-ACK is multi-carrier, with energy at ~1311/1500/1600 Hz.
kestrel's token synth is **single-carrier**: it carries the
correct differential bits (its own detector, and the demod that read them off real audio,
score it at Hamming 0) and round-trips kestrel↔kestrel, but its *waveform* correlates only
~0.1 with a real token. This is not what VARA specifies; it is a stopgap until a
waveform-faithful synth exists. Such a synth is achievable offline — reconstructing each
constant token from its measured per-column OFDM cells (≈125 complex cells) hits
correlation **0.965** against the real averaged token — but whether it is *needed* is an
on-air question: if VARA's receiver demodulates the DBPSK bits (as kestrel's RX does), the
bit-faithful single-carrier token already decodes and the multi-carrier waveform is
surplus. It is not built, because it cannot be validated without a real VARA to accept it
and may be unnecessary; the on-air `vara_compat` test resolves which synth is required.

**What a control-burst demod needs**: (1) a **burst-sync /
acquisition** front-end keyed on the 1500 Hz centre-carrier energy — the
DATA-burst stride-256 active/guard energy-concentration metric (§2.2) relies
on the ±1 alternating comb and does **not** apply to these non-alternating bursts,
so a distinct detector/template is required; (2) the same joint CFO+timing search
can likely re-use the 1500 Hz centre reference; (3) once synced, the modulation
riding the control burst (esp. the connect-req/resp, which must carry the
callsigns + BW request + speed level) must be demodulated — this is the gate on
reading the ARQ seq/ACK/NAK fields and the negotiated level
([`05 §5.7`](05-arq-session-state-machine.md), [`06 §6.5`](06-speed-gearshift.md)).
This also frames the **DATA-burst preamble question**: whether DATA bursts begin
with a short slice of this same control/acquisition waveform (the ~6–10-symbol
onset transient in §2.1) or acquire blind — correlating the DATA onset against the
short ACK burst decides it.

## 2.6 Open questions / provisional facts

- **Blind acquisition is solved for DATA bursts** without any preamble template:
  the stride-256 energy-concentration metric locks
  every burst, so no preamble detector is needed for demodulation. A dedicated
  preamble template is still wanted for fast, robust *acquisition* (esp. the
  CONTROL bursts).
- **Onset marker**: DATA bursts show a ~6–10-symbol transient (incl. a k=−3 tone
  excursion) before the steady alternating-tone body; correlating this head across
  many bursts would expose any fixed SOF/training symbol.
- **CONTROL/ACK burst waveform** is characterised at the envelope level (§2.5):
  responder-sent, centre-carrier-dominated, **no k=±1 alternation** (alt ≈0.2 vs
  DATA ≈0.99), ~0.36 s data-ACK up to ~1.7 s connect-request. **Still open:** its
  internal preamble/payload split. It needs a dedicated burst-sync
  front-end (the DATA energy-concentration lock does not apply).
- **No CP**: acquisition cannot use a CP Schmidl-Cox metric; the
  energy-concentration lock is the working substitute.
