# Sabir capability negotiation

This document defines session capability exchange. [SPEC.md](SPEC.md) defines
waveforms, frame layouts, profiles and stream integrity. The implementation is
`hfmodem.sabir.arq.wire` and `hfmodem.sabir.arq.fsm`.

## Establishing a session

A wideband caller with fast controls enabled first sends CONNECT on the coherent
fast waveform when its offer needs no optional extensions. An eligible listener
replies with CONNECT_ACK on the same waveform. All implemented profile permissions
fit the fixed capability bitmap;
these two blocks establish the session. Either message may declare optional CAPS
blocks in the same transmission when an extension is needed. Both messages advertise the sender's receive capabilities. Each sender
independently intersects the peer's advertisement with its own enabled features
and bandwidth limit. A capability bit grants permission to transmit that feature;
it does not command the peer's rate-selection policy.

The fast bootstrap is a bounded probe before capabilities are known. A listener
restricted below 1500 Hz, configured for floor-only control, or requiring reply
extensions ignores the probe. An unanswered probe is followed by the robust floor
handshake; all later retries use the floor. The probe deadline is twice its burst
duration plus two turnarounds and the ACK margin (3.421 s with default settings).
The probe is additional to the floor retry budget. An offer with extensions starts
on the floor. Identical capability/session contents are retained across retries,
so a lost fast reply can recover through a floor retry without resetting the
responder's stream state. Established-session control tier selection remains
separate from this bootstrap policy.

The floor reply window accommodates CONNECT_ACK and the maximum three optional
CAPS blocks, plus two turnarounds and the ACK margin. The configured connection
timeout can increase this bound, but cannot shorten it. With default settings the
reply budget is 36.880 seconds after the request airtime; a one-block unanswered
floor request is retried at 45.600 seconds from its start. This bound prevents a
retry from colliding with a legal extension reply. It increases the wait when no
peer answers, without delaying a successfully decoded response. Five unanswered
floor attempts take 228 seconds; the default fast probe adds 3.421 seconds.

Every session block is 44 bytes. Bytes 0–9 contain packet type, wire discriminator
2 and a nonzero 64-bit session identifier. The caller chooses a fresh random
identifier for each new connection attempt; retries retain it. Integer fields are
big-endian unless explicitly stated otherwise. Every block ends in CRC-16/CCITT-
FALSE, big-endian, over all preceding bytes.

CONNECT and CONNECT_ACK use this payload:

| Bytes | Field |
|---|---|
| 10 | Reserved zero |
| 11 | Low nibble: CAPS count, 0–3. Bit 7: CFAIL, CONNECT_ACK only. Other bits zero. |
| 12–15 | Exact capability bitmap |
| 16–27 | Sender identity, printable ASCII, space-padded to 12 bytes |
| 28–39 | Destination identity, printable ASCII, space-padded to 12 bytes |
| 40–41 | Reserved zero |
| 42–43 | CRC16 |

The listener checks the explicit destination against its configured identity.
The caller checks session, source and destination on CONNECT_ACK. Session IDs
isolate accidental delayed traffic; they do not authenticate identities or prevent
replay across restarts.

A complete offer requires exactly the declared CAPS positions, all carrying the
same session identifier and declared total. Conflicting duplicates invalidate the
offer; identical duplicates count once. Missing or mismatched extensions leave
session state unchanged and the caller retries. No feature set is inferred from
an incomplete offer. A CAPS block by itself cannot modify an active session.

An established listener answers an identical CONNECT retry without resetting its
stream state. A changed capability image under the same session ID is ignored.
Renegotiation requires a new session. The immediately retired session's CONNECT
is ignored while listening.

## Capability bitmap

Each DATA profile is granted by the bit matching its absolute profile ID.

| Bits | Meaning |
|---|---|
| 3–7 | robust, workhorse, workhorse34, fast, max, respectively |
| 16–21 | doppler, sparse34, narrow, narrow2, narrow4, wide256, respectively |
| 24 | FASTCTL: coherent OFDM controls accepted |
| 25 | PBACK: final ACK may be piggybacked before reverse DATA |
| 26 | LOADING: per-group loading maps accepted |
| 27 | DEFLATE: compressed host records accepted |
| 28 | BEACON: presence capability indicator |
| Other bits | Unassigned; ignored on receive |

Each profile bit grants only that profile. Advertising a higher-rate waveform
never implies support for lower-rate waveforms. The sender intersects explicit
permissions with implemented profiles and its local bandwidth limit. Profile IDs
0–2 name control waveforms and supply no DATA permission. A bitmap with no usable
DATA profile can be decoded, but attempting to send DATA to that peer fails the
session unless an optional profile extension supplies a usable profile.

The local `max_rung`, `bandwidth_hz` and additional receive-profile configuration
select which bits to advertise; they are not separate negotiation mechanisms.
The sender's rate-selection policy skips unsupported profiles. Optional profiles
are chosen explicitly or as fallbacks. No total robustness ordering across
waveform families is assumed.

FASTCTL, PBACK, LOADING and DEFLATE are enabled only when locally implemented and
permitted by the peer. The receiver tests both supported control waveforms and
their bounded block lengths; no extra control-tier negotiation is required.
Unknown capability bits do not enable behavior. A protocol discriminator mismatch
is rejected; incompatible changes require a different discriminator.

CONNECT and the compact presence beacon encode this same bitmap in big-endian
order. The beacon does not carry optional extensions, so it is an observation of
base permissions rather than a complete session negotiation transcript.

Every stream record carries a SHA-256 integrity trailer, including records
queued before connection. Integrity is mandatory and has no capability bit.
Compression is optional and used only after a connected peer advertises DEFLATE.
A cryptographic digest detects corruption but supplies no authentication.

## CAPS and TLVs

CAPS is optional. Current profiles and features need no extension blocks.
An extension uses the same 44-byte session envelope, with type 10:

| Bytes | Field |
|---|---|
| 10 | High nibble: total blocks, 1–3; low nibble: index, 0 through total−1 |
| 11–41 | 31-byte TLV area, zero-padded |
| 42–43 | CRC16 |

The three-block bound limits handshake airtime and decoder work. A block holds
up to 14 uint16 profile IDs in one GEARSET TLV. The count bound does not
limit TLV type assignments; new semantics can use unknown optional identifiers
or an explicitly required critical identifier. Two blocks require no padding
block. TLVs cannot cross block boundaries; the encoder rejects overflow rather
than truncating it.

A zero byte is PAD and occupies one byte. Other TLVs are `[type:1][length:1]
[value:length]`. The low seven type bits identify the extension and bit 7 is
CRIT. CRIT on PAD is invalid. Length must fit the remaining block; malformed
lengths invalidate the block. An unknown optional TLV is ignored. An unsupported
critical TLV causes CFAIL and prevents session establishment in either direction.
There are no extension-specific exceptions to that rule.

| ID | Value | Implemented semantics |
|---|---|---|
| 0 | No length/value | PAD |
| 1 | Even-length list of big-endian uint16 IDs, each ≥24 | GEARSET: profiles outside the bitmap |
| 5 | Opaque bytes | IMPL: implementation identifier, no feature permissions |

GEARSET cannot name IDs below 24: the fixed bitmap is authoritative for those
profiles. GEARSET order and repeated IDs have no semantic effect. The sender uses only
profiles that it implements and the peer explicitly names. Unrecognized IDs do
not select a substitute waveform. Profile IDs name complete waveform/coding
combinations; receiver-local algorithms do not require new IDs.

An unsupported critical requirement produces CONNECT_ACK with CFAIL set. The
responder then returns to listening; the caller reports refusal. An incomplete
critical extension is handled as an incomplete offer, without assuming consent.

## Data, turns and closure

DATA names an absolute uint16 profile, a uint32 transmission generation, a
uint64 stream offset, codeword geometry, a present-codeword bitmap and a loading
map. Retransmissions retain generation/profile/geometry/offset. Rebuilding at a
different profile starts a fresh generation at the same stream offset. Generation
counters do not wrap within a session.

The receiver combines soft evidence only within that frame identity. It rejects
stream gaps and commits a frame only when all codewords decode. Previously
committed prefixes are discarded, including across rebuilds with different
codeword boundaries. A duplicate of the last completed frame receives a full ACK
without another delivery. ACK bitmaps report codewords of one physical generation;
they do not implicitly commit a prefix of another frame.

Only the send-role holder starts DATA. A receiver with queued traffic sets the
traffic bit in its ACK, or sends a bounded sequence of TURN_REQ controls on an
idle link. TURN grants the role. An unanswered grant times out; unanswered
TURN_REQ retries eventually report link failure instead of retaining an
undrainable queue. A receiver wanting a graceful disconnect also requests the
role before sending DISC.

With mutual PBACK support, an eligible DATA header may offer handover. A receiver
that completes the frame and has reverse traffic can prefix its DATA with the
final ACK and role-taken flag. The normal retransmission and duplicate-ACK rules
recover lost piggybacked transmissions. Failed fast controls fall back to the
robust floor. Rate and tier thresholds are implementation policies, not negotiated
wire constants.

DISC and DISC_ACK carry station identity. DISC is retried if its ACK is lost.
The receiver retains the immediately closed session ID to answer a repeated DISC;
that courtesy response cannot close a different active session. Once a session
ends, an application may still be uncertain whether its last unacknowledged
message arrived. Cross-session transaction deduplication belongs to the application.

Each connected endpoint has a receiver-local inactivity deadline, 180 seconds by
default and configurable to a positive finite value. Only accepted peer session
traffic refreshes it; noise, wrong-session traffic, stale ACK generations and
invalid DATA do not. A validated DATA header can extend the lease through a long
legal body and its reply window. A local transmission similarly protects the
period during which a peer cannot reply. The front end derives those durations
from checked codeword geometry, not an unchecked symbol count.

Expiry clears the session and queued state with a link-failure indication and one
courtesy DISC. Idle sessions therefore expire without periodic valid peer traffic.
The 180-second default tolerates ordinary multi-round frame recovery while bounding
a silent peer's hold on a listener; it is an implementation policy, not a wire
permission or an experimentally established optimum.

## Validation and limits

Deterministic tests cover exact profile sets including holes, bandwidth policy,
encoded field widths, malformed and unsupported optional extensions, complete and
incomplete handshakes, lost CONNECT_ACK, changed offers,
stale controls, role requests/grants, dropped DISC_ACK, selective retransmission,
rebuild/resegmentation and duplicate suppression. Sample-level tests additionally
exercise the actual control waveforms and DATA codecs.

The common fast handshake occupies 1.421 seconds of transmitted airtime, or 1.921
seconds with two modeled 250 ms turnarounds. The floor handshake occupies 17.440
seconds of airtime, or 17.940 seconds including those turnarounds. Each optional
CAPS adds 8.72 seconds in the direction that sends it. These totals include guards,
synchronization, FEC and CRC, and carry no application payload.

In ten deterministic trials per condition (seeds 0–9, AWGN referenced to transmitted
active power in 3 kHz), fast-first setup completed 10/10 at +20 dB in 1.921 seconds.
At −10 dB, with a +20/−10 dB asymmetric path, or with the first fast ACK erased, it
completed 10/10 through floor recovery in 21.362 seconds. Floor-only setup completed
10/10 in the same noise cases in 17.940 seconds. The probe therefore reduces the
strong-link setup time and costs 3.422 seconds when floor recovery is needed.
These are simulator times, excluding CPU scheduling and real radio switching.
Reproduce with `python -m hfmodem.sabir.sim.negotiation_campaign`.

The session simulator accounts for both endpoints' transmitted airtime, turnaround
and timer idle time, and declares success only after exact delivery and terminal
states at both peers. Direction-specific impairments and burst erasures are
supported. These tests establish implementation behavior under the modeled
conditions; live-radio session integration remains incomplete.
