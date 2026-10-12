# Implementation evidence

Results for hfmodem's data, linking and text modes.
PACTOR, VARA and ARDOP are alpha; all other modes are experimental.
RF reception, gateway transfers, independent bench decodes and simulations
are identified separately.

## PACTOR

Winlink mail sent and received through WS8EOC over hfmodem’s PACTOR-3.
Four messages also received through KB5LZK over PACTOR-1, with a normal session close.

| Result | Evidence and conditions | Record |
|---|---|---|
| PACTOR-3 outbound body accepted | W9SSJ via WS8EOC; gateway returned `FS Y` and `FF`; local `FQ` queued | Station log |
| PACTOR-3 message received | W9SSJ via WS8EOC; complete message saved | Station log |
| PACTOR-3 outbound mail delivered | W9SSJ via WS8EOC; receipt confirmed by the operator | Message `9WQNVWBDLVOY` |
| Four messages fetched; session closed normally | KB5LZK; PACTOR-1 throughout the exchange. Message IDs: `JWKY65C2OZES`, `FP8L9JM4Q2L3`, `YQKB3UDBH1KG`, `J7VQ9ZYDX9L5` | Station log |
| PACTOR-1 payload delivered byte-exact to the peer's host | Virtual audio with Sailer's (HB9JNX) hf-pactor software; frozen-counter and altered-payload negative controls | Bench record |
| PACTOR-1 rate reduction in both directions | Bench tests cover local fallback and requests for the peer to halve its rate | Test suite |
| PACTOR-2 SL1 entry frame decoded | Independent monitor on the bench | Station bench record |
| PACTOR-3 payload prefixes decoded at all six speed levels | Reference decoder over virtual audio; idle/CR normalization, wider lock search at SL1/2 | Bench record |
| PACTOR-3 grants and entry packets received | Station summary of 68 attempts: four reached a grant and two reached a grant plus a decoded entry packet. Separate 40 m and 80 m records contain banner and packet payload fragments | Station logs |

PACTOR-3 mail transfer is demonstrated in both directions through WS8EOC.
The outbound body acceptance and inbound saved message establish mail transfer;
a confirmed PACTOR link shutdown is a separate result. Reliability across
gateways remains open. PACTOR-2 evidence is at the bench
level. The association between status-byte bits 4–5 and a PACTOR-3 grant is an
inference from two gateways; see
[capability negotiation](protocols/pactor/pactor-capability.md).

<!-- site:omit:start -->
Experimental PACTOR-4 entry, waveform and mail paths are also present. The
current offline replay reaches a CMS prompt and sends a B2F login/proposal;
that result does not establish on-air PACTOR-4 mail delivery. See the
[C2 runtime and offline derivation](protocols/pactor/pactor4-c2-derivation.md)
for the distinction between runtime representation and original-generator proof.
<!-- site:omit:end -->

[Mode guide](viz/pactor-explained.html) ·
[ARQ implementation](../packages/hfmodem/hfmodem/shrike/arq.py)

## VARA

Mail reception through K0SI and confirmed outbound delivery through KB8AY.
Multiple deliveries confirmed by the operator. VARA supports high speed capable
and speed limited links. The transmit and receive codecs cover the level tables
for BW500, BW2300 and BW2750; the named RF sessions below do not establish
RF performance at every level.

| Result | Evidence and conditions | Record |
|---|---|---|
| Three complete message files and one partial file saved | K0SI; four mail offers, BW2300, 40 m, 7101.8 kHz center, 10 W; acknowledged close | Station log |
| Secure login accepted and outbound body accepted | KB8AY; BW2750, 40 m. CMS challenge answered, proposal `V15F3FF1BAF4` accepted with `FS Y`, body followed by `FF`, session closed with `FQ` | Station log |
| Outbound message delivered to its internet address | KB8AY; BW2300, 40 m, 7101.5 kHz center; gateway acceptance and destination receipt confirmed | Message `NL03TXWQ8SGL` |
| Multiple outbound deliveries confirmed | Receipt in Gmail confirmed by the operator; gateway and session details unspecified | Operator report |
| All 41 connect tones recovered | Michigan KiwiSDR recording of a KC9GHZ call on 7103.5 kHz | Remote recording |

Inbox confirmation establishes delivery of the named message through KB8AY.

[Mode guide](viz/vara-explained.html) ·
[Session implementation](../packages/hfmodem/hfmodem/kestrel/vara/vara_arq.py)

## ARDOP

The besra implementation has transferred mail in both directions, in separate
gateway sessions.

| Result | Evidence and conditions | Record |
|---|---|---|
| Message fetched | WW2MI gateway session | Station log |
| Outbound message delivered | WW2MI gateway session with delivery confirmation | Message `2KX2HWM919HR` |
| Three messages fetched in one session | W6IDS gateway session | Station log |
| Bidirectional cross-decode across all 59 frame types | ardopcf reference codec | Bench record |

Reference-codec and recording-dependent tests report skips when those inputs are
absent. `pytest -rs` lists them.

[Mode guide](viz/ardop-explained.html)

## sabir

Sabir is experimental and unfinished. Its protocol and wire format are subject
to change. Beacons have been decoded from two remote receivers; session traffic
and host integration are exercised through simulated channels.

| Component | Demonstrated behavior | Evidence |
|---|---|---|
| Beacon reception | 40 m transmissions decoded from public receivers 103 and 302 miles away | Remote recordings, decoded with hfmodem |
| ARQ and retransmission | Selective codeword acknowledgement, timeout retry and payload resegmentation without duplicate delivery | State-machine tests and channel simulations |
| Turn-taking | Sender handover, receiver turn requests and turn recovery | State-machine tests; bidirectional Watterson channel simulation |
| Disconnect | Host-requested close, remote close and abort; stale-session controls rejected | State-machine and host-interface tests |
| Host interface | Single framed-CBOR connection per station; bidirectional payload transfer | Paired servers over simulated channels |
| Data profiles and object transfer | OFDM and narrow FSK profiles; connectionless object framing and integrity checks | Error-free and Watterson channel simulations |

The radio entry point transmits beacons and prepared datagrams. Connecting the
two-way session engine
to live audio and PTT remains incomplete. Carrier groups masked during a session
remain masked; re-probing is unfinished.

[Mode guide](viz/sabir-explained.html) ·
[ARQ implementation](../packages/hfmodem/hfmodem/sabir/arq/fsm.py) ·
[Host interface](../packages/hfmodem/hfmodem/sabir/host/run_server.py)

## Robust Packet (brubru)

Brubru provides Robust Packet transmit/receive, AX.25 framing and a station
KISS host on a local TCP port. A KISS client owns the higher link layers;
connected-mode state, retries and mail transport are not implemented by brubru.
Station integration is tested with simulated audio and fake resources.

The tests decode four published corpus frames from three stations with valid FCS.
These are corpus results, not observed results from this station. Regeneration
agrees on 1327/1328 differentials for RPR_Frame with limiting enabled, and
1328/1328 for Robust-packet.mp3 and the first radioscanner frame without it.
The second radioscanner frame is pinned for its payload, FCS and 1808
differentials; no regeneration agreement is claimed for it. Containers and
level 1 are covered by synthetic tests, without an independent station result.

[Waveform and evidence](protocols/robust-packet/waveform.md) ·
[Framing](protocols/robust-packet/framing.md) ·
[Station attachment](protocols/robust-packet/station.md)

## bateleur

MIL-STD-188-110 serial, 39-tone and 16-tone, 110C appendix D (3 kHz),
STANAG 4285, 4529 and 4539, the
STANAG 5066 link layer, 2G ALE and HF selcall run through station lanes and
host ports. The console adds automatic ALE scanning/sounding, manual calls and
native AMD/DTM/DBM text, plus held-channel selcall calls and pages. These console
workflows have software and station integration tests; their two-way external
radio interoperability remains to be tested. The evidence column distinguishes
local simulations from published recordings and earlier external decodes.

| Component | Demonstrated behavior | Evidence |
|---|---|---|
| 188-110 serial, 75–4800 bit/s | Short, long and zero interleave; rate and interleave read from the preamble; message recovered to its EOM | Loopback and Watterson channel simulations |
| 188-110 Table XX | Short simulations: eleven rows, zero errors in 128–4,800 bits per row; conditional independent-bit 95% upper bounds 0.0231–0.000624 | One trial and one second of requested payload per condition, seed 0; [conditions and counts](protocols/bateleur/README.md#simulation-measurements) |
| STANAG 4285, 75–3600 bit/s | Coded and uncoded streams, including SOM-less reception; Annex E framing remains foreign-unverified. Short simulations: 21 rows, zero errors in 600–2,400 bits per row; conditional independent-bit 95% upper bounds 0.00498–0.00125 | Recorded short `annex-b` run; [conditions and counts](protocols/bateleur/README.md#simulation-measurements) |
| STANAG 4539, 3200–12800 bit/s | Local round trips at every rate and interleaver. Short simulations: fifteen rows, zero errors in 3,200–12,800 bits per row; conditional independent-bit 95% upper bounds 0.000936–0.000234 | Recorded short `c-xvii`, `c6-3`, `c6-4` runs; [conditions and counts](protocols/bateleur/S4539.md#simulation-measurements) |
| 188-110 39-tone, 75–2400 bit/s | Short simulation at 1200L, 15 dB, two equal paths 2 ms apart with 2 Hz spread, +23 Hz offset: 2/1,200 bit errors; conditional independent-bit 95% upper bound 0.00524. Six other short rows have zero errors | Recorded short `b-xii` run; [conditions, counts and clock-test limitation](protocols/bateleur/TONE39.md#simulation-measurements) |
| 188-110 16-tone DPSK mode (appendix A), 75–2400 bit/s | Local round trips, offset and clock tests, and station lane integration. No performance table is recorded for this waveform | Loopback and channel tests; [limits](protocols/bateleur/TONE16.md#how-it-is-tested) |
| 188-110C appendix D, 3 kHz (waveforms 0-13), 75–16000 bit/s | Every waveform and interleaver recovered, the waveform read off the preamble; table D-LII met at 23 of 26 points (8 transmissions, 15 kbit–3.2 Mbit a point), the three misses each about 1 dB (600 bit/s poor, 4800 poor, 12000 white noise); D.6.2's static channels, D.6.4's ±75 Hz and D.6.5's sweep decoded; on the station and its host port (`MODE WBHF`) | Loopback, channel simulations, station tests on replayed audio |
| 188-110A "mod" (receive only) | Worked out from seven off-air recordings, with no description to go on: tone header, parallel section, and a 188-110A data phase with a 640-symbol sync phase; every burst in them found, its data phase followed to the channel bits, both formats (bits once, or twice with an end marker) told apart; the bits' coding not found (not 110's, not an uninterleaved K<=7 rate-1/2 code), so they are output raw | Off-air recordings, synthesized bursts in the test suite |
| STANAG 4529, 75–1800 bit/s | Every coded rate and both interleavers, uncoded 600–1800, any sub-carrier 800–2400 Hz; the conformance test procedures' sync pattern, flush counts and 99% power in 1240 Hz met; 75 bit/s at 3 dB 30 Hz off and 150 bit/s on CCIR poor at 12 dB decoded whole; on the station and its host port (`MODE 4529 <mode>`) | Loopback, channel simulations, station tests on replayed audio; written from the JITC conformance test procedures, not the STANAG's text |
| Binary FSK (STANAG 4481 annex C, 110B 5.1), 50–1200 Bd | 850 and 170 Hz shifts and the 16 multi-channel sub-carriers, with diversity; no character errors on white noise to 6 dB with 25 Hz off; 2–11% character errors uncoded on CCIR poor at 10 dB, under 1% with diversity at 6 dB; on the station and its host port (`MODE FSK <mode>`), keyed only inside the declared span | Loopback, channel simulations, station tests on replayed audio; no broadcast recording |
| STANAG 5066 ed. 1.2, annexes A-C | D_PDUs against the annex's worked CRC example; selective-repeat ARQ through bit errors to 2e-3; expedited and non-ARQ delivery, groups, errored delivery; soft and hard links, crossing requests, full reset, data rate change; nodes on the modem through rendered audio and a channel, and on the station with clients on TCP 5066 | Nodes against each other in simulated channels (bytes and audio), station tests on replayed audio with a fake rig; no other implementation |
| 2G ALE | Calls, scanning, AMD, directory lookup, DTM/DBM and linking-protected reception and transmission; the operator enables scanning and supplies its channels | Local coding and channel tests. Earlier LinuxALE decode reported without retained WAV, command or output; DTM CRC and protected wire details remain open. [Evidence](protocols/bateleur/ALE.md#interoperability-evidence) |
| HF selcall (CCIR 493-4 4-digit, UN open 6-digit) | Local coding and channel tests; three published recordings decode; a station lane and host dialect; no call-success table is recorded | Tests and published recordings; [limits](protocols/bateleur/SELCALL.md#measured) |
| Occupied bandwidth | Declared rows 350–2995 Hz trimmed and 350–3140 Hz untrimmed, including parallel modes; 39-tone short bursts filtered | Welch PSD at 30 Hz equivalent noise bandwidth, −26 dB, short and 30-second payloads in `tests/bateleur/test_phy.py` |
| Station modem | Replayed 188-110 audio reaches an attached application through the station; a message sent from the host port keys through the arbiter and decodes back; messages longer than `max_key_s` split into whole transmissions | Station tests on replayed audio with a fake rig |

[Waveform description](protocols/bateleur/README.md) ·
[Receiver](../packages/hfmodem/hfmodem/bateleur/phy.py) ·
[Host interface](protocols/bateleur/HOST-API.md) ·
[STANAG 5066](protocols/bateleur/S5066.md) ·
[FSK](protocols/bateleur/FSK.md) ·
[STANAG 4529](protocols/bateleur/S4529.md) ·
[Appendix D](protocols/bateleur/WBHF.md) ·
[188-110A mod](protocols/bateleur/MOD110.md)


## Lanner: RTTY, SITOR-B and NAVTEX

Lanner supplies ITA2 RTTY and collective SITOR-B waveform encoding and
receive decoding, plus NAVTEX bulletin extraction. The console provides
separate passive `rtty`, `sitor-b` and `navtex` monitors. They support automatic
tone-pair acquisition through a receive audio passband up to 4 kHz.

The tests cover code tables, local audio round trips, offset/noise and
time-diversity behavior, chunked reception and bulletin parsing. Those tests
establish software behavior; no named independent on-air Lanner decode is
claimed here. SITOR-A ARQ and a two-way Lanner console terminal are not provided
by these monitor commands.

[Implementation and limits](protocols/sitor/README.md) ·
[Passive monitoring](console-monitor.md)

## Source records

RF transfer results come from station logs and operator receipt confirmations.
The delivery references are message `9WQNVWBDLVOY` for PACTOR-3 through WS8EOC
and message `NL03TXWQ8SGL` for VARA through KB8AY, plus message `2KX2HWM919HR`
for ARDOP through WW2MI. Named reference decoders and
remote receivers provide the independent observations identified above.
Simulation results use two instances of hfmodem.
