# Implementation evidence

Results for hfmodem's four modes. RF reception, gateway transfers, independent
bench decodes and simulations are identified separately.

## PACTOR

Confirmed outbound mail delivery through WS8EOC over PACTOR-3. Four messages
received through KB5LZK over PACTOR-1, with a normal session close.

| Result | Evidence and conditions | Record |
|---|---|---|
| PACTOR-3 outbound mail delivered | W9SSJ via WS8EOC; receipt confirmed by the operator | Message `9WQNVWBDLVOY` |
| Four messages fetched; session closed normally | KB5LZK; PACTOR-1 throughout the exchange. Message IDs: `JWKY65C2OZES`, `FP8L9JM4Q2L3`, `YQKB3UDBH1KG`, `J7VQ9ZYDX9L5` | Station log |
| PACTOR-1 payload delivered byte-exact to the peer's host | Virtual audio with Sailer's (HB9JNX) hf-pactor software; frozen-counter and altered-payload negative controls | Bench record |
| PACTOR-1 rate reduction in both directions | Bench tests cover local fallback and requests for the peer to halve its rate | Test suite |
| PACTOR-2 SL1 entry frame decoded | Independent monitor on the bench | Station bench record |
| PACTOR-3 payload prefixes decoded at all six speed levels | Reference decoder over virtual audio; idle/CR normalization, wider lock search at SL1/2 | Bench record |
| PACTOR-3 grants and entry packets received | Station summary of 68 attempts: four reached a grant and two reached a grant plus a decoded entry packet. Separate 40 m and 80 m records contain banner and packet payload fragments | Station logs |

PACTOR-3 mail delivery is demonstrated through one gateway; speed-level coverage
and reliability across gateways remain open. PACTOR-2 evidence is at the bench
level. The association between status-byte bits 4–5 and a PACTOR-3 grant is an
inference from two gateways; see
[capability negotiation](protocols/pactor/pactor-capability.md).

[Mode guide](viz/pactor-explained.html) ·
[ARQ implementation](../packages/hfmodem/hfmodem/shrike/arq.py)

## VARA

Mail reception through K0SI and confirmed outbound delivery through KB8AY.
Multiple deliveries confirmed by the operator.

| Result | Evidence and conditions | Record |
|---|---|---|
| Three complete message files and one partial file saved | K0SI; four mail offers, BW2300, 40 m, 7101.8 kHz center, 10 W; acknowledged close | Station log |
| Secure login accepted and outbound body accepted | KB8AY; BW2750, 40 m. CMS challenge answered, proposal `V15F3FF1BAF4` accepted with `FS Y`, body followed by `FF`, session closed with `FQ` | Station log |
| Outbound message delivered to its internet address | KB8AY; BW2300, 40 m, 7101.5 kHz center; gateway acceptance and destination receipt confirmed | Message `NL03TXWQ8SGL` |
| Multiple outbound deliveries confirmed | Receipt in Gmail confirmed by the operator; gateway and session details unspecified | Operator report |
| All 41 connect tones recovered | Michigan KiwiSDR recording of a KC9GHZ call on 7103.5 kHz | Remote recording |

The secure-login and confirmed-delivery records used the registered third-party
SID `[Pat-1.0.0-B2FHM$]`. They establish the challenge-response and B2F exchange
under that client identity. Inbox confirmation establishes delivery of the named
message through KB8AY.

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

## Source records

RF transfer results come from station logs and operator receipt confirmations.
The delivery references are message `9WQNVWBDLVOY` for PACTOR-3 through WS8EOC
and message `NL03TXWQ8SGL` for VARA through KB8AY, plus message `2KX2HWM919HR`
for ARDOP through WW2MI. Named reference decoders and
remote receivers provide the independent observations identified above.
Simulation results use two instances of hfmodem.
