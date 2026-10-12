# PACTOR-3 speed adaptation

## Entry and direction of speed requests

The receiver requests a higher transmit speed with CS4. Our `--p3-speed-up
auto` policy asks the peer to transmit faster after three consecutive cycles
delivering new payload; `hold` suppresses those outgoing requests. Neither
setting prevents us from following a CS4 received from the peer.

A granted P3 entry is SL1. An ordinary acknowledgment retains that level by
default; it does not authorize an unsolicited jump to SL3. An uninvited entry
retains its configured entry level after acknowledgment. The explicit
`--p3-traffic-sl` override remains available for experiments after entry.
If CS4 itself answers entry, the command takes precedence over that override:
the next packet is one level above the entry actually sent, bounded by the
configured maximum. Entry completion cannot reset the commanded level. A
payload packet sent at that higher level receives the same MAXTry protection
as a speed trial later in the transfer.

This fixes the ordering observed in `onair-0920-1024`: VE3KPG answered an SL1
entry with CS4, ARQ selected SL2, and entry completion reset it to the mail
preset's SL1 before rendering. Subsequent CS4s are still interpreted using the
ordinary policy; `--p3-tx-gear-hold` remains an opt-in experiment, not a proven
replacement for that policy.

Packet length is separate from speed level. Status bit 5 requests long packets
when application bytes remain queued; CS6 grants a cycle-length change. The
console keeps long cycles opt-in (`--long-cycle`); without it, bit 5 is clear
and we do not grant incoming requests to enter long cycles. Received CS6 is
still followed to preserve the peer's clock. This is not a speed-up solicitation
flag, and changing it is not evidence that a peer will issue CS4.

## Trial and receive-error limits

The SCS PTC-IIIusb manual separates a sender's speed-up attempt from the
receiver's decision to ask for a lower speed. Both settings are exposed by
`hfmodem.shrike.onair`:

| Flag | SCS command | Default | Range | Meaning |
| --- | --- | --- | --- | --- |
| `--p3-max-try N` | MAXTry | 2 | 1–9 | Total emitted transmissions of the first packet at a trial speed |
| `--p3-max-down N` | MAXDown | 6 | 2–30 | Consecutive receive errors before requesting one lower level with CS5 |

Pass `--p3-max-try 2 --p3-max-down 6` to `python -m hfmodem.shrike.onair`
to select these defaults explicitly. Run the module with `--help` for the
station configuration required for an on-air session.

Each accepted changeover to the transmitting role resets P3 to the configured
minimum speed (SL1 by default). This is a conservative local policy: a speed
requested in an earlier transmit turn is not carried into the new one. The
changeover packet retains its fixed two-carrier format; subsequent data starts
at the minimum unless the peer requests a higher level with CS4. A refused
changeover does not reset the speed or take the transmitting role.
When a goodbye is queued behind data, acknowledged payload renews its placement
budget so a progressing SL1 transfer can drain; repeat requests do not.

MAXTry includes the original transmission: the default is one initial attempt
and one retransmission. After CS4 acknowledges the preceding packet and requests
a higher speed, the next packet is a trial. A matching acknowledgment establishes
the new speed. If the trial remains unacknowledged after N actual transmissions,
its next attempt returns to the preceding speed with the same packet counter.
The lower-capacity renderer puts the unused payload suffix back at the front of
the queue. Refused transmissions spend no attempts. Once back at the previous
speed, repeated CS1/CS2 requests alone do not force further descent. CS5 still
requests an immediate one-level drop, regardless of MAXTry.

MAXDown applies while receiving. Failed CRCs count toward the error run; any
valid packet, including a duplicate, clears it. In the live driver, an occupied
receive window with no CRC-valid packet also counts after all its readers finish.
Multiple failed candidates in one window count once. Skipped historical slots,
quiet windows and energy outside the expected slot do not invent received error
packets. The next available reply carries CS5 once the threshold is reached.
A refused or superseded CS5 does not spend the request; actual emission starts
a new error run. SL1 cannot request a lower P3 speed.

The driver accounting above is our mapping of observations to the documented
packet counters. The manual does not specify SCS's internal detector or scheduling
implementation. These settings do not claim equivalence of those internals.

`--link-retries` remains the independent link timeout. P3 CRC failures use that
budget rather than the old three-bad-packet abort. A timeout shorter than MAXDown
can still end the link first; for a large MAXDown, choose a sufficient
`--link-retries` value. PACTOR-1/2 retain their existing behavior.

Sources: SCS PTC-IIIusb 4.1 manual, §§6.55 and 6.58
([English](https://www.scs-pactor.com/assets/downloads/SCS_Manual_PTC-IIIusb_4.1.pdf),
[German](https://www.p4dragon.com/download/SCS_Handbuch_PTC-IIIusb_4.1.pdf)).
The German MAXTry description explicitly counts how often the higher-speed packet
is transmitted in total, resolving the English word “repeats.”

An offline replay of the WS8EOC control sequence verifies bounded fallback and byte accounting, not remote acceptance of a
lower-speed retransmission that was never sent in the original recording.


### Turn cycle and stalled handover recovery (September 20)

A long receive turn does not authorize long data in our next transmit turn.
The old cycle still places outgoing changeover attempts. Once the peer
acknowledges that changeover, ordinary data starts short; a fresh CS6 enables
long data. A CS6 answering the changeover itself grants long operation from the
new turn's short baseline. CS4 changes speed without granting a long cycle.

An established speed can stall on an empty ordinary packet offering handover,
after its preceding idle packet was acknowledged. For that case we now retry
one level lower after six actual transmissions without acknowledgment at the
current level, down to the configured minimum. This is a local recovery policy,
separate from MAXTry and MAXDown. It preserves the packet counter, cycle length,
handover status, and queued bytes. Refused emissions spend no attempts. ACKs
start the next packet's count; CS5 still lowers speed immediately. The station
remains ISS until the peer actually takes the channel. Entry, CS3 changeover,
QRT, and ordinary loaded packets retain their existing policies. The independent
silence timeout still applies.

This addresses the 21:14 WS8EOC empty-handover repeat train and the 21:19 VE3KPG
long-frame transmission without an incoming CS6. Offline tests verify the new
behavior and decoded waveforms; acceptance by the remote modem still needs an
on-air trial.

### Repeated cycle grants

After accepting CS6, retain the following packet and the granted cycle length
when CS6 repeats. The repeated command may still refer to the preceding packet;
it is not independent evidence that the first frame at the new length arrived.
A distinct ACK or gear command clears this hold, as does a role change. A repeat
request alone does not. This applies in both cycle directions and also when the
first attempted transmission at the new length was refused locally.

This is a conservative ambiguity policy prompted by three failed-login captures
(WS8EOC 20:59 and 22:07, VE3KPG 21:59). Their zero-error CS6 pairs previously
retired the 36-byte login frame and toggled short. Recorded-control replay now
preserves those bytes and retransmits the same counter at the granted length.
The replay proves byte accounting, not the peer's hidden acceptance state.
Consecutive intentional opposite cycle commands without an intervening distinct
acknowledgment remain an interoperability risk to check in on-air trials.
