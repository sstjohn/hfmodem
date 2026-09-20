# PACTOR-3 speed adaptation

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
