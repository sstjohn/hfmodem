# PACTOR-3 console traffic

`tools/console` displays these records by default. The arm transcript retains
them too. They describe observations; they do not change ARQ decisions.

- `[traffic] TX/RX P3 SLn ...` identifies the actual data speed level, short
  (1.25 s) or long (3.75 s) cycle, sequence, type, and status byte. CHANGEOVER
  uses a fixed two-carrier waveform; it does not establish the ordinary data
  speed. TX and RX speeds are tracked separately.
- `payload ... hex=... bytes=...` gives the complete packet payload, with
  escaped control/non-ASCII bytes. These are modem packet bytes, before host
  decompression or B2F parsing. RX `information` additionally preserves fill
  and status exactly as decoded. TX `field` includes fill, status, and CRC,
  before whitening and FEC. CRC-ENCODED on TX does not claim peer reception.
- Controls carry no application payload. Their 20-bit codeword appears in
  hexadecimal and in temporal, least-significant-bit-first order. Both
  DBPSK carriers send that word at 100 baud; controls do not have a data SL.
  RX reports the decoded/corrected word, not the noisy demodulated bits.
- CS1 acknowledges even counters; CS2 acknowledges odd counters. The log
  interprets ACK/REPEAT against the packet counter when it is known.
- A keyed CS4 logs a speed-up **request**. The next decoded ordinary packet
  reports whether that speed has been observed. A packet still at the old SL
  leaves the request unconfirmed; it does not prove the peer never tried an
  unreadable higher-speed packet. CHANGEOVER is not speed confirmation.
- TX `attempt=1 retry=0` is the first emitted copy. Retrying the same packet
  increments these counters; a refused transmission does not. RX
  `copy=2 repeat=1` means two copies were decoded, not that only two were sent.
  Counters reset for a different packet or a change of data direction.
  Controls use separate `reply-copy`/`reply-repeat` counters for repeated
  replies to the same packet. These counts are not the ARQ silence budget.
- Dry-run TX records are explicitly marked `TX(dry-run)`.

For example, after a CS4 request from SL1 to SL2, another CRC-valid SL1
packet produces `speed-up unconfirmed`. A subsequent SL2 packet produces
`speed-up confirmed`. The requested level alone never changes the displayed
observed RX level.
