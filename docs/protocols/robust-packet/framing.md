# Robust Packet framing

AX.25 values at the library boundary exclude FCS bytes. A bare radio packet
carries the AX.25 FCS; a container instead carries one CRC over its header,
length words and contents. No FCS is synthesized for delivered container frames.
Bare framing is measured on four public frames. [P: 3 stations] Container
framing is covered by generated vectors. [T: 1 implementation, synthetic frames, 2026-09-30]

Addresses compress from seven bytes to five using a base-40 callsign integer
and the original SSID/flags byte. The bare packet prepends a byte carrying
padding and speed suggestion. PN9 scrambling excludes the final zero tail byte;
a K=9 rate-1/2 convolutional code and a length-dependent interleaver follow.
[P: 3 stations, level 0]

Classes 0–14 hold 15, 21, 45, 75, 105, 126, 147, 168, 189, 210, 231, 252,
273, 294 and 312 bytes including the tail. Class 15 adds a second header and
holds 345 + 30e bytes for extensions e = 0–16. The smallest fitting class is
selected, including exact-capacity extended packets. Bare padding must still
fit its five-bit field. [T: 1 implementation, synthetic frames, 2026-09-30]
Public frames cover class 3 at three stations [P: 3 stations, 3 frames] and
class 4 at one station. [P: 1 station, 1 frame]

`brubru.frame.pack_frames` returns packets in input order. A frame that cannot
share the container's address representation goes bare. A frame that fits no
permitted bare class is logged and skipped; other frames retain their order.
Later container frames store only one C bit, so both destination/source
combinations 00 and 11 must
stay bare. One-frame containers are accepted on receive and are never emitted.
[T: 1 implementation, synthetic frames, 2026-09-30]

Both boundaries validate all addresses, control/PID presence, declared lengths
and the applicable CRC before returning frames. Receive results contain only
validated, unchanged AX.25 bytes. Invalid headers, unsupported extensions,
truncation, malformed containers and failed checks produce no delivered frame.
[T: 1 implementation, malformed-input tests, 2026-09-30]

Padding outside the CRC is discarded; its byte values do not change the
delivered frame. [T: 1 implementation, 2 padding cases, 2026-09-30]

The Python literal vectors pin the preamble, convolutional encoder pair,
interleaver checks, all 64 header codewords, compressed callsign examples and
CRC check value. Public corpus tests additionally pin each recording's digest,
payload and measured regeneration extent. The second radioscanner frame is
227 symbols and 1808 differentials; its acceptance currently requires a valid
FCS and exact packet reconstruction, without a regeneration-count claim.

Container header bits 3–4 and length-word bits 9–14 are ignored on receive;
length uses bits 0–8. The first word's C bit is ignored: its addresses carry
the original flags. Later words supply the destination C bit, with its complement
used for the source. The container CRC still covers every header and length
bit. The public frames do not establish container behaviour.
[T: 1 implementation, 10 bit/flag cases, 2026-09-30]
