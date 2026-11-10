<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Four recorded PACTOR-4 chirp packets

Derived from [SigIDWiki PACTOR IV](https://www.sigidwiki.com/wiki/PACTOR_IV),
[Pactor_IV_chirps.ogg](https://www.sigidwiki.com/images/3/3f/Pactor_IV_chirps.ogg).
Original SHA-256: `246d15c3592741782ad8f2e1b56c162a384dc040e65b62bb14867f9d4b4f77de`.
Attribution: Signal Identification Wiki contributors. Source and these adapted
audio excerpts are [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
The source page does not identify the recording operator or capture date.

Four packets from one recording, not independent stations or sessions.
`fixtures/sigidwiki/p4-chirps/fields.json` holds their channel-decoded 25-byte fields, in transmission order
and as hex: 22 payload bytes, the status byte, then the CRC-16. No bytes have
been selected from our transmitter. Marker words are 9, 9, 8, 8; channel rank
zero is lower, lower, upper, upper. The recorded tuning is 242.4 Hz low.

Reproduction: FFmpeg decode original OGG to mono 48000 Hz signed 16-bit PCM.
Take 158760 samples at starts 178944, 538944, 718944, 1078944 (3.728, 11.228,
14.978, 22.478 seconds). Convert to float by division by 32768; apply SciPy
`resample_poly(x, 1, 6)`; clip to [-1,1], multiply by 32767, round and cast to
int16. `audio.npz` contains the four resulting 26460-sample rows and `fs=8000`.
No frequency shift, denoising or transmitted sample substitution is applied.

Tests require exact reproduction of the original 25-byte channel-decoded fields,
normal X-25 after dewhitening, both marker/lane combinations, and rejection of
truncated or mistuned input. This does not establish P4 setup or ACK semantics.
