<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Two recorded normal-long PACTOR-4 packets

Source: [Signal Identification Wiki PACTOR IV](https://www.sigidwiki.com/wiki/PACTOR_IV),
[PACTORIV_audio2.mp3](https://www.sigidwiki.com/images/a/aa/PACTORIV_audio2.mp3). Original SHA-256:
`945608057c39fbcc11110ddf2bd78ad384760adc4d6052d247500e2bcd008074`.
Attribution: Signal Identification Wiki contributors. Source and adapted audio
excerpts are [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
The page does not identify the operator or recording date.

Reproduce with FFmpeg: decode the MP3 to mono 28800-Hz signed 16-bit PCM.
For training origins 134460 and 242461, retain samples from origin minus 2880
through origin plus 95168, exclusive. Store both 98048-sample rows as `audio`
in `fixtures/sigidwiki/p4-long/audio.npz`. No denoising or synthetic waveform replacement is applied.

The same recording carries the two short packets in `fixtures/sigidwiki/p4-short`. Those two
origins are 1.25 seconds apart and these two are 3.7500 seconds apart, each
variant on its own published raster. Their CRCs, status bytes and
203-byte payloads were established by the separately retained research decoder
before this package's long receiver was written. Both are from one station and
recording. They validate normal-long BPSK reception at supplied origins, not
header interpretation, setup, other modulation or connected ARQ.
