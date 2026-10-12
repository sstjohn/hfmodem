# Two recorded normal-short PACTOR-4 packets

Source: [Signal Identification Wiki PACTOR IV](https://www.sigidwiki.com/wiki/PACTOR_IV),
[PACTORIV_audio2.mp3](https://www.sigidwiki.com/images/a/aa/PACTORIV_audio2.mp3). Original SHA-256:
`945608057c39fbcc11110ddf2bd78ad384760adc4d6052d247500e2bcd008074`.
Attribution: Signal Identification Wiki contributors. Source and adapted audio
excerpts are [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
The page does not identify the operator or recording date.

Reproduce with FFmpeg: decode the MP3 to mono 28800-Hz signed 16-bit PCM.
For training origins 62459 and 98459, retain samples from origin minus 2880
through origin plus 22848, exclusive. Store both 25728-sample rows as `audio`
in `audio.npz`. No denoising or synthetic waveform replacement is applied.

The two packet hashes and CRCs were established by the separately retained
research decoder before the package receiver was written. Both are from one
station and recording. They validate normal-short BPSK reception at supplied
origins, not header interpretation, setup, other modulation or connected ARQ.
