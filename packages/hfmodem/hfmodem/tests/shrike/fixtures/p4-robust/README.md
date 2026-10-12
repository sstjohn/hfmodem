# Six recorded robust-mode short packets, as dibits

`dibits.npz` holds six rows of 84 quarter-turn steps, `int8`, one per packet.
The seventh known packet is `p4robusttx.SHORT_DIBITS` and is not repeated here.

`B5`, `B6` — the second and third caller bursts of
[Signal Identification Wiki PACTOR IV](https://www.sigidwiki.com/wiki/PACTOR_IV),
[PACTORIVaudio1.mp3](https://www.sigidwiki.com/images/c/ce/PACTORIVaudio1.mp3),
SHA-256 `e9d14a178426ad9c82a62e5bc7272bbcb152c13a48821ceb89a02965d6816a76`, at
6.113 s and 7.364 s. Attribution: Signal Identification Wiki contributors, source
and adapted excerpts [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
`SHORT_DIBITS` is the first burst of the same tape, at 4.863 s.

`T0`–`T3` — the four short packets of
[signals.taunus.de PACTOR-IV](http://www.signals.taunus.de/WAV/PACTOR-IV.HTML),
`PACTOR-IV_112-5_DQPSK.WAV`, SHA-256
`c4e191d4888f0d15ebd2cfecdb5f6503c99a34b29c40fd8285c04eb58d0f47a7`, 11025 Hz mono,
at 0.257 s, 1.507 s, 2.757 s and 4.007 s. A different station from the tape. The
file is kept under the ignored `working/pactor4-decode-0916/normal-short/local-audio/`
with the rest of that page's provenance.

Reproduce: decode to mono float, take each burst with 0.12–0.23 s of lead, mix to
complex baseband at 28800 Hz, lock chip phase and carrier on the SF-16 despread
energy fraction over all 85 symbols, refine the carrier on the fourth-power tone
of the despread symbols, then hard-decide the 84 differentials with the residual
common rotation removed. Every row is unchanged across four such windows; the
taunus locks land at 0.905–0.912 despread fraction, the tape's at 0.35.

These are symbol decisions, not audio: they validate the datablock reading, the
two header words and the mask, and nothing about acquisition from a raw window.
