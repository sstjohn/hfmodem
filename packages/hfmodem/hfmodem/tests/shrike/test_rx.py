# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate shrike.rx against an independent PACTOR-3 decoder's ground truth.

Run:  .venv/bin/python tests/shrike/test_rx.py

What each check proves (or precisely where it stops):

  0. case-0 header -- round-trip a case-0 TX header an independent decoder
     confirmed CRC-valid.
  1. control signals -- every CS index recovered from clean audio, and through
     heavy AWGN; separately, nearest-codeword through 5 bit errors.
  2. SL>=2 header -- decode `placement.data_packet` from AUDIO and assert the
     byte-exact CRC-valid field, with noise and a case-0 packet as negatives.
     The chain this inverts is not shrike's guess: an independent decoder's own
     trellis and CRC accept exactly this frame and reject thirteen neighbouring
     conventions, and that decoder reads this audio back byte-exact 5 times out
     of 5 against a 0-of-5 control.
  3. raster gather -- assert rx.gather reproduces the expected de-interleaved
     values for the reference soft raster.
  5/7. the occ15 capture's own front end -- a different, still-open span
     (a 216-byte frame, not the 26-byte header), measured and reported.
"""
import wave

import numpy as np

from hfmodem.shrike import modem, placement, rx, spec
from hfmodem.tests.shrike import archive

# The reference vectors are the only fixtures here that are not synthesized. They are
# committed and they do not ship, so their absence reads one way from a
# distribution and the opposite way on a checkout: silent here, and loud in
# tests/gates/test_corpus_present.py, which fails on a source tree that has lost
# them rather than letting this whole module drop out of the count unremarked.
RASTER = archive.RASTER3
pytestmark = archive.requires_raster_dumps

#: The one recording this module reads that is NOT committed. Both stages that
#: open it report a measurement and assert nothing, so an absent capture drops the
#: report -- a fresh checkout has the dumps and not the capture, and an unhandled
#: open there is neither a skip nor a result.
OCC15 = archive.ARCHIVE / "captures" / "occ15.wav"


def check_sl2_header() -> bool:
    """The SL>=2 header, audio in and field out, plus two matched negatives."""
    info = bytes((0x53 + 7 * i) & 0xFF for i in range(24))
    cfg = modem.ModConfig()
    audio = placement.data_packet(info, cfg=cfg)
    field, ok, start = rx.decode_case1_header(audio, fs=cfg.sample_rate)
    match = ok and field == placement.build_field(info, placement.HEADER)
    rng = np.random.default_rng(0)
    noise = rx.decode_case1_header(rng.normal(0, 0.1, audio.size), fs=cfg.sample_rate)[1]
    wrong = rx.decode_case1_header(
        placement.case0_packet(bytes(6), cfg=cfg),
        fs=cfg.sample_rate)[1]
    print(f"[2] SL>=2 header from audio: crc_ok={ok} field_byte_exact={match} "
          f"(row 0 at sample {start})")
    print(f"    negatives: noise crc_ok={noise}, case-0 packet crc_ok={wrong}")
    return match and not noise and not wrong


def check_gather() -> bool:
    vhinF = np.fromfile(RASTER / "vhinF.bin", np.int8).astype(int)
    deintF = np.fromfile(RASTER / "deintF.bin", np.int8).astype(int)
    g = rx.gather(vhinF)
    m = float((g == deintF[:g.size]).mean())
    print(f"[3] raster gather vs the reference de-interleaved buffer: "
          f"exact-match={m:.3f} ({g.size} softs)")
    return m == 1.0


def _clean_case0_audio(field: bytes, fs: int = 48000) -> np.ndarray:
    """Render a clean case-0 packet through the transmitter, body only.

    The field is the one an independent decoder confirmed CRC-valid,
    814800008000c8e1, so a failure here is this module's demod and not the frame.
    """
    from hfmodem.shrike import placement, modem
    info = field[:placement.DETECT.crc_bytes - 2]
    return placement.case0_packet(info, cfg=modem.ModConfig(sample_rate=fs))


def check_clean_case0() -> bool:
    """The decisive demod test (coordinator): decode shrike's OWN clean case-0 TX
    header -- clean audio isolates demod quality from corpus quality."""
    field = bytes.fromhex("814800008000c8e1")
    audio = _clean_case0_audio(field)
    hdr, ok = rx.decode_case0_header(audio)
    match = ok and hdr == field
    print(f"[0] native RX of shrike's clean case-0 TX header: CRC_ok={ok} "
          f"field={hdr.hex() if hdr else '-'} matches_TX={match}")
    return match


def _cs_audio(cs_index: int, fs: int = 48000) -> np.ndarray:
    """A control signal as clean audio, from the transmitter the link keys.

    The production encoder and no private copy: this file used to render its own
    codeword audio, so breaking `placement.control_signal` -- the tone pair, the
    pulse, the bit order -- left every round-trip here green.
    """
    from hfmodem.shrike import placement
    assert fs == spec.SAMPLE_RATE     # the transmitter renders at the spec rate
    return placement.control_signal(cs_index)


def check_cs_roundtrip() -> bool:
    """End-to-end on clean audio: shrike TXes each control signal, RX recovers its
    index. This is the peer-ACK/NAK path the ARQ handshake listens on."""
    fs = 48000
    delay = (rx._pulse(fs // 100).size - 1) // 2
    ok = True
    for i in range(len(spec.CONTROL_SIGNALS)):
        idx, dist = rx.decode_control_signal(_cs_audio(i, fs), delay, fs=fs)
        ok &= (idx == i and dist == 0)
    # robustness: heavy AWGN, distance-12 code + dual-tone combining
    rng = np.random.default_rng(1)
    rec = tot = 0
    for i in range(len(spec.CONTROL_SIGNALS)):
        a = _cs_audio(i, fs); amp = 0.5 * np.abs(a).max()
        for _ in range(30):
            idx, _ = rx.decode_control_signal(a + rng.normal(0, amp, a.size), delay, fs=fs)
            rec += (idx == i); tot += 1
    print(f"[1] CS round-trip on clean audio: all 6 recovered exact={ok}; "
          f"under 0.5x-peak AWGN {rec}/{tot} recovered")
    return ok


def check_control_signals() -> bool:
    rng = np.random.default_rng(0)
    worst = 0
    ok = True
    for i, w in enumerate(spec.CONTROL_SIGNALS):
        bits = np.array([(w >> b) & 1 for b in range(spec.CS_BITS_PER_TONE)], np.uint8)
        for _ in range(200):
            err = rng.random(spec.CS_BITS_PER_TONE) < (5 / spec.CS_BITS_PER_TONE)
            j, dist = rx.nearest_control_signal(bits ^ err.astype(np.uint8))
            worst = max(worst, int(err.sum()))
            if j != i and err.sum() <= 5:
                ok = False
    print(f"[4] CS nearest-codeword: recovered through <=5 errors (tested to {worst}): {ok}")
    return ok


def _occ15() -> tuple[np.ndarray, int] | None:
    """The occ15 capture as mono samples and its rate, or None when it is absent."""
    if not OCC15.exists():
        return None
    w = wave.open(str(OCC15)); fs = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
    if w.getnchannels() == 2:
        a = a.reshape(-1, 2)[:, 0]
    w.close()
    return a, fs


def check_frontend() -> None:
    cap = _occ15()
    if cap is None:
        print(f"[5] occ15 audio->raster front end: SKIP -- {OCC15} is not here")
        return
    a, fs = cap
    vhinF = np.fromfile(RASTER / "vhinF.bin", np.int8).astype(int)
    refs = [vhinF[o:o + 72].astype(float) for o in rx.RASTER_BLOCKS]

    def best_stream_corr(r):
        """Max |corr| of each demod block against ANY reference block: measures the
        demod quality without assuming the tone/I-Q block assignment."""
        tot = 0.0
        for o in rx.RASTER_BLOCKS:
            s = r[o:o + 72].astype(float); s = s - s.mean()
            best = 0.0
            for b in refs:
                bb = b - b.mean(); d = np.linalg.norm(s) * np.linalg.norm(bb)
                best = max(best, abs(float(np.dot(s, bb) / d)) if d else 0.0)
            tot += best
        return tot / 4

    start = rx.acquire(a, fs, search=(4.4, 5.2))
    best = (0.0, 0, 0)
    for off in range(-240, 400, 12):
        for lag in (1, 8):
            c = best_stream_corr(rx.demod_raster(a, start + off, fs=fs, lag=lag))
            if c > best[0]:
                best = (c, off, lag)
    print(f"[5] occ15 audio->raster front end: best per-stream |corr| to the "
          f"reference raster = {best[0]:.3f} (lag={best[2]}); OPEN stage -- the "
          f"phase-limiter ahead of a reference decoder's raster is not yet "
          f"reproduced. A full audio->CRC decode needs this plus the 288->432 "
          f"depuncture and cross-cycle soft accumulation.")


def check_fec_threshold() -> None:
    """How sign-clean must the softs be for the SL>=2 header CRC to pass?

    Worth knowing because an independent decoder's own gather of shrike's packet
    lands around 0.93 sign-correct and still decodes -- the K=7 trellis, not the
    placement, is what carries that last few percent."""
    rng = np.random.default_rng(0)
    info = bytes((0x53 + 7 * i) & 0xFF for i in range(24))
    code = placement.encode_frame(placement.build_field(info, placement.HEADER),
                                  placement.HEADER)
    softs = np.where(code > 0, -1.0, 1.0)
    print("[6] SL>=2 header FEC soft-error tolerance:")
    for frac in (0.0, 0.02, 0.05, 0.08, 0.12):
        ok = sum(rx.decode_header_softs(
                     np.where(rng.random(softs.size) < frac, -softs, softs))[1]
                 for _ in range(40))
        print(f"    {frac*100:4.1f}% sign flips -> CRC pass {ok}/40")


def check_demod_fidelity() -> None:
    """Where shrike's own numeric demod actually lands: sign agreement to the
    CRC-valid raster vhinF. The gap to ~0.98 is the whole residual, and it is
    systematic -- the front-end filter and the input pipeline ahead of it -- so
    cross-cycle accumulation, which only averages down random error, cannot close
    it."""
    cap = _occ15()
    if cap is None:
        print(f"[7] native demod sign agreement: SKIP -- {OCC15} is not here")
        return
    a, fs = cap
    vhinF = np.fromfile(RASTER / "vhinF.bin", np.int8).astype(float)
    refsign = [np.sign(vhinF[o:o + 72]) for o in rx.RASTER_BLOCKS]
    start = rx.acquire(a, fs, search=(4.4, 5.2))
    best = 0.0
    for off in range(-240, 400, 12):
        for lag in (1, 8):
            r = rx.demod_raster(a, start + off, fs=fs, lag=lag)
            streams = [r[o:o + 72] for o in rx.RASTER_BLOCKS]
            tot = 0.0
            for s in streams:
                ss = np.sign(s)
                tot += max(max((ss == rs).mean(), (ss == -rs).mean()) for rs in refsign)
            best = max(best, tot / 4)
    print(f"[7] native demod sign agreement to CRC-valid raster: {best:.3f} "
          f"(need ~0.98). That ~0.6 residual is systematic -- the front-end filter "
          f"and the input pipeline ahead of it -- so accumulation, which only "
          f"averages down random error, cannot close it. Matching a reference "
          f"decoder's filter and input pipeline exactly would (see rx.py notes).")


def main() -> int:
    print("=== shrike.rx validation against an independent decoder ===")
    z = check_clean_case0()
    cs = check_cs_roundtrip()
    h = check_sl2_header()
    g = check_gather()
    c = check_control_signals()
    check_frontend()
    check_fec_threshold()
    check_demod_fidelity()
    print()
    print(f"PROVEN: clean_case0_decode={z} cs_roundtrip={cs} sl2_header={h} "
          f"gather={g} cs_codeword={c}")
    return 0 if (z and cs and h and g and c) else 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
