# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Assemble a full PACTOR-3 transmit frame and render it to audio.

This is the transmit path an independent decoder judges. It deliberately exposes
every UNKNOWN (unknowns.py) as a parameter, so searching for the combination that
decodes is a loop over parameter choices rather than an edit.

Frame structure (spec §4, cycle graphic):
  [phase-ref pulse | header (8 sym) | data field | CS] per the ARQ cycle.
Here we build the ISS data packet: phase-ref, then per-tone headers, then the
coded/interleaved data field. Control signals (reverse direction) are separate.

Header mapping (U6), read from the spec: a variable header is a 32-bit code sent
as "8 symbols" across tones 5 and 12 -> 16 bits per tone / 8 symbols = 2 bits per
symbol = DQPSK. Constant headers are 16-bit -> 8 symbols x 2 bits on their tone.
This is the leading hypothesis, and only a decode by an independent receiver
settles it either way.
"""

from __future__ import annotations

import numpy as np

from . import coding, modem, spec


def _bits_msb(value: int, width: int) -> np.ndarray:
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def header_symbols_for_channel(cn: int, vh_index: int, bits_per_symbol: int) -> np.ndarray:
    """The 8 header symbols carried on channel `cn` (complex, differential).

    Tones 5 and 12 carry the two 16-bit halves of a 32-bit variable header;
    every other active tone carries its 16-bit constant header.
    """
    if cn == 5:
        code16 = (spec.VARIABLE_HEADERS[vh_index] >> 16) & 0xFFFF
    elif cn == 12:
        code16 = spec.VARIABLE_HEADERS[vh_index] & 0xFFFF
    else:
        # constant header: index by position among the non-VH channels
        non_vh = [c for c in range(spec.N_CHANNELS) if c not in spec.VH_CHANNELS]
        code16 = spec.CONSTANT_HEADERS[non_vh.index(cn) % len(spec.CONSTANT_HEADERS)]
    bits = _bits_msb(code16, 16)
    # differential_encode prepends the phase-reference symbol, so 16 bits / 2 = 8
    # data symbols + 1 reference = 9 symbols on the wire.
    return modem.differential_encode(bits, bits_per_symbol=2)


def build_frame(sl: int, payload: bytes, *,
                vh_index: int = 0,
                generators: dict | None = None,
                puncture: coding.Puncture | None = None,
                interleaver: coding.BlockInterleaver | None = None,
                crc_variant: str = "ccitt-false",
                cfg: modem.ModConfig | None = None) -> np.ndarray:
    """Render one short-cycle PACTOR-3 data frame for speed level `sl` to audio.

    Every argument after `payload` is an UNKNOWN, pinned down by whether an
    independent decoder accepts the frame it produces. Defaults use the current
    best hypotheses from unknowns.py.
    """
    s = spec.SPEED_LEVELS[sl]
    cfg = cfg or modem.ModConfig()

    # --- information packet: payload + status + CRC-16 ---
    if len(payload) > s.payload_short:
        raise ValueError(f"SL{sl} short cycle holds {s.payload_short} payload bytes")
    payload = payload.ljust(s.payload_short, b"\x00")
    info = payload + bytes([spec.status_byte(1, spec.DataType.ASCII_8BIT)])
    info += coding.crc16(info, crc_variant).to_bytes(2, "big")

    # --- coding chain: conv-encode -> puncture -> interleave ---
    code = coding.conv_code_for(sl, generators)
    if puncture is None:
        puncture = {(1, 2): coding.RATE_1_2,
                    (3, 4): coding.CANDIDATE_PUNCTURES["3/4-yasuda"],
                    (8, 9): coding.CANDIDATE_PUNCTURES["8/9-yasuda"]}[s.code_rate]
    coded = puncture.apply(code.encode(coding.bytes_to_bits(info), terminate=True))

    # spread the coded bits across (data tones x symbols); geometry is U3/U8
    n_data_tones = s.n_tones
    bits_per_sym = s.bits_per_symbol
    sym_per_tone = int(np.ceil(coded.size / (n_data_tones * bits_per_sym)))
    need = n_data_tones * bits_per_sym * sym_per_tone
    coded = np.concatenate([coded, np.zeros(need - coded.size, dtype=np.uint8)])
    if interleaver is None:
        interleaver = coding.BlockInterleaver(sym_per_tone, n_data_tones * bits_per_sym)
    woven = interleaver.interleave(coded)

    # column c (one symbol slot) holds bits for all tones; split per tone
    grid = woven.reshape(sym_per_tone, n_data_tones * bits_per_sym)

    # --- per-tone symbol streams: header symbols then data symbols ---
    tone_syms: dict[int, np.ndarray] = {}
    for ti, cn in enumerate(s.channels):
        hdr = header_symbols_for_channel(cn, vh_index, bits_per_sym)   # incl. phase ref
        # data bits for this tone, across all symbol slots
        col = grid[:, ti * bits_per_sym:(ti + 1) * bits_per_sym].ravel()
        data = modem.differential_encode(col, bits_per_sym)[1:]        # drop the extra ref
        tone_syms[cn] = np.concatenate([hdr, data])

    # pad every tone to equal length (header symbol counts already match)
    n = max(len(v) for v in tone_syms.values())
    for cn in tone_syms:
        if len(tone_syms[cn]) < n:
            tone_syms[cn] = np.concatenate([
                tone_syms[cn],
                np.full(n - len(tone_syms[cn]), tone_syms[cn][-1])])

    return modem.modulate_tones(tone_syms, cfg)


def write_wav(path: str, signal: np.ndarray, sample_rate: int = 48000,
              stereo: bool = True) -> None:
    """Write a signal as 16-bit PCM, stereo by default because that is how monitor
    software opens a loopback capture."""
    import wave
    x = np.clip(signal, -1, 1)
    pcm = (x * 32767).astype("<i2")
    if stereo:
        pcm = np.repeat(pcm.reshape(-1, 1), 2, axis=1).ravel()
    with wave.open(path, "w") as w:
        w.setnchannels(2 if stereo else 1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
