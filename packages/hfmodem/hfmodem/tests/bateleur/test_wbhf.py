# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""MIL-STD-188-110C appendix D in 3 kHz: its tables against the text, and end
to end through the channels in ``bateleur.sim``.

Self-consistency only: no other implementation has decoded this one, nor has
it decoded another.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import phy, sim, wbhf as W
from hfmodem.bateleur.spec import KNOWN_POINTS

MSG = bytes(np.random.default_rng(110).integers(0, 256, 300, dtype=np.uint8))


def _air(data, mode, **ch):
    a = np.concatenate([np.zeros(3000), W.transmit(data, mode), np.zeros(3000)])
    return sim.channel(a, **ch)


# -- the text's tables -----------------------------------------------------------

def test_the_walsh_scrambler_is_d514s_example():
    """D.5.1.4: the first 32 values, and the dibit 01 scrambled with them."""
    assert list(W.TRI[:32]) == [5, 6, 2, 1, 7, 3, 1, 1, 6, 0, 5, 4, 0, 7, 7, 0,
                                5, 3, 1, 3, 3, 2, 2, 5, 5, 4, 7, 3, 5, 4, 3, 0]
    chips = W.walsh_block(np.array([1, 0]))[:32]
    assert list(chips) == [5, 2, 2, 5, 7, 7, 1, 5, 6, 4, 5, 0, 0, 3, 7, 4,
                           5, 7, 1, 7, 3, 6, 2, 1, 5, 0, 7, 7, 5, 0, 3, 4]


def test_the_shifted_mini_probe_is_table_d22c():
    """The shifted 24-symbol probe: the base's last 7, all 13, its first 4."""
    m = W.MODES_WBHF["12000L"]
    base = W.BASES[24][0]
    assert list(W.probe(m, True)) == list(base[6:]) + list(base) + list(base[:4])
    assert list(W.probe(m)) == list(base) + list(base[:11])


def test_the_mini_probes_are_unit_psk():
    for k, (base, _) in W.BASES.items():
        assert np.allclose(np.abs(KNOWN_POINTS[base]), 1.0)
    # Table D-XXV's sixth symbol on is a fifth root of unity.
    assert np.isclose(KNOWN_POINTS[W.BASES[48][0][6]], 0.309017 - 0.951057j, atol=1e-5)


def test_every_mode_has_its_rate():
    """Table D-II: an input block takes as long to send as its bits at the rate."""
    for m in W.MODES_WBHF.values():
        secs = m.frames * m.frame_symbols / 2400
        assert m.block_bits / secs == pytest.approx(m.w.rate)
    assert len(W.MODES_WBHF) == 14 * 4 - 1               # waveform 0 has no US


def test_the_code_fills_the_interleaver_and_decodes():
    rng = np.random.default_rng(5)
    for m in W.MODES_WBHF.values():
        b = rng.integers(0, 2, m.block_bits).astype(np.uint8)
        e = W.encode_block(b, m)
        assert e.size == m.interleaver_bits
        assert (W.decode_block(1.0 - 2.0 * e, m) == b).all()


def test_one_third_is_two_thirds_after_repeating():
    """D.5.3.2.4's example: T1 T2 T1 P for each input bit."""
    m = W.MODES_WBHF["600US"]
    b = np.random.default_rng(2).integers(0, 2, m.block_bits).astype(np.uint8)
    from hfmodem.bateleur import fec
    pairs = fec.encode_tailbiting(b).reshape(-1, 2)
    want = np.stack([pairs[:, 0], pairs[:, 1], pairs[:, 0]], axis=1).ravel()
    size = m.interleaver_bits
    got = W.encode_block(b, m)[(np.arange(size) * m.increment) % size]
    assert (got == want).all()


def test_the_waveform_id_is_table_d15_to_d17():
    for m in W.MODES_WBHF.values():
        d = W.wid_dibits(m)
        bits = [(x >> s) & 1 for x in d for s in (1, 0)]      # d9 ... d0
        assert bits[0] * 8 + bits[1] * 4 + bits[2] * 2 + bits[3] == m.waveform
        assert bits[4] * 2 + bits[5] == W.INTERLEAVERS.index(m.interleaver)
        assert bits[6] == 0
        d9, d8, d7, d6, d5, d4, d3, d2, d1, d0 = bits
        assert (d2, d1, d0) == (d9 ^ d8 ^ d7, d7 ^ d6 ^ d5, d5 ^ d4 ^ d3)


def test_the_count_carries_its_parity():
    for c in range(32):
        d = W.count_dibits(c)
        b = [(x >> s) & 1 for x in d for s in (1, 0)]          # b7 ... b0
        b7, b6, b5, b4, b3, b2, b1, b0 = b
        assert b4 * 16 + b3 * 8 + b2 * 4 + b1 * 2 + b0 == c
        assert (b7, b6, b5) == (b1 ^ b2 ^ b3, b2 ^ b3 ^ b4, b0 ^ b1 ^ b2)


def test_the_constellations_fit_the_unit_circle():
    assert W.QAM256.size == 256 and np.unique(W.QAM256).size == 256
    assert np.max(np.abs(W.QAM256)) == pytest.approx(1.0, abs=1e-5)


def test_symbol_count_matches_what_is_sent():
    for name, n in (("9600L", 0), ("75S", 33), ("150US", 500), ("16000M", 7000)):
        m = W.MODES_WBHF[name]
        assert W.symbol_count(m, n, superframes=3, tlc=2) == \
            W.symbols(bytes(n), m, superframes=3, tlc=2).size


# -- end to end ---------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["75S", "150US", "300S", "600US", "1200S", "1600US",
                                  "3200S", "4800US", "6400US", "8000US", "9600US",
                                  "12000US", "16000US", "2400US"])
def test_every_waveform_round_trips_and_names_itself(mode):
    r = W.receive(_air(MSG, mode, snr_db=35, freq_hz=-23.0, seed=1))
    assert r.mode.name == mode and r.eom and r.data == MSG


def test_a_longer_preamble_is_counted_down():
    """Three superframes and two level-control blocks: found at the first,
    and its count says two more come before the data."""
    a = np.concatenate([np.zeros(3000), W.transmit(MSG, "4800S", superframes=3, tlc=2),
                        np.zeros(3000)])
    r = W.receive(sim.channel(a, snr_db=25, freq_hz=12.0, seed=4))
    assert r.mode.name == "4800S" and r.data == MSG


def test_a_zero_checksum_bit_is_accepted():
    """The sentence after table D-XVI has w1's last bit 0; a sender that
    obeys it is understood."""
    m = W.MODES_WBHF["6400S"]
    hdr = W.wid_dibits(m, checksum=False)
    assert hdr != W.wid_dibits(m)
    orig = W.wid_dibits
    try:
        W.wid_dibits = lambda mode, checksum=True: orig(mode, False)
        a = np.concatenate([np.zeros(3000), W.transmit(MSG, m), np.zeros(3000)])
    finally:
        W.wid_dibits = orig
    r = W.receive(sim.channel(a, snr_db=30, seed=2))
    assert r.mode == m and r.data == MSG


def test_walsh_on_the_poor_channel():
    """Waveform 0 at table D-LII's poor-channel SNR, -1 dB."""
    r = W.receive(_air(MSG[:60], "75M", snr_db=-1, preset="poor", freq_hz=17.0, seed=3))
    assert r.mode.name == "75M" and r.data == MSG[:60]


def test_64qam_on_the_poor_channel():
    r = W.receive(_air(MSG * 8, "9600L", snr_db=35, preset="poor", freq_hz=17.0, seed=3))
    assert r.mode.name == "9600L" and r.data == MSG * 8


def test_256qam_on_white_noise():
    r = W.receive(_air(MSG * 4, "16000S", snr_db=32, freq_hz=9.0, seed=2))
    assert r.data == MSG * 4


def test_the_modem_knows_the_standard():
    from hfmodem.bateleur.modem import Mode
    m = Mode.parse("wbhf", "16000us")
    assert (m.standard, m.name) == ("WBHF", "16000US")
    for name in ("75S", "16000L", "2400M"):
        m = Mode.parse("WBHF", name)
        assert abs(m.seconds(1000) - m.render(bytes(1000)).size / phy.FS) < 0.2


def test_a_5066_node_changes_rate_onto_it():
    from hfmodem.bateleur.modem import Mode
    from hfmodem.bateleur.s5066 import onair
    assert onair.mode_for(16000, "long", "WBHF") == Mode("WBHF", "16000L")
    assert onair.mode_for(9600, "short", "WBHF") == Mode("WBHF", "9600S")
    assert onair.rate_of(Mode("WBHF", "150L")) == (150, "long")


def test_75_hz_off():
    """D.6.4's offset."""
    r = W.receive(_air(MSG, "3200S", snr_db=25, preset="poor", freq_hz=75.0, seed=6))
    assert r.data == MSG


# -- streaming ------------------------------------------------------------------------

def _stream(y):
    from hfmodem.bateleur import stream
    from hfmodem.core.resample import for_card
    card = for_card(y.astype(np.float32), phy.FS)
    fe = stream.FrontEnd(48000)
    rx = stream.ReceiverWbhf()
    out, events = b"", []
    for i in range(0, card.size, 4800):
        for e in rx.push(fe.push(card[i:i + 4800])):
            if isinstance(e, stream.Data):
                out += e.data
            else:
                events.append(e)
    return out, events


@pytest.mark.parametrize("mode", ["8000S", "75S"])
def test_the_stream_receiver_agrees(mode):
    msg = MSG * 3 if mode != "75S" else MSG[:40]
    y = np.concatenate([_air(msg, mode, snr_db=28, freq_hz=9.0, seed=4),
                        _air(msg[:20], "1200US", snr_db=28, freq_hz=-5.0, seed=5),
                        sim.channel(np.zeros(phy.FS), snr_db=28, seed=6)])     # the band goes on
    out, events = _stream(y)
    assert out == msg + msg[:20]
    assert [(e.standard, e.mode) for e in events[0::2]] == [("WBHF", mode), ("WBHF", "1200US")]
    assert all(e.complete and e.why == "EOM" for e in events[1::2])


@pytest.mark.parametrize("mode", ["75S", "6400S"])
def test_without_an_eom_the_reception_ends_when_the_signal_goes(mode):
    """D.5.4.3 makes the EOM optional; the message is whole blocks, and the
    receiver hands them all on once the signal has gone."""
    m = W.MODES_WBHF[mode]
    msg = bytes(np.random.default_rng(4).integers(0, 256, 2 * m.block_bits // 8,
                                                  dtype=np.uint8))
    a = np.concatenate([np.zeros(3000), W.transmit(msg, m, eom=False), np.zeros(phy.FS * 6)])
    r = W.receive(sim.channel(a, snr_db=20, freq_hz=5.0, seed=8))
    assert not r.eom and r.data[:len(msg)] == msg
    assert r.end_sample < a.size - 2 * phy.FS


def test_through_the_corner_of_d65s_doppler_sweep():
    """D.6.5: waveform 10 at 24 dB, swept at 3.5 Hz/s between -75 and 75 Hz.
    Joined 61 s into the sweep, at -63 Hz, the corner comes 3.3 s later."""
    a = np.concatenate([np.zeros(3000), W.transmit(MSG * 20, "9600S"), np.zeros(3000)])
    skip = 61 * phy.FS
    f = sim.sweep(a.size + skip, rate_hz_s=3.5, limit_hz=75.0)[skip:]
    r = W.receive(sim.channel(a, snr_db=24, freq_hz=f, seed=2))
    assert r.mode.name == "9600S" and r.data == MSG * 20


def test_75_hz_off_on_the_poor_channel():
    """D.6.4: waveform 10, long interleave, 75 Hz off on 2 ms / 1 Hz at
    30 dB."""
    r = W.receive(_air(MSG * 12, "9600L", snr_db=30, preset="poor", freq_hz=-75.0, seed=1))
    assert r.mode.name == "9600L" and r.data == MSG * 12
