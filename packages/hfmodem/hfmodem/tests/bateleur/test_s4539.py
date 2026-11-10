# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""STANAG 4539's high-rate waveform (MIL-STD-188-110B appendix C): its tables
against the text, and end to end through the channels in ``bateleur.sim``.

Self-consistency only, as for 188-110 and 4285: no other implementation has
decoded this one, nor has it decoded another.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import fec, phy, s4539 as S, sim, stream
from hfmodem.bateleur.modem import Mode

MSG = bytes(np.random.default_rng(39).integers(0, 256, 600, dtype=np.uint8))


def _air(data, mode, **ch):
    a = np.concatenate([np.zeros(3000), S.transmit(data, mode), np.zeros(3000)])
    return sim.channel(a, **ch)


# -- the text's tables -----------------------------------------------------------

def test_table_c9_d_values_are_the_dibits_of_table_c10():
    """Table C-IX, a row of it and a column, from table C-X's bit patterns
    through table C-III."""
    assert S.d_values(S.Mode4539(3200, 1)) == (0, 0, 4)
    assert S.d_values(S.Mode4539(4800, 1)) == (0, 6, 2)
    assert S.d_values(S.Mode4539(9600, 72)) == (4, 2, 6)
    assert S.d_values(S.Mode4539(8000, 18)) == (4, 0, 0)
    assert S.d_values(S.Mode4539(12800, 1)) == (6, 6, 2)


def test_table_c12_probe_signs():
    """Table C-XII: 9600 bit/s, 36 frames is "- + - - + -", then table
    C-XIII's set count, framed by seven "-" and a "+" either side."""
    signs = S.probe_signs(S.Mode4539(9600, 36))
    sym = "".join("+" if s else "-" for s in signs)
    assert sym[:18] == "-------+" + "-+--+-" + "++-" + "+"
    assert sym[18 + 14: 18 + 17] == "+-+"          # the second set's count
    assert sym[54 + 14: 54 + 17] == "-++"          # the fourth's


def test_the_scrambler_is_figure_c5s():
    """Period 511, and the first symbol scrambled by the initial value 1."""
    v = S.scramble_values(6, 600)
    assert v[0] == 1 and S.scramble_values(3, 1)[0] == 1
    r = [0] * 8 + [1]
    seen = {}
    for i in range(600):
        t = tuple(r)
        if t in seen:
            assert i - seen[t] == 511
            break
        seen[t] = i
        r = [r[4] ^ r[8]] + r[:8]


def test_the_interleaver_loads_as_c5332s_example():
    """3200 bit/s, one frame: 512 bits, increment 97, the first eight load
    locations 0, 97, 194, 291, 388, 485, 70, 167."""
    m = S.Mode4539(3200, 1)
    assert m.interleaver_bits == 512 and m.increment == 97
    assert list((np.arange(8) * m.increment) % m.interleaver_bits) == \
        [0, 97, 194, 291, 388, 485, 70, 167]


def test_block_sizes_are_table_c14s():
    assert S.Mode4539(3200, 72).block_bits == 27648
    assert S.Mode4539(9600, 1).block_bits == 1152
    assert S.Mode4539(6400, 18).block_bits == 13824


def test_tail_biting_decodes_what_it_encodes():
    rng = np.random.default_rng(0)
    b = rng.integers(0, 2, 1152)
    c = fec.encode_tailbiting(b)
    assert c.size == 2 * b.size
    soft = (1 - 2.0 * c) + rng.normal(0, 0.6, c.size)
    assert np.array_equal(fec.viterbi_tailbiting(soft), b)


def test_the_constellations_are_the_tables_and_fit_the_unit_circle():
    assert S.QAM16.size == 16 and S.QAM32.size == 32 and S.QAM64.size == 64
    for t in (S.QAM16, S.QAM32, S.QAM64):
        assert np.max(np.abs(t)) == pytest.approx(1.0, abs=1e-5)
        assert len({complex(round(p.real, 5), round(p.imag, 5)) for p in t}) == t.size
    assert S.QAM64[35] == pytest.approx(-0.932897 + 0.360142j)    # table C-VII
    assert S.QAM64[36] == pytest.approx(-1 + 0j) and S.QAM64[4] == pytest.approx(-1j)


def test_symbol_count_matches_what_is_sent():
    for name, n in (("9600VL", 0), ("3200US", 5000), ("12800US", 777), ("4800M", 200000)):
        assert S.symbol_count(S.MODES4539[name], n) == S.symbols(bytes(n), name).size


# -- end to end ---------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["3200S", "4800VS", "6400US", "8000S", "9600US", "12800US"])
def test_every_rate_round_trips_and_names_itself(mode):
    r = S.receive(_air(MSG, mode, snr_db=30, freq_hz=-23.0, seed=1))
    assert r.mode.name == mode and r.eom and r.data == MSG


def test_64qam_through_the_poor_channel():
    """9600 bit/s, long interleave, on the ITU-R poor channel (2 ms, 1 Hz)
    at 40 dB: the block equalizer's gains come from the probes, so a fade's
    wrong decisions cannot turn them."""
    r = S.receive(_air(MSG * 6, "9600L", snr_db=40, preset="poor", freq_hz=17.0, seed=3))
    assert r.mode.name == "9600L" and r.data == MSG * 6


def test_16qam_on_the_poor_channel_at_table_c17s_snr():
    r = S.receive(_air(MSG * 4, "6400L", snr_db=24, preset="poor", freq_hz=17.0, seed=2))
    assert r.data == MSG * 4


def test_late_entry_joins_at_a_reinserted_preamble():
    """C.4: a receiver that missed the preamble reads the mode from the next
    reinserted one and decodes the interleaver blocks after it."""
    m = S.MODES4539["3200M"]
    per = m.block_bits // 8
    msg = bytes(np.random.default_rng(3).integers(0, 256, 6 * per - 8, dtype=np.uint8))
    tx = S.transmit(msg, m)
    cut = (287 + 40 * S.FRAME) * 4 + 400                  # the preamble and 40 frames gone
    a = np.concatenate([np.zeros(2000), tx[cut:], np.zeros(2000)])
    r = S.receive(sim.channel(a, snr_db=20, freq_hz=11.0, seed=5))
    assert r.mode.name == "3200M" and r.eom and r.data == msg[4 * per:]


def test_the_stream_receiver_agrees():
    from hfmodem.core.resample import for_card
    y = _air(MSG * 3, "8000S", snr_db=28, freq_hz=9.0, seed=4)
    card = for_card(y.astype(np.float32), phy.FS)
    fe = stream.FrontEnd(48000)
    rx = stream.Receiver4539()
    out, events = b"", []
    for i in range(0, card.size, 4800):
        for e in rx.push(fe.push(card[i:i + 4800])):
            if isinstance(e, stream.Data):
                out += e.data
            else:
                events.append(e)
    assert out == MSG * 3
    assert events[0].standard == "4539" and events[0].mode == "8000S"
    assert events[-1].complete and events[-1].why == "EOM"


def test_the_modem_knows_the_standard():
    m = Mode.parse("4539", "9600l")
    assert m.name == "9600L"
    assert abs(m.seconds(1000) - m.render(bytes(1000)).size / phy.FS) < 0.2


def test_the_stream_receiver_joins_late_too():
    from hfmodem.core.resample import for_card
    m = S.MODES4539["4800S"]
    per = m.block_bits // 8
    msg = bytes(np.random.default_rng(8).integers(0, 256, 16 * per - 8, dtype=np.uint8))
    tx = S.transmit(msg, m)
    cut = (287 + 30 * S.FRAME) * 4 + 400                  # the preamble and 30 frames gone
    y = sim.channel(np.concatenate([np.zeros(2000), tx[cut:], np.zeros(9600)]),
                    snr_db=25, freq_hz=-8.0, seed=6)
    card = for_card(y.astype(np.float32), phy.FS)
    fe = stream.FrontEnd(48000)
    rx = stream.Receiver4539()
    out, events = b"", []
    for i in range(0, card.size, 4800):
        for e in rx.push(fe.push(card[i:i + 4800])):
            if isinstance(e, stream.Data):
                out += e.data
            else:
                events.append(e)
    assert events[0].mode == "4800S" and events[-1].complete
    assert out == msg[8 * per:]                           # from the block after frame 72


def test_75_hz_off_on_the_poor_channel():
    """C.6.3's offset: the preamble found from a search shifted towards it,
    and the pulse modelled as it leaves the matched filter 75 Hz off."""
    r = S.receive(_air(MSG * 6, "9600L", snr_db=33, preset="poor", freq_hz=75.0, seed=1))
    assert r.mode.name == "9600L" and r.data == MSG * 6


def test_through_the_corner_of_c64s_doppler_sweep():
    """C.6.4: 3.5 Hz/s, turning at -75 Hz. Joined 61 s into the sweep, at
    -63 Hz, the corner comes 3.3 s later."""
    a = np.concatenate([np.zeros(3000), S.transmit(MSG * 16, "9600S"), np.zeros(3000)])
    skip = 61 * phy.FS
    f = sim.sweep(a.size + skip, rate_hz_s=3.5, limit_hz=75.0)[skip:]
    r = S.receive(sim.channel(a, snr_db=24, freq_hz=f, seed=2))
    assert r.mode.name == "9600S" and r.data == MSG * 16
