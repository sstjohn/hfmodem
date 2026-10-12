# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Start-stop characters over the bit-synchronous waveforms (110B C.5.4.1.1)."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import charframe as CF, mil110, sim
from hfmodem.bateleur.modem import BateleurModem, Mode, ModemError, Observer
from hfmodem.core.resample import for_card

TEXT = b"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 0123456789\r\n"


def test_a_word_is_start_character_parity_stop():
    """'A' (0x41) as 7E1: start 0, 1000001 least significant first, even
    parity 0, stop 1."""
    assert list(CF.frame(b"A", CF.CharFormat.parse("7E1"))) == [0, 1, 0, 0, 0, 0, 0, 1, 0, 1]
    assert CF.CharFormat.parse("8N2").word_bits == 11


@pytest.mark.parametrize("bad", ["9N1", "8E1", "4N1", "8N3", "8X1", "81"])
def test_the_formats_the_standard_allows(bad):
    """5 to 8 character bits, parity included; 1 or 2 stop bits."""
    with pytest.raises(ValueError):
        CF.CharFormat.parse(bad)


@pytest.mark.parametrize("fmt", ["8N1", "7E1", "7O2", "5N2", "6M1"])
def test_framing_round_trips_a_piece_at_a_time(fmt):
    f = CF.CharFormat.parse(fmt)
    mask = (1 << f.data_bits) - 1
    data = bytes(v & mask for v in np.random.default_rng(1).integers(0, 256, 300))
    bits = np.concatenate([np.ones(13, np.uint8), CF.frame(data, f), np.ones(5, np.uint8)])
    d = CF.Deframer(f)
    out = b"".join(d.push(bits[i:i + 37]) for i in range(0, bits.size, 37))
    assert out == data and d.framing_errors == 0 and d.parity_errors == 0


def test_a_framing_error_is_dropped_and_the_stream_found_again():
    f = CF.CharFormat.parse("8N1")
    bits = CF.frame(TEXT, f)
    bits[49] ^= 1                                  # the fifth word's stop bit
    d = CF.Deframer(f)
    out = d.push(bits)
    assert d.framing_errors >= 1
    assert out.endswith(TEXT[-20:])                # back in step well before the end


def test_parity_errors_are_counted_not_dropped():
    f = CF.CharFormat.parse("7E1")
    bits = CF.frame(b"HELLO", f)
    bits[2] ^= 1                                   # a data bit of the H
    d = CF.Deframer(f)
    assert len(d.push(bits)) == 5 and d.parity_errors == 1


def test_seven_e_one_over_188_110():
    f = CF.CharFormat.parse("7E1")
    tx = mil110.transmit(CF.pack(CF.frame(TEXT, f)), "600S")
    a = np.concatenate([np.zeros(3000), tx, np.zeros(3000)])
    r = mil110.receive(sim.channel(a, snr_db=15, freq_hz=6.0, seed=3))
    assert CF.Deframer(f).push(CF.unpack(r.data)) == TEXT


class _Heard(Observer):
    def __init__(self):
        self.data = b""

    def rx_data(self, data):
        self.data += data


def _modems(fmt):
    sent = []
    tx = BateleurModem(lambda audio, fs: sent.append(audio) or "", card_rate=48000, max_key_s=60)
    heard = _Heard()
    rx = BateleurModem(lambda *_: "", card_rate=48000, max_key_s=60, observer=heard)
    for m in (tx, rx):
        m.set_framing(fmt)
    return tx, rx, heard, sent


@pytest.mark.parametrize("std, mode", [("110", "1200S"), ("4539", "3200S"), ("WBHF", "600S")])
def test_the_modem_frames_and_deframes(std, mode):
    tx, rx, heard, sent = _modems("8N1")
    tx.set_mode(Mode.parse(std, mode))
    tx.transmit(TEXT)
    card = for_card(np.concatenate([np.zeros(4800), *sent, np.zeros(48000)]).astype(np.float32),
                    9600)
    for i in range(0, card.size, 4800):
        rx._audio(card[i:i + 4800])
    assert heard.data == TEXT


def test_the_host_command():
    from hfmodem.bateleur.host import BateleurHost
    m = BateleurModem(lambda *_: "", card_rate=48000, max_key_s=60)
    h = BateleurHost(m, cmd_port=0, data_port=0)
    h._on_modem = lambda fn: fn()
    assert h.command("FRAMING ASYNC 7e1") == "OK FRAMING ASYNC 7E1"
    assert m.framing == CF.CharFormat.parse("7E1")
    assert h.command("FRAMING ASYNC 9N1").startswith("ERR")
    assert h.command("FRAMING SYNC") == "OK FRAMING SYNC" and m.framing is None
    with pytest.raises(ModemError):
        m.set_framing("bogus")
