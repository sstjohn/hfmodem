# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The ALE modem and its receiver, through ``bateleur.sim``'s channels."""
from __future__ import annotations

import numpy as np

from hfmodem.bateleur.ale import fsk, sim, words as W


def _receive(audio) -> list:
    r = fsk.DualReceiver()
    out = []
    for i in range(0, audio.size, 800):
        out += r.push(audio[i:i + 800])
    return [o.frame for o in out if isinstance(o, fsk.Received)]


def test_a_redundant_word_is_49_tones_of_8_ms():
    """A.5.1.3: 125 tones a second, 49 to a triple word, 392 ms."""
    a = fsk.modulate([W.word(W.TO, "JOE")])
    assert a.size == 49 * 64 and a.size / fsk.FS == 0.392


def test_every_tone_is_whole_cycles_and_where_the_table_puts_it():
    """A.5.1.2's frequencies; each 8 ms tone starts and ends at a peak."""
    for bits, hz in fsk.TONE_HZ.items():
        assert (hz * 8) % 1000 == 0
    a = fsk.modulate([W.word(W.DATA, bits=0)])
    spec = np.abs(np.fft.rfft(a[:64]))
    assert abs(np.argmax(spec) * fsk.FS / 64 - 750) < 1          # 000 -> 750 Hz
    assert a[0] == 0 and a[-1] == 0
    assert np.all(np.isclose(a[64::64], 1.0))                      # zero slope, at maxima


def test_a_call_with_a_message_is_read_through_noise():
    ws = W.call("JOE", "SAMUEL", scan_words=4, message=W.amd_words("hello from sam"))
    frames = _receive(sim.air(fsk.modulate(ws), snr_db=0, seed=1))
    assert any(f.to == ["JOE"] and f.caller == "SAMUEL" and f.amd == ["HELLO FROM SAM"]
               for f in frames)


def test_a_60_hz_offset_is_found_and_followed():
    """A.4.2.3: 60 Hz of Doppler shift may cost no more than a decibel."""
    ws = W.call("SAM", "JOE", scan_words=4)
    for hz in (60.0, -60.0):
        frames = _receive(sim.air(fsk.modulate(ws), snr_db=0, freq_hz=hz, seed=2))
        assert any(f.to == ["SAM"] and f.caller == "JOE" for f in frames), hz


def test_the_poor_channel():
    """Table A-II's modified CCIR poor channel, 2.2 ms and 1 Hz: a quarter of
    each tone is echo, which the receiver's shorter window steps around."""
    ws = W.call("SAM", "JOE")
    got = sum(any(f.to == ["SAM"] and f.caller == "JOE"
                  for f in _receive(sim.air(fsk.modulate(ws), snr_db=11,
                                            preset="ale-poor", seed=s)))
              for s in range(6))
    assert got >= 5


def test_noise_is_not_a_call():
    rng = np.random.default_rng(5)
    r = fsk.DualReceiver()
    out = []
    x = rng.normal(0, 1, fsk.FS * 30)
    for i in range(0, x.size, 800):
        out += r.push(x[i:i + 800])
    assert out == []


def test_an_expected_word_is_taken_only_when_it_is_nearly_exact():
    """The receiver compares a word it could not decode with what the
    protocol expects -- but noise must not match."""
    r = fsk.Receiver()
    r.expect(W.call("SAM", "JOE"))
    rng = np.random.default_rng(9)
    out = []
    x = rng.normal(0, 1, fsk.FS * 20)
    for i in range(0, x.size, 800):
        out += r.push(x[i:i + 800])
    assert out == []


def test_a_dtm_block_is_read_through_the_whole_of_it():
    """A.5.7.3: once a CMD DTM is read the receiver stays with the data block
    for its whole length, whatever the words hold."""
    import numpy as np
    from hfmodem.bateleur.ale import fsk, sim, words as W
    data = bytes(range(256))
    blocks = W.dtm_blocks(data, max_words=40)
    tx = fsk.modulate(W.call("JOE", "SAM", message=sum(blocks, [])))
    ok = 0
    for seed in range(3):
        y = sim.air(tx, snr_db=4.0, preset="ale-poor", seed=seed)
        r = fsk.DualReceiver()
        got = []
        for i in range(0, y.size, 800):
            got += r.push(y[i:i + 800])
        got += r.push(np.zeros(3 * fsk.FS))
        frames = [o.frame for o in got if isinstance(o, fsk.Received) and o.frame.dtm]
        if frames and all(d.ok for d in frames[0].dtm):
            bits = sum((d.data_bits for d in frames[0].dtm), [])
            ok += bytes(int("".join(map(str, bits[i:i + 8])), 2)
                        for i in range(0, len(bits), 8)) == data
    assert ok >= 2


def test_a_dbm_block_is_read_at_the_frames_word_phase():
    """A.5.7.4: the block starts exactly where the CMD DBM ends, so the
    word phase must be exact -- a phase a redundant copy out reads every
    word, and loses the block."""
    import numpy as np
    from hfmodem.bateleur.ale import fsk, sim, words as W
    data = bytes(range(256)) * 2
    blocks = W.dbm_blocks(data, max_depth=49 * 4)
    tx = fsk.modulate(W.call("JOE", "SAM", message=sum(blocks, [])))
    ok = 0
    for seed in range(4):
        y = sim.air(tx, snr_db=-3.0, seed=seed)
        r = fsk.DualReceiver()
        got = []
        for i in range(0, y.size, 800):
            got += r.push(y[i:i + 800])
        got += r.push(np.zeros(3 * fsk.FS))
        frames = [o.frame for o in got if isinstance(o, fsk.Received) and o.frame.dbm]
        if frames and all(d.ok for d in frames[0].dbm):
            bits = sum((d.data_bits for d in frames[0].dbm), [])
            ok += bytes(int("".join(map(str, bits[i:i + 8])), 2)
                        for i in range(0, len(bits), 8)) == data
    assert ok >= 3
