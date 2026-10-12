# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The streaming receivers, fed the card's rate in uneven blocks, against the
batch receivers that decode the same audio whole."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import mil110, s4285, sim, stream
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.resample import for_card

MSG = bytes(np.random.default_rng(21).integers(0, 256, 700, dtype=np.uint8))


def _card(audio_fs, **ch):
    y = sim.channel(audio_fs, **ch)
    return for_card(y * 0.3, 9600)


def _feed(rx, card, seed=0):
    """Blocks of 256-4096 card samples, as a sound card's callbacks vary."""
    fe = stream.FrontEnd(CARD_RATE_HZ)
    rng = np.random.default_rng(seed)
    events, i = [], 0
    while i < card.size:
        n = int(rng.integers(256, 4096))
        events += rx.push(fe.push(card[i:i + n]))
        i += n
    # Silence after, so a receiver that waits on the signal to end sees it end.
    for _ in range(40):
        events += rx.push(fe.push(np.zeros(4096, np.float32)))
    return events


def _payload(events):
    return b"".join(e.data for e in events if isinstance(e, stream.Data))


def _pad(a, lead=4000, tail=4000):
    return np.concatenate([np.zeros(lead), a, np.zeros(tail)])


@pytest.mark.parametrize("mode", ["2400S", "600S", "4800N", "150S", "600Z", "75S"])
def test_110_stream_matches_the_message(mode):
    data = MSG[:60] if mode.startswith("75") else MSG
    card = _card(_pad(mil110.transmit(data, mode)), snr_db=18, freq_hz=-22.0, seed=1)
    ev = _feed(stream.Receiver110(zero_interleave=mode.endswith("Z")), card)
    kinds = [type(e).__name__ for e in ev]
    assert kinds[0] == "Started" and ev[0].mode == mode
    assert kinds[-1] == "Ended" and ev[-1].complete
    assert _payload(ev) == data


def test_110_bytes_arrive_before_the_end():
    """A long-interleave message is released block by block, not at the EOM."""
    data = MSG * 3
    card = _card(_pad(mil110.transmit(data, "1200L")), snr_db=20, seed=2)
    ev = _feed(stream.Receiver110(), card)
    datas = [i for i, e in enumerate(ev) if isinstance(e, stream.Data)]
    assert len(datas) > 1
    assert _payload(ev) == data


def test_110_back_to_back_transmissions():
    a = _pad(mil110.transmit(b"one", "600S"), tail=2000)
    b = _pad(mil110.transmit(b"two", "2400S"), lead=0)
    ev = _feed(stream.Receiver110(), _card(np.concatenate([a, b]), snr_db=25, seed=3))
    starts = [e.mode for e in ev if isinstance(e, stream.Started)]
    ends = [e.complete for e in ev if isinstance(e, stream.Ended)]
    assert starts == ["600S", "2400S"] and ends == [True, True]
    assert _payload(ev) == b"onetwo"


def test_110_on_the_poor_channel():
    card = _card(_pad(mil110.transmit(MSG, "600S")), snr_db=12, preset="poor",
                 freq_hz=30.0, clock_ppm=-60, seed=4)
    assert _payload(_feed(stream.Receiver110(), card)) == MSG


def test_110_msb_first():
    card = _card(_pad(mil110.transmit(MSG, "1200S")), snr_db=25, seed=5)
    got = _payload(_feed(stream.Receiver110(bit_order="msb"), card))
    flip = bytes(int(f"{b:08b}"[::-1], 2) for b in MSG)
    assert got == flip


@pytest.mark.parametrize("mode", ["1200S", "2400L", "300S"])
def test_4285_stream_frames_the_message(mode):
    audio = _pad(s4285.transmit(MSG, mode))
    card = _card(audio, snr_db=20, freq_hz=17.0, seed=6)
    ev = _feed(stream.Receiver4285(mode), card)
    assert isinstance(ev[0], stream.Started)
    assert isinstance(ev[-1], stream.Ended) and ev[-1].complete
    assert _payload(ev) == MSG


def test_4285_uncoded_stream():
    audio = _pad(s4285.transmit(MSG, "3600U"))
    ev = _feed(stream.Receiver4285("3600U"), _card(audio, snr_db=20, seed=6))
    assert _payload(ev)[:len(MSG)] == MSG


def test_4285_back_to_back():
    a = _pad(s4285.transmit(b"first", "600S"), tail=2000)
    b = _pad(s4285.transmit(b"second", "600S"), lead=0)
    ev = _feed(stream.Receiver4285("600S"), _card(np.concatenate([a, b]), snr_db=25, seed=8))
    assert [e.complete for e in ev if isinstance(e, stream.Ended)] == [True, True]
    assert _payload(ev) == b"firstsecond"


def test_noise_starts_nothing():
    card = for_card(np.random.default_rng(7).normal(0, 0.05, 9600 * 8), 9600)
    ev = _feed(stream.Receiver110(), card) + _feed(stream.Receiver4285("600S"), card)
    assert ev == []


# --------------------------------------------------------------------------
# The 39-tone mode

from hfmodem.bateleur import tone39  # noqa: E402

TEXT39 = b"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 0123456789\r\n" * 3


def _feed39(rx, card, seed=0, tail_s=10):
    """As ``_feed``, handing the receiver the front end's real audio."""
    fe = stream.FrontEnd(CARD_RATE_HZ)
    rng = np.random.default_rng(seed)
    events, i = [], 0
    while i < card.size:
        n = int(rng.integers(256, 4096))
        fe.push(card[i:i + n])
        events += rx.push(fe.audio)
        i += n
    for _ in range(tail_s * CARD_RATE_HZ // 4096):
        fe.push(np.zeros(4096, np.float32))
        events += rx.push(fe.audio)
    return events


@pytest.mark.parametrize("mode,given", [("600/33/A10.8", True), ("600/33/A10.8", False),
                                        ("2400/36/A9.7", False), ("75/12/A10.8", False)])
def test_39_stream_matches_the_message(mode, given):
    """Given the mode, or finding it from the data."""
    audio = _pad(tone39.transmit(TEXT39, mode), lead=9600)
    ev = _feed39(stream.Receiver39(mode if given else None),
                 _card(audio, snr_db=20, freq_hz=14.0, seed=2))
    starts = [e for e in ev if isinstance(e, stream.Started)]
    ends = [e for e in ev if isinstance(e, stream.Ended)]
    assert [(s.standard, s.mode) for s in starts] == [("39", tone39.mode(mode).name)]
    assert abs(starts[0].freq_hz - 14.0) < 1.0
    assert [e.complete for e in ends] == [True]
    assert abs(ends[0].snr_db - 20) < 3.0
    want = bytes(b & ((1 << tone39.mode(mode).data_bits) - 1) for b in TEXT39)
    assert _payload(ev) == want


def test_39_synchronous_through_fading():
    audio = _pad(tone39.transmit(MSG, "1200/63/S"), lead=9600)
    ev = _feed39(stream.Receiver39("1200/63/S"),
                 _card(audio, snr_db=20, freq_hz=-30.0, paths=sim.two_paths(2.0, 2.0), seed=3))
    assert _payload(ev)[:len(MSG)] == MSG
    assert [e.why for e in ev if isinstance(e, stream.Ended)] == ["signal lost"]


def test_39_stream_follows_the_sender_s_clock():
    """400 ppm walks a 30 s message 115 samples, past the guard: the reading
    follows the clock as it measures it."""
    data = MSG * 3
    audio = _pad(tone39.transmit(data, "600/33/S"), lead=9600)
    ev = _feed39(stream.Receiver39("600/33/S"),
                 _card(audio, snr_db=15, freq_hz=6.0, clock_ppm=400.0, seed=4))
    assert _payload(ev)[:len(data)] == data


def test_39_back_to_back_and_msb_first():
    a = _pad(tone39.transmit(b"first", "600/33/A10.8", order="msb"), lead=9600, tail=9600)
    b = _pad(tone39.transmit(b"second", "600/33/A10.8", order="msb"), lead=0)
    ev = _feed39(stream.Receiver39("600/33/A10.8", bit_order="msb"),
                 _card(np.concatenate([a, b]), snr_db=25, seed=8))
    assert [e.complete for e in ev if isinstance(e, stream.Ended)] == [True, True]
    assert _payload(ev) == b"firstsecond"


def test_39_noise_starts_nothing():
    card = for_card(np.random.default_rng(7).normal(0, 0.05, 9600 * 12), 9600)
    assert _feed39(stream.Receiver39(None), card, tail_s=1) == []
