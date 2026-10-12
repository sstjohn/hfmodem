# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The called station's search for a call to itself, as SCS's standby search
runs it.

Eleven tone pairs across +-250 Hz, eight timing candidates a bit, both shifts.
The Normal and Longpath address section is compared exactly; the 88-bit robust
image of our call is forgiven 15 bit errors, 6 under CONIntegrity 1.

Run:  pytest packages/hfmodem/hfmodem/tests/shrike/test_callsearch.py
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p1rx, pactor1
from hfmodem.shrike.arq import State
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_callb import (NEGATIVES, SELCALL, TRAINS, _load,
                                             _load_mp3, _shifted)
from hfmodem.tests.kestrel import corpora

FS = p1rx.FS
CALL = "W9SSJ"


def _robust(call: str = CALL, flips: int = 0, invert: bool = False,
            seed: int = 0) -> np.ndarray:
    bits = p1rx._bits(pactor1.call_b_bytes(call)).astype(np.uint8)
    if flips:
        rng = np.random.default_rng(seed)
        bits[rng.choice(bits.size, flips, replace=False)] ^= 1
    f = pactor1._fsk_freqs(bits, FS // 100, invert)
    burst = 0.11 * np.cos(1.22 + 2 * np.pi * np.cumsum(f) / FS)
    return np.concatenate([np.zeros(FS // 2), burst, np.zeros(FS // 2)])


def _call(variant: str, invert: bool = False, call: str = CALL) -> np.ndarray:
    return pactor1.connect_signal(call, variant=variant, invert=invert).astype(float)


@pytest.mark.parametrize("variant", ["normal", "longpath", "robust"])
@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("hz", [-250.0, -137.0, 0.0, 90.0, 250.0])
def test_a_call_is_found_across_the_grid(variant: str, invert: bool, hz: float) -> None:
    rng = np.random.default_rng(int(hz) + 300)
    audio = _shifted(_call(variant, invert), hz)
    audio += rng.normal(0, 0.02, audio.size)
    [(t, got)] = p1rx.search_calls(audio, CALL)
    assert (got.variant, got.callsign, got.inverted) == (variant.title(), CALL, invert)
    assert abs(got.offset_hz - hz) <= 8
    frame = p1rx.CALL_B_N if variant == "robust" else p1rx.CALL_N
    assert abs(t - (FS // 2 + frame) / FS) <= 0.005


@pytest.mark.parametrize("flips, strict, found", [
    (15, False, True), (16, False, False), (6, True, True), (7, True, False)])
def test_the_robust_image_forgives_up_to_the_threshold(flips: int, strict: bool,
                                                      found: bool) -> None:
    for seed, invert in ((1, False), (2, True)):
        hits = p1rx.search_calls(_robust(flips=flips, invert=invert, seed=seed),
                                 CALL, strict=strict)
        assert bool(hits) is found
        assert all(c.errors == flips for _, c in hits)


def test_the_address_section_is_compared_exactly() -> None:
    addr, red = pactor1.connect_frame_bytes(CALL)
    addr = bytes([*addr[:3], addr[3] ^ 0x04, *addr[4:]])
    f = np.concatenate([pactor1._fsk_freqs(p1rx._bits(addr), FS // 100),
                        pactor1._fsk_freqs(p1rx._bits(red), FS // 200)])
    burst = 0.11 * np.cos(1.22 + 2 * np.pi * np.cumsum(f) / FS)
    assert p1rx.search_calls(np.concatenate([np.zeros(FS // 2), burst]), CALL) == []


def test_a_call_to_someone_else_is_not_ours() -> None:
    for audio in (_call("normal", call="W9SSK"), _call("longpath", call="W9SS"),
                  _call("robust", call="K6SDR"), _call("normal", call="W9SSJA")):
        assert p1rx.search_calls(audio, CALL) == []


def test_offair_normal_connects_moved_off_our_pair() -> None:
    """KE5YTA called three times on tape. Moved 139 Hz, the monitor's decode reads
    none of them and the search reads all three, and says where they were."""
    audio = _load(corpora.REGRESS_FIXTURES / "pos_pactor1_local.wav")
    assert len(p1rx.search_calls(audio, "KE5YTA")) == 3
    moved = _shifted(audio, 139.0)
    hits = p1rx.search_calls(moved, "KE5YTA")
    assert len(hits) == 3 and all(abs(c.offset_hz - 139) < 10 for _, c in hits)
    assert p1rx.decode_connect(moved[:3 * FS]) is None


def test_offair_selcall(tmp_path: Path) -> None:
    """SCS's published sample, which its monitor reads as ten `Normal Call: OL1A`,
    sits near 2215 Hz; brought inside the grid, all ten are read."""
    audio = _shifted(_load_mp3(SELCALL, tmp_path), 1500.0 + 120.0 - 2215.0)
    hits = p1rx.search_calls(audio, "OL1A")
    assert len(hits) == 10
    assert {c.variant for _, c in hits} == {"Normal"}


@pytest.mark.parametrize("path, span, centre, call, variant, floor", TRAINS,
                         ids=[t[3] for t in TRAINS])
def test_offair_robust_trains(path: Path, span, centre: float, call: str,
                              variant: str, floor: int) -> None:
    """Every Robust Call train on file, moved inside the grid, is read at least
    as often as SCS's own monitor reads it, by nobody else's call, and placed
    in frequency to within a couple of hertz burst after burst."""
    audio = _shifted(_load(path, span), 1500.0 - 60.0 - centre)
    hits = p1rx.search_calls(audio, call)
    assert len(hits) >= floor
    assert {c.variant for _, c in hits} == {"Robust"}
    assert np.ptp([c.offset_hz for _, c in hits]) < 2
    assert p1rx.search_calls(audio, CALL) == []


@pytest.mark.parametrize("path", NEGATIVES, ids=lambda p: p.stem)
def test_no_call_where_there_is_none(path: Path) -> None:
    audio = _load(path)
    assert p1rx.search_calls(audio, CALL) == []


def _listening(**cfg) -> PtcHost:
    host = PtcHost(mycall=CALL)
    host.arq.cfg = replace(host.arq.cfg, **cfg)
    host.arq.on_host_listen(True)
    return host


def test_a_listening_host_answers_a_longpath_call_off_its_pair() -> None:
    host = _listening()
    host.feed_audio(_shifted(_call("longpath", True), 120.0), fs=FS)
    assert host.arq.state is State.CONNECTED and host.arq.answering
    assert host.arq.connect_variant == "longpath"
    assert abs(host.called_by.offset_hz - 120) < 8
    assert abs(host.called_at - (FS // 2 + p1rx.CALL_N) / FS) <= 0.005
    assert host._df() == int(host.called_by.offset_hz) & 0xFF


def test_a_listening_host_answers_a_robust_call() -> None:
    host = _listening()
    host.feed_audio(_shifted(_robust(flips=10), -200.0), fs=FS)
    assert host.arq.state is State.CONNECTED
    assert host.arq.connect_variant == "robust"


def test_conintegrity_tightens_the_robust_search() -> None:
    host = _listening(conintegrity=1)
    host.feed_audio(_robust(flips=10), fs=FS)
    assert host.arq.state is State.LISTENING
    host.feed_audio(_robust(flips=6), fs=FS)
    assert host.arq.state is State.CONNECTED
