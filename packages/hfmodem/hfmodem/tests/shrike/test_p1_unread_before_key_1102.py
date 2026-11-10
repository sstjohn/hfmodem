# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A burst keyed from a decode behind the key left the reply in front of it unread.

WS8EOC-80, onair-20261102T015731. TX[13] keyed slot 12 and dropped at 769304.
Hold 1's grid read then took the 14.83 s CS1 behind it and the ARQ keyed TX[14]
at once, PTT up at 780765. `_tx` drained 769304 onwards into the rolling
decoder, which reads nothing there, and no hold window covers the span:
hold_01 ends at 720256 and hold_02 starts at 829304. WS8EOC's CS1 at 16.12 s,
inverted again after 14.83 and so its acceptance, sat in it at 0 bit errors.

Fixtures: stream samples 769304-780765 and hold_02, 829304-840064, of that arm.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair, rxfront
from hfmodem.shrike.arq import ISS
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol

from .recorded_pcm import recorded_pcm
from .test_tx_latency_correction import _Stream

FS = onair.FS
CARRIER_END, KEY_UP, HOLD_02 = 769304, 780765, 829304
GAP = {"file": "ws8eoc-1102-0157-unread-before-key.wav",
       "sha256": "f7546ba85f44f16b6e17db94a102608b1b107e1dd83e2b2e2dc7fdbf703337da"}
HOLD = {"file": "ws8eoc-1102-0157-hold-02.wav",
        "sha256": "c0bb8b00551147fcef20dae179237f7a66490b214e5a942328ebebd8fb5069c1"}
BURST, BURST_N = 773912, 5520
"""Where `_peer_bursts` puts the 16.12 s answer, and how long."""


class _Seam:
    def attach(self, host) -> None:
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


def _sessrx():
    host = PtcHost(peer=_Seam(), mycall="W9SSJ")
    host.arq.role, host.arq.dxcall = ISS, "WS8EOC"
    host.arq._enter_connected()
    seen: list = []
    host.on_rx_event = seen.append
    return host, onair._SessionRx(host, tag="TEST"), seen


def _words(seen):
    return [(ev.cs, ev.sense, round(ev.t, 2)) for ev in seen if ev.kind == "cs"]


@pytest.fixture(scope="module")
def gap() -> np.ndarray:
    return recorded_pcm(GAP)


def _keyed(tmp_path, audio, host, *, pos=CARRIER_END, loss=False):
    """TX[14] through `RadioTx._tx`, from the read position TX[13] left."""
    class Stream(_Stream):
        def take_until(self, index):
            seg = audio[self.pos - CARRIER_END:max(0, index - CARRIER_END)]
            self.pos += seg.size
            return seg

    stream = Stream()
    stream.samples, stream.pos = KEY_UP, pos
    rig = SimpleNamespace(ptt=lambda keyed: True, key_failure=lambda: None)
    tx = onair.RadioTx(rig, transmit=True, outdir=tmp_path, settle=.04)
    tx.live, tx.tx_end, tx.host = stream, CARRIER_END, host
    tx.boundary = KEY_UP + round(tx.settle * FS)
    if loss:
        tx._observe_input_loss = lambda where: True
    tx._tx(np.ones(46080, np.float32), "P1 pkt#1 200Bd 7B")
    return tx


def _read(sessrx, tx, host, sending=True):
    onair._read_unread_before_key(sessrx, tx, host, SimpleNamespace(sending=sending))


def _span(host, audio, origin, carrier_end, live):
    return (origin, audio, carrier_end, host.arq._link_generation, onair._loss_epoch(live))


def _deliver(sessrx, t, sense):
    """A CS1 some other reader of the hold window delivered."""
    sessrx._on(rxfront.Event(t, "cs", "CS1  (test)", protocol="PACTOR-1",
                             sense=sense, cs=0))


def test_the_key_keeps_what_it_drained(tmp_path, gap):
    host, _, _ = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    [(origin, audio, carrier_end, generation, epoch)] = tx.unread_before_key
    assert (origin, carrier_end, generation, epoch) == (
        CARRIER_END, CARRIER_END, host.arq._link_generation, onair._loss_epoch(tx.live))
    assert BURST + BURST_N <= origin + audio.size <= KEY_UP


def test_the_next_hold_reads_the_acceptance(tmp_path, gap):
    host, sessrx, seen = _sessrx()
    sessrx.cs_seen = True       # hold 1 delivered 14.83 s
    tx = _keyed(tmp_path, gap, host)
    _read(sessrx, tx, host)
    words = _words(seen)
    assert len(words) == 1 and words[0][:2] == (0, 1), words
    assert abs(words[0][2] - 16.12) <= .03, words
    assert tx.unread_before_key == []


def test_it_leaves_the_next_window_its_own_codeword(tmp_path, gap):
    host, sessrx, seen = _sessrx()
    sessrx.cs_seen = True
    tx = _keyed(tmp_path, gap, host)
    _read(sessrx, tx, host)
    sessrx.new_cycle()
    hold = recorded_pcm(HOLD)
    onair._read_codeword_at_bursts(
        sessrx, hold, HOLD_02, [at for at, _ in onair._peer_bursts(hold, HOLD_02)])
    assert [w[:2] for w in _words(seen)] == [(0, 1), (0, 0)], _words(seen)
    assert abs(_words(seen)[1][2] - 17.37) <= .03, _words(seen)


def test_a_span_keyed_by_the_read_is_read_before_the_next_window(tmp_path, gap):
    """16.12 keys the next burst, whose drain holds 17.37: both go in order,
    in this one call, before the hold window that would hold 18.61."""
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    hold = recorded_pcm(HOLD)
    def keyed(ev):
        seen.append(ev)
        if len(seen) == 1:
            tx.unread_before_key.append(_span(host, hold, HOLD_02, HOLD_02, tx.live))
    host.on_rx_event = keyed
    _read(sessrx, tx, host)
    assert [(w[0], w[1]) for w in _words(seen)] == [(0, 1), (0, 0)], _words(seen)
    assert [round(w[2], 1) for w in _words(seen)] == [16.1, 17.4], _words(seen)
    assert tx.unread_before_key == []


def test_a_span_behind_a_newer_word_is_dropped(tmp_path, gap):
    """16.12, then 18.61 from the window, then 17.36 left queued: never delivered."""
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    _read(sessrx, tx, host)
    tx.unread_before_key.append(_span(host, recorded_pcm(HOLD), HOLD_02, HOLD_02, tx.live))
    _deliver(sessrx, 18.611, 1)
    _read(sessrx, tx, host)
    assert [round(w[2], 2) for w in _words(seen)] == [16.12, 18.61], _words(seen)
    assert tx.unread_before_key == []


def test_a_word_another_reader_delivered_is_not_delivered_again(tmp_path, gap):
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    _read(sessrx, tx, host)
    tx.unread_before_key.extend(_keyed(tmp_path, gap, host).unread_before_key)
    _read(sessrx, tx, host)
    assert len(_words(seen)) == 1, _words(seen)


def test_only_the_delivered_burst_is_left_out_of_a_long_span(tmp_path, gap):
    """A re-aimed drain over two answers, the first already delivered: the
    second is still read. Zeros stand in for our muted carrier between them."""
    host, sessrx, seen = _sessrx()
    hold = recorded_pcm(HOLD)
    long = np.concatenate([gap, np.zeros(HOLD_02 - KEY_UP, np.float32), hold])
    tx = _keyed(tmp_path, gap, host)
    tx.unread_before_key[:] = [_span(host, long, CARRIER_END, CARRIER_END, tx.live)]
    _deliver(sessrx, BURST / FS, 1)
    _read(sessrx, tx, host)
    assert [(w[1], round(w[2], 1)) for w in _words(seen)] == [(1, 16.1), (0, 17.4)], _words(seen)


def _long(gap):
    """Two answers in one re-aimed drain. Zeros stand in for our muted carrier."""
    return np.concatenate([gap, np.zeros(HOLD_02 - KEY_UP, np.float32), recorded_pcm(HOLD)])


def test_every_answer_in_a_long_span_is_read_in_order(tmp_path, gap):
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    tx.unread_before_key[:] = [_span(host, _long(gap), CARRIER_END, CARRIER_END, tx.live)]
    _read(sessrx, tx, host)
    assert [(w[1], round(w[2], 1)) for w in _words(seen)] == [(1, 16.1), (0, 17.4)], _words(seen)


def test_an_unassigned_word_hides_only_its_own_burst(tmp_path, gap):
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    tx.unread_before_key[:] = [_span(host, _long(gap), CARRIER_END, CARRIER_END, tx.live)]
    sessrx.cs_log.append(SimpleNamespace(t=BURST / FS, kind="unassigned"))
    _read(sessrx, tx, host)
    assert [(w[1], round(w[2], 1)) for w in _words(seen)] == [(0, 17.4)], _words(seen)


def test_the_grid_read_does_not_deliver_a_word_again(tmp_path, gap, monkeypatch):
    """Below the onset detector the grid read is the reader: an unassigned
    word another reader took 19 ms after our carrier is not taken twice."""
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    monkeypatch.setattr(onair, "_peer_bursts", lambda audio, origin: [])
    sessrx.cs_log.append(SimpleNamespace(t=(CARRIER_END + round(.019 * FS)) / FS,
                                         kind="unassigned"))
    _read(sessrx, tx, host)
    assert not seen


def test_the_grid_read_still_takes_a_word_nothing_delivered(tmp_path, gap, monkeypatch):
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    monkeypatch.setattr(onair, "_peer_bursts", lambda audio, origin: [])
    _read(sessrx, tx, host)
    assert [w[:2] for w in _words(seen)] == [(0, 1)], _words(seen)


def test_a_span_behind_a_newer_unassigned_word_is_dropped(tmp_path, gap):
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    sessrx.cs_log.append(SimpleNamespace(t=18.611, kind="unassigned"))
    _read(sessrx, tx, host)
    assert not seen and tx.unread_before_key == []


def test_a_loss_in_what_a_delivery_keys_ends_the_span(tmp_path, gap):
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    tx.unread_before_key[:] = [_span(host, _long(gap), CARRIER_END, CARRIER_END, tx.live)]
    def keyed(ev):
        seen.append(ev)
        tx.live.lost = getattr(tx.live, "lost", 0) + 480
    host.on_rx_event = keyed
    _read(sessrx, tx, host)
    assert [round(w[2], 1) for w in _words(seen)] == [16.1], _words(seen)


def test_the_same_span_without_the_burst_reads_nothing(tmp_path, gap):
    quiet = gap.copy()
    quiet[BURST - CARRIER_END:BURST - CARRIER_END + BURST_N] = 0
    host, sessrx, seen = _sessrx()
    _read(sessrx, _keyed(tmp_path, quiet, host), host)
    assert not seen


def test_an_ordinary_cycle_keeps_nothing(tmp_path, gap):
    host, _, _ = _sessrx()
    assert _keyed(tmp_path, gap, host, pos=KEY_UP - round(.015 * FS)).unread_before_key == []


def test_a_capture_loss_refusal_keeps_nothing(tmp_path, gap):
    host, _, _ = _sessrx()
    tx = _keyed(tmp_path, gap, host, loss=True)
    assert tx.refused and tx.unread_before_key == []


def test_a_span_queued_before_a_capture_loss_is_dropped(tmp_path, gap):
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    tx.live.lost = getattr(tx.live, "lost", 0) + 480
    _read(sessrx, tx, host)
    assert not seen and tx.unread_before_key == []


def test_a_span_from_another_link_is_dropped(tmp_path, gap):
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    host.arq._link_generation += 1
    _read(sessrx, tx, host)
    assert not seen and tx.unread_before_key == []


@pytest.mark.parametrize("sending, protocol", [(False, Protocol.PACTOR1),
                                                (True, Protocol.PACTOR3)])
def test_kept_audio_is_consumed_and_read_only_by_a_p1_sender(
        tmp_path, gap, sending, protocol):
    host, sessrx, seen = _sessrx()
    tx = _keyed(tmp_path, gap, host)
    host.protocol = protocol
    _read(sessrx, tx, host, sending)
    assert not seen and tx.unread_before_key == []
