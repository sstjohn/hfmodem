# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`--p3-changeover-p1-cs`: a PACTOR-1 codeword in the PACTOR-3 answer slot.

A gateway reads our PACTOR-3 controls -- WS8EOC obeyed our CS4 sixteen times and
stepped its counter on every one of them -- and peers stuck at seq=0 still do not
advance on our PACTOR-3 CS1. The PACTOR-1 CS1 is the codeword with a record:
KB5LZK 2026-08-22 and WS8EOC 2026-08-30 keyed it and their peers advanced, VE1YZ
2026-09-02 keyed CS2 and its peer repeated the same packet 45 times. So this
swaps the renderer and nothing else -- the link, the packet counter, the
alternation, the answer slot and the guards all stay where they were -- and the
burst is read back through our own PACTOR-1 reader.

Run: python -m pytest hfmodem/tests/shrike/test_changeover_p1_cs.py
"""
import sys

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p1rx, pactor1, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_control_profiles import (NOMINAL_REPLY_N,
                                                        _received_reply)


def _keyed(tmp_path, monkeypatch, **kw):
    rendered = []
    real = pactor1.control_signal

    def spy(*a, **k):
        rendered.append((a, k, real(*a, **k)))
        return rendered[-1][2]

    monkeypatch.setattr(onair.pactor1, "control_signal", spy)
    session, tx, live, grid, slot, emitted = _received_reply(
        tmp_path, monkeypatch, None, None, **kw)
    at = grid._p3_peer[0]
    assert grid.boundary(slot) == at + NOMINAL_REPLY_N
    tx.emit_pending_cs()
    assert len(emitted) == 1 and not tx.refused
    return session, tx, grid, at, emitted[0], rendered


def test_the_recorded_changeover_is_answered_with_a_pactor1_codeword(
        tmp_path, monkeypatch):
    _, tx, _, _, audio, rendered = _keyed(tmp_path, monkeypatch,
                                          p1_changeover=True)
    assert len(rendered) == 1
    (index,), kw, codeword = rendered[0]
    assert index == arq.CS_ACK and kw["msb_first"] == tx.p1_ack_msb
    raw = onair._trim_silence(codeword)
    want = (raw * (tx.drive * tx.p1_drive / max(abs(raw)))).astype(np.float32)
    np.testing.assert_allclose(audio, want, rtol=0, atol=2e-12)
    # 120 ms of FSK against the 248.7 ms of DBPSK it replaces.
    assert round(len(audio) / onair.FS, 3) == round(spec.P1_CS_S, 3)


def test_our_own_reader_takes_it_back_off_the_air(tmp_path, monkeypatch):
    _, _, _, _, audio, rendered = _keyed(tmp_path, monkeypatch,
                                         p1_changeover=True)
    got = p1rx.decode_control_signal(audio.astype(np.float64), 0.0, spec.P1_CS_S)
    assert got is not None and got.index == arq.CS_ACK and got.errors == 0
    # The shift the grid gave this cycle is the shift that reached the air.
    assert got.sense == int(rendered[0][1]["invert"])


def test_the_key_instant_stays_on_the_pactor3_raster(tmp_path, monkeypatch):
    """The peer read the entry packet, so its window is the LINK's -- 0.890 s
    past its own packet, not the PACTOR-1 turnaround the codeword usually keys
    on (`docs/protocols/pactor/pactor3.md`, "The slot moves when the LINK does")."""
    _, tx, grid, at, audio, _ = _keyed(tmp_path, monkeypatch, p1_changeover=True)
    assert tx.tx_audio_start == at + NOMINAL_REPLY_N
    assert tx.tx_end - tx.tx_audio_start == len(audio)
    # An FSK codeword has no pulse centers, and the reply-timing chain still
    # needs the instant the burst was aimed at.
    assert tx.tx_pulse_offsets is None
    assert grid._p3_controls[-1] == (at + NOMINAL_REPLY_N, grid.cycle_n)


def test_the_tape_can_be_scored_without_guesswork(tmp_path, monkeypatch, capsys):
    _keyed(tmp_path, monkeypatch, p1_changeover=True)
    assert "CS1 ACK [pactor-1]" in capsys.readouterr().out


def test_it_is_off_unless_asked(tmp_path, monkeypatch):
    session, tx, grid, at, audio, _ = _keyed(tmp_path, monkeypatch)
    assert not tx._pending_p3_cs_p1
    assert tx.tx_pulse_offsets is not None
    assert len(audio) > round(0.2 * onair.FS)


class _Peer:
    """A driver seam that records the renderer it was asked for."""

    def __init__(self):
        self.sent = []

    def attach(self, host):
        pass

    def send_cs(self, index, *, p1_codeword=False):
        self.sent.append((index, p1_codeword))

    def send_p1_cs(self, index):
        self.sent.append(("p1", index))


def _host():
    peer = _Peer()
    host = PtcHost(peer, mycall="W9SSJ")
    host.protocol = spec.Protocol.PACTOR3
    host.p3_changeover_p1_cs = True
    return host, peer


@pytest.mark.parametrize("cs_index", [arq.CS_BREAKIN, arq.CS_SPEED_UP,
                                      arq.CS_NAK, arq.CS_CYCLE_TOG])
def test_only_the_two_codewords_both_tables_share_change_renderer(cs_index):
    """PACTOR-1 has four codewords, its CS4 is a speed-DOWN to 100 Bd where
    PACTOR-3's demands a speed-up, and it has no CS5 or CS6 at all."""
    host, peer = _host()
    host._p3_changeover_answer = True
    host.send_cs(cs_index)
    assert peer.sent == [(host._counter_cs_for(cs_index), False)]


@pytest.mark.parametrize("answering", [False, True])
def test_the_renderer_changes_only_on_the_changeover_cycle(answering):
    host, peer = _host()
    host._p3_changeover_answer = answering
    host.send_cs(arq.CS_ACK)
    assert peer.sent == [(host._counter_cs_for(arq.CS_ACK), answering)]


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_requires_explicit_opt_in(monkeypatch, enabled):
    monkeypatch.setattr(sys, "argv", ["onair", "--dxcall", "WS8EOC"]
                        + (["--p3-changeover-p1-cs"] if enabled else []))
    monkeypatch.setattr(onair.config, "mail_password", lambda *args: None)
    observed = []
    monkeypatch.setattr(onair, "run",
                        lambda args: observed.append(args.p3_changeover_p1_cs) or 0)
    assert onair.main() == 0
    assert observed == [enabled]
