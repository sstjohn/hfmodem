# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A PACTOR-1 packet is keyed 10 ms past its last bit; nothing else is.

The 1213 witness saw our RF end 1.2-2.4 ms before the 960 ms packet did, which
half-cuts the last CRC bit at 200 Bd.
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import onair, pactor1

TAIL_N = round(0.010 * onair.FS)


def _keyed(tmp_path, monkeypatch, send):
    keyed = []
    monkeypatch.setattr(onair.session, "write_wav",
                        lambda path, audio: keyed.append(np.asarray(audio)))
    send(onair.RadioTx(None, transmit=False, outdir=tmp_path, drive=0.8))
    assert len(keyed) == 1
    return keyed[0]


def _bare(audio):
    return len(onair._trim_silence(audio))


def test_p1_packet_holds_the_key_ten_ms_past_the_last_bit(tmp_path, monkeypatch):
    for baud in (100, 200):
        audio = _keyed(tmp_path, monkeypatch,
                       lambda tx: tx.send_p1_packet(b"1W9SSJ\r", baud, 1))
        bare = _bare(pactor1.packet_signal(b"1W9SSJ\r", baud))
        assert len(audio) == bare + TAIL_N
        assert audio[bare - 1] != 0
        assert not audio[bare:].any()


def test_p1_breakin_holds_the_key_too(tmp_path, monkeypatch):
    audio = _keyed(tmp_path, monkeypatch,
                   lambda tx: tx.send_p1_breakin(b"", 200, 0))
    assert len(audio) == _bare(pactor1.breakin_signal(b"", 200)) + TAIL_N


def test_p1_control_signal_and_p3_keep_their_length(tmp_path, monkeypatch):
    cs = _keyed(tmp_path, monkeypatch, lambda tx: tx.send_p1_cs(0))
    assert len(cs) == _bare(pactor1.control_signal(0))
    for send in (lambda tx: tx.send_entry_packet(1, b"", 0x1A),
                 lambda tx: tx.send_cs(0)):
        audio = _keyed(tmp_path, monkeypatch, send)
        assert audio[-1] != 0


def test_1213_tx6_to_tx8_duplex_tail_crc_polarity_and_peer_raster(tmp_path, monkeypatch):
    from hfmodem.shrike import p1rx, spec
    from hfmodem.tests.shrike.test_p3_breakin_timing import duplex
    fs = onair.FS
    grid = onair._MasterGrid(round(7.555*fs), 60000, 0, packet_n=46080,
                             cs_n=5760, d_max_n=6240)
    tx = duplex(grid, tmp_path, seconds=16)
    tx.host.protocol = spec.Protocol.PACTOR1
    tx.p1_status_bits45 = 2
    tx.p1_status_from = 1
    waves = []
    transmit = tx.live.transmit
    def capture(audio, **kw):
        waves.append(audio.copy())
        return transmit(audio, **kw)
    monkeypatch.setattr(tx.live, 'transmit', capture)
    monkeypatch.setattr(onair.session, 'write_wav', lambda *a: None)
    for slot, inverted, gap in ((0, False, .072), (1, True, .0732), (3, True, .072)):
        tx.aim(grid, slot)
        monkeypatch.setattr(tx, '_flip', lambda: inverted)
        tx.send_p1_packet(b'1w9ssj\r', 200, 1)
        start, end = tx.live.emissions[-1]
        assert start == round(7.555*fs)+slot*60000
        wave = waves[-1]
        signal_n = tx.live.signal_ends[-1]-start
        assert end-start == signal_n+480
        # The pulse taper may remove less than one bit, never a terminal symbol.
        assert signal_n == 46080
        packets = p1rx.decode_p1_packets(np.pad(wave, (4800, 4800)), fs)
        assert len(packets) == 1
        packet = packets[0]
        assert (packet.payload, packet.status, packet.baud, packet.inverted) == (
            b'1w9ssj\r', 0x21, 200, inverted)
        peer_at = start+46080+round(gap*fs)
        peer = pactor1.control_signal(pactor1.CS_ACK_A, invert=not inverted)
        received = np.pad(peer, (peer_at-start, 4800))
        cs = p1rx.decode_control_signal(received, (peer_at-start)/fs, .120)
        assert cs is not None and cs[0] == pactor1.CS_ACK_A
        assert cs.sense == (not inverted)
        assert peer_at-end == 46080-signal_n+round(gap*fs)-480
    assert tx.rig.edges == [True, False]*3
