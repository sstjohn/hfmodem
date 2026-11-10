# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The default P3 entry restores 234 trimmed samples; explicit zero is retained.

The actual DAC start changes, while waveform samples, slot selection and the
ordinary transmission guards remain intact. Arbitrary bounded delays still work.

Run:  pytest hfmodem/tests/shrike/test_entry_delay.py
"""
from __future__ import annotations

import sys
from unittest import mock

import numpy as np

from hfmodem.shrike import onair, placement, spec
from hfmodem.tests.shrike.test_grid import FS, SLOT_N, _Bench, _Rig

ENTRY_STATUS = 0x1A         # the reference entry packet's own status byte
P1_PACKET_N = round(spec.P1_PACKET_S * FS)
P1_CS_N = round(spec.P1_CS_S * FS)

#: An arbitrary delay that is not the default, so a test passing on the default
#: alone cannot pass by accident.
PROBE_MS = 17.5

BASE = ["onair", "--dxcall", "WS8EOC", "--mycall", "W9SSJ"]


def _parsed(*argv: str):
    """The namespace `run` is handed, straight off the real parser."""
    got = []
    with mock.patch.object(onair, "run", lambda a: got.append(a) or 0), \
            mock.patch.object(sys, "argv", BASE + list(argv)):
        onair.main()
    return got[0]


def _rendered(tmp_path, monkeypatch, delay_ms: float) -> np.ndarray:
    """The samples a dry run would key, at this delay."""
    out: list[np.ndarray] = []
    monkeypatch.setattr(onair.session, "write_wav",
                        lambda path, audio: out.append(np.asarray(audio)))
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path, drive=0.8)
    tx.entry_delay_n = round(delay_ms / 1e3 * FS)
    tx.send_entry_packet(1, b"", ENTRY_STATUS)
    assert len(out) == 1
    return out[0]


def _keyed(tmp_path, delay_ms: float | None = None, slot: int = 4) -> dict:
    """Key one entry packet against the duplex bench and report where it landed."""
    bench = _Bench(seconds=40.0)
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path, drive=0.8)
    tx.live = bench
    if delay_ms is not None:
        tx.entry_delay_n = round(delay_ms / 1e3 * FS)
    raster = onair._MasterGrid(0, SLOT_N, round(onair.TX_OFFSET_S * FS),
                               packet_n=P1_PACKET_N, cs_n=P1_CS_N,
                               d_max_n=round(0.13 * FS))
    tx.aim(raster, slot)
    tx.send_entry_packet(1, b"", ENTRY_STATUS)
    return {"emissions": bench.emissions, "slot": tx.slot,
            "refused": tx.refused, "boundary": raster.boundary(slot)}


def test_default_restores_trimmed_entry_epoch(tmp_path) -> None:
    assert onair.RadioTx.entry_delay_n == 234
    assert _parsed().p3_entry_delay == 4.875
    got = _keyed(tmp_path)
    assert not got['refused'] and len(got['emissions']) == 1
    assert got['emissions'][0][0]-got['boundary'] == 234
    assert _parsed('--p3-entry-delay', '0').p3_entry_delay == 0.0


def test_a_bare_flag_uses_the_default_delay() -> None:
    assert _parsed("--p3-entry-delay").p3_entry_delay == 4.875
    assert _parsed("--p3-entry-delay", str(PROBE_MS)).p3_entry_delay == PROBE_MS


def test_the_packet_is_the_same_packet(tmp_path, monkeypatch) -> None:
    flush = _rendered(tmp_path, monkeypatch, 0.0)
    delayed = _rendered(tmp_path, monkeypatch, PROBE_MS)
    assert delayed.tobytes() == flush.tobytes()


def test_off_keys_where_it_always_keyed(tmp_path) -> None:
    got = _keyed(tmp_path, 0.0)
    assert len(got["emissions"]) == 1
    assert got["emissions"][0][0] == got["boundary"]
    assert not got["refused"]


def test_on_moves_the_carrier_by_exactly_the_delay(tmp_path) -> None:
    flush, delayed = _keyed(tmp_path, 0.0), _keyed(tmp_path, PROBE_MS)
    (a, a_end), (b, b_end) = delayed["emissions"][0], flush["emissions"][0]
    assert a - b == round(PROBE_MS / 1e3 * FS)
    assert a_end - a == b_end - b       # the same audio, moved whole


def test_a_delayed_entry_is_keyed_and_not_deferred(tmp_path) -> None:
    flush, delayed = _keyed(tmp_path, 0.0), _keyed(tmp_path, PROBE_MS)
    for got in (flush, delayed):
        assert len(got["emissions"]) == 1, "the burst went out"
        assert not got["refused"]
    # NOT ONE SLOT ON. A delay that cost the slot it was aimed at would read as a
    # success here -- the carrier moved, the packet was keyed -- while spending a
    # cycle of the grant budget every time.
    assert delayed["slot"] == flush["slot"]


def test_the_measured_default_is_admitted_too(tmp_path) -> None:
    got = _keyed(tmp_path, placement.ENTRY_DELAY_S * 1e3)
    assert len(got["emissions"]) == 1 and not got["refused"]
    assert (got["emissions"][0][0] - got["boundary"]
            == round(placement.ENTRY_DELAY_S * FS))
